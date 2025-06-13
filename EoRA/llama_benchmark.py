# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import time

import torch
import torch.nn as nn

from gptq import *
from modelutils import *
from quant import *
import math
import quant_cuda

def get_llama(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(model, torch_dtype='auto')
    model.seqlen = 2048
    return model

class Quant3Linear_dummy(nn.Module): 

    def __init__(self, infeatures, outfeatures, faster=False):
        super().__init__()
        self.register_buffer('zeros', torch.zeros((outfeatures, 1)))
        self.register_buffer('scales', torch.zeros((outfeatures, 1)))
        self.register_buffer('bias', torch.zeros(outfeatures))
        self.register_buffer(
            'qweight', torch.zeros((infeatures // 32 * 3, outfeatures), dtype=torch.int)
        )
        self.faster = faster

    def forward(self, x):
        # print(x.shape)
        # if x.shape[-1] == x.numel():
        outshape = list(x.shape)
        y = self.bias.clone()
        outshape[-1] = self.bias.numel()
        dtype = x.dtype

        x = x.half()
        quant_cuda.vecquant3matmul_faster(x, self.qweight, y, self.scales, self.zeros)

        y = y.to(dtype)
        return y.reshape(outshape)
        # raise ValueError('Only supports a single token currently.')

class Quant3Linear_normal_eora_dummy(nn.Module): 

    def __init__(self, infeatures, outfeatures, faster=True, rank=64):
        super().__init__()
        self.register_buffer('zeros', torch.zeros((outfeatures, 1)))
        self.register_buffer('scales', torch.zeros((outfeatures, 1)))
        self.register_buffer('bias', torch.zeros(outfeatures))
        self.register_buffer(
            'qweight', torch.zeros((infeatures // 32 * 3, outfeatures), dtype=torch.int)
        )
        self.faster = faster
        self.register_buffer('down', torch.zeros((infeatures, rank), dtype=torch.half))
        self.register_buffer('up', torch.zeros((rank, outfeatures), dtype=torch.half))
        self.rank = rank
        self.register_buffer('down_proj', torch.zeros((1, self.rank), dtype=torch.half))
        


    def forward(self, x):
        if x.shape[-1] == x.numel():
            outshape = list(x.shape)
            y = self.bias.clone()
            outshape[-1] = self.bias.numel()
            dtype = x.dtype

            x = x.half()
            quant_cuda.vecquant3matmul_faster(x, self.qweight, y, self.scales, self.zeros)
            
            torch.matmul(x, self.down, out=self.down_proj)
            y += (self.down_proj @ self.up).flatten()

            y = y.to(dtype)
            return y.reshape(outshape)
        raise ValueError('Only supports a single token currently.')

class Quant3Linear_fused_eora_dummy(nn.Module): 

    def __init__(self, infeatures, outfeatures, faster=True, rank=64):
        super().__init__()
        self.register_buffer('zeros', torch.zeros((outfeatures, 1)))
        self.register_buffer('scales', torch.zeros((outfeatures, 1)))
        self.register_buffer('bias', torch.zeros(outfeatures))
        self.register_buffer(
            'qweight', torch.zeros((infeatures // 32 * 3, outfeatures), dtype=torch.int)
        )
        self.faster = faster
        self.register_buffer('down', torch.zeros((infeatures, rank), dtype=torch.half))
        self.register_buffer('up', torch.zeros((rank, outfeatures), dtype=torch.half))
        self.rank = rank
        self.register_buffer('down_proj', torch.zeros((1, self.rank), dtype=torch.half))
        
    def forward(self, x):
        if x.shape[-1] == x.numel():
            outshape = list(x.shape)
            y = self.bias.clone()
            outshape[-1] = self.bias.numel()
            dtype = x.dtype

            x = x.half()

            torch.matmul(x, self.down, out=self.down_proj)
            quant_cuda.vecquant3matmul_lora_faster(
                x, self.qweight, self.down_proj, self.up, y, self.scales, self.zeros
            )
            y = y.to(dtype)
            return y.reshape(outshape)
        

        raise ValueError('Only supports a single token currently.')

def make_quant3_llama(module, names, name='', faster=False):
    if isinstance(module, Quant3Linear_dummy):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        name1 = name + '.' + attr if name != '' else attr
        if name1 in names:
            setattr(
                module, attr, Quant3Linear_dummy(tmp.in_features, tmp.out_features, faster=faster)
            )
    for name1, child in module.named_children():
        make_quant3_llama(child, names, name + '.' + name1 if name != '' else name1, faster=faster)

def make_quant3_normal_eora_llama(module, names, name='', faster=False):
    if isinstance(module, Quant3Linear_normal_eora_dummy):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        name1 = name + '.' + attr if name != '' else attr
        if name1 in names:
            setattr(
                module, attr, Quant3Linear_normal_eora_dummy(tmp.in_features, tmp.out_features, faster=faster, rank=args.rank)
            )
    for name1, child in module.named_children():
        make_quant3_normal_eora_llama(child, names, name + '.' + name1 if name != '' else name1, faster=faster)

def make_quant3_fused_eora_llama(module, names, name='', faster=False):
    if isinstance(module, Quant3Linear_fused_eora_dummy):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        name1 = name + '.' + attr if name != '' else attr
        if name1 in names:
            setattr(
                module, attr, Quant3Linear_fused_eora_dummy(tmp.in_features, tmp.out_features, faster=faster, rank=args.rank)
            )
    for name1, child in module.named_children():
        make_quant3_fused_eora_llama(child, names, name + '.' + name1 if name != '' else name1, faster=faster)

def load_quant3_llama(model):
    from transformers import LlamaConfig, LlamaForCausalLM
    config = LlamaConfig.from_pretrained(model)
    def noop(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = noop 
    torch.nn.init.uniform_ = noop 
    torch.nn.init.normal_ = noop 

    torch.set_default_dtype(torch.half)
    transformers.modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)
    model = LlamaForCausalLM(config)
    torch.set_default_dtype(torch.float)
    model = model.eval()
    layers = find_layers(model)
    for name in ['lm_head']:
        if name in layers:
            del layers[name]
    #  '3_bit', '3_bit_wo_fuse', '3_bit_fuse',
    if args.inference_type == '3_bit':
        make_quant3_llama(model, layers, faster=args.faster_kernel)
    elif args.inference_type == '3_bit_wo_fuse':
        make_quant3_normal_eora_llama(model, layers, faster=args.faster_kernel)
    elif args.inference_type == '3_bit_fuse':
        make_quant3_fused_eora_llama(model, layers, faster=args.faster_kernel)
    print(model)
    model.seqlen = model.config.max_position_embeddings
    print('Done.')

    return model

class Quant4Linear_dummy(nn.Module): 

    def __init__(self, infeatures, outfeatures, faster=False):
        super().__init__()
        self.register_buffer('zeros', torch.zeros((1, outfeatures)))
        self.register_buffer('scales', torch.zeros((1, outfeatures)))
        self.register_buffer('bias', torch.zeros(outfeatures))
        self.register_buffer(
            'qweight', torch.zeros((infeatures // 32 * 3, outfeatures), dtype=torch.int)
        )
        self.faster = faster

    def forward(self, x):
        outshape = list(x.shape)
        y = self.bias.clone()
        outshape[-1] = self.bias.numel()
        dtype = x.dtype

        x = x.float()

        quant_cuda.vecquant4matmul(x, self.qweight, y, self.scales, self.zeros)

        y = y.to(dtype)
        return y.reshape(outshape)

class Quant4Linear_normal_eora_dummy(nn.Module): 

    def __init__(self, infeatures, outfeatures, faster=True, rank=64):
        super().__init__()
        self.register_buffer('zeros', torch.zeros((1, outfeatures)))
        self.register_buffer('scales', torch.zeros((1, outfeatures)))
        self.register_buffer('bias', torch.zeros(outfeatures))
        self.register_buffer(
            'qweight', torch.zeros((infeatures // 32 * 3, outfeatures), dtype=torch.int)
        )
        self.faster = faster
        self.register_buffer('down', torch.zeros((infeatures, rank), dtype=torch.half))
        self.register_buffer('up', torch.zeros((rank, outfeatures), dtype=torch.half))
        self.rank = rank
        self.register_buffer('down_proj', torch.zeros((1, self.rank), dtype=torch.half))


    def forward(self, x):
        if x.shape[-1] == x.numel():
            outshape = list(x.shape)
            y = self.bias.clone()
            outshape[-1] = self.bias.numel()
            dtype = x.dtype

            x = x.float()
            xh = x.half()
            quant_cuda.vecquant4matmul(x, self.qweight, y, self.scales, self.zeros)
            
            torch.matmul(xh, self.down, out=self.down_proj)
            y += (self.down_proj @ self.up).flatten()

            y = y.to(dtype)
            return y.reshape(outshape)
        raise ValueError('Only supports a single token currently.')

class Quant4Linear_fused_eora_dummy(nn.Module): 

    def __init__(self, infeatures, outfeatures, faster=True, rank=64):
        super().__init__()
        self.register_buffer('zeros', torch.zeros((outfeatures, 1)))
        self.register_buffer('scales', torch.zeros((outfeatures, 1)))
        self.register_buffer('bias', torch.zeros(outfeatures))
        self.register_buffer(
            'qweight', torch.zeros((infeatures // 32 * 3, outfeatures), dtype=torch.int)
        )
        self.faster = faster
        self.register_buffer('down', torch.zeros((infeatures, rank), dtype=torch.half))
        self.register_buffer('up', torch.zeros((rank, outfeatures), dtype=torch.half))
        self.rank = rank
        self.register_buffer('down_proj', torch.zeros((1, self.rank), dtype=torch.half))

    def forward(self, x):
        if x.shape[-1] == x.numel():
            outshape = list(x.shape)
            y = self.bias.clone()
            outshape[-1] = self.bias.numel()
            dtype = x.dtype

            x = x.float()
            xh = x.half()
            torch.matmul(xh, self.down, out=self.down_proj)
            quant_cuda.vecquant4matmul_lora(
                x, self.qweight, self.down_proj, self.up, y, self.scales, self.zeros
            )
            y = y.to(dtype)
            return y.reshape(outshape)
        

        raise ValueError('Only supports a single token currently.')

def make_quant4_llama(module, names, name='', faster=False):
    if isinstance(module, Quant4Linear_dummy):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        name1 = name + '.' + attr if name != '' else attr
        if name1 in names:
            setattr(
                module, attr, Quant4Linear_dummy(tmp.in_features, tmp.out_features, faster=faster)
            )
    for name1, child in module.named_children():
        make_quant4_llama(child, names, name + '.' + name1 if name != '' else name1, faster=faster)

def make_quant4_normal_eora_llama(module, names, name='', faster=False):
    if isinstance(module, Quant4Linear_normal_eora_dummy):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        name1 = name + '.' + attr if name != '' else attr
        if name1 in names:
            setattr(
                module, attr, Quant4Linear_normal_eora_dummy(tmp.in_features, tmp.out_features, faster=faster, rank=args.rank)
            )
    for name1, child in module.named_children():
        make_quant4_normal_eora_llama(child, names, name + '.' + name1 if name != '' else name1, faster=faster)

def make_quant4_fused_eora_llama(module, names, name='', faster=False):
    if isinstance(module, Quant4Linear_fused_eora_dummy):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        name1 = name + '.' + attr if name != '' else attr
        if name1 in names:
            setattr(
                module, attr, Quant4Linear_fused_eora_dummy(tmp.in_features, tmp.out_features, faster=faster, rank=args.rank)
            )
    for name1, child in module.named_children():
        make_quant4_fused_eora_llama(child, names, name + '.' + name1 if name != '' else name1, faster=faster)

def load_quant4_llama(model):
    from transformers import LlamaConfig, LlamaForCausalLM
    config = LlamaConfig.from_pretrained(model)
    def noop(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = noop 
    torch.nn.init.uniform_ = noop 
    torch.nn.init.normal_ = noop 

    torch.set_default_dtype(torch.half)
    transformers.modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)
    model = LlamaForCausalLM(config)
    torch.set_default_dtype(torch.float)
    model = model.eval()
    layers = find_layers(model)
    for name in ['lm_head']:
        if name in layers:
            del layers[name]
    if args.inference_type == '4_bit':
        make_quant4_llama(model, layers, faster=args.faster_kernel)
    elif args.inference_type == '4_bit_wo_fuse':
        make_quant4_normal_eora_llama(model, layers, faster=args.faster_kernel)
    elif args.inference_type == '4_bit_fuse':
        make_quant4_fused_eora_llama(model, layers, faster=args.faster_kernel)
    print(model)
    model.seqlen = model.config.max_position_embeddings
    print('Done.')

    return model

def llama_multigpu(model, gpus):
    model.model.embed_tokens = model.model.embed_tokens.to(gpus[0])
    
    model.model.norm = model.model.norm.to(gpus[-1])
    import copy
    model.lm_head = copy.deepcopy(model.lm_head).to(gpus[-1])

    cache = {'mask': None, 'position_ids': None}

    class MoveModule(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.dev = next(iter(self.module.parameters())).device
        def forward(self, *inp, **kwargs):
            inp = list(inp)
            if inp[0].device != self.dev:
                inp[0] = inp[0].to(self.dev)
            if kwargs['attention_mask'] != None:
                if cache['mask'] is None or cache['mask'].device != self.dev:
                    cache['mask'] = kwargs['attention_mask'].to(self.dev)
                kwargs['attention_mask'] = cache['mask']
            
            if kwargs['position_ids'].device != self.dev:
                if cache['position_ids'] is None or cache['position_ids'].device != self.dev:
                    cache['position_ids'] = kwargs['position_ids'].to(self.dev)
                kwargs['position_ids'] = cache['position_ids']

            tmp = self.module(*inp, **kwargs)
            return tmp

    layers = model.model.layers
    pergpu = math.ceil(len(layers) / len(gpus))
    for i in range(len(layers)):
        layers[i] = MoveModule(layers[i].to(gpus[i // pergpu]))

    model.gpus = gpus
    print(model)

def benchmark_llama(model, input_ids, check=False):
    input_ids = input_ids.to(model.gpus[0] if hasattr(model, 'gpus') else DEV)
    torch.cuda.synchronize()

    cache = {'past': None}
    def clear_past(i):
        def tmp(layer, inp, out):
            if cache['past']:
                cache['past'][i] = None
        return tmp
    for i, layer in enumerate(model.model.layers):
        layer.register_forward_hook(clear_past(i))

    print('Benchmarking ...')

    if check:
        loss = nn.CrossEntropyLoss()
        tot = 0.

    def sync():
        if hasattr(model, 'gpus'):
            for gpu in model.gpus:
                torch.cuda.synchronize(gpu)
        else:
            torch.cuda.synchronize()
    with torch.no_grad():
        attention_mask = torch.ones((1, input_ids.numel()), device=DEV)
        print(f"attention_mask: {attention_mask[:, :(i + 1)].reshape((1, -1))}")
        times = []
        for i in range(input_ids.numel()):
            tick = time.time()
            out = model(
                input_ids[:, i].reshape((1,-1)),
                past_key_values=cache['past'],
                attention_mask=attention_mask[:, :(i + 1)].reshape((1, -1))
            )
            sync()
            times.append(time.time() - tick)
            print(i, times[-1])
            if check and i != input_ids.numel() - 1:
                tot += loss(out.logits[0].to(DEV), input_ids[:, (i + 1)].to(DEV)).float()
            cache['past'] = list(out.past_key_values)
            del out
        sync()
        import numpy as np
        print('Median:', np.median(times))
        if check:
            print('PPL:', torch.exp(tot / (input_ids.numel() - 1)).item())

if __name__ == '__main__':
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='OPT model to load; pass `facebook/opt-X`.'
    )
    parser.add_argument(
        'dataset', type=str, choices=['wikitext2'],
        help='Where to extract calibration data from.'
    )
    parser.add_argument(
        '--seed',
        type=int, default=0, help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--rank',
        type=int, default=0, help='rank of eora'
    )
    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        '--benchmark', type=int, default=0,
        help='Number of tokens to use for benchmarking.'
    )
    parser.add_argument(
        '--inference_type', type=str, choices=['full_precision', '3_bit', '3_bit_wo_fuse', '3_bit_fuse', '4_bit', '4_bit_wo_fuse', '4_bit_fuse'],
    )

    args = parser.parse_args()

    if args.inference_type == 'full_precision':
        model = get_llama(args.model)
    elif "3_bit" in args.inference_type:
        model = load_quant3_llama(args.model)
    elif "4_bit" in args.inference_type:
        model = load_quant4_llama(args.model)
    model.eval()


    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    )


    if args.benchmark:
        gpus = [torch.device('cuda:%d' % i) for i in range(torch.cuda.device_count())]
        if len(gpus) > 1:
            llama_multigpu(model, gpus)
            print("triggered multi gpu")
        else:
            model = model.to(DEV)
        if args.benchmark:
            input_ids = next(iter(dataloader))[0][:, :args.benchmark]
            benchmark_llama(model, input_ids)