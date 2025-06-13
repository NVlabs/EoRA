import time

import torch
import torch.nn as nn
from wanda import *
from sparsegpt import *
from modelutils import *
from quant import *
from evaluate_utils import evaluate_model
from gptq import * 
from safetensors.torch import save_file
import json
import tqdm
from torch import device

CPU = device("cpu")
CUDA_0 = device("cuda:0")

try:
    import wandb
    has_wandb = True
except:
    has_wandb = False


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

@torch.no_grad()
def llama_sequential_gptq(model, dataloader, dev):
    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    quantizers = {}
    quantized_weights = {}
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        full = find_layers(layer)

        if args.true_sequential:
            sequential = [
                ['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj'],
                ['self_attn.o_proj'],
                ['mlp.up_proj', 'mlp.gate_proj'],
                ['mlp.down_proj']
            ]
        else:
            sequential = [list(full.keys())]
       
        for names in sequential:
            subset = {n: full[n] for n in names}

            gptq = {}
            for name in subset:
                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer = Quantizer()
                gptq[name].quantizer.configure(
                    args.wbits, perchannel=True, sym=args.sym, mse=False
                )

            def add_batch(name):
                def tmp(_, inp, out):
                    # print(f"gptq inp[0] {inp[0].shape}")
                    # print(f"gptq inp[0][0] {inp[0][0]}")
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp
            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            for h in handles:
                h.remove()

            for name in subset:
                print(i, name)
                print('Quantizing ...')
                quantize_weight = gptq[name].fasterquant_wo_replcaing_weight(
                    percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order, static_groups=args.static_groups
                )
                gptq[name].layer.weight.data = quantize_weight

                quantized_weights['model.layers.%d.%s' % (i, name)] = quantize_weight.cpu()

                quantizers['model.layers.%d.%s' % (i, name)] = gptq[name].quantizer
                gptq[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer.cpu()
        del layer
        del gptq 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    
    return quantizers, quantized_weights

@torch.no_grad()
def llama_sequential_sparsegpt(model, dataloader, dev):
    print("Starting...")

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    # print(f"attention_mask {attention_mask}")
    print("Ready.")

    quantizers = {}
    pruned_weights = {}
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        full = find_layers(layer)

        if args.true_sequential:
            sequential = [
                ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
                ["self_attn.o_proj"],
                ["mlp.up_proj", "mlp.gate_proj"],
                ["mlp.down_proj"],
            ]
        else:
            sequential = [list(full.keys())]

        for names in sequential:
            subset = {n: full[n] for n in names}

            gpts = {}
            for name in subset:
                if (
                    not (args.minlayer <= i < args.maxlayer and args.prune_only in name)
                ) == (not args.invert):
                    continue
                gpts[name] = SparseGPT(subset[name])
                if args.wbits < 16:
                    gpts[name].quantizer = Quantizer()
                    gpts[name].quantizer.configure(
                        args.wbits, perchannel=True, sym=False, mse=False
                    )

            def add_batch(name):
                def tmp(_, inp, out):
                    gpts[name].add_batch(inp[0].data, out.data)

                return tmp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
            for h in handles:
                h.remove()

            for name in subset:
                print(i, name)
                print("Pruning ...")
                sparsity = args.sparsity
                pruned_weight = gpts[name].fasterquant_wo_replcaing_weight(
                    sparsity,
                    prunen=args.prunen,
                    prunem=args.prunem,
                    percdamp=args.percdamp,
                    blocksize=args.blocksize,
                )

                gpts[name].layer.weight.data = pruned_weight
                pruned_weights['model.layers.%d.%s' % (i, name)] = pruned_weight.cpu()
                gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        layers[i] = layer.cpu()
        del layer
        del gpts
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache

    return quantizers, pruned_weights


@torch.no_grad()
def llama_sequential_wanda(model, dataloader, dev):
    print("Starting wanda...")
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps).to(dev)
    attention_mask = cache["attention_mask"]
    
    print("Ready.")

    
    pruned_weights = {}
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        subset = find_layers(layer)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False

            args.prunen
            if args.prunen != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % args.prunem == 0:
                        tmp = W_metric[:,ii:(ii+args.prunem)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, args.prunen,dim=1, largest=False)[1], True)
            else:
                print(f"unstructured pruning {args.sparsity * 100.0}%")
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                # unstructured pruning
                indices = sort_res[1][:,:int(W_metric.shape[1]*args.sparsity)]
                W_mask.scatter_(1, indices, True)

            subset[name].weight.data[W_mask] = 0  ## set weights to zero 
            pruned_weights['model.layers.%d.%s' % (i, name)] = subset[name].weight.data.cpu()

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        
        layers[i] = layer.cpu()
        del layer
        del wrapped_layers
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    model.config.use_cache = use_cache

    return pruned_weights, pruned_weights

@torch.no_grad()
def llama_sequential_eigen(model, dataloader, compressed_weights, dev):
    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.eigen_nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )

    ## this only apply to normal attention (flash attention will require different shape)
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError
        
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']

    print('Ready.')
    lowrank_dict = {}
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        full = find_layers(layer)
        
        sequential = [list(full.keys())]
       
        for names in sequential:
            subset = {n: full[n] for n in names}
            
            subset_eigen_scaling_diag_matrix = {}
            for name in subset:
                subset_eigen_scaling_diag_matrix[name] = 0

            def hook(name):

                def tmpp(_, input, output):
                    inp = input[0].detach().float()
                    if inp.dim() == 2:
                        inp = inp.unsqueeze(0)
                    tmp = inp.shape[0]
                    adds = torch.matmul(inp.transpose(1,2), inp)
                    adds_sum = torch.sum(adds, dim=0)
                    subset_eigen_scaling_diag_matrix[name] *= args.eigen_nsamples / (args.eigen_nsamples+tmp)
                    
                    subset_eigen_scaling_diag_matrix[name] += adds_sum / args.eigen_nsamples

                    del inp, adds, adds_sum, output
                    torch.cuda.empty_cache()
                return tmpp
            
            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(hook(name)))

            for j in range(args.eigen_nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
            for h in handles:
                h.remove()

            for name in subset:
                print(i, name)
                print('Start eigen projection ...')
                original_weight = subset[name].weight.data
                
                compressed_weight = compressed_weights['model.layers.%d.%s' % (i, name)].to(dev)

                delta = original_weight - compressed_weight

                raw_scaling_diag_matrix = subset_eigen_scaling_diag_matrix[name].double().to("cuda")
                
                L, Q = torch.linalg.eigh(raw_scaling_diag_matrix)
                if (L < 0).any().item():
                    print(f"found negative eigenvalues in {name}")
                    minimum = torch.min(L[L > 0])
                    L[L < 0] = minimum

                sqrtEigenvalues = torch.sqrt(L)
                scaling_diag_matrix = Q @ torch.diag(sqrtEigenvalues)
                scaling_matrix_inv = torch.diag(1/sqrtEigenvalues) @ Q.T

                scaling_diag_matrix = scaling_diag_matrix.float()
                scaling_matrix_inv = scaling_matrix_inv.float()
                
                delta_scale = torch.matmul(delta.to(torch.float32), scaling_diag_matrix)

                r=args.eigen_r

                U, S, VT = torch.linalg.svd(delta_scale, full_matrices=False)
                num_s_after_trunc = r
                truc_s = S[:num_s_after_trunc]
                truc_u = U[:, :num_s_after_trunc]
                truc_v = torch.matmul(VT[:num_s_after_trunc, :], scaling_matrix_inv)
                truc_sigma = torch.diag(truc_s)
                
                sqrtSigma = torch.sqrt(truc_sigma)
                B = torch.matmul(truc_u, sqrtSigma).to(compressed_weight.dtype)
                A = torch.matmul(sqrtSigma, truc_v).to(compressed_weight.dtype)

                comp_weight = compressed_weight + B@A

                subset[name].weight.data = comp_weight.to(subset[name].weight.data.dtype)
                
                lowrank_dict[f'base_model.model.model.layers.{i}.{name}.lora_A.weight'] = A.cpu()
                lowrank_dict[f'base_model.model.model.layers.{i}.{name}.lora_B.weight'] = B.cpu()
                del B, A, compressed_weight, U, S, VT, L, Q

        for j in range(args.eigen_nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]


        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache

    return lowrank_dict

@torch.no_grad()
def llama_sequential_svd(model, compressed_weights, dev):
    print('Starting svd compensation...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    dtype = next(iter(model.parameters())).dtype
    
    lowrank_dict = {}

    for i in range(len(layers)):
        layer = layers[i].to(dev)
        full = find_layers(layer)


        sequential = [list(full.keys())]
       
        for names in sequential:
            subset = {n: full[n] for n in names}


            for name in subset:
                print(i, name)
                print('SVD compensation ...')
                original_weight = subset[name].weight.data
                quantize_weight = compressed_weights['model.layers.%d.%s' % (i, name)].to(dev)


                delta = original_weight - quantize_weight

                ##
                r=args.eigen_r
                U, S, VT = torch.linalg.svd(delta.to(torch.float32), full_matrices=False)
                num_s_after_trunc = r
                truc_s = S[:num_s_after_trunc]
                truc_u = U[:, :num_s_after_trunc]
                truc_v = VT[:num_s_after_trunc, :]
                truc_sigma = torch.diag(truc_s)
                #### Replace Attn, MLP ####
                sqrtSigma = torch.sqrt(truc_sigma)
                B = torch.matmul(truc_u, sqrtSigma).to(quantize_weight.dtype)
                A = torch.matmul(sqrtSigma, truc_v).to(quantize_weight.dtype)

                final_weight = quantize_weight + B @ A
                subset[name].weight.data = final_weight.to(subset[name].weight.data.dtype)
                lowrank_dict[f'base_model.model.model.layers.{i}.{name}.lora_A.weight'] = A.cpu()
                lowrank_dict[f'base_model.model.model.layers.{i}.{name}.lora_B.weight'] = B.cpu()
                del B, A, original_weight





        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()

    model.config.use_cache = use_cache
    return lowrank_dict

def optimize_BA(weight_original, compressed_weight, data=None, residual_rank=64, 
                   optim_iterations=1000,  lr=1.0e-3, device = "cuda", opt_weights=None,
                    total_global_iters=1):

    device = device

    weight = weight_original.clone()
                
    B = torch.nn.Parameter(torch.empty(weight.shape[0], residual_rank, device=device, dtype=torch.float32))
    A = torch.nn.Parameter(torch.empty(residual_rank, weight.shape[1], device=device, dtype=torch.float32))

    torch.nn.init.xavier_normal_(A)
    torch.nn.init.zeros_(B)
    
    params_to_optimize = [A, B]

    for iters in range(total_global_iters):
        # $ first do with learnable mask

        optimizer = torch.optim.AdamW(params_to_optimize, lr=lr, weight_decay=0.0)

        #create cosine learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=optim_iterations, eta_min=1e-6)

        for step in range(optim_iterations):
            #decrease temperature for Gumbel softmax
            
            # convert the code to closure function
            def closure():
                optimizer.zero_grad()
                
              
                #forward pass
                with torch.cuda.amp.autocast(enabled=True):
                    
                    
                    all_zero = True
                    while all_zero == True:
                        random_sample = data[torch.randint(0, len(data), (1,))]
                        all_zero = torch.all(random_sample == 0.0) 

                    random_sample = random_sample.to(device)
   
                    with torch.no_grad():
                        output_teacher = random_sample @ weight_original.T
                        
                    output = random_sample @ (compressed_weight + B @ A).T
                    
                    loss = torch.norm(output_teacher - output, p=2).mean()
         
                    
                loss.backward()
                    
                divider = 500
                if step % divider == 0 or step == optim_iterations-1:
                    print(f"iters {iters} step {step} loss {loss.item():.3f}")
                    
                del random_sample
                
                return loss
                
            
            optimizer.step(closure)
            scheduler.step()
            
                
        del optimizer    
        torch.cuda.empty_cache()

    # switch to eval model to remove Gumbel noise

    
    new_weight = (compressed_weight + B @ A).clone()
    
    return new_weight, B, A

@torch.no_grad()
def llama_sequential_learn_BA(model, dataloader, compressed_weights, dev):
    print("Starting...")

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]

    print("Ready.")

    lowrank_dict = {}
    for i in range(len(layers)):

        layer = layers[i].to(dev)
        full = find_layers(layer)
        print(full)
        if args.true_sequential:
            sequential = [
                ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
                ["self_attn.o_proj"],
                ["mlp.up_proj", "mlp.gate_proj"],
                ["mlp.down_proj"],
            ]
        else:
            sequential = [list(full.keys())]
        for names in sequential:
            subset = {n: full[n] for n in names}

            inps_cache = {}
            for name in subset:
                inps_cache[name] = []

            # for name in subset:  
            #         if name in gpts: gpts[name].inps = []

            def add_batch(name):
                def tmp(_, inp, out):
                    ## 
                    inps_cache[name].append(inp[0].data.half())

                return tmp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
            for h in handles:
                h.remove()

            for name in subset:
                print(i, name)

                weight_original = subset[name].weight.data.detach().clone()
                
                optim_iterations = 3000
                total_global_iters = 1
                rank = args.eigen_r

       
                compressed_weight = compressed_weights['model.layers.%d.%s' % (i, name)].to(dev)
          
                new_weight, B, A =  optimize_BA(weight_original, compressed_weight, data=inps_cache[name],
                                            residual_rank= rank, optim_iterations=optim_iterations, 
                                            lr=0.0005, total_global_iters=total_global_iters,)

                B = B.to(compressed_weight.dtype)
                A = A.to(compressed_weight.dtype)

                # subset[name].weight.data = comp_weight.to(subset[name].weight.data.dtype)

                subset[name].weight.data = (compressed_weight + B@A).to(subset[name].weight.data.dtype)

                lowrank_dict[f'base_model.model.model.layers.{i}.{name}.lora_A.weight'] = A.cpu()
                lowrank_dict[f'base_model.model.model.layers.{i}.{name}.lora_B.weight'] = B.cpu()

                del new_weight, A, B
                del weight_original
                
                torch.cuda.empty_cache()
            
            del inps_cache

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        layers[i] = layer.cpu()
        del layer

        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache

    return lowrank_dict

@torch.no_grad()
def llama_sequential_activation(model, dataloader, compressed_weights, dev):
    print('Starting activation svd compensation...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.eigen_nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )

    ## this only apply to normal attention (flash attention will require different shape)
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            # cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    # position_ids = cache['position_ids']

    print('Ready.')
    lowrank_dict = {}
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        full = find_layers(layer)
        
        sequential = [list(full.keys())]
       
        for names in sequential:
            subset = {n: full[n] for n in names}
            
            subset_act_scaling_diag_matrix = {}
            for name in subset:
                subset_act_scaling_diag_matrix[name] = 0

            def hook(name):

                def tmpp(_, input, output):
                    inp = input[0].detach().float()
                    if inp.dim() == 2:
                        inp = inp.unsqueeze(0)
                    

                    tmp = inp.shape[0]
                    adds = inp
                    adds_sum = torch.sum(adds.abs(), dim=0)
                    
                    subset_act_scaling_diag_matrix[name] += adds_sum
                    del inp, adds, adds_sum, output
                    torch.cuda.empty_cache()
                return tmpp
            
            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(hook(name)))

            for j in range(args.eigen_nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
            for h in handles:
                h.remove()

            for name in subset:
                print(i, name)
                print('Start act projection ...')
                original_weight = subset[name].weight.data
                
                compressed_weight = compressed_weights['model.layers.%d.%s' % (i, name)].to(dev)

                delta = original_weight - compressed_weight

                ## save this later for SVD

                raw_scaling_diag_matrix = subset_act_scaling_diag_matrix[name].double().to("cuda")

                raw_scaling_diag_matrix = torch.mean(raw_scaling_diag_matrix,dim=0)

                scaling_diag_matrix = torch.diag(raw_scaling_diag_matrix)

                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)

                scaling_diag_matrix = scaling_diag_matrix.float()
                scaling_matrix_inv = scaling_matrix_inv.float()

                delta_scale = torch.matmul(delta.to(torch.float32), scaling_diag_matrix)

                r=args.eigen_r

                U, S, VT = torch.linalg.svd(delta_scale, full_matrices=False)
                num_s_after_trunc = r
                truc_s = S[:num_s_after_trunc]
                truc_u = U[:, :num_s_after_trunc]
                truc_v = torch.matmul(VT[:num_s_after_trunc, :], scaling_matrix_inv)
                truc_sigma = torch.diag(truc_s)

                sqrtSigma = torch.sqrt(truc_sigma)
                B = torch.matmul(truc_u, sqrtSigma).to(compressed_weight.dtype)
                A = torch.matmul(sqrtSigma, truc_v).to(compressed_weight.dtype)

                comp_weight = compressed_weight + B@A

                subset[name].weight.data = comp_weight.to(subset[name].weight.data.dtype)
                lowrank_dict[f'base_model.model.model.layers.{i}.{name}.lora_A.weight'] = A.cpu()
                lowrank_dict[f'base_model.model.model.layers.{i}.{name}.lora_B.weight'] = B.cpu()
                del B, A, compressed_weight, U, S, VT

        for j in range(args.eigen_nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]


        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache

    return lowrank_dict

@torch.no_grad()
def llama_replace(model, compressed_weights, dev):
   
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    dtype = next(iter(model.parameters())).dtype
    


    for i in range(len(layers)):
        layer = layers[i].to(dev)
        full = find_layers(layer)

        # if args.true_sequential:
        #     sequential = [
        #         ['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj'],
        #         ['self_attn.o_proj'],
        #         ['mlp.up_proj', 'mlp.gate_proj'],
        #         ['mlp.down_proj']
        #     ]
        # else:
        sequential = [list(full.keys())]
       
        for names in sequential:
            subset = {n: full[n] for n in names}


            for name in subset:
                original_weight = subset[name].weight.data
                quantize_weight = compressed_weights['model.layers.%d.%s' % (i, name)].to(dev)
                subset[name].weight.data = quantize_weight.to(subset[name].weight.data.dtype)

                del original_weight

        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()

    model.config.use_cache = use_cache

@torch.no_grad()
def llama_eval(model, testenc, dev,  dataset: str, log_wandb: bool = False):
    print("Evaluating ...")

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]

    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(f"Perplexity: {ppl.item():3f}")

    model.config.use_cache = use_cache

class compressed_lowrankLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        is_llama3 = False,
        **kwargs,
    ):
        super().__init__()

        if is_llama3:
            self.compressed_weight = torch.nn.Parameter(torch.zeros((out_features, in_features),dtype=torch.bfloat16,device="cuda"),requires_grad=False)
            self.lora_A = torch.nn.Linear(in_features=in_features, out_features= r, bias=False, device="cuda",  dtype=torch.bfloat16)
            self.lora_B = torch.nn.Linear(in_features=r, out_features=out_features, bias=False, device="cuda", dtype=torch.bfloat16)
        else:
            self.compressed_weight = torch.nn.Parameter(torch.zeros((out_features, in_features),dtype=torch.float16,device="cuda"),requires_grad=False)
            self.lora_A = torch.nn.Linear(in_features=in_features, out_features= r, bias=False, device="cuda",  dtype=torch.float16)
            self.lora_B = torch.nn.Linear(in_features=r, out_features=out_features, bias=False, device="cuda", dtype=torch.float16)

        self.is_llama3 = is_llama3
        
    def forward(self, x: torch.Tensor):

        output = x @ self.compressed_weight.T + self.lora_B(self.lora_A(x))
        
        return output
        
@torch.no_grad()
def llama_sequential_gptq_lowrank(model, dataloader, dev):
    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.eigen_nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    quantizers = {}
    quantized_weights = {}
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        full = find_layers(layer)


        sequential = [list(full.keys())]
        print(f"lowrank modules {sequential}")
        for names in sequential:
            subset = {n: full[n] for n in names}

            gptq = {}
            for name in subset:
                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer = Quantizer()
                gptq[name].quantizer.configure(
                    args.lowrank_wbits, perchannel=True, sym=args.sym, mse=False
                )

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp
            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.eigen_nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            for h in handles:
                h.remove()

            for name in subset:
                print(i, name)
                print('Quantizing ...')
                quantize_weight = gptq[name].fasterquant_wo_replcaing_weight(
                    percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order, static_groups=args.static_groups
                )
                gptq[name].layer.weight.data = quantize_weight

                quantizers['model.layers.%d.%s' % (i, name)] = gptq[name].quantizer
                gptq[name].free()

        for j in range(args.eigen_nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer.cpu()
        del layer
        del gptq 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    
@torch.no_grad()
def llama_sequential_gptq_lowrank_original_dataloader(model, dataloader, dev):
    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    quantizers = {}
    quantized_weights = {}
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        full = find_layers(layer)


        sequential = [list(full.keys())]
        print(f"lowrank modules {sequential}")
        for names in sequential:
            subset = {n: full[n] for n in names}

            gptq = {}
            for name in subset:
                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer = Quantizer()
                gptq[name].quantizer.configure(
                    args.lowrank_wbits, perchannel=True, sym=args.sym, mse=False
                )

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp
            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            for h in handles:
                h.remove()

            for name in subset:
                print(i, name)
                print('Quantizing ...')
                quantize_weight = gptq[name].fasterquant_wo_replcaing_weight(
                    percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order, static_groups=args.static_groups
                )
                gptq[name].layer.weight.data = quantize_weight

                quantizers['model.layers.%d.%s' % (i, name)] = gptq[name].quantizer
                gptq[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer.cpu()
        del layer
        del gptq 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    

    
if __name__ == "__main__":
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument("model", type=str, help="LlaMA model to load")
    parser.add_argument(
        "dataset",
        type=str,
        choices=["wikitext2", "c4"],
        help="Where to extract calibration data from.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for sampling the calibration data."
    )
    parser.add_argument(
        "--nsamples", type=int, default=128, help="Number of calibration data samples."
    )
    parser.add_argument(
        "--percdamp",
        type=float,
        default=0.01,
        help="Percent of the average Hessian diagonal to use for dampening.",
    )
    parser.add_argument("--sparsity", type=float, default=0, help="Target sparsity")
    parser.add_argument("--prunen", type=int, default=0, help="N for N:M pruning.")
    parser.add_argument("--prunem", type=int, default=0, help="M for N:M pruning.")
    parser.add_argument(
        "--blocksize",
        type=int,
        default=128,
        help="Blocksize to use for adaptive mask selection.",
    )
    parser.add_argument(
        "--gmp", action="store_true", help="Whether to run the GMP baseline."
    )
    parser.add_argument(
        "--wbits", type=int, default=16, help="Whether to quantize as well."
    )
    parser.add_argument(
        '--groupsize', type=int, default=-1,
        help='Groupsize to use for quantization; default uses full row.'
    )
    parser.add_argument(
        '--sym', action='store_true',
        help='Whether to perform symmetric quantization.'
    )
    parser.add_argument(
        "--minlayer", type=int, default=-1, help="Prune all layers with id >= this."
    )
    parser.add_argument(
        "--maxlayer", type=int, default=1000, help="Prune all layers with id < this."
    )
    parser.add_argument(
        "--prune_only",
        type=str,
        default="",
        help="Prune only layers that contain this text.",
    )
    parser.add_argument("--invert", action="store_true", help="Invert subset.")
    parser.add_argument(
        '--act-order', action='store_true',
        help='Whether to apply the activation order GPTQ heuristic'
    )
    parser.add_argument(
        "--true-sequential",
        action="store_true",
        help="Whether to run in true sequential model.",
    )
    parser.add_argument(
        '--static-groups', action='store_true',
        help='Whether to use static groups; recommended when using `--actorder` for more efficient inference.'
    )
    parser.add_argument(
        "--log_wandb", action="store_true", help="Whether to log to wandb."
    )
    parser.add_argument(
        '--compression_method', type=str, default="sparsegpt", choices=["gptq", "sparsegpt", "wanda", "full-precision"]
    )
    parser.add_argument(
        '--lowrank_method', type=str, default="eigen", choices=["svd", "eigen", "activation", "learn", "no"]
    )
    parser.add_argument(
        "--eigen_dataset",
        type=str,
        default= "wikitext2",
        choices=["wikitext2", "arc", "mathqa", "gsm8k"],
        help="Where to extract calibration data from.",
    )
    parser.add_argument(
        '--eigen_nsamples', type=int, default=256,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        '--eigen_r', type=int, default=512
    )
    parser.add_argument(
        '--lowrank_wbits', type=int, default=16
    )
    parser.add_argument(
        '--eval_ppl', action="store_true", help="Whether to run wikitext2 ppl eval"
    )
    parser.add_argument(
        '--eval_arc', action="store_true", help="Whether to run zero-shot arc evaluation"
    )
    parser.add_argument(
        '--eval_mathqa', action="store_true", help="Whether to run zero-shot mathqa evaluation"
    )
    parser.add_argument(
        '--eval_gsm8k', action="store_true", help="Whether to run gsm8k evaluation"
    )

    args = parser.parse_args()
    if args.lowrank_method != "no":
        print(f"using rank {args.eigen_r}")

    model = get_llama(args.model)
    model.eval()

    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    )

    if args.compression_method == "gptq":
        assert args.wbits < 16
        print("start quantizing model")
        quantizers, compressed_weights = llama_sequential_gptq(model, dataloader, DEV)

    elif args.compression_method == "sparsegpt": 
        assert args.sparsity > 0 
        quantizers, compressed_weights = llama_sequential_sparsegpt(model, dataloader, DEV)
    
    elif args.compression_method == "wanda": 
        assert args.sparsity > 0 
        quantizers, compressed_weights = llama_sequential_wanda(model, dataloader, DEV)

    elif args.compression_method == "full-precision":
        print("Full-precision model")
    else:
        raise NotImplementedError

    if args.compression_method != "full-precision":
        del model

        model = get_llama(args.model)
        model.eval()
    
    eigen_dataloader = None
    if args.lowrank_method == "eigen":

        eigen_dataloader, _ = get_loaders(
            args.eigen_dataset, nsamples=args.eigen_nsamples, seed=args.seed + 1, model=args.model, seqlen=model.seqlen
        )  
        lowrank_save_weight = llama_sequential_eigen(model, eigen_dataloader, compressed_weights, DEV)

    elif args.lowrank_method == "svd":

        lowrank_save_weight = llama_sequential_svd(model, compressed_weights, DEV)

    elif args.lowrank_method == "activation":

        act_dataloader, _ = get_loaders(
            args.eigen_dataset, nsamples=args.eigen_nsamples, seed=args.seed + 1, model=args.model, seqlen=model.seqlen
        )
        lowrank_save_weight = llama_sequential_activation(model, act_dataloader, compressed_weights, DEV)
    
    elif args.lowrank_method == "learn":

        learn_dataloader, _ = get_loaders(
            args.eigen_dataset, nsamples=args.eigen_nsamples, seed=args.seed + 1, model=args.model, seqlen=model.seqlen
        )   
        
        lowrank_save_weight = llama_sequential_learn_BA(model, learn_dataloader, compressed_weights, DEV)
    
    elif args.lowrank_method == "no":
        ## original compressed model
        if args.compression_method != "full-precision":
            print("replace the original model weight with compressed model weight")
            llama_replace(model, compressed_weights, DEV)
        

    if args.lowrank_wbits < 16:
        print(f"Quantize lowrank weights to {args.lowrank_wbits} bits")
        print("Replacing linear layers with compressed + lowrank module")

        def _get_submodules(model, key):
            parent = model.get_submodule(".".join(key.split(".")[:-1]))
            target_name = key.split(".")[-1]
            target = model.get_submodule(key)
            return parent, target, target_name
        
        def _replace_module( parent_module, child_name, new_module, old_module):
            setattr(parent_module, child_name, new_module)

            # dispatch to correct device
            for name, module in new_module.named_modules():    
                module.to(old_module.weight.device)
            
            del old_module

        is_llama3 = False
        if "Llama-3" in args.model:
            is_llama3 = True

        key_list = [(key,module) for (key, module) in model.named_modules()]
        for key,module in key_list:
            if isinstance(module, nn.Linear) and key!= "lm_head":
                parent, target, target_name = _get_submodules(model, key)
                print(f"replacing {key}")
                new_module = compressed_lowrankLinear(in_features=target.in_features, out_features= target.out_features, r = args.eigen_r, is_llama3 = is_llama3)
                _replace_module(parent, target_name, new_module, target)

        print("Load the compressed weight and lowrank weight")
        layers = model.model.layers

        def find_compressed_lowrank_layers(module, layers=[compressed_lowrankLinear], name=''):
            if type(module) in layers:
                return {name: module}
            res = {}
            for name1, child in module.named_children():
                # print(f"name1: {name1}")
                res.update(find_compressed_lowrank_layers(
                    child, layers=layers, name=name + '.' + name1 if name != '' else name1
                ))
            return res

        for i in range(len(layers)):
            layer = layers[i]
            full = find_compressed_lowrank_layers(layer)
            sequential = [list(full.keys())]
        
            for names in sequential:
                subset = {n: full[n] for n in names}

                for name in subset:
                    print(f"loading weight to {name}")
                    subset[name].lora_A.weight.data = lowrank_save_weight[f'base_model.model.model.layers.{i}.{name}.lora_A.weight']
                    subset[name].lora_B.weight.data = lowrank_save_weight[f'base_model.model.model.layers.{i}.{name}.lora_B.weight']
                    subset[name].compressed_weight.data = compressed_weights['model.layers.%d.%s' % (i, name)]


        torch.cuda.empty_cache()
        print("start quantizing the lowrank path")
        
        if eigen_dataloader != None:
            print(f"use eigen dataloader for applying gptq to quantize the lowrank path {args.eigen_dataset}")
            llama_sequential_gptq_lowrank(model, eigen_dataloader, DEV)
        else:
            print(f"use gptq dataloader for applying gptq to quantize the lowrank path {args.eigen_dataset}")
            llama_sequential_gptq_lowrank_original_dataloader(model, dataloader, DEV)

    if args.eval_ppl:

        for dataset in ["wikitext2"]:
            dataloader, testloader = get_loaders(
                dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
            )
            llama_eval(model, testloader, DEV, dataset, args.log_wandb)


    if args.eval_arc or args.eval_mathqa or args.eval_gsm8k:
        from transformers import LlamaTokenizer, AutoTokenizer
        is_llama3 = False
        if "llama-3" in args.model.lower():
            tokenizer = AutoTokenizer.from_pretrained(args.model,use_fast=False,legacy=False)
            is_llama3 = True
        else:
            tokenizer = LlamaTokenizer.from_pretrained(args.model, use_fast=False)

        model = model.to("cuda")

        if args.eval_arc:
            result = evaluate_model(
                model,
                tokenizer,
                limit=-1,
                is_llama3=is_llama3,
                tasks="arc_challenge"
            )
        if args.eval_mathqa:
            result = evaluate_model(
                model,
                tokenizer,
                limit=-1,
                is_llama3=is_llama3,
                tasks="mathqa"
            )
        if args.eval_gsm8k:
            result = evaluate_model(
                model,
                tokenizer,
                num_fewshot=5,
                limit=-1,
                is_llama3=is_llama3,
                tasks="gsm8k"
            )

        
