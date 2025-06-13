
    
import numpy as np
import torch
import transformers
from typing import Dict, Optional, Sequence
import re


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

def get_mathqa_c4(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata_mathqa = load_dataset('math_qa', split='train')
    from transformers import AutoTokenizer 
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False, seqlen=2048)

    import random
    random.seed(seed)
    trainloader = []

    if nsamples == 64:
        mathqa_namsples = int(20)
        c4_nsamples = nsamples - mathqa_namsples
    elif nsamples == 32:
        mathqa_namsples = int(16)
        c4_nsamples = nsamples - mathqa_namsples
    elif nsamples ==16:
        mathqa_namsples = int(8)
        c4_nsamples = nsamples - mathqa_namsples
    else:
        mathqa_namsples = int(20)
        c4_nsamples = nsamples - mathqa_namsples

    i = 0
    for _ in range(mathqa_namsples):

        cur_len = 0
        input = ""
        while cur_len < seqlen:


            doc = traindata_mathqa[i]
            cur_input = "Question: " + doc["Problem"] + " Choices: " + doc["options"] + ". Rationale: " + doc["Rationale"] + ". "

            input = input + cur_input
            trainenc = tokenizer(input, return_tensors='pt')
            ## comback later to see if need padding
            cur_len = (trainenc.input_ids.shape[1]) ## neglect the bos token
            i += 1

        ## reach seq_len
        final_inp = tokenizer(input, return_tensors='pt')
        inp = final_inp.input_ids[:, :seqlen]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    traindata = load_dataset("sliuau/c4-train", split='train')


    
    for _ in range(c4_nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader, trainloader

def get_arc_c4(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata_arc_easy = load_dataset('ai2_arc', 'ARC-Easy', split='train')
    traindata_arc_challenge = load_dataset('ai2_arc', 'ARC-Challenge', split='train')
    from transformers import AutoTokenizer 
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False, seqlen=2048)


    import random
    random.seed(seed)
    trainloader = []
    arc_e_namsples = int(20)
    print(f"arc_e_namsples {arc_e_namsples}")
    i = 0
    for _ in range(arc_e_namsples):
        
        cur_len = 0
        input = ""
        while cur_len < seqlen:
            answer = traindata_arc_easy[i]['choices']['label'].index(traindata_arc_easy[i]['answerKey'])
            cur_input = traindata_arc_easy[i]['question'] +" "+ traindata_arc_easy[i]['choices']['text'][answer] + ". "
            # print(cur_input)
            input = input + cur_input
            trainenc = tokenizer(input, return_tensors='pt')
            ## comback later to see if need padding
            cur_len = (trainenc.input_ids.shape[1]) ## neglect the bos token
            # print(cur_len)
            i += 1
        
        ## reach seq_len
        final_inp = tokenizer(input, return_tensors='pt')
        # print(f"final_inp.input_ids {final_inp.input_ids.shape}")
        inp = final_inp.input_ids[:, :seqlen]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

        # print(f"inp {inp.shape}")

    arc_c_namsples = int(10)
    print(f"arc_c_namsples {arc_c_namsples}")
    i = 0
    for _ in range(arc_c_namsples):
        
        cur_len = 0
        input = ""
        while cur_len < seqlen:
            answer = traindata_arc_challenge[i]['choices']['label'].index(traindata_arc_challenge[i]['answerKey'])
            cur_input = traindata_arc_challenge[i]['question'] +" "+ traindata_arc_challenge[i]['choices']['text'][answer] + ". "
            # print(cur_input)
            input = input + cur_input
            trainenc = tokenizer(input, return_tensors='pt')
            ## comback later to see if need padding
            cur_len = (trainenc.input_ids.shape[1]) ## neglect the bos token
            # print(cur_len)
            i += 1

        ## reach seq_len
        final_inp = tokenizer(input, return_tensors='pt')
        # print(f"final_inp.input_ids {final_inp.input_ids.shape}")
        inp = final_inp.input_ids[:, :seqlen]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))


    traindata = load_dataset("sliuau/c4-train", split='train')
    print(f"traindata {traindata[0]}")
    c4_nsamples = nsamples - arc_c_namsples - arc_e_namsples
    for _ in range(c4_nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            # print(len(traindata[i]['text']))
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        # print(f"inp {inp.shape}")
        trainloader.append((inp, tar))

    return trainloader, trainloader

def get_gsm8k_c4(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata_gsm8k = load_dataset('gsm8k', 'main', split='train')
    
    from transformers import AutoTokenizer 
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False, seqlen=2048)


    import random
    random.seed(seed)
    trainloader = []
    gsm8k_namsples = int(32)
    print(f"gsm8k {gsm8k_namsples}")
    i = 0
    for _ in range(gsm8k_namsples):
        
        cur_len = 0
        input = ""
        while cur_len < seqlen:
            answer = traindata_gsm8k[i]["answer"]
            cur_input = "Question: " + traindata_gsm8k[i]["question"] + "\nAnswer:" + answer
            # print(cur_input)
            input = input + cur_input
            trainenc = tokenizer(input, return_tensors='pt')
            ## comback later to see if need padding
            cur_len = (trainenc.input_ids.shape[1]) ## neglect the bos token
            # print(cur_len)
            i += 1
            print(f"i {i}")
        
        ## reach seq_len
        final_inp = tokenizer(input, return_tensors='pt')
        # print(f"final_inp.input_ids {final_inp.input_ids.shape}")
        inp = final_inp.input_ids[:, :seqlen]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

        # print(f"inp {inp.shape}")


    traindata = load_dataset("sliuau/c4-train", split='train')

    c4_nsamples = nsamples - gsm8k_namsples
    for _ in range(c4_nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            # print(len(traindata[i]['text']))
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        # print(f"inp {inp.shape}")
        trainloader.append((inp, tar))

    return trainloader, trainloader

def get_wikitext2(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    from transformers import AutoTokenizer 
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_c4(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata = load_dataset("sliuau/c4-train", split='train')
    valdata = load_dataset("sliuau/c4-val", split='train')

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)

    import random
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            # print(len(traindata[i]['text']))
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        # print(f"inp {inp.shape}")
        trainloader.append((inp, tar))

    import random
    random.seed(0)
    valenc = []
    for _ in range(256):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
            if tmp.input_ids.shape[1] > seqlen:
                break
        # print(f" tmp.input_ids.shape[1] {tmp.input_ids.shape[1]}")
        i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])
    valenc = torch.hstack(valenc)
    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids
    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc 

def get_loaders(
    name, nsamples=128, seed=0, seqlen=2048, model=''
):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, model)
    if 'c4' in name:
        return get_c4(nsamples, seed, seqlen, model)
    
    if "mathqa" in name:
        return get_mathqa_c4(nsamples, seed, seqlen, model)
    
    if "arc" in name:
        return get_arc_c4(nsamples, seed, seqlen, model)

    if "gsm8k" in name:
        return get_gsm8k_c4(nsamples, seed, seqlen, model)

    
