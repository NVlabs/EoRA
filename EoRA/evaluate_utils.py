import torch
import torch.nn as nn
from tqdm import tqdm
import os

from lm_eval.base import BaseLM
from lm_eval import evaluator
from lm_eval import utils
from datasets import load_dataset
import time
import re


class EvalLM(BaseLM):
    def __init__(
        self,
        model,
        tokenizer,
        # device="cuda:0",
        batch_size=1,
    ):
        super().__init__()

        # assert isinstance(device, str)
        assert isinstance(batch_size, int)

        # self._device = torch.device(device)
        self._device = model.device

        # self.model = model.to(self.device)
        self.model = model
        self.model.eval()

        self.tokenizer = tokenizer

        self.vocab_size = self.tokenizer.vocab_size

        self.batch_size_per_gpu = batch_size  # todo: adaptive batch size

        self.seqlen = 2048
        # print("vocab size: ", self.vocab_size)

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        try:
            return self.model.config.n_ctx
        except AttributeError:
            # gptneoconfig doesn't have n_ctx apparently
            return self.model.config.max_position_embeddings

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        # TODO: fix multi-gpu
        return self.batch_size_per_gpu  # * gpus

    @property
    def device(self):
        # TODO: fix multi-gpu
        return self._device

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():
            return self.model(inps)[0][:, :, :50257]

    def _model_generate(self, context, max_length, eos_token_id):
        return self.model.generate(
            context, max_length=max_length, eos_token_id=eos_token_id, do_sample=False
        )

class GSM8K_EvalLM(BaseLM):
    def __init__(
        self,
        model,
        tokenizer,
        # device="cuda:0",
        batch_size=1,
    ):
        super().__init__()

        # assert isinstance(device, str)
        assert isinstance(batch_size, int)

        # self._device = torch.device(device)
        self._device = model.device

        # self.model = model.to(self.device)
        self.model = model
        self.model.eval()

        self.tokenizer = tokenizer

        self.vocab_size = self.tokenizer.vocab_size

        self.batch_size_per_gpu = batch_size  # todo: adaptive batch size

        self.seqlen = 2048
        # print("vocab size: ", self.vocab_size)

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        try:
            return self.model.config.n_ctx
        except AttributeError:
            # gptneoconfig doesn't have n_ctx apparently
            return self.model.config.max_position_embeddings

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        # TODO: fix multi-gpu
        return self.batch_size_per_gpu  # * gpus

    @property
    def device(self):
        # TODO: fix multi-gpu
        return self._device

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        
        with torch.no_grad():
            return self.model(inps)[0][:, :, :50257]

    def _model_generate(self, context, max_length, eos_token_id):
        return self.model.generate(
            context, max_length=1500, do_sample=False
        )

    
    def greedy_until(self, requests):
        # TODO: implement fully general `until` that handles until that are
        #       multiple tokens or that span multiple tokens correctly

        # TODO: extract to TokenizedLM?
        res = []

        def _collate(x):
            toks = self.tok_encode(x[0])
            return len(toks), x[0]

        re_ord = utils.Reorderer(requests, _collate)

        for context, until in tqdm(re_ord.get_reordered()):
            if isinstance(until, str):
                until = [until]
            # print(f"until: {until[0]}")
            # (primary_until,) 
            primary_until = self.tok_encode(until[0])[1]
            # print(f"primary_until {primary_until}")
            # print(f"Question {context}")
            context_enc = torch.tensor(
                [self.tok_encode(context)[self.max_gen_toks - self.max_length :]]
            ).to(self.device)
            
            # print(f"check: {self.tok_decode(self.tok_encode(context))[self.max_gen_toks - self.max_length :]}")
            cont = self._model_generate(
                context_enc, context_enc.shape[1] + self.max_gen_toks, primary_until
            )
            # print(cont)
            s = self.tok_decode(cont[0].tolist()[context_enc.shape[1] :])
            # s = self.tok_decode(cont[0].tolist())
            # s = self.tok_decode(cont)
            # print(f"response {s}")
            s = s.split('\n\n')[0]
            # partial caching
            self.cache_hook.add_partial("greedy_until", (context, until), s)
            # print(f"response {s}")
            res.append(s)

        return re_ord.get_original(res)




class EvalLM_llama3(BaseLM):
    def __init__(
        self,
        model,
        tokenizer,
        # device="cuda:0",
        batch_size=1,
    ):
        super().__init__()

        # assert isinstance(device, str)
        assert isinstance(batch_size, int)

        # self._device = torch.device(device)
        self._device = model.device

        # self.model = model.to(self.device)
        self.model = model
        self.model.eval()

        self.tokenizer = tokenizer

        self.vocab_size = self.tokenizer.vocab_size

        self.batch_size_per_gpu = batch_size  # todo: adaptive batch size

        self.seqlen = 2048
        print("vocab size: ", self.vocab_size)

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        try:
            return self.model.config.n_ctx
        except AttributeError:
            # gptneoconfig doesn't have n_ctx apparently
            return self.model.config.max_position_embeddings

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        # TODO: fix multi-gpu
        return self.batch_size_per_gpu  # * gpus

    @property
    def device(self):
        # TODO: fix multi-gpu
        return self._device

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():
            return self.model(inps)["logits"]

    def _model_generate(self, context, max_length, eos_token_id):
        return self.model.generate(
            context, max_length=max_length, eos_token_id=eos_token_id, do_sample=False
        )
    
class GSM8K_EvalLM_llama3(BaseLM):
    def __init__(
        self,
        model,
        tokenizer,
        # device="cuda:0",
        batch_size=1,
    ):
        super().__init__()

        # assert isinstance(device, str)
        assert isinstance(batch_size, int)

        # self._device = torch.device(device)
        self._device = model.device

        # self.model = model.to(self.device)
        self.model = model
        self.model.eval()

        self.tokenizer = tokenizer

        self.vocab_size = self.tokenizer.vocab_size

        self.batch_size_per_gpu = batch_size  # todo: adaptive batch size

        self.seqlen = 2048
        print("vocab size: ", self.vocab_size)

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        try:
            return self.model.config.n_ctx
        except AttributeError:
            # gptneoconfig doesn't have n_ctx apparently
            return self.model.config.max_position_embeddings

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        # TODO: fix multi-gpu
        return self.batch_size_per_gpu  # * gpus

    @property
    def device(self):
        # TODO: fix multi-gpu
        return self._device

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call

        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():
            return self.model(inps)["logits"]

    def _model_generate(self, context, max_length, eos_token_id):
        return self.model.generate(
            context, max_length=max_length, eos_token_id=eos_token_id, do_sample=False, pad_token_id=self.tokenizer.eos_token_id
        )

    def greedy_until(self, requests):
        # TODO: implement fully general `until` that handles until that are
        #       multiple tokens or that span multiple tokens correctly

        # TODO: extract to TokenizedLM?
        res = []

        def _collate(x):
            toks = self.tok_encode(x[0])
            return len(toks), x[0]

        re_ord = utils.Reorderer(requests, _collate)

        for context, until in tqdm(re_ord.get_reordered()):
            if isinstance(until, str):
                until = [until]
            # print(f"until: {until[0]}")
            (primary_until,) = self.tok_encode(until[0])
            # test = self.tok_encode(until[0])
            # print(f"primary_until test{test}")
            # print(f"Question {context}")
            context_enc = torch.tensor(
                [self.tok_encode(context)[self.max_gen_toks - self.max_length :]]
            ).to(self.device)

            cont = self._model_generate(
                context_enc, context_enc.shape[1] + self.max_gen_toks, primary_until
            )

            s = self.tok_decode(cont[0].tolist()[context_enc.shape[1] :])
            # print(f"response {s}")
            s = s.split('\n\n')[0]
            # partial caching
            self.cache_hook.add_partial("greedy_until", (context, until), s)
            # print(f"response {s}")
            res.append(s)

        return re_ord.get_original(res)




    
@torch.no_grad()
def evaluate_perplexity(model, dataset, limit):
    """
    dataset: input ids tensor of shape [batch, sequence length]
    """
    nsamples, seqlen = dataset.size()

    nlls = []

    for i in range(nsamples):
        if i == limit:
            break
        input_ids = dataset[i:i+1,:-1].to(model.device)
        labels = dataset[i:i+1,1:].contiguous()
        logits = model(input_ids=input_ids)[0]
        shift_logits = logits[:, :, :]
        shift_labels = labels.to(model.device)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        neg_log_likelihood = loss.float() * seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (len(nlls) * seqlen))
    return ppl.item()


@torch.no_grad()
def evaluate_model(
    model,
    tokenizer,
    num_fewshot=0,
    limit=-1,
    batch_size=1,
    is_llama3 = False,
    tasks = "arc_challenge"
):
    """
    model: model name
    limit: number of test samples for debug, set to -1 is no limit
    tasks: str tasks are split by ,
    num_fewshot: Number of examples in few-shot context
    eval_ppl: str datasets are split by , such as 'wikitext2,ptb,c4'
    """
    if is_llama3:
        if "gsm8k" in tasks:
            print("gsm8k model config!")
            lm = GSM8K_EvalLM_llama3(model, tokenizer, batch_size=batch_size)
        else:
            lm = EvalLM_llama3(model, tokenizer, batch_size=batch_size)
    else:
        if "gsm8k" in tasks:
            print("gsm8k model config!")
            lm = GSM8K_EvalLM(model, tokenizer, batch_size=batch_size)
        else:
            lm = EvalLM(model, tokenizer, batch_size=batch_size)
    results = {}


    print(f"tasks {tasks}")
    if tasks != "":
        t_results = evaluator.simple_evaluate(
            lm,
            tasks=tasks.split(","),
            batch_size=batch_size,
            num_fewshot=num_fewshot,
            limit=None if limit == -1 else limit,
            no_cache=True,
        )
        t_results = t_results["results"]
        acc_list = [
            t_results[key]["acc"] for key in t_results.keys() if "acc" in t_results[key]
        ]
        t_results["mean"] = sum(acc_list) / len(acc_list)
        results.update(t_results)
        print(results)
        # print mean
        print(f"\n\n===== mean acc: {sum(acc_list)/len(acc_list)} =====\n\n")

    return results
