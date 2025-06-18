# 🔧 Environment Setup

## Step 1: Install Dependencies
```bash
conda create -n eora python=3.10
conda activate eora
pip install -r requirements.txt
```

## Step 2: Compile CUDA Kernels (Optional)
```bash
python setup_cuda.py install
```

> 💡 If you're using Conda, install `nvcc` and the CUDA toolkit via:
```bash
conda install nvidia::cuda-cudart-dev
```

---

# 📊 Reproduce MathQA Results (LLaMA3-8B Quantized with GPTQ)

```bash
# EoRA (rank=128) with 3-bit GPTQ
python eora.py meta-llama/Meta-Llama-3-8B wikitext2 \
  --wbits 3 --true-sequential --act-order \
  --eigen_r 128 --compression_method gptq \
  --lowrank_method eigen \
  --eigen_dataset mathqa --eval_mathqa --eigen_nsamples 64

# ACT-S (activation-based) method
python eora.py meta-llama/Meta-Llama-3-8B wikitext2 \
  --wbits 3 --true-sequential --act-order \
  --eigen_r 128 --compression_method gptq \
  --lowrank_method activation \
  --eigen_dataset mathqa --eval_mathqa --eigen_nsamples 64

# SVD-based low-rank compensation
python eora.py meta-llama/Meta-Llama-3-8B wikitext2 \
  --wbits 3 --true-sequential --act-order \
  --eigen_r 128 --compression_method gptq \
  --lowrank_method svd --eval_mathqa

# GPTQ-only baseline
python llama_eigen_fix.py meta-llama/Meta-Llama-3-8B wikitext2 \
  --wbits 3 --true-sequential --act-order \
  --compression_method gptq --eval_mathqa \
  --lowrank_method no

# Full-precision baseline
python llama_eigen_fix.py meta-llama/Meta-Llama-3-8B wikitext2 \
  --compression_method full-precision \
  --lowrank_method no --eval_mathqa
```

---

# 🌿 Reproduce SparseGPT (2:4) Results on MathQA

```bash
# EoRA (rank=128) with SparseGPT (2:4)
python eora.py meta-llama/Meta-Llama-3-8B wikitext2 \
  --sparsity 0.5 --prunen 2 --prunem 4 \
  --eigen_r 128 --compression_method sparsegpt \
  --lowrank_method eigen \
  --eigen_dataset mathqa --eval_mathqa --eigen_nsamples 64

# ACT-S method
python eora.py meta-llama/Meta-Llama-3-8B wikitext2 \
  --sparsity 0.5 --prunen 2 --prunem 4 \
  --eigen_r 128 --compression_method sparsegpt \
  --lowrank_method activation \
  --eigen_dataset mathqa --eval_mathqa --eigen_nsamples 64

# SVD method
python eora.py meta-llama/Meta-Llama-3-8B wikitext2 \
  --sparsity 0.5 --prunen 2 --prunem 4 \
  --eigen_r 128 --compression_method sparsegpt \
  --lowrank_method svd --eval_mathqa

# SparseGPT-only baseline
python llama_eigen_fix.py meta-llama/Meta-Llama-3-8B wikitext2 \
  --true-sequential --sparsity 0.5 --prunen 2 --prunem 4 \
  --compression_method sparsegpt --eval_mathqa --lowrank_method no

# Full-precision baseline
python llama_eigen_fix.py meta-llama/Meta-Llama-3-8B wikitext2 \
  --compression_method full-precision --lowrank_method no --eval_mathqa
```

---

# 🔬 Reproduce Wanda (2:4) Results on MathQA

```bash
# EoRA (rank=128) with Wanda (2:4)
python eora.py meta-llama/Meta-Llama-3-8B wikitext2 \
  --sparsity 0.5 --prunen 2 --prunem 4 \
  --eigen_r 128 --compression_method wanda \
  --lowrank_method eigen \
  --eigen_dataset mathqa --eval_mathqa --eigen_nsamples 64

# ACT-S method
python eora.py meta-llama/Meta-Llama-3-8B wikitext2 \
  --sparsity 0.5 --prunen 2 --prunem 4 \
  --eigen_r 128 --compression_method wanda \
  --lowrank_method activation \
  --eigen_dataset mathqa --eval_mathqa --eigen_nsamples 64

# SVD method
python eora.py meta-llama/Meta-Llama-3-8B wikitext2 \
  --sparsity 0.5 --prunen 2 --prunem 4 \
  --eigen_r 128 --compression_method wanda \
  --lowrank_method svd --eval_mathqa

# Wanda-only baseline
python llama_eigen_fix.py meta-llama/Meta-Llama-3-8B wikitext2 \
  --true-sequential --sparsity 0.5 --prunen 2 --prunem 4 \
  --compression_method wanda --eval_mathqa --lowrank_method no

# Full-precision baseline
python llama_eigen_fix.py meta-llama/Meta-Llama-3-8B wikitext2 \
  --compression_method full-precision --lowrank_method no --eval_mathqa
```

---

# 🧪 Evaluating on Other Datasets

To reproduce results on other tasks such as **GSM8K** or **ARC-Challenge**, simply modify:
- `--eigen_dataset gsm8k` or `--eigen_dataset arc`
- Add `--eval_gsm8k` or `--eval_arc` accordingly

---

# 🚀 EoRA CUDA Kernel Benchmarks

```bash
# Benchmark q_proj CUDA kernel
python test_kernel.py

# Full-precision LLaMA3-70B (128 tokens)
CUDA_VISIBLE_DEVICES=0,1 python llama_benchmark.py meta-llama/Meta-Llama-3-70B wikitext2 --benchmark 128

# 3-bit GPTQ
CUDA_VISIBLE_DEVICES=0 python llama_benchmark.py meta-llama/Meta-Llama-3-70B wikitext2 \
  --benchmark 128 --inference_type "3_bit"

# 3-bit + EoRA (no fusion)
CUDA_VISIBLE_DEVICES=0 python llama_benchmark.py meta-llama/Meta-Llama-3-70B wikitext2 \
  --benchmark 128 --inference_type "3_bit_wo_fuse" --rank 64

# 3-bit + EoRA (fused)
CUDA_VISIBLE_DEVICES=0 python llama_benchmark.py meta-llama/Meta-Llama-3-70B wikitext2 \
  --benchmark 128 --inference_type "3_bit_fuse" --rank 64

# 4-bit GPTQ
CUDA_VISIBLE_DEVICES=0 python llama_benchmark.py meta-llama/Meta-Llama-3-70B wikitext2 \
  --benchmark 128 --inference_type "4_bit"

# 4-bit + EoRA (no fusion)
CUDA_VISIBLE_DEVICES=0 python llama_benchmark.py meta-llama/Meta-Llama-3-70B wikitext2 \
  --benchmark 128 --inference_type "4_bit_wo_fuse" --rank 64

# 4-bit + EoRA (fused)
CUDA_VISIBLE_DEVICES=0 python llama_benchmark.py meta-llama/Meta-Llama-3-70B wikitext2 \
  --benchmark 128 --inference_type "4_bit_fuse" --rank 64
```
