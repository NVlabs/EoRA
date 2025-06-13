// Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/all.h>
#include <torch/python.h>

constexpr int BLOCKWIDTH = 256;
constexpr int BLOCKHEIGHT = 24;
constexpr int NUM_THREADS = BLOCKWIDTH;

__device__ inline unsigned int as_unsigned(int i) {
  return *reinterpret_cast<unsigned int *>(&i);
}
// const int LORA_K_BLOCKSZ = 64;

template <size_t LORA_K_BLOCKSZ>
__global__ void
VecQuant3MatMulKernelLoraFaster(const half2 *vec, const int *mat,
                                const half2 *down_proj, // expect M, K
                                const half *up, // expect N, K (transposed)
                                float *mul, const float *scales,
                                const float *zeros, int height, int width) {
  const int blockwidth2 = BLOCKWIDTH / 2;

  int row = BLOCKHEIGHT * blockIdx.x;
  int col = BLOCKWIDTH * blockIdx.y + threadIdx.x;

  // printf("row: %d, col: %d\n", row, col);
  __shared__ half2 blockvec[blockwidth2];
  if (threadIdx.x < blockwidth2)
    blockvec[threadIdx.x] =
        vec[(row / BLOCKHEIGHT) * blockwidth2 + threadIdx.x];

  constexpr size_t LORA_SIZE = LORA_K_BLOCKSZ; // lora representations dim
  constexpr size_t LORA_SIZE2 = LORA_SIZE / 2;
  __shared__ half2 down_sh[LORA_SIZE2]; // expected to be small

  static_assert(NUM_THREADS >= LORA_SIZE2);

  // assume thread_size > lora_size2
  // int lora_col = threadIdx.x % LORA_SIZE2;
  // int lora_row = threadIdx.x / LORA_SIZE2;
  constexpr int load_iter = (LORA_SIZE2 + NUM_THREADS - 1) / NUM_THREADS;
#pragma unroll
  for (int i = 0; i < load_iter; i++) {
    int low_rank_i = i * NUM_THREADS + threadIdx.x;
    if (low_rank_i < LORA_SIZE2)
      down_sh[low_rank_i] = down_proj[low_rank_i];
  }

  __shared__ half2 deq2[64][32];
  int val = threadIdx.x / 32;
  int off = threadIdx.x % 32;
  for (; val < 64; val += BLOCKWIDTH / 32) {
    deq2[val][off] =
        __halves2half2(__int2half_rn(val & 0x7), __int2half_rn(val >> 3));
  }

  half2 scale = __float2half2_rn(scales[col]);
  half2 zero = __float2half2_rn(-zeros[col]);

  int i = width * row + col;
  int k = 0;

  float res = 0;
  half2 res2;

  unsigned int tmp1;
  unsigned int tmp2;
  unsigned int tmp;

  __syncthreads();

  while (k < blockwidth2) {
    res2 = {};
    tmp1 = as_unsigned(mat[i]);
    res2 = __hfma2(__hfma2(deq2[(tmp1 >> 0) & 0x3f][off], scale, zero),
                   blockvec[k + 0], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp1 >> 6) & 0x3f][off], scale, zero),
                   blockvec[k + 1], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp1 >> 12) & 0x3f][off], scale, zero),
                   blockvec[k + 2], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp1 >> 18) & 0x3f][off], scale, zero),
                   blockvec[k + 3], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp1 >> 24) & 0x3f][off], scale, zero),
                   blockvec[k + 4], res2);
    i += width;
    tmp2 = as_unsigned(mat[i]);
    tmp = (tmp1 >> 30) | ((tmp2 << 2) & 0x3c);
    res2 = __hfma2(__hfma2(deq2[tmp][off], scale, zero), blockvec[k + 5], res2);
    tmp2 >>= 4;
    k += 6;
    res2 = __hfma2(__hfma2(deq2[(tmp2 >> 0) & 0x3f][off], scale, zero),
                   blockvec[k + 0], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp2 >> 6) & 0x3f][off], scale, zero),
                   blockvec[k + 1], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp2 >> 12) & 0x3f][off], scale, zero),
                   blockvec[k + 2], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp2 >> 18) & 0x3f][off], scale, zero),
                   blockvec[k + 3], res2);
    i += width;
    tmp1 = as_unsigned(mat[i]);
    tmp = (tmp2 >> 24) | ((tmp1 << 4) & 0x30);
    res2 = __hfma2(__hfma2(deq2[tmp][off], scale, zero), blockvec[k + 4], res2);
    tmp1 >>= 2;
    k += 5;
    res2 = __hfma2(__hfma2(deq2[(tmp1 >> 0) & 0x3f][off], scale, zero),
                   blockvec[k + 0], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp1 >> 6) & 0x3f][off], scale, zero),
                   blockvec[k + 1], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp1 >> 12) & 0x3f][off], scale, zero),
                   blockvec[k + 2], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp1 >> 18) & 0x3f][off], scale, zero),
                   blockvec[k + 3], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp1 >> 24) & 0x3f][off], scale, zero),
                   blockvec[k + 4], res2);
    i += width;
    k += 5;
    res += __half2float(res2.x) + __half2float(res2.y);
  }

  if (blockIdx.x == 0) {
    half2 lr_acc = {};
#pragma unroll
    for (int i = 0; i < LORA_SIZE2; i++) {
      half2 down_proj_el = down_sh[i];
      half2 up_el = __halves2half2(up[col + i * 2 * width],
                                   up[col + (i * 2 + 1) * width]);
      lr_acc = __hfma2(down_proj_el, up_el, lr_acc);
    }
    res += __half2float(lr_acc.x) + __half2float(lr_acc.y);
  }

  atomicAdd(&mul[col], res);
}

void vecquant3matmul_lora_faster_cuda(torch::Tensor vec, torch::Tensor mat,
                                      torch::Tensor down_proj, torch::Tensor up,
                                      torch::Tensor mul, torch::Tensor scales,
                                      torch::Tensor zeros) {
  int height = mat.size(0);
  int width = mat.size(1);
  int r = down_proj.size(-1);

  dim3 blocks((height + BLOCKHEIGHT - 1) / BLOCKHEIGHT,
              (width + BLOCKWIDTH - 1) / BLOCKWIDTH);
  dim3 threads(BLOCKWIDTH);

  if (r == 64) {
    VecQuant3MatMulKernelLoraFaster<64><<<blocks, threads>>>(
        (half2 *)vec.data_ptr(), mat.data_ptr<int>(),
        (half2 *)down_proj.data_ptr(), (half *)up.data_ptr(),
        mul.data_ptr<float>(), scales.data_ptr<float>(),
        zeros.data_ptr<float>(), height, width);
  } else if (r == 128) {
    VecQuant3MatMulKernelLoraFaster<128><<<blocks, threads>>>(
        (half2 *)vec.data_ptr(), mat.data_ptr<int>(),
        (half2 *)down_proj.data_ptr(), (half *)up.data_ptr(),
        mul.data_ptr<float>(), scales.data_ptr<float>(),
        zeros.data_ptr<float>(), height, width);
  } else if (r == 256) {
    VecQuant3MatMulKernelLoraFaster<256><<<blocks, threads>>>(
        (half2 *)vec.data_ptr(), mat.data_ptr<int>(),
        (half2 *)down_proj.data_ptr(), (half *)up.data_ptr(),
        mul.data_ptr<float>(), scales.data_ptr<float>(),
        zeros.data_ptr<float>(), height, width);
  } else if (r == 512) {
    VecQuant3MatMulKernelLoraFaster<512><<<blocks, threads>>>(
        (half2 *)vec.data_ptr(), mat.data_ptr<int>(),
        (half2 *)down_proj.data_ptr(), (half *)up.data_ptr(),
        mul.data_ptr<float>(), scales.data_ptr<float>(),
        zeros.data_ptr<float>(), height, width);
  } else {
    throw std::runtime_error("Unsupported r value");
  }
}
