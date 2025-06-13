// Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>
#include <torch/python.h>

void vecquant3matmul_cuda(torch::Tensor vec, torch::Tensor mat,
                          torch::Tensor mul, torch::Tensor scales,
                          torch::Tensor zeros);

void vecquant3matmul_faster_cuda(torch::Tensor vec, torch::Tensor mat,
                                 torch::Tensor mul, torch::Tensor scales,
                                 torch::Tensor zeros);

void vecquant3matmul_lora_faster_cuda(torch::Tensor vec, torch::Tensor mat,
                                      torch::Tensor down_proj, torch::Tensor up,
                                      torch::Tensor mul, torch::Tensor scales,
                                      torch::Tensor zeros);

void vecquant3matmul(torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
                     torch::Tensor scales, torch::Tensor zeros) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant3matmul_cuda(vec, mat, mul, scales, zeros);
}

void vecquant3matmul_faster(torch::Tensor vec, torch::Tensor mat,
                            torch::Tensor mul, torch::Tensor scales,
                            torch::Tensor zeros) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant3matmul_faster_cuda(vec, mat, mul, scales, zeros);
}

void vecquant3matmul_lora_faster(torch::Tensor vec, torch::Tensor mat,
                                 torch::Tensor down_proj, torch::Tensor up,
                                 torch::Tensor mul, torch::Tensor scales,
                                 torch::Tensor zeros) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant3matmul_lora_faster_cuda(vec, mat, down_proj, up, mul, scales, zeros);
}

void vec_mm_s4_lora_cuda(torch::Tensor vec, torch::Tensor mat,
                         torch::Tensor down_proj, torch::Tensor up,
                         torch::Tensor mul, torch::Tensor scales,
                         torch::Tensor zeros);

void vecquant4matmul_lora(torch::Tensor vec, torch::Tensor mat,
                          torch::Tensor down_proj, torch::Tensor up,
                          torch::Tensor mul, torch::Tensor scales,
                          torch::Tensor zeros) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vec_mm_s4_lora_cuda(vec, mat, down_proj, up, mul, scales, zeros);
}
void vec_mm_s4_cuda(torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
                    torch::Tensor scales, torch::Tensor zeros);
void vec_mm_s4(torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
               torch::Tensor scales, torch::Tensor zeros) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vec_mm_s4_cuda(vec, mat, mul, scales, zeros);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("vecquant3matmul", &vecquant3matmul,
        "Vector 3-bit Quantized Matrix Multiplication (CUDA)");
  m.def("vecquant3matmul_lora_faster", &vecquant3matmul_lora_faster,
        "Vector 3-bit Quantized Matrix Multiplication with EORA (CUDA)");
  m.def("vecquant3matmul_faster", &vecquant3matmul_faster,
        "Vector 3-bit Quantized Matrix Multiplication (CUDA), faster version");
  m.def("vecquant4matmul_lora", &vecquant4matmul_lora,
        "Vector 4-bit Quantized Matrix Multiplication with EORA(CUDA)");
  m.def("vecquant4matmul", &vec_mm_s4,
        "Vector 4-bit Quantized Matrix Multiplication with EORA(CUDA)");
}
