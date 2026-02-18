/*
 * Copyright (C) 2026 Max Kleinegger (mkleinegger@gmail.com). All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#include <ATen/ATen.h>
#include <torch/types.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#ifndef MATGPTQ_CUDA_DISABLE_PYBIND
#include <torch/extension.h>
#endif

#include <cuda.h>

#include <vector>
#include <iostream>
#include <utility>

#include "include/common.cuh"
#include "include/gemm.h"
#include "include/utils.h"
#include "include/benchmark.h"

namespace MATGPTQ_CUDA {

/*********************************** PACKING ************************************/

torch::Tensor pack_matgptq_qweights(torch::Tensor const &qweights, 
                                    int64_t n_out_features, 
                                    int64_t n_in_features, 
                                    int64_t bitwidth, 
                                    int64_t group_size, 
                                    int64_t feature_flag) 
  {
  TORCH_CHECK(qweights.dtype() == torch::kUInt8, "qweights must be uint8");
  TORCH_CHECK(qweights.is_contiguous(), "qweights must be contiguous");
  TORCH_CHECK(qweights.dim() == 2, "qweights must be 2D (n_out_features, n_in_features)");
  TORCH_CHECK(qweights.size(0) == n_out_features, "n_out_features does not match qweights.size(0)");
  TORCH_CHECK(qweights.size(1) == n_in_features, "n_in_features does not match qweights.size(1)");
  TORCH_CHECK(bitwidth == 2 || bitwidth == 3 || bitwidth == 4 || bitwidth == 6 || bitwidth == 8, "bitwidth must be one of {2, 3, 4, 6, 8}");
  TORCH_CHECK(group_size == -1 || group_size == 32 || group_size == 128, "group_size must be one of {-1, 32, 128}");
  if (group_size != -1) {
    TORCH_CHECK(n_in_features % group_size == 0, "group_size must divide n_in_features. Got group_size=", group_size,", n_in_features=", n_in_features);
  }
  
  Features features = Features{._ = static_cast<u32>(feature_flag)};
  int dev = qweights.get_device();
  auto buffer0 = torch::zeros({n_out_features * n_in_features / 32}, torch::dtype(torch::kUInt64).device(torch::kCPU));
  auto buffer1 = torch::zeros({n_out_features * n_in_features / 32}, torch::dtype(torch::kUInt32).device(torch::kCPU));
  auto buffer2 = torch::zeros(n_out_features * n_in_features / 32, torch::dtype(torch::kUInt32).device(torch::kCPU));
  auto buffer3 = torch::zeros(n_out_features * n_in_features / 32, torch::dtype(torch::kUInt64).device(torch::kCPU));
  auto buffer4 = torch::zeros(n_out_features * n_in_features / 32, torch::dtype(torch::kUInt64).device(torch::kCPU));
  auto matgptq_weights = torch::zeros({bitwidth * n_out_features * n_in_features / 32}, torch::dtype(torch::kUInt32).device(torch::kCPU));

  u8 *w_swizzled = new u8[n_out_features * n_in_features];
  swizzle_weights(bitwidth, n_out_features, n_in_features, (u8 *)qweights.to(torch::kCPU).contiguous().data_ptr(), w_swizzled, features.flags.is_bf16);
  split_up_n_bits(bitwidth, n_out_features, n_in_features, w_swizzled, (u64 *)buffer0.data_ptr(), (u32 *)buffer1.data_ptr(), (u32 *)buffer2.data_ptr(), (u64 *)buffer3.data_ptr(),
                  (u64 *)buffer4.data_ptr());

  int offset = (n_out_features * n_in_features) / 32;
  u32 *v_base = (u32 *)matgptq_weights.data_ptr();

  swizzle_matgptq(bitwidth, offset, (u64 *)buffer0.data_ptr(), (u32 *)buffer1.data_ptr(), (u32 *)buffer2.data_ptr(), (u64 *)buffer3.data_ptr(), (u64 *)buffer4.data_ptr(), v_base, v_base + offset,
                  v_base + 2 * offset, v_base + 3 * offset, v_base + 4 * offset, v_base + 5 * offset, v_base + 6 * offset, v_base + 7 * offset);

  delete[] w_swizzled;

  return matgptq_weights.to(torch::kCUDA, dev);
}

torch::Tensor reorder_matgptq_scales(torch::Tensor const &scales, 
                                     int64_t n_out_features, 
                                     int64_t n_in_features, 
                                     int64_t group_size) 
  {
  // TORCH_CHECK(scales.dtype() == torch::kFloat16, "scales must be float16");
  TORCH_CHECK(scales.is_contiguous(), "scales must be contiguous");
  TORCH_CHECK(scales.dim() == 2, "qweights must be 2D (n_out_features, n_in_features)");
  TORCH_CHECK(group_size == -1 || group_size == 32 || group_size == 128, "group_size must be one of {-1, 32, 128}");
  if (group_size != -1) {
    TORCH_CHECK(n_in_features % group_size == 0, "group_size must divide n_in_features. Got group_size=", group_size,", n_in_features=", n_in_features);
  }

  int dev = scales.get_device();
  auto scales_cpu = scales.to(torch::kCPU).contiguous();
  auto matgptq_scales =  torch::zeros_like(scales_cpu);

  if (scales_cpu.dtype() == torch::kFloat16) {
    auto* scales_ptr = scales_cpu.data_ptr<at::Half>();
    auto* matgptq_scale_ptr = matgptq_scales.data_ptr<at::Half>();

    if (group_size == -1) {
      std::memcpy(matgptq_scale_ptr, scales_ptr, scales_cpu.nbytes());
    } else {
      int64_t groups = n_in_features / group_size;
      int64_t out_idx = 0;

      for (int64_t i = 0; i < n_out_features; i += 16) {
        for (int64_t j = 0; j < groups; j++) {
          for (int64_t k = 0; k < 16; k++) {
            matgptq_scale_ptr[out_idx++] = scales_ptr[(i + k) * groups + j];
          }
        }
      }
    }
  } else {
    auto* scales_ptr = scales_cpu.data_ptr<at::BFloat16>();
    auto* matgptq_scale_ptr = matgptq_scales.data_ptr<at::BFloat16>();

    if (group_size == -1) {
      std::memcpy(matgptq_scale_ptr, scales_ptr, scales_cpu.nbytes());
    } else {
      int64_t groups = n_in_features / group_size;
      int64_t out_idx = 0;

      for (int64_t i = 0; i < n_out_features; i += 16) {
        for (int64_t j = 0; j < groups; j++) {
          for (int64_t k = 0; k < 16; k++) {
            matgptq_scale_ptr[out_idx++] = scales_ptr[(i + k) * groups + j];
          }
        }
      }
    }
  }

  return matgptq_scales.to(torch::Device(torch::kCUDA, dev));
}

/*********************************** Inference ************************************/

torch::Tensor matmul_matgptq(torch::Tensor const &input, 
                             torch::Tensor const &matgptq_weights, 
                             torch::Tensor const &matgptq_scales, 
                             int64_t n_out_features, 
                             int64_t n_in_features, 
                             int64_t bitwidth,
                             int64_t group_size, 
                             int64_t feature_flag) 
  {
  torch::checkDeviceType("matmul_matgptq", {input, matgptq_weights, matgptq_scales}, at::DeviceType::CUDA);
  torch::checkAllSameGPU("matmul_matgptq", {{input, "input", 0}, {matgptq_weights, "matgptq_weights", 1}, {matgptq_scales, "matgptq_scales", 2}});
  TORCH_CHECK(matgptq_weights.dtype() == torch::kUInt32, "matgptq_weights must be uint32");
  TORCH_CHECK(matgptq_weights.is_contiguous(), "matgptq_weights must be contiguous");
  TORCH_CHECK(matgptq_weights.dim() == 1, "qweights must be 1D");
  TORCH_CHECK(matgptq_weights.size(0) == (bitwidth * n_out_features * n_in_features / 32), "matgptq_weights has the wrong size");
  // TORCH_CHECK(matgptq_scales.dtype() == torch::kFloat16, "matgptq_scales must be float16");
  TORCH_CHECK(bitwidth == 2 || bitwidth == 3 || bitwidth == 4 || bitwidth == 6 || bitwidth == 8, "bitwidth must be one of {2, 3, 4, 6, 8}");
  TORCH_CHECK(group_size == -1 || group_size == 32 || group_size == 128, "group_size must be one of {-1, 32, 128}");
  if (group_size != -1) {
    TORCH_CHECK(n_in_features % group_size == 0, "group_size must divide n_in_features. Got group_size=", group_size,", n_in_features=", n_in_features);
  }

  Features features = Features{._ = static_cast<u32>(feature_flag)};
  int dev = matgptq_weights.get_device();

  auto k = input.size(-2);

  auto output_dev = (features.flags.is_bf16) ? torch::kBFloat16 : torch::kFloat16;
  auto output = torch::empty({k, n_out_features}, torch::dtype(output_dev).device(torch::kCUDA, dev));

  matmul_matgptq_host(
    bitwidth, 
    group_size, 
    n_out_features, 
    n_in_features, 
    k, 
    matgptq_weights.data_ptr(), 
    matgptq_scales.data_ptr(), 
    input.data_ptr(), 
    output.data_ptr(),
    at::cuda::getCurrentCUDAStream(dev), 
    features
  );

  return output;
}

#ifdef TORCH_EXPOSE_DEFINITIONS
TORCH_EXPOSE_DEFINITIONS
#endif

TORCH_LIBRARY(_matgptq_cuda_C, m) {
  m.def("pack_matgptq_qweights(Tensor qweights, int n_out_features, int n_in_features, int bitwidth, int group_size, int feature_flag) -> Tensor");
  m.def("reorder_matgptq_scales(Tensor scales, int n_out_features, int n_in_features, int group_size) -> Tensor");
  m.def("matmul_matgptq(Tensor input, Tensor matgptq_weights, Tensor matgptq_scales, int n_out_features, int n_in_features, int bitwidth, int group_size, int feature_flag) -> Tensor");
}

TORCH_LIBRARY_IMPL(_matgptq_cuda_C, CUDA, m) {
  m.impl("pack_matgptq_qweights",     TORCH_FN(&pack_matgptq_qweights));
  m.impl("reorder_matgptq_scales",    TORCH_FN(&reorder_matgptq_scales));
  m.impl("matmul_matgptq",            TORCH_FN(&matmul_matgptq));
}

//====== pybind ======

#define DEFINE_pybind(name) m.def(#name, &name, #name);

#ifndef MATGPTQ_CUDA_DISABLE_PYBIND
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pack_matgptq_qweights",    &pack_matgptq_qweights,    "pack_matgptq_qweights");
  m.def("reorder_matgptq_scales",   &reorder_matgptq_scales,   "reorder_matgptq_scales");
  m.def("matmul_matgptq",           &matmul_matgptq,           "matmul_matgptq_batched");

}
#endif
} // namespace MATGPTQ_CUDA