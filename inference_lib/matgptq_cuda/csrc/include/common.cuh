/*
 * Copyright (C) 2026 Elvir Crncevic (elvircrn@gmail.com). All Rights Reserved.
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

#pragma once
#include <iostream>
#include <stdexcept>
#include <cstdint>
#include <climits>

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cuda.h>

#ifndef __host__
#define __host__
#endif

#ifndef __device__
#define __device__
#endif

#ifndef __forceinline__
#define __forceinline__
#endif

#define DEVICE_INLINE __forceinline__ __device__

#define CHECK_CUDA(func)                                                                                                                                                                               \
  {                                                                                                                                                                                                    \
    cudaError_t status = (func);                                                                                                                                                                       \
    if (status != cudaSuccess) {                                                                                                                                                                       \
      printf("CUDA API failed at line %d with error: %s (%d)\n", __LINE__, cudaGetErrorString(status), status);                                                                                        \
      exit(1);                                                                                                                                                                                         \
    }                                                                                                                                                                                                  \
  }

using u64 = unsigned long long;
using s32 = int;
using s64 = long long int;
using u32 = unsigned int;
using u8 = unsigned char;
using u16 = unsigned short;

union ColVal {
  u32 _;

  struct {
    unsigned short c;
    half v;
  } members;
};

template <class T> union ColValT {
  u32 _;

  struct {
    unsigned short c;
    T v;
  } members;
};

union Features {
  u32 _;

  struct {
    u32 is_async : 1;
    u32 is_bf16 : 1;
    u32 stub : 29;
  } flags;
};

// Instances of `Vec` are used to organize groups of >>registers<<, as needed
// for instance as inputs to tensor core operations. Consequently, all
// corresponding index accesses must be compile-time constants, which is why we
// extensively use `#pragma unroll` throughout the kernel code to guarantee
// this.
template <typename T, int n> struct Vec {
  T elems[n];

  DEVICE_INLINE T &operator[](int i) { return elems[i]; }
  DEVICE_INLINE const T operator[](int i) const { return elems[i]; }
};

namespace MATGPTQ_CUDA {
struct Float16Weight {
  static constexpr const char *name = "fp16";
  using Scalar_t = half;
  using Group_t = half2;
  using FragA = Vec<half2, 4>;
  using FragB = Vec<half2, 2>;
};

struct Bfloat16Weight {
  static constexpr const char *name = "bf16";
  using Scalar_t = nv_bfloat16;
  using Group_t = nv_bfloat162;
  using FragA = Vec<nv_bfloat162, 4>;
  using FragB = Vec<nv_bfloat162, 2>;
};

}