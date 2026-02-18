/*
 * Copyright (C) 2026 Elvir Crncevic (elvircrn@gmail.com). All Rights Reserved.
 * Modified by Max Kleinegger (mkleinegger@gmail.com).
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

#include "include/common.cuh"
#include "include/gemm.h"
#include "include/benchmark.h"

#include <cassert>
#include <cstdio>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp16.hpp>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>

namespace MATGPTQ_CUDA {

#define UPDIV(X, Y) (((X) + (Y) - 1) / (Y))

static constexpr u32 FULL_MASK = 0xFFFFFFFFu;
static constexpr int TC_K = 8;

using Load_t = __int128_t;

DEVICE_INLINE void cp_async(half2 *__restrict__ dst, const half2 *__restrict__ src) {
  u32 s_dst = u32(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(s_dst), "l"(src));
}

DEVICE_INLINE void cp_async128(Load_t *__restrict__ dst, const Load_t *__restrict__ src) {
  u32 s_dst = u32(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" ::"r"(s_dst), "l"(src));
}

DEVICE_INLINE void cp_async128_scales(Load_t *__restrict__ dst, const Load_t *__restrict__ src) {
  u32 s_dst = u32(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" ::"r"(s_dst), "l"(src));
}

DEVICE_INLINE void cp_async_wait_all() { asm volatile("cp.async.wait_all;\n"); }

template <class T> DEVICE_INLINE auto fp16_to_fp162(const T &x) {
  if constexpr (std::is_same_v<T, half>) {
    return __half2half2(x);
  } else {
    return __bfloat162bfloat162(x);
  }
}

template <class T> DEVICE_INLINE T fp32_to_fp16(float x) {
  if constexpr (std::is_same_v<T, half>) {
    return __float2half_rn(x);
  } else {
    return __float2bfloat16_rn(x);
  }
}

template <class T> DEVICE_INLINE float fp16_to_float(const T &x) {
  if constexpr (std::is_same_v<T, half>) {
    return __half2float(x);
  } else {
    return __bfloat162float(x);
  }
}

template <class T> DEVICE_INLINE float2 fp162_to_float2(const T &x) {
  if constexpr (std::is_same_v<T, half2>) {
    return __half22float2(x);
  } else {
    return __bfloat1622float2(x);
  }
}

// Stolen from CUTLASS.
// Produces [0, 15] ranges.
template <u32 BITS> struct FastInterleavedAndUnbiasedNumericArrayConverterBF16 {
  using result_type = Bfloat16Weight::FragA;
  using source_type = Vec<u32, 8>;
  DEVICE_INLINE
  static result_type convert(u32 source) {
    result_type result;
#if 0
    auto bf = &result.elems->x;
#pragma unroll
    for (u32 i = 0; i < 8; i++) {
      bf[i] = __uint2bfloat16_rd((int(source & 0b1111u)));
      source >>= 4u;
    }
    return result;
#else
    auto *h = reinterpret_cast<uint32_t *>(&result);
    constexpr u32 MAGIC_MASK = 0b00000000011110000000000001111000u;
    h[0] = ((source >> 1u) & MAGIC_MASK) | 0x41804180u;
    h[1] = ((source >> 5u) & MAGIC_MASK) | 0x41804180u; //
    h[2] = ((source >> 9u) & MAGIC_MASK) | 0x41804180u;
    h[3] = ((source << 3u) & MAGIC_MASK) | 0x41804180u; //
    u32 DIFF_U32 = 0b01000001100000000100000110000000u;
    const auto DIFF = *reinterpret_cast<__nv_bfloat162 *>(&DIFF_U32);

    result.elems[0] -= DIFF;
    result.elems[1] -= DIFF;
    result.elems[2] -= DIFF;
    result.elems[3] -= DIFF;
#endif
    return result;
  }

  DEVICE_INLINE result_type operator()(u32 s) const { return convert(s); }
};

// Stolen from CUTLASS.
// Produces [0, 15] ranges.
template <u32 BITS> struct FastInterleavedAndUnbiasedNumericArrayConverter {
  using result_type = Float16Weight::FragA;
  using source_type = Vec<u32, 8>;

  DEVICE_INLINE
  static result_type convert(u32 const &source) {
    result_type result;

    uint32_t *h = reinterpret_cast<uint32_t *>(&result);
    uint32_t const i4s = reinterpret_cast<uint32_t const &>(source);

    // First, we extract the i4s and construct an intermediate fp16 number.
    static constexpr uint32_t immLut = (0xf0 & 0xcc) | 0xaa;
    static constexpr uint32_t BOTTOM_MASK = 0x000f000f;
    static constexpr uint32_t TOP_MASK = 0x00f000f0;
    static constexpr uint32_t I4s_TO_F16s_MAGIC_NUM = 0x64006400;

    // Note that the entire sequence only requires 1 shift instruction. This is
    // thanks to the register packing format and the fact that we force our
    // integers to be unsigned, and account for this in the fp16 subtractions.
    // In addition, I exploit the fact that sub and fma have the same throughput
    // in order to convert elt_23 and elt_67 to fp16 without having to shift
    // them to the bottom bits before hand.

    // Shift right by 8 to now consider elt_45 and elt_67. Issue first to hide
    // RAW dependency if we issue immediately before required.
    const uint32_t top_i4s = i4s >> 8;
    // Extract elt_01 - (i4s & 0x000f000f) | 0x64006400
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n" : "=r"(h[0]) : "r"(i4s), "n"(BOTTOM_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
    // Extract elt_23 (i4s & 0x00f000f0) | 0x64006400
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n" : "=r"(h[1]) : "r"(i4s), "n"(TOP_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
    // Extract elt_45 (top_i4s & 0x000f000f) | 0x64006400
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n" : "=r"(h[2]) : "r"(top_i4s), "n"(BOTTOM_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));
    // Extract elt_67 (top_i4s & 0x00f000f0) | 0x64006400
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n" : "=r"(h[3]) : "r"(top_i4s), "n"(TOP_MASK), "n"(I4s_TO_F16s_MAGIC_NUM), "n"(immLut));

    // I use inline PTX below because I am not sure if the compiler will emit
    // float2half instructions if I use the half2 ctor. In this case, I chose
    // performance reliability over code readability.

    // This is the half2 {1032, 1032} represented as an integer.
    // Haotian: subtract {1024, 1024} instead, we do not need to map to [-8, 7]
    static constexpr uint32_t FP16_TOP_MAGIC_NUM = 0x64006400;
    // This is the half2 {1 / 16, 1 / 16} represented as an integer.
    static constexpr uint32_t ONE_SIXTEENTH = 0x2c002c00;
    // Haotian: Let's use {-64, -64}.
    static constexpr uint32_t NEG_64 = 0xd400d400;

    // Finally, we construct the output numbers.
    // Convert elt_01
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[0]) : "r"(h[0]), "r"(FP16_TOP_MAGIC_NUM));
    // Convert elt_23
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(h[1]) : "r"(h[1]), "r"(ONE_SIXTEENTH), "r"(NEG_64));
    // Convert elt_45
    asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(h[2]) : "r"(h[2]), "r"(FP16_TOP_MAGIC_NUM));
    // Convert elt_67
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(h[3]) : "r"(h[3]), "r"(ONE_SIXTEENTH), "r"(NEG_64));

    return result;
  }

  DEVICE_INLINE result_type operator()(u32 s) const { return convert(s); }
};

[[nodiscard]] __device__ __host__ __forceinline__ int updiv(int x, int y) { return (x + y - 1) / y; }

// Wait until at most `n` async copy stages are still pending.
template <int n> DEVICE_INLINE void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(n)); }

template <class Conv, int BITS, class T, class U>
DEVICE_INLINE auto convert(const u32 j, const u32 v0_lower, const u32 v0_higher, const u32 v1, const u32 v2, const u32 v3_lower, const u32 v3_higher, const u32 v4_lower, const u32 v4_higher,
                           const T sixteen, const U minus) {
  static constexpr u32 ZERO_TWO_BIT = 0b00110011001100110011001100110011u;
  static constexpr u32 ZERO_ONE_BIT0 = 0b01000100010001000100010001000100u;
  static constexpr u32 ZERO_ONE_BIT1 = 0b10001000100010001000100010001000u;
  static constexpr u32 ZERO_TWO_BIT_UPPER = 0b11001100110011001100110011001100u;
  u32 bits;

  switch (j) {
  case 0:
    bits = (v0_lower & ZERO_TWO_BIT);
    if constexpr (BITS >= 3) {
      bits |= ((v1 >> 1u) & ZERO_ONE_BIT0);
    }
    if constexpr (BITS >= 4) {
      bits |= (v2 & ZERO_ONE_BIT1);
    }
    break;
  case 1:
    bits = ((v0_lower >> 2u) & ZERO_TWO_BIT);
    if constexpr (BITS >= 3) {
      bits |= (v1 & ZERO_ONE_BIT0);
    }
    if constexpr (BITS >= 4) {
      bits |= ((v2 << 1u) & ZERO_ONE_BIT1);
    }
    break;
  case 2:
    bits = (v0_higher & ZERO_TWO_BIT);
    if constexpr (BITS >= 3) {
      bits |= ((v1 << 1u) & ZERO_ONE_BIT0);
    }
    if constexpr (BITS >= 4) {
      bits |= ((v2 << 2u) & ZERO_ONE_BIT1);
    }
    break;
  case 3:
    bits = ((v0_higher >> 2u) & ZERO_TWO_BIT);
    if constexpr (BITS >= 3) {
      bits |= ((v1 << 2u) & ZERO_ONE_BIT0);
    }
    if constexpr (BITS >= 4) {
      bits |= ((v2 << 3u) & ZERO_ONE_BIT1);
    }
    break;
  default:
    bits = 0;
  }

  auto frag_a = Conv()(bits);

  if constexpr (BITS >= 6) {
    u32 bits2 = 0;

    switch (j) {
    case 0:
      bits2 = (v3_lower & ZERO_TWO_BIT);
      if constexpr (BITS >= 8) {
        bits2 |= ((v4_lower << 2u) & ZERO_TWO_BIT_UPPER);
      }
      break;
    case 1:
      bits2 = ((v3_lower >> 2u) & ZERO_TWO_BIT);
      if constexpr (BITS >= 8) {
        bits2 |= (v4_lower & ZERO_TWO_BIT_UPPER);
      }
      break;
    case 2:
      bits2 = (v3_higher & ZERO_TWO_BIT);
      if constexpr (BITS >= 8) {
        bits2 |= ((v4_higher << 2u) & ZERO_TWO_BIT_UPPER);
      }
      break;
    case 3:
      bits2 = ((v3_higher >> 2u) & ZERO_TWO_BIT);
      if constexpr (BITS >= 8) {
        bits2 |= (v4_higher & ZERO_TWO_BIT_UPPER);
      }
      break;
    default:
      bits2 = 0;
    }

    auto other_frag = Conv()(bits2);

#pragma unroll
    for (int i = 0; i < 4; i++) {
      other_frag.elems[i] = __hmul2(other_frag.elems[i], sixteen);
      frag_a.elems[i] = __hadd2(frag_a.elems[i], other_frag.elems[i]);
    }
  }

#pragma unroll
    for (int i = 0; i < 4; i++) {
      frag_a.elems[i] = __hsub2(frag_a.elems[i], minus);
    }

  return frag_a;
}

template <int K> constexpr u32 calc_out_dim() {
  if constexpr (K >= 8) {
    return 4;
  } else {
    return 2 * K;
  }
}

template <u32 OUT_K = 4> using FragC = Vec<float, OUT_K>;

__device__ inline void mma(const Float16Weight::FragA &a_frag, const Float16Weight::FragB &b_frag, FragC<> &frag_c) {
  const uint32_t *a = reinterpret_cast<const uint32_t *>(&a_frag);
  const uint32_t *b = reinterpret_cast<const uint32_t *>(&b_frag);
  float *c = reinterpret_cast<float *>(&frag_c);

  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
               : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

__device__ inline void mma(const Bfloat16Weight::FragA &a_frag, const Bfloat16Weight::FragB &b_frag, FragC<> &frag_c) {
  const uint32_t *a = reinterpret_cast<const uint32_t *>(&a_frag);
  const uint32_t *b = reinterpret_cast<const uint32_t *>(&b_frag);
  float *c = reinterpret_cast<float *>(&frag_c);

  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
               : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

template <class T> __device__ inline void __ldsm2(T &frag_b, const void *smem_ptr) {
  uint32_t *b = reinterpret_cast<uint32_t *>(&frag_b);
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(b[0]), "=r"(b[1]) : "r"(smem));
}

template <class T> __device__ __host__ const T &__min(const T &a, const T &b) { return (b < a) ? b : a; }

template <int BITS, int THREAD_COUNT, int GROUP_SIZE, int BLOCK_HEIGHT, int BLOCK_WIDTH, int K, int PAGE_SIZE_FP32, class Conv_t, class Weight_t>
__global__ void gptq_matryoshka(u32 m, u32 n,
                                // W 1st order stats
                                const u32 *__restrict__ v0_lower_ptr, const u32 *__restrict__ v0_higher_ptr, const u32 *__restrict__ v1_ptr, const u32 *__restrict__ v2_ptr,
                                const u32 *__restrict__ v3_lower_ptr, const u32 *__restrict__ v3_higher_ptr, const u32 *__restrict__ v4_lower_ptr, const u32 *__restrict__ v4_higher_ptr,
                                const typename Weight_t::Group_t *__restrict__ scales, const Load_t *__restrict__ x,
                                // Output
                                typename Weight_t::Scalar_t *__restrict__ y_fp16) {
  /*
                              K         K
           ┌─────────────┐ ┌─────┐   ┌─────┐
           │   block 0   │ │     │   │     │
           ├─────────────┤ │     │   │     │
           │   block 1   │ │     │   │     │
           └─────────────┘ │  X  │ = │  Y  │
           │    ...      │ │     │   │     │
           ┌─────────────┐ │     │   │     │
           │  block m-1  │ │     │   │     │
           └─────────────┘ └─────┘   └─────┘
  */
  extern __shared__ __align__(16) half2 s_x2[];

  auto *s_x2_tc = reinterpret_cast<typename Weight_t::Group_t *>(s_x2);

  static constexpr u32 WARP_SIZE = 32;
  static constexpr int BETA1 = 16;
  static constexpr u32 ROW_OFFSETS_SIZE = BLOCK_HEIGHT;
  static constexpr int NUM_FRAG_C_ELEMENTS = 4;
  static constexpr u32 WARP_COUNT = UPDIV(THREAD_COUNT, WARP_SIZE);

  // We now need to accumulate the results from dense and sparse computations.
  static constexpr u32 WARPS_PER_TILE_WIDTH = THREAD_COUNT / 32;

  const u32 t_x = threadIdx.x;
  const u32 thread_xy = t_x + (threadIdx.y * blockDim.x);
  const u32 lane_id = t_x & 0x1f;
  const u32 tile_row_id = blockIdx.x * BLOCK_HEIGHT + threadIdx.y;
  const u32 total_x_fp32 = n * K / 2;
  const u32 pipeline_stages = UPDIV(total_x_fp32, PAGE_SIZE_FP32);

  const u32 weights_offset0 = tile_row_id * n / 2 + thread_xy; // Simplified from tile_row_id * (n / 16) * 16 + thread_xy;
  const u32 subwarp_id = t_x / WARP_SIZE;
  const u32 warp_id = thread_xy / WARP_SIZE;

  y_fp16 += blockIdx.y * m * K + tile_row_id * BETA1;
  auto s_acc = reinterpret_cast<float *>(s_x2_tc) + threadIdx.y * WARPS_PER_TILE_WIDTH * WARP_SIZE * NUM_FRAG_C_ELEMENTS;
  x += blockIdx.y * n * K / 8 + warp_id * (n / 8) + lane_id;
  auto s_b_ptr_base = s_x2_tc + (4 * subwarp_id) * BETA1 * K / 2 + 4 * lane_id;

  v0_lower_ptr += weights_offset0;
  v0_higher_ptr += weights_offset0;

  if constexpr (BITS >= 3) {
    v1_ptr += weights_offset0;
  }

  if constexpr (BITS >= 4) {
    v2_ptr += weights_offset0;
  }

  if constexpr (BITS >= 6) {
    v3_lower_ptr += weights_offset0;
    v3_higher_ptr += weights_offset0;
  }

  if constexpr (BITS >= 8) {
    v4_lower_ptr += weights_offset0;
    v4_higher_ptr += weights_offset0;
  }

  static constexpr u32 SMEM_HEIGHT_FP16 = (2 * PAGE_SIZE_FP32) / TC_K;
  static constexpr u32 HEIGHT_OFFSET = PAGE_SIZE_FP32 / (K * 4) * TC_K;

  static constexpr u32 S_SCALE_COUNT_32 = (GROUP_SIZE == -1) ? 8 : ((SMEM_HEIGHT_FP16 / GROUP_SIZE) * 8);
  const u32 S_SCALE_COUNT_128 = min(((n / GROUP_SIZE) * 8) / 4, S_SCALE_COUNT_32 / 4);

  __shared__ __align__(16) typename Weight_t::Group_t s_scales[S_SCALE_COUNT_32];

  typename Weight_t::FragA _frag_a;
  typename Weight_t::FragB _frag_b;
  FragC<> _frag_c;

#pragma unroll
  for (int i = 0; i < 4; i++) {
    _frag_c[i] = 0.0f;
  }

  // TODO: This is horrible - scales need to be eventually reordered so that the loads can be done in Group_t with bank-conflict free loads.
  typename Weight_t::Scalar_t *s_scales_fp16 = reinterpret_cast<typename Weight_t::Scalar_t *>(s_scales); // We use this in case group_size == -1.
  typename Weight_t::Scalar_t *s_scales_fp16_lower_ptr = nullptr, *s_scales_fp16_higher_ptr = nullptr;

  if constexpr (GROUP_SIZE == -1) {
    scales += tile_row_id * 8;
  } else {
    scales += tile_row_id * 8 * (n / GROUP_SIZE);
  }

  if constexpr (GROUP_SIZE == -1) {
    for (u32 i = thread_xy; i < S_SCALE_COUNT_32; i += THREAD_COUNT) {
      s_scales[i] = scales[i];
    }
  }

  __syncthreads();

  ColValT<typename Weight_t::Scalar_t> colval{._ = 0u};

  auto scale128_ptr = reinterpret_cast<const Load_t *>(scales) + thread_xy;
  for (u32 iteration_id = 0, height_offset = 0, height_fp128_iter = n, max_glob_loads = n; iteration_id < pipeline_stages;
       iteration_id++, height_offset += HEIGHT_OFFSET, height_fp128_iter -= HEIGHT_OFFSET, max_glob_loads -= SMEM_HEIGHT_FP16) {
    if constexpr (GROUP_SIZE != -1) {
      if constexpr (GROUP_SIZE < 128) {
        s_scales_fp16_lower_ptr = reinterpret_cast<typename Weight_t::Scalar_t *>(s_scales) + warp_id * 16 * 2 + (lane_id / 4);
        s_scales_fp16_higher_ptr = reinterpret_cast<typename Weight_t::Scalar_t *>(s_scales) + warp_id * 16 * 2 + 8 + (lane_id / 4);
      } else {
        s_scales_fp16_lower_ptr = reinterpret_cast<typename Weight_t::Scalar_t *>(s_scales) + (warp_id / 2) * 16 + (lane_id / 4);
        s_scales_fp16_higher_ptr = reinterpret_cast<typename Weight_t::Scalar_t *>(s_scales) + (warp_id / 2) * 16 + 8 + (lane_id / 4);
      }
    }

    const auto height_fp128 = min(height_fp128_iter, HEIGHT_OFFSET);

    auto s_x_128 = reinterpret_cast<Load_t *>(s_x2_tc);

    static constexpr int MEM_OFFSET = WARP_SIZE;
    static constexpr int SMEM_OFFSET = THREAD_COUNT;
    int SCALE_SMEM_OFFSET = min(S_SCALE_COUNT_128, THREAD_COUNT);

    u32 t = thread_xy;

    auto s_scale128_ptr = reinterpret_cast<Load_t *>(s_scales) + thread_xy;

    auto move_scales_pred = [&]() {
      if constexpr (GROUP_SIZE != -1) {
        if (t < S_SCALE_COUNT_128) {
          cp_async128_scales(s_scale128_ptr, scale128_ptr);
          s_scale128_ptr += SCALE_SMEM_OFFSET;
          scale128_ptr += SCALE_SMEM_OFFSET;
          __pipeline_commit();
        }
      }
    };

    auto s_x_128_ptr = s_x_128 + lane_id * TC_K + warp_id;

#pragma unroll
    for (int rep = 0; rep < 4; rep++) {
      move_scales_pred();
      if (t < height_fp128) {
        cp_async128(s_x_128_ptr, x);
        t += THREAD_COUNT;
        s_x_128_ptr += SMEM_OFFSET;
        x += MEM_OFFSET;
      }
    }
    __pipeline_commit();

    u32 num_fp16_per_page_height = min(SMEM_HEIGHT_FP16, max_glob_loads);
    u32 num_iterations = UPDIV(num_fp16_per_page_height, (WARP_COUNT * 4 * 16));
    u32 num_iterations_pred = warp_id * 4 * 16;

    auto s_b_ptr = s_b_ptr_base;
    Conv_t conv;

    for (int it = 0; it < num_iterations; it++) {
      u32 v0_lower, v0_higher, v1, v2, v3_lower, v3_higher, v4_lower, v4_higher;
      if (num_iterations_pred < num_fp16_per_page_height) {
        v0_lower = *v0_lower_ptr;
        v0_higher = *v0_higher_ptr;
        v0_lower_ptr += THREAD_COUNT;
        v0_higher_ptr += THREAD_COUNT;

        if constexpr (BITS >= 3) {
          v1 = *v1_ptr;
          v1_ptr += THREAD_COUNT;
        }

        if constexpr (BITS >= 4) {
          v2 = *v2_ptr;
          v2_ptr += THREAD_COUNT;
        }

        if constexpr (BITS >= 6) {
          v3_lower = *v3_lower_ptr;
          v3_higher = *v3_higher_ptr;
          v3_lower_ptr += THREAD_COUNT;
          v3_higher_ptr += THREAD_COUNT;
        }

        if constexpr (BITS >= 8) {
          v4_lower = *v4_lower_ptr;
          v4_higher = *v4_higher_ptr;
          v4_lower_ptr += THREAD_COUNT;
          v4_higher_ptr += THREAD_COUNT;
        }

        for (int rep = 0; rep < 4; rep++) {
          move_scales_pred();
          if (t < height_fp128) {
            cp_async128(s_x_128_ptr, x);
            __pipeline_commit();
            t += THREAD_COUNT;
            s_x_128_ptr += SMEM_OFFSET;
            x += MEM_OFFSET;
          }
        }
      }

      cp_async_wait_all(); // TODO: Move this somewhere useful.
      __syncthreads();     // TODO: Figure out why this is necessary?

      if (num_iterations_pred < num_fp16_per_page_height) {
#pragma unroll
        for (int j = 0; j < 4; j++) {
          const auto SIXTEEN = fp16_to_fp162<Weight_t::Scalar_t>(fp32_to_fp16<Weight_t::Scalar_t>(16.f));
          const auto MINUS = fp16_to_fp162<Weight_t::Scalar_t>(fp32_to_fp16<Weight_t::Scalar_t>((1 << (BITS - 1))));
          _frag_a = convert<Conv_t, BITS, Weight_t::Group_t, Weight_t::Group_t>(j, v0_lower, v0_higher, v1, v2, v3_lower, v3_higher, v4_lower, v4_higher, SIXTEEN, MINUS);

          if constexpr (GROUP_SIZE != -1) {
            if constexpr (GROUP_SIZE < 128) {
              _frag_a.elems[0] = __hmul2(_frag_a.elems[0], fp16_to_fp162(*(s_scales_fp16_lower_ptr + (j / (128 / (GROUP_SIZE * 2))) * 16)));
              _frag_a.elems[2] = __hmul2(_frag_a.elems[2], fp16_to_fp162(*(s_scales_fp16_lower_ptr + (j / (128 / (GROUP_SIZE * 2))) * 16)));
              _frag_a.elems[1] = __hmul2(_frag_a.elems[1], fp16_to_fp162(*(s_scales_fp16_higher_ptr + (j / (128 / (GROUP_SIZE * 2))) * 16)));
              _frag_a.elems[3] = __hmul2(_frag_a.elems[3], fp16_to_fp162(*(s_scales_fp16_higher_ptr + (j / (128 / (GROUP_SIZE * 2))) * 16)));
            } else {
              _frag_a.elems[0] = __hmul2(_frag_a.elems[0], fp16_to_fp162(*s_scales_fp16_lower_ptr));
              _frag_a.elems[2] = __hmul2(_frag_a.elems[2], fp16_to_fp162(*s_scales_fp16_lower_ptr));
              _frag_a.elems[1] = __hmul2(_frag_a.elems[1], fp16_to_fp162(*s_scales_fp16_higher_ptr));
              _frag_a.elems[3] = __hmul2(_frag_a.elems[3], fp16_to_fp162(*s_scales_fp16_higher_ptr));
            }
          }

          __ldsm2(_frag_b, s_b_ptr + j * K * 16 / 2);
          mma(_frag_a, _frag_b, _frag_c);
        }

        if constexpr (GROUP_SIZE != -1) {
          s_scales_fp16_lower_ptr += (16 * (WARP_COUNT * 64)) / GROUP_SIZE;
          s_scales_fp16_higher_ptr += (16 * (WARP_COUNT * 64)) / GROUP_SIZE;
        }

        s_b_ptr += (WARP_COUNT * 4 * K * 16) / 2;
        num_iterations_pred += WARP_COUNT * 64;
      }
    }

    __syncthreads();
  }

  
  if (subwarp_id) {
#pragma unroll
    for (int i = 0; i < NUM_FRAG_C_ELEMENTS; i++) {
      s_acc[(subwarp_id - 1) * NUM_FRAG_C_ELEMENTS * WARP_SIZE + lane_id * NUM_FRAG_C_ELEMENTS + i] = _frag_c[i];
    }
  }

  __syncthreads();

  if (!subwarp_id) {
#pragma unroll
    for (int i = 0; i < WARPS_PER_TILE_WIDTH - 1; i++) {
#pragma unroll
      for (int j = 0; j < NUM_FRAG_C_ELEMENTS; j++) {
        _frag_c.elems[j] += s_acc[i * NUM_FRAG_C_ELEMENTS * WARP_SIZE + lane_id * NUM_FRAG_C_ELEMENTS + j];
      }
    }
  }


  __syncthreads();
  if (!warp_id) {
    auto c = _frag_c.elems;
    auto groupID = lane_id >> 2;
    auto threadID_in_group = lane_id % 4;
    auto row = groupID;
    auto col = (threadID_in_group * 2) + (0 & 0x1);

    if constexpr (GROUP_SIZE == -1) {
      c[0] *= fp16_to_float(s_scales_fp16[lane_id / 4]);
      c[1] *= fp16_to_float(s_scales_fp16[lane_id / 4]);
      c[2] *= fp16_to_float(s_scales_fp16[lane_id / 4 + 8]);
      c[3] *= fp16_to_float(s_scales_fp16[lane_id / 4 + 8]);
    }

    s_acc[col * BETA1 + row] = c[0];
    s_acc[(col + 1) * BETA1 + row] = c[1];
    row = groupID + 8;
    s_acc[col * BETA1 + row] = c[2];
    s_acc[(col + 1) * BETA1 + row] = c[3];
  }
  __syncthreads();

  for (u32 i = t_x; i < BETA1 * K; i += THREAD_COUNT) {
    static constexpr float SCALE = static_cast<float>(1 << (8 - BITS)); // NOLINT(*-use-auto)
    y_fp16[m * (i / BETA1) + i % BETA1] = fp32_to_fp16<typename Weight_t::Scalar_t>(s_acc[i] * SCALE);
  }
}

template <int BITS, int THREAD_COUNT, int GROUP_SIZE, int BLOCK_HEIGHT, int BLOCK_WIDTH, int K, int PAGE_SIZE_FP32, class Conv_t, class Weight_t>
__global__ void gptq_matryoshka_simt(u32 m, u32 n,
                                     // W 1st order stats
                                     const u32 *__restrict__ v0_lower_ptr, const u32 *__restrict__ v0_higher_ptr, const u32 *__restrict__ v1_ptr, const u32 *__restrict__ v2_ptr,
                                     const u32 *__restrict__ v3_lower_ptr, const u32 *__restrict__ v3_higher_ptr, const u32 *__restrict__ v4_lower_ptr, const u32 *__restrict__ v4_higher_ptr,
                                     const typename Weight_t::Group_t *__restrict__ scales, const Load_t *__restrict__ x,
                                     // Output
                                     typename Weight_t::Scalar_t *__restrict__ y_fp16) {
  /*
                              K         K
           ┌─────────────┐ ┌─────┐   ┌─────┐
           │   block 0   │ │     │   │     │
           ├─────────────┤ │     │   │     │
           │   block 1   │ │     │   │     │
           └─────────────┘ │  X  │ = │  Y  │
           │    ...      │ │     │   │     │
           ┌─────────────┐ │     │   │     │
           │  block m-1  │ │     │   │     │
           └─────────────┘ └─────┘   └─────┘
  */
  static constexpr u32 WARP_SIZE = 32;
  static constexpr int BETA1 = 16;
  static constexpr u32 ROW_OFFSETS_SIZE = BLOCK_HEIGHT;
  static constexpr u32 WARP_COUNT = UPDIV(THREAD_COUNT, WARP_SIZE);
  extern __shared__ __align__(16) half2 s_x2[];
  auto *s_x2_simt = reinterpret_cast<typename Weight_t::Group_t *>(s_x2);

  // We now need to accumulate the results from dense and sparse computations.
  static constexpr u32 WARPS_PER_TILE_WIDTH = THREAD_COUNT / 32;

  const u32 t_x = threadIdx.x;
  const u32 thread_xy = t_x + (threadIdx.y * blockDim.x);
  const u32 lane_id = t_x & 0x1f;
  const u32 tile_row_id = blockIdx.x * BLOCK_HEIGHT + threadIdx.y;

  const u32 weights_offset0 = tile_row_id * n / 2 + thread_xy; // Simplified from tile_row_id * (n / 16) * 16 + thread_xy;
  const u32 subwarp_id = t_x / WARP_SIZE;
  const u32 warp_id = thread_xy / WARP_SIZE;

  y_fp16 += blockIdx.y * m * K + tile_row_id * BETA1;

  static constexpr u32 WARPS_PER_COLUMN = WARP_COUNT / K;
  u32 s_t = thread_xy % (WARPS_PER_COLUMN * WARP_SIZE);
  x += (warp_id / WARPS_PER_COLUMN) * (n / 8) + s_t;
  static constexpr int MEM_OFFSET = WARP_SIZE * WARPS_PER_COLUMN;
  static constexpr int SMEM_OFFSET = WARP_SIZE * WARPS_PER_COLUMN;
  static constexpr u32 SMEM_HEIGHT_FP16 = (2 * PAGE_SIZE_FP32) / K;

  const u32 pipeline_stages = UPDIV(n, SMEM_HEIGHT_FP16);

  // / 4 gives us 128 bits.
  static constexpr u32 HEIGHT_OFFSET_128 = (PAGE_SIZE_FP32 / 4) / K;
  static constexpr u32 S_SCALE_COUNT_32 = (GROUP_SIZE == -1) ? 8 : ((SMEM_HEIGHT_FP16 / GROUP_SIZE) * 8);
  auto s_b_ptr_base = s_x2_simt + warp_id * 4 * 8;

  v0_lower_ptr += weights_offset0;
  v0_higher_ptr += weights_offset0;

  if constexpr (BITS >= 3) {
    v1_ptr += weights_offset0;
  }

  if constexpr (BITS >= 4) {
    v2_ptr += weights_offset0;
  }

  if constexpr (BITS >= 6) {
    v3_lower_ptr += weights_offset0;
    v3_higher_ptr += weights_offset0;
  }

  if constexpr (BITS >= 8) {
    v4_lower_ptr += weights_offset0;
    v4_higher_ptr += weights_offset0;
  }

  __shared__ __align__(16) typename Weight_t::Group_t s_scales[S_SCALE_COUNT_32];

  typename Weight_t::FragA _frag_a;
  typename Weight_t::FragB _frag_b;
  FragC<calc_out_dim<K>()> _frag_c;

#pragma unroll
  for (int i = 0; i < calc_out_dim<K>(); i++) {
    _frag_c[i] = 0.0f;
  }

  // TODO: This is horrible - scales need to be eventually reordered so that the loads can be done in Group_t with bank-conflict free loads.
  typename Weight_t::Scalar_t *s_scales_fp16 = reinterpret_cast<typename Weight_t::Scalar_t *>(s_scales); // We use this in case group_size == -1.
  typename Weight_t::Scalar_t *s_scales_fp16_lower_ptr = nullptr, *s_scales_fp16_higher_ptr = nullptr;

  if constexpr (GROUP_SIZE == -1) {
    scales += tile_row_id * 8;
  } else {
    scales += tile_row_id * 8 * (n / GROUP_SIZE);
  }

  if constexpr (GROUP_SIZE == -1) {
    for (u32 i = thread_xy; i < S_SCALE_COUNT_32; i += THREAD_COUNT) {
      s_scales[i] = scales[i];
    }
  }

  __syncthreads();

  ColValT<typename Weight_t::Scalar_t> colval{._ = 0u};

  auto scale128_ptr = reinterpret_cast<const Load_t *>(scales) + thread_xy;
  for (u32 iteration_id = 0, height_fp128_iter = (n / 8), max_glob_loads = n; iteration_id < pipeline_stages;
       iteration_id++, height_fp128_iter -= HEIGHT_OFFSET_128, max_glob_loads -= SMEM_HEIGHT_FP16) {
    u32 s_t = thread_xy % (WARPS_PER_COLUMN * WARP_SIZE);
    u32 num_fp16_per_page_height = min(SMEM_HEIGHT_FP16, max_glob_loads);
    const u32 S_SCALE_COUNT_128 = min((16 * (num_fp16_per_page_height / GROUP_SIZE)) / 8, S_SCALE_COUNT_32 / 4);
    int SCALE_SMEM_OFFSET = min(S_SCALE_COUNT_128, THREAD_COUNT);
    u32 t_scale = thread_xy;
    if constexpr (GROUP_SIZE != -1) {
      if constexpr (GROUP_SIZE < 128) {
        s_scales_fp16_lower_ptr = reinterpret_cast<typename Weight_t::Scalar_t *>(s_scales) + warp_id * 16 * 2 + (lane_id / 4);
        s_scales_fp16_higher_ptr = reinterpret_cast<typename Weight_t::Scalar_t *>(s_scales) + warp_id * 16 * 2 + 8 + (lane_id / 4);
      } else {
        s_scales_fp16_lower_ptr = reinterpret_cast<typename Weight_t::Scalar_t *>(s_scales) + (warp_id / 2) * 16 + (lane_id / 4);
        s_scales_fp16_higher_ptr = reinterpret_cast<typename Weight_t::Scalar_t *>(s_scales) + (warp_id / 2) * 16 + (lane_id / 4) + 8;
      }
    }

    const auto height_fp128 = min(height_fp128_iter, HEIGHT_OFFSET_128);

    auto s_x_128_ptr = reinterpret_cast<Load_t *>(s_x2_simt + (warp_id / WARPS_PER_COLUMN) * (PAGE_SIZE_FP32 / K)) + s_t;
    auto s_scale128_ptr = reinterpret_cast<Load_t *>(s_scales) + thread_xy;

    auto move_scales_pred = [&]() {
      if constexpr (GROUP_SIZE != -1) {
        if (t_scale < S_SCALE_COUNT_128) {
          cp_async128_scales(s_scale128_ptr, scale128_ptr);
          s_scale128_ptr += SCALE_SMEM_OFFSET;
          scale128_ptr += SCALE_SMEM_OFFSET;
          __pipeline_commit();
          t_scale += THREAD_COUNT;
        }
      }
    };

    move_scales_pred();
    if (s_t < height_fp128) {
      cp_async128(s_x_128_ptr, x);
      s_t += SMEM_OFFSET;
      s_x_128_ptr += SMEM_OFFSET;
      x += MEM_OFFSET;
    }

    __pipeline_commit();

    u32 num_iterations = UPDIV(num_fp16_per_page_height, (WARP_COUNT * 4 * 16));
    u32 num_iterations_pred = warp_id * 4 * 16;

    auto *__restrict__ s_b_compute_fp32 = s_b_ptr_base;
    Conv_t conv;

    for (int it = 0; it < num_iterations; it++) {
      u32 v0_lower, v0_higher, v1, v2, v3_lower, v3_higher, v4_lower, v4_higher;
      if (num_iterations_pred < num_fp16_per_page_height) {
        v0_lower = *v0_lower_ptr;
        v0_higher = *v0_higher_ptr;
        v0_lower_ptr += THREAD_COUNT;
        v0_higher_ptr += THREAD_COUNT;

        if constexpr (BITS >= 3) {
          v1 = *v1_ptr;
          v1_ptr += THREAD_COUNT;
        }

        if constexpr (BITS >= 4) {
          v2 = *v2_ptr;
          v2_ptr += THREAD_COUNT;
        }

        if constexpr (BITS >= 6) {
          v3_lower = *v3_lower_ptr;
          v3_higher = *v3_higher_ptr;
          v3_lower_ptr += THREAD_COUNT;
          v3_higher_ptr += THREAD_COUNT;
        }

        if constexpr (BITS >= 8) {
          v4_lower = *v4_lower_ptr;
          v4_higher = *v4_higher_ptr;
          v4_lower_ptr += THREAD_COUNT;
          v4_higher_ptr += THREAD_COUNT;
        }

        move_scales_pred();
        if (s_t < height_fp128) {
          cp_async128(s_x_128_ptr, x);
          __pipeline_commit();
          s_t += SMEM_OFFSET;
          s_x_128_ptr += SMEM_OFFSET;
          x += MEM_OFFSET;
        }
      }

      cp_async_wait_all(); // TODO: Move this somewhere useful.
      __syncthreads();     // TODO: Figure out why this is necessary?

      if (num_iterations_pred < num_fp16_per_page_height) {
#pragma unroll
        for (int j = 0; j < 4; j++) {
          const auto SIXTEEN = fp16_to_fp162<Weight_t::Scalar_t>(fp32_to_fp16<Weight_t::Scalar_t>(16.f));
          const auto MINUS = fp16_to_fp162<Weight_t::Scalar_t>(fp32_to_fp16<Weight_t::Scalar_t>((1 << (BITS - 1))));
          _frag_a = convert<Conv_t, BITS, Weight_t::Group_t, Weight_t::Group_t>(j, v0_lower, v0_higher, v1, v2, v3_lower, v3_higher, v4_lower, v4_higher, SIXTEEN, MINUS);

          if constexpr (GROUP_SIZE != -1) {
            if constexpr (GROUP_SIZE < 128) {
              _frag_a.elems[0] = __hmul2(_frag_a.elems[0], fp16_to_fp162(*(s_scales_fp16_lower_ptr + (j / (128 / (GROUP_SIZE * 2))) * 16)));
              _frag_a.elems[2] = __hmul2(_frag_a.elems[2], fp16_to_fp162(*(s_scales_fp16_lower_ptr + (j / (128 / (GROUP_SIZE * 2))) * 16)));
              _frag_a.elems[1] = __hmul2(_frag_a.elems[1], fp16_to_fp162(*(s_scales_fp16_higher_ptr + (j / (128 / (GROUP_SIZE * 2))) * 16)));
              _frag_a.elems[3] = __hmul2(_frag_a.elems[3], fp16_to_fp162(*(s_scales_fp16_higher_ptr + (j / (128 / (GROUP_SIZE * 2))) * 16)));
            } else {
              _frag_a.elems[0] = __hmul2(_frag_a.elems[0], fp16_to_fp162(*s_scales_fp16_lower_ptr));
              _frag_a.elems[2] = __hmul2(_frag_a.elems[2], fp16_to_fp162(*s_scales_fp16_lower_ptr));
              _frag_a.elems[1] = __hmul2(_frag_a.elems[1], fp16_to_fp162(*s_scales_fp16_higher_ptr));
              _frag_a.elems[3] = __hmul2(_frag_a.elems[3], fp16_to_fp162(*s_scales_fp16_higher_ptr));
            }
          }

          auto matmul_simt = [&] {
            static constexpr u32 SMEM_HEIGHT_FP32 = HEIGHT_OFFSET_128 * 4;
#pragma unroll
            for (int k = 0; k < K; k++) {
              auto b_ptr = s_b_compute_fp32 + (lane_id % 4) + j * 8;
              float2 w_fp32, x_fp32;

              // TODO: Can this be replaced with an LDSM2 /4
              x_fp32 = fp162_to_float2(b_ptr[SMEM_HEIGHT_FP32 * k]);

              w_fp32 = fp162_to_float2(_frag_a[0]);
              _frag_c[k] = fmaf(x_fp32.x, w_fp32.x, _frag_c[k]);
              _frag_c[k] = fmaf(x_fp32.y, w_fp32.y, _frag_c[k]);

              w_fp32 = fp162_to_float2(_frag_a[1]);
              _frag_c[K + k] = fmaf(x_fp32.x, w_fp32.x, _frag_c[K + k]);
              _frag_c[K + k] = fmaf(x_fp32.y, w_fp32.y, _frag_c[K + k]);

              x_fp32 = fp162_to_float2(b_ptr[SMEM_HEIGHT_FP32 * k + 4]);

              w_fp32 = fp162_to_float2(_frag_a[2]);
              _frag_c[k] = fmaf(x_fp32.x, w_fp32.x, _frag_c[k]);
              _frag_c[k] = fmaf(x_fp32.y, w_fp32.y, _frag_c[k]);

              w_fp32 = fp162_to_float2(_frag_a[3]);
              _frag_c[K + k] = fmaf(x_fp32.x, w_fp32.x, _frag_c[K + k]);
              _frag_c[K + k] = fmaf(x_fp32.y, w_fp32.y, _frag_c[K + k]);
            }
          };

          matmul_simt();
        }

        if constexpr (GROUP_SIZE != -1) {
          // Every warp will access 4 tiles, each of which has 16x16 weights.
          static constexpr u32 SCALE_OFFSET = ((WARP_COUNT * 4 * 16 * 16) / GROUP_SIZE);
          s_scales_fp16_lower_ptr += SCALE_OFFSET;
          s_scales_fp16_higher_ptr += SCALE_OFFSET;
        }

        s_b_compute_fp32 += (WARP_COUNT * 4 * 16) / 2;
        num_iterations_pred += WARP_COUNT * 64;
      }
    }

    __syncthreads();
  }

  auto s_accumulator = reinterpret_cast<float *>(s_x2_simt);

  if (subwarp_id) {
#pragma unroll
    for (int i = 0; i < calc_out_dim<K>(); i++) {
      s_accumulator[(subwarp_id - 1) * calc_out_dim<K>() * WARP_SIZE + lane_id * calc_out_dim<K>() + i] = _frag_c[i];
    }
  }

  __syncthreads();

  if (!warp_id) {
#pragma unroll
    for (int i = 0; i < WARPS_PER_TILE_WIDTH - 1; i++) {
#pragma unroll
      for (int j = 0; j < calc_out_dim<K>(); j++) {
        _frag_c.elems[j] += s_accumulator[i * calc_out_dim<K>() * WARP_SIZE + lane_id * calc_out_dim<K>() + j];
      }
    }
  }

  __syncthreads();

  if (!warp_id) {
    for (int j = 0; j < calc_out_dim<K>(); j++) {
      for (u32 i = 1; i <= 2; i++) {
        _frag_c[j] += __shfl_down_sync(FULL_MASK, _frag_c[j], i);
      }

      if (lane_id % 4 == 0) {
        auto local_offset = 8 * (j / (calc_out_dim<K>() / 2));
        auto offset_height = lane_id / 4 + local_offset;
        auto offset_width = (j % (calc_out_dim<K>() / 2));
        float res = _frag_c[j];

        if constexpr (GROUP_SIZE == -1) {
          res *= fp16_to_float(s_scales_fp16[lane_id / 4 + local_offset]);
        }

        static constexpr float SCALE = static_cast<float>(1 << (8 - BITS)); // NOLINT(*-use-auto)
        res *= SCALE;
        y_fp16[m * offset_width + offset_height] = fp32_to_fp16<typename Weight_t::Scalar_t>(res);
      }
    }
  }
}

#define CALL_BATCHED_V2                                                                                                                                                                                \
  if (k < 8) {                                                                                                                                                                                         \
    gptq_matryoshka_simt<BITS, BLOCK_WIDTH * 16, GROUP_SIZE, 1, BLOCK_WIDTH, K, PAGE_SIZE_FP32, Conv, W_t>                                                                                             \
        <<<dim3(updiv(m, 16 * BLOCK_HEIGHT), 1, 1), dim3(__min(updiv(n, 16), BLOCK_WIDTH) * 16, __min(updiv(m, 16), BLOCK_HEIGHT), 1), PAGE_SIZE_FP32 * sizeof(float), stream>>>(                      \
            m, n, v0_lower_ptr, v0_higher_ptr, v1_ptr, v2_ptr, v3_lower_ptr, v3_higher_ptr, v4_lower_ptr, v4_higher_ptr, scales_ptr, reinterpret_cast<const Load_t *>(input), y_ptr);                  \
  } else {                                                                                                                                                                                             \
    gptq_matryoshka<BITS, BLOCK_WIDTH * 16, GROUP_SIZE, 1, BLOCK_WIDTH, K, PAGE_SIZE_FP32, Conv, W_t>                                                                                                  \
        <<<dim3(updiv(m, 16 * BLOCK_HEIGHT), k / 8, 1), dim3(__min(updiv(n, 16), BLOCK_WIDTH) * 16, __min(updiv(m, 16), BLOCK_HEIGHT), 1), PAGE_SIZE_FP32 * sizeof(float), stream>>>(                  \
            m, n, v0_lower_ptr, v0_higher_ptr, v1_ptr, v2_ptr, v3_lower_ptr, v3_higher_ptr, v4_lower_ptr, v4_higher_ptr, scales_ptr, reinterpret_cast<const Load_t *>(input), y_ptr);                  \
  }

#define CALL_GROUP_SIZE                                                                                                                                                                                \
  if (group_size == -1) {                                                                                                                                                                              \
    static constexpr int GROUP_SIZE = -1;                                                                                                                                                              \
    CALL_BATCHED_V2;                                                                                                                                                                                   \
  } else if (group_size == 32) {                                                                                                                                                                       \
    static constexpr int GROUP_SIZE = 32;                                                                                                                                                              \
    CALL_BATCHED_V2;                                                                                                                                                                                   \
  } else if (group_size == 128) {                                                                                                                                                                      \
    static constexpr int GROUP_SIZE = 128;                                                                                                                                                             \
    CALL_BATCHED_V2;                                                                                                                                                                                   \
  } 

#define CALL_TYPE                                                                                                                                                                                      \
  if (features.flags.is_bf16) {                                                                                                                                                                        \
  using Conv = FastInterleavedAndUnbiasedNumericArrayConverterBF16<BITS>;                                                                                                                              \
  using W_t = Bfloat16Weight;                                                                                                                                                                          \
  const auto *scales_ptr = (const __nv_bfloat162 *)matgptq_scales;                                                                                                                                     \
  __nv_bfloat16 *y_ptr = ((__nv_bfloat16 *)output);                                                                                                                                                    \
  CALL_GROUP_SIZE                                                                                                                                                                                      \
} else {                                                                                                                                                                                               \
  using Conv = FastInterleavedAndUnbiasedNumericArrayConverter<BITS>;                                                                                                                                  \
  const auto *scales_ptr = (const half2 *)matgptq_scales;                                                                                                                                              \
  using W_t = Float16Weight;                                                                                                                                                                           \
  half *y_ptr = ((half *)output);                                                                                                                                                                      \
  CALL_GROUP_SIZE                                                                                                                                                                                      \
}                                                                                                                                                                                                      \

#define CALL_BITS                                                                                                                                                                                      \
  if (bitwidth == 2) {                                                                                                                                                                                 \
    static constexpr u32 BITS = 2;                                                                                                                                                                     \
    CALL_TYPE                                                                                                                                                                                          \
  } else if (bitwidth == 3) {                                                                                                                                                                          \
    static constexpr u32 BITS = 3;                                                                                                                                                                     \
    CALL_TYPE                                                                                                                                                                                          \
  } else if (bitwidth == 4) {                                                                                                                                                                          \
    static constexpr u32 BITS = 4;                                                                                                                                                                     \
    CALL_TYPE                                                                                                                                                                                          \
  } else if (bitwidth == 6) {                                                                                                                                                                          \
    static constexpr u32 BITS = 6;                                                                                                                                                                     \
    CALL_TYPE                                                                                                                                                                                          \
  } else if (bitwidth == 8) {                                                                                                                                                                          \
    static constexpr u32 BITS = 8;                                                                                                                                                                     \
    CALL_TYPE                                                                                                                                                                                          \
  } 

#define CALL_K                                                                                                                                                                                         \
  if (k == 1) {                                                                                                                                                                                        \
    static constexpr int K = 1;                                                                                                                                                                        \
    CALL_BITS                                                                                                                                                                                          \
  } else if (k == 2) {                                                                                                                                                                                 \
    static constexpr int K = 2;                                                                                                                                                                        \
    CALL_BITS                                                                                                                                                                                          \
  } else if (k == 4) {                                                                                                                                                                                 \
    static constexpr int K = 4;                                                                                                                                                                        \
    CALL_BITS                                                                                                                                                                                          \
  } else {                                                                                                                                                                                             \
    static constexpr int K = 8;                                                                                                                                                                        \
    CALL_BITS                                                                                                                                                                                          \
  }

int matmul_matgptq_host(   
    int64_t bitwidth,
    int64_t group_size,
    int64_t m,  // n_out_features
    int64_t n,  // n_in_features
    int64_t k,  // batch size
    const void* matgptq_weights,
    const void* matgptq_scales,
    const void* input,
    void* output,
    cudaStream_t stream,
    Features features) 
  {
  const u32 *_matgptq_weights = reinterpret_cast<const u32 *>(matgptq_weights);
  const int offset = (m * n) / 32;
  
  const auto *v0_lower_ptr = _matgptq_weights;
  const auto *v0_higher_ptr = _matgptq_weights + offset;
  const auto *v1_ptr = _matgptq_weights + 2 * offset;
  const auto *v2_ptr = _matgptq_weights + 3 * offset;
  const auto *v3_lower_ptr = _matgptq_weights + 4 * offset;
  const auto *v3_higher_ptr = _matgptq_weights + 5 * offset;
  const auto *v4_lower_ptr = _matgptq_weights + 6 * offset;
  const auto *v4_higher_ptr = _matgptq_weights + 7 * offset;

  constexpr int PAGE_SIZE_FP32 = 4096;

  auto F = [&] {
    constexpr int BLOCK_HEIGHT = 1;
    constexpr int BLOCK_WIDTH = 16;
    CALL_K
  };

  F();
  if (!features.flags.is_async) {
    CHECK_CUDA(cudaDeviceSynchronize());
  }

  return 0;
}
}