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

#include "include/utils.h"

namespace MATGPTQ_CUDA {

void swizzle_weights(int64_t bitwidth, 
                     int64_t m, 
                     int64_t n, 
                     const u8 *input, 
                     u8 *output, 
                     bool is_bf16) 
  {
  auto output_host_updated_base = output;
  for (int _i = 0; _i < m; _i += 16) {
    for (int _j = 0; _j < n; _j += 16) {
      for (u64 j = 0; j < 8; j++) {
        for (u64 k = 0; k < 4; k++) {
          *(output++) = input[n * (_i + j) + _j + 2 * k + 0];
          *(output++) = input[n * (_i + j) + _j + 2 * k + 1];

          *(output++) = input[n * (_i + j + 8) + _j + 2 * k + 0];
          *(output++) = input[n * (_i + j + 8) + _j + 2 * k + 1];

          *(output++) = input[n * (_i + j) + 8 + _j + 2 * k + 0];
          *(output++) = input[n * (_i + j) + 8 + _j + 2 * k + 1];

          *(output++) = input[n * (_i + j + 8) + 8 + _j + 2 * k + 0];
          *(output++) = input[n * (_i + j + 8) + 8 + _j + 2 * k + 1];
        }
      }
    }
  }

  if (!is_bf16) {
    // We do this for fast int -> fp16 conversion.
    for (int i = 0; i < m * n; i += 8) {
      u8 base[8] = {output_host_updated_base[0], output_host_updated_base[1], output_host_updated_base[2], output_host_updated_base[3],
                    output_host_updated_base[4], output_host_updated_base[5], output_host_updated_base[6], output_host_updated_base[7]};

      output_host_updated_base[0] = base[0];
      output_host_updated_base[4] = base[1];
      output_host_updated_base[1] = base[2];
      output_host_updated_base[5] = base[3];
      output_host_updated_base[2] = base[4];
      output_host_updated_base[6] = base[5];
      output_host_updated_base[3] = base[6];
      output_host_updated_base[7] = base[7];

      output_host_updated_base += 8;
    }
  } else {
    // We do this for fast int -> bf16 conversion.
    //    7    6    5    4    3    2     1   0
    // 0b0000 0000 1111 0000 0000 0000 1111 0000
    //              1                    0
    //         3                    2
    //     4                   5
    //                    7                   6
    for (int i = 0; i < m * n; i += 8) {
      auto w = output_host_updated_base;
      std::swap(w[2 * 2], w[2 * 2 + 1]);

      const u8 base[8] = {output_host_updated_base[0], output_host_updated_base[1], output_host_updated_base[2], output_host_updated_base[3],
                          output_host_updated_base[4], output_host_updated_base[5], output_host_updated_base[6], output_host_updated_base[7]};

      output_host_updated_base[0] = base[6];
      output_host_updated_base[1] = base[0];
      output_host_updated_base[2] = base[2];
      output_host_updated_base[3] = base[5];
      output_host_updated_base[4] = base[7];
      output_host_updated_base[5] = base[1];
      output_host_updated_base[6] = base[3];
      output_host_updated_base[7] = base[4];

      output_host_updated_base += 8;
    }
  }
}

void swizzle_matgptq(int64_t bitwidth, 
                     int64_t buffer_size,
                     const u64 *buffer0, // 2 bit 2
                     const u32 *buffer1, // 1 bit 3
                     const u32 *buffer2, // 1 bit 4
                     const u64 *buffer3, // 2 bit 6
                     const u64 *buffer4, // 2 bit 8
                     u32 *v0_lower, u32 *v0_higher, 
                     u32 *v1, 
                     u32 *v2, 
                     u32 *v3_lower, u32 *v3_higher, 
                     u32 *v4_lower, u32 *v4_higher) 
  {
  auto reduce = [](u32 b) -> u32 {
    u32 res = ((b) & 0b11u) | (((b >> 2u) & 0b11u) << 4u) | (((b >> 4u) & 0b11u) << 8u) | (((b >> 6u) & 0b11u) << 12u) | (((b >> 8u) & 0b11u) << 16u) | (((b >> 10u) & 0b11u) << 20u) |
              (((b >> 12u) & 0b11u) << 24u) | (((b >> 14u) & 0b11u) << 28u);

    b >>= 16u;

    res |= (((b) & 0b11u) | (((b >> 2u) & 0b11u) << 4u) | (((b >> 4u) & 0b11u) << 8u) | (((b >> 6u) & 0b11u) << 12u) | (((b >> 8u) & 0b11u) << 16u) | (((b >> 10u) & 0b11u) << 20u) |
            (((b >> 12u) & 0b11u) << 24u) | (((b >> 14u) & 0b11u) << 28u))
           << 2u;

    return res;
  };

  auto reduce_1_0 = [](u32 b) {
    u32 res = 0;

    for (int j = 0; j < 2; j++) {
      for (int i = 0; i < 2; i++) {
        res |= (((b >> (8 * i + 0u)) & 0b1) << (0u + (3 - (2 * j + i)))) | (((b >> (8 * i + 1u)) & 0b1) << (4u + (3 - (2 * j + i)))) | (((b >> (8 * i + 2u)) & 0b1) << (8u + (3 - (2 * j + i)))) |
               (((b >> (8 * i + 3u)) & 0b1) << (12u + (3 - (2 * j + i)))) | (((b >> (8 * i + 4u)) & 0b1) << (16u + (3 - (2 * j + i)))) | (((b >> (8 * i + 5u)) & 0b1) << (20u + (3 - (2 * j + i)))) |
               (((b >> (8 * i + 6u)) & 0b1) << (24u + (3 - (2 * j + i)))) | (((b >> (8 * i + 7u)) & 0b1) << (28u + (3 - (2 * j + i))));
      }
      b >>= 16u;
    }
    return res;
  };

  for (u32 i = 0; i < buffer_size; i++) {
    u64 b = buffer0[i];
    v0_lower[i] = reduce(b);
    v0_higher[i] = reduce(static_cast<u32>(b >> 32ull));

    if (bitwidth >= 3) {
      v1[i] = reduce_1_0(buffer1[i]);
      if (bitwidth >= 4) {
        v2[i] = reduce_1_0(buffer2[i]);

        if (bitwidth >= 6) {
          u64 b = buffer3[i];
          v3_lower[i] = reduce(b);
          v3_higher[i] = reduce(static_cast<u32>(b >> 32ull));

          if (bitwidth >= 8) {
            u64 b = buffer4[i];
            v4_lower[i] = reduce(b);
            v4_higher[i] = reduce(static_cast<u32>(b >> 32ull));
          }
        }
      }
    }
  }
}

void split_up_n_bits(int64_t bitwidth, 
                     int64_t m, 
                     int64_t n, 
                     const u8 *input, 
                     u64 *buffer0, 
                     u32 *buffer1, 
                     u32 *buffer2, 
                     u64 *buffer3, 
                     u64 *buffer4) 
  {
  // Assumes buffers are zero'd before calling.
  int cnt = 0;
  for (u32 i = 0; i < (m * n) / (4 * 16 * 16); i++) {
    // We organize data lane_id-wise.
    // This needs to be done on a per-gpu basis.
    const u8 *__w = input + i * 16 * 16 * 4;
    for (int lane_id = 0; lane_id < 32; lane_id++) {
      // Each warp is responsible locally for 8 values in a tile.
      const u8 *_w = __w + lane_id * 8;
      // A single tensor core warp tile load is represented with 16 bits of b0, b1 and b2
      // ... but we need four of these to fill up the 1-bit buffers.
      for (u64 t = 0; t < 4; t++) {
        const u8 *w = _w + t * 16 * 16;
        for (u64 j = 0; j < 8; j++) {
          u32 val = *(w++);
          u64 lsb = (val & 0b11ull);
          u32 msb0 = (val & 0b100u) >> 2ull;
          u32 msb1 = (val & 0b1000u) >> 3ull;
          (*buffer0) |= (lsb << ((t * 8ull + j) * 2ull));
          (*buffer1) |= (msb0 << (t * 8ull + j));
          if (bitwidth >= 4) {
            (*buffer2) |= (msb1 << (t * 8ull + j));
            if (bitwidth >= 6) {
              val >>= 4u;
              msb0 = val & 0b11ull;
              (*buffer3) |= (static_cast<u64>(msb0) << (t * 8ull + j) * 2ull);
              if (bitwidth >= 8) {
                msb1 = (val >> 2ull) & 0b11ull;
                (*buffer4) |= (static_cast<u64>(msb1) << ((t * 8ull + j) * 2ull));
              }
            }
          }
        }
      }
      cnt++;
      buffer0++;
      buffer1++;
      if (bitwidth >= 4)
        buffer2++;
      if (bitwidth >= 6)
        buffer3++;
      if (bitwidth >= 8)
        buffer4++;
    }
  }
}
} // namespace MATGPTQ_CUDA