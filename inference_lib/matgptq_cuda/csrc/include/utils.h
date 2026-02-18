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

#include "common.cuh"

namespace MATGPTQ_CUDA {

void swizzle_weights(int64_t bitwidth, 
                     int64_t m, 
                     int64_t n, 
                     const u8 *input, 
                     u8 *output, 
                     bool is_bf16);

void split_up_n_bits(int64_t bitwidth, 
                     int64_t m, 
                     int64_t n, 
                     const u8 *input, 
                     u64 *buffer0, 
                     u32 *buffer1, 
                     u32 *buffer2, 
                     u64 *buffer3, 
                     u64 *buffer4);

void swizzle_matgptq(int64_t bitwidth, 
                     int64_t buffer_size, 
                     const u64 *buffer0, 
                     const u32 *buffer1, 
                     const u32 *buffer2, 
                     const u64 *buffer3, 
                     const u64 *buffer4, 
                     u32 *v0_lower, u32 *v0_higher, 
                     u32 *v1,
                     u32 *v2, 
                     u32 *v3_lower, u32 *v3_higher, 
                     u32 *v4_lower, u32 *v4_higher);

} // namespace MATGPTQ_CUDA