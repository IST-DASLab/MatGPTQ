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

#pragma once
#include "common.cuh"

namespace MATGPTQ_CUDA {

int matmul_matgptq_host(int64_t bitwidth, 
                        int64_t group_size,
                        int64_t m, // n_out_features
                        int64_t n, // n_in_features
                        int64_t k, // batch size
                        const void *matgptq_weights, 
                        const void *matgptq_scales, 
                        const void *input, 
                        void *output, 
                        cudaStream_t stream,
                        Features features);

}