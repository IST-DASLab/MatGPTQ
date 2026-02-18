
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

#define IS_PROFILER_MODE 0

#if IS_PROFILER_MODE
static constexpr int BENCHMARK_SPQR_NUM_RUNS = 16;
static constexpr int BENCHMARK_SPQR_WARMUPS = 0;
#else
static constexpr int BENCHMARK_SPQR_NUM_RUNS = 2000;
static constexpr int BENCHMARK_SPQR_WARMUPS = 1900;
#endif

namespace MATGPTQ_CUDA {
struct Timer {
  cudaEvent_t ce_start{}, ce_stop{};
  cudaStream_t stream;

  inline void start() { cudaEventRecord(ce_start, stream); }

  inline float end_and_measure() {
    float time_ms{};
    cudaEventRecord(ce_stop, stream);
    cudaEventSynchronize(ce_stop);
    cudaEventElapsedTime(&time_ms, ce_start, ce_stop);
    // Returns ms
    return time_ms;
  }

  inline Timer(cudaStream_t stream) : stream(stream) {
    cudaEventCreate(&ce_start);
    cudaEventCreate(&ce_stop);
  }

  inline Timer(Timer &&timer) = delete;

  inline Timer(const Timer &timer) = delete;

  ~Timer() {
    cudaEventDestroy(ce_start);
    cudaEventDestroy(ce_stop);
  }
};
}