#
# Copyright (C) 2026 Max Kleinegger (mkleinegger@gmail.com). All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import torch.utils.cpp_extension as torch_cpp_ext
import os
import pathlib
import torch
import re

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup_dir = os.path.dirname(os.path.realpath(__file__))
HERE = pathlib.Path(__file__).absolute().parent
torch_version = torch.__version__


def remove_unwanted_pytorch_nvcc_flags():
    REMOVE_NVCC_FLAGS = [
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
    ]
    for flag in REMOVE_NVCC_FLAGS:
        try:
            torch_cpp_ext.COMMON_NVCC_FLAGS.remove(flag)
        except ValueError:
            pass


def detect_cc():
    dev = torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(dev)
    return major * 10 + minor


cc = detect_cc()


def get_cuda_arch_flags():
    flags = [
        "-gencode",
        "arch=compute_86,code=sm_86",
        "--expt-relaxed-constexpr",
        "--use_fast_math",
        "-std=c++20",
        "-O3",
        "-DNDEBUG",
        "-Xcompiler",
        "-funroll-loops",
        "-Xcompiler",
        "-ffast-math",
        "-Xcompiler",
        "-finline-functions",
    ]
    return flags


def third_party_cmake():
    import subprocess
    import sys
    import shutil

    cmake = shutil.which("cmake")
    if cmake is None:
        raise RuntimeError("Cannot find CMake executable.")

    retcode = subprocess.call([cmake, HERE])
    if retcode != 0:
        sys.stderr.write("Error: CMake configuration failed.\n")
        sys.exit(1)


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA is not available!"
    device = torch.cuda.current_device()
    print(f"Current device: {torch.cuda.get_device_name(device)}")
    print(f"Current CUDA capability: {torch.cuda.get_device_capability(device)}")

    major, minor = map(int, torch.cuda.get_device_capability(device))
    assert (major > 8) or (major >= 8 and minor >= 6), f"CUDA capability must be >= 8.6, yours is {torch.cuda.get_device_capability(device)}"
    
    print(f"PyTorch version: {torch_version}")
    m = re.match(r"^(\d+)\.(\d+)", torch_version)
    if not m:
        raise RuntimeError(f"Cannot parse PyTorch version '{torch_version}'")
    major, minor = map(int, m.groups())
    if major < 2 or (major == 2 and minor < 7):
        raise RuntimeError(f"PyTorch version must be >= 2.7, but found {torch_version}")

    third_party_cmake()
    remove_unwanted_pytorch_nvcc_flags()
    setup(
        name="matgptq_cuda",
        version="0.1.0",
        author="Elvir Crncevic",
        author_email="elvircrn@gmail.com",
        description="Efficient CUDA kernel with support for 2-8 'nested' bit-widths.",
        packages=["matgptq_cuda"],
        package_dir={"matgptq_cuda": "inference_lib/matgptq_cuda"},
        ext_modules=[
            CUDAExtension(
                name="matgptq_cuda._CUDA",
                sources=[
                    "inference_lib/matgptq_cuda/csrc/bindings.cpp",
                    "inference_lib/matgptq_cuda/csrc/utils.cpp",
                    "inference_lib/matgptq_cuda/csrc/matgptq_gemm.cu",
                ],
                include_dirs=[
                    os.path.join(setup_dir, "inference_lib/matgptq_cuda/csrc/include"),
                ],
                define_macros=[("TARGET_CUDA_ARCH", str(cc))],
                extra_compile_args={
                    "cxx": ["-std=c++20"],
                    "nvcc": get_cuda_arch_flags(),
                },
                extra_link_args=[
                    "-lcudart",
                    "-lcuda",
                ],
            )
        ],
        cmdclass={"build_ext": BuildExtension},
    )