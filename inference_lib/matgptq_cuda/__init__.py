from typing import Tuple
import torch
import matgptq_cuda._CUDA 

@torch.library.custom_op("matgptq_cuda::pack_matgptq_qweights", mutates_args=())
def pack_matgptq_qweights(
    qweights: torch.Tensor,
    n_out_features: int,
    n_in_features: int,
    bitwidth: int,
    group_size: int,
    feature_flag: int,
) -> torch.Tensor:
    return matgptq_cuda._CUDA.pack_matgptq_qweights(qweights, n_out_features, n_in_features, bitwidth, group_size, feature_flag)


@torch.library.custom_op("matgptq_cuda::reorder_matgptq_scales", mutates_args=())
def reorder_matgptq_scales(
    scales: torch.Tensor, 
    m: int, 
    n: int, 
    group_size: int
) -> torch.Tensor:
    return matgptq_cuda._CUDA.reorder_matgptq_scales(scales, m, n, group_size)


@torch.library.custom_op("matgptq_cuda::matmul_matgptq", mutates_args=())
def matmul_matgptq(
    input: torch.Tensor,
    matgptq_weights: torch.Tensor,
    matgptq_scales: torch.Tensor,
    n_out_features: int,
    n_in_features: int,
    bitwidth: int,
    group_size: int,
    feature_flag: int,
) -> torch.Tensor:
    return matgptq_cuda._CUDA.matmul_matgptq(
        input,
        matgptq_weights,
        matgptq_scales,
        n_out_features,
        n_in_features,
        bitwidth,
        group_size,
        feature_flag,
    )

@matmul_matgptq.register_fake
def matmul_matgptq_fake(
    input: torch.Tensor,
    matgptq_weights: torch.Tensor,
    matgptq_scales: torch.Tensor,
    n_out_features: int,
    n_in_features: int,
    bitwidth: int,
    group_size: int,
    feature_flag: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return input.new_empty(
        (input.shape[-2], n_out_features), 
        dtype=torch.bfloat16 if feature_flag & 2 else torch.float16
    )