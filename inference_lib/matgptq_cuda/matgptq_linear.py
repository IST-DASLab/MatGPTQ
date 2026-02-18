import torch
from enum import IntEnum

import matgptq_cuda

class MatGTPQ_FeatureFlags(IntEnum):
    SYNC = 0
    ASYNC = 1
    IS_BF16 = 2

class MatGPTQLinear(torch.nn.Module):
    
    def __init__(
            self,
            qweight: torch.Tensor,
            scales: torch.Tensor,
            bitwidth: int,
            group_size: int
    ):
        super().__init__()
        self.n_out_features: int = qweight.shape[0]
        self.n_in_features: int = qweight.shape[1]
        self.bitwidth: int = bitwidth
        self.group_size: int = group_size

        self.flag: int = MatGTPQ_FeatureFlags.ASYNC + MatGTPQ_FeatureFlags.IS_BF16 * (scales.dtype == torch.bfloat16)

        self.matgptq_qweight: torch.Tensor = torch.nn.Parameter(
            torch.ops.matgptq_cuda.pack_matgptq_qweights(
                qweight, 
                self.n_out_features, 
                self.n_in_features, 
                self.bitwidth, 
                self.group_size, 
                self.flag), 
            requires_grad=False)
        self.matgptq_scales: torch.Tensor = torch.nn.Parameter(
            torch.ops.matgptq_cuda.reorder_matgptq_scales(
                scales, 
                self.n_out_features, 
                self.n_in_features, 
                self.group_size
            ), 
            requires_grad=False
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.ops.matgptq_cuda.matmul_matgptq(
            x,
            self.matgptq_qweight,
            self.matgptq_scales,
            self.n_out_features,
            self.n_in_features,
            self.bitwidth,
            self.group_size,
            self.flag,
        )
        return out
