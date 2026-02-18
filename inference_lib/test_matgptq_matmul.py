import torch
import pytest
import torch.nn.functional as F

from matgptq_cuda.matgptq_linear import MatGPTQLinear

# ── Deterministic / precision settings ──────────────────────────────────────
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

@pytest.mark.parametrize("seed", [1])
@pytest.mark.parametrize("do_round", [True])
@pytest.mark.parametrize("bitwidth", [2, 3, 4, 6, 8])
@pytest.mark.parametrize("n_out_features", [16])
@pytest.mark.parametrize("n_in_features", [1024, 4096, 8192])
@pytest.mark.parametrize("batch_size", [1, 8, 16])
@pytest.mark.parametrize("group_size", [-1, 32, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_matgptq_matmul_random(
    seed: int,
    do_round: bool,
    bitwidth: int,
    n_out_features: int,
    n_in_features: int,
    batch_size: int,
    group_size: int,
    dtype: torch.dtype,
):
    torch.random.manual_seed(seed)

    gs = group_size if group_size != -1 else n_in_features

    qweights: torch.Tensor = torch.randint(0, 2 ** min(8, bitwidth), (n_out_features, n_in_features), dtype=torch.uint8)
    scales: torch.Tensor = torch.rand(n_out_features, n_in_features // gs, dtype=dtype)
    x: torch.Tensor = torch.randn(batch_size, n_in_features, dtype=dtype)

    if do_round:
        scales, x = scales.round(), x.round()

    weight_dq: torch.Tensor = ((qweights.to(dtype=torch.int32).unflatten(dim=-1, sizes=(-1, gs))- 2 ** (bitwidth - 1)) * scales.to(dtype=dtype)[..., None]).flatten(start_dim=-2) * (2 ** (8 - bitwidth))
    weight_dq, x = weight_dq.cuda(), x.cuda()

    matgptq_linear = MatGPTQLinear(qweights, scales, bitwidth, group_size)

    y_true: torch.Tensor = F.linear(x.to(dtype=weight_dq.dtype), weight_dq)
    y: torch.Tensor = matgptq_linear(x)
    torch.cuda.synchronize()

    assert torch.allclose(y, y_true), f"FAILED  m={n_out_features} n={batch_size} k={n_in_features} dtype={dtype} group_size={group_size} bits={bitwidth}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
