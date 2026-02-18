import argparse

import torch
import triton
import triton.testing

from matgptq_cuda.matgptq_linear import MatGPTQLinear

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
torch.backends.cudnn.benchmark = False
torch.set_float32_matmul_precision("highest")
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

bitwidthS = [2, 3, 4, 6, 8]


def make_benchmark(n_out_features, n_in_features, group_size, dtype, no_cudagraph):
    @triton.testing.perf_report(
        triton.testing.Benchmark(
            x_names=["batch_size"],
            x_vals=[1, 2, 4, 8, 16, 32, 64],
            x_log=True,
            line_arg="bitwidth",
            line_vals=bitwidthS,
            line_names=[f"{b}-bit" for b in bitwidthS],
            ylabel="Speedup over torch (larger is better)",
            plot_name=f"matgptq speedup  |  M={n_out_features} N={n_in_features} gs={group_size} {dtype}",
            args={"n_out_features": n_out_features, "n_in_features": n_in_features, "group_size": group_size, "dtype": dtype, "no_cudagraph": no_cudagraph},
        )
    )
    def benchmark(batch_size, bitwidth, n_out_features, n_in_features, group_size, dtype, no_cudagraph):
        torch.random.manual_seed(1)
        gs = group_size if group_size != -1 else n_in_features

        qweights = torch.randint(0, 2**bitwidth, (n_out_features, n_in_features), dtype=torch.uint8)
        scales = torch.randn(n_out_features, n_in_features // gs, dtype=dtype).round()
        x = torch.randn(batch_size, n_in_features, dtype=dtype).round().cuda()

        m8a_linear = MatGPTQLinear(qweights, scales, bitwidth, group_size).cuda()
        weight_dq = ((qweights.to(torch.int32).unflatten(-1, (-1, gs)) - 2 ** (bitwidth - 1)) * scales.to(dtype)[..., None]).flatten(-2).cuda() * (2 ** (8 - bitwidth))

        quantiles = [0.5, 0.2, 0.8]
        benchmark = triton.testing.do_bench if no_cudagraph else triton.testing.do_bench_cudagraph

        torch_ms, torch_min, torch_max = benchmark(lambda: torch.nn.functional.linear(x, weight_dq), rep=200, quantiles=quantiles)
        kernel_ms, kernel_min, kernel_max = benchmark(lambda: m8a_linear(x), rep=200, quantiles=quantiles)

        return torch_ms / kernel_ms, torch_min / kernel_max, torch_max / kernel_min

    return benchmark


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--n_out_features", type=int, default=4096)
    parser.add_argument("--n_in_features", type=int, default=4096)
    parser.add_argument("--group-size", type=int, default=128, choices=[-1, 32, 128])
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--no_cudagraph", action="store_true", default=False)
    args = parser.parse_args()

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    bench = make_benchmark(args.n_out_features, args.n_in_features, args.group_size, dtype, args.no_cudagraph)
    bench.run(show_plots=True, print_data=True)
