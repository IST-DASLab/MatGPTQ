import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import time
import numpy as np
from enum import IntEnum
from typing import Optional, Optional, Tuple, Union

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, StaticCache

import torch
torch.autograd.set_grad_enabled(False)
torch.set_printoptions(sci_mode=False)

from src.model_utils import load_mat_gptq_weights
from src.quant_utils_matgpqt import matryoshka_slice
from matgptq_cuda.matgptq_linear import MatGPTQLinear

def replace_with_matgptq_linear(
    model: AutoModelForCausalLM,
    quantized_weights_path: Union[str, os.PathLike],
    quantized_config_path: Optional[str] = None,
    master_bitwidth: int = 8,
    slice_bitwidth: int = 8,
    group_size: int = 128,
):
    def _replace(layer_name: str, bitwidth: int):
        layer_data = torch.load(
            os.path.join(quantized_weights_path, layer_name, "data.pt"),
            weights_only=True,
        )
        m_layer = MatGPTQLinear(
            matryoshka_slice(
                layer_data["qweight"],
                master_bitwidth=master_bitwidth,
                slice_bitwidth=bitwidth,
            ).cuda(),
            layer_data["scale"].cuda(),
            bitwidth,
            group_size
        )
        # Derive parent module and attribute name from the dotted layer path
        *parent_path, attr = layer_name.split(".")
        parent = model.get_submodule(".".join(parent_path)) if parent_path else model
        setattr(parent, attr, m_layer)

    if quantized_config_path:
        with open(quantized_config_path, "r") as f:
            for line in f:
                layer_name, bitwidth = line.strip().split(":")
                _replace(layer_name.strip(), int(bitwidth.strip()))
    else:
        for layer_name in sorted(os.listdir(quantized_weights_path)):
            if not os.path.isdir(os.path.join(quantized_weights_path, layer_name)):
                continue
            _replace(layer_name, slice_bitwidth)

    return model


def time_prof(func, init=lambda: (), sync=lambda: (), benchmark_runs=16, number_of_runs=128, warmups=64):
    times = []
    for _ in range(warmups):
        func()
    sync()

    for _ in range(benchmark_runs):
        t1 = time.time()
        for _ in range(number_of_runs):
            func()
        sync()
        t2 = time.time()
        mean = (t2 - t1) / number_of_runs
        print(mean)
        times.append(mean)
    return np.array(times)

def decode_one_tokens(model, cur_token, input_pos, cache_position, past_key_values):
    logits = model(
        cur_token,
        position_ids=input_pos,
        cache_position=cache_position,
        past_key_values=past_key_values,
        return_dict=False,
        use_cache=False,
    )[0]
    new_token = torch.argmax(logits[:, -1], dim=-1)[:, None]
    return new_token

class Mode(IntEnum):
    DENSE = 0
    FAKE_QUANTIZED = 1
    KERNEL_QUANTIZED = 2

class BenchEnd2End:
    def __init__(
        self,
        pretrained_model_path: str,
        flag: Mode,
        *,
        device: str = "cuda",
        compile_forward: bool = True,
        # Quantization args (used for FAKE_QUANTIZED / KERNEL_QUANTIZED):
        quant_weights_path: Optional[str] = None,
        quantized_config_path: Optional[str] = None,
        quant_master_level: int = 8,
        quant_uniform_bitwidth: int = 8,
        group_size: int = 128,
        dtype: torch.dtype = torch.float16,
        verbose: bool = False,
    ):
        self.flag = flag
        self.device = device
        self.dtype = dtype
        self.compile_forward = compile_forward

        # ---- Load config/model along the 3 paths ----
        from_pretrained_orig = AutoModelForCausalLM.from_pretrained
        from_pretrained_overriden = from_pretrained_orig
        # Override from_pretrained
        if flag == Mode.KERNEL_QUANTIZED:
            def from_pretrained_overriden(*args, **kwargs):
                model = from_pretrained_orig(*args, **kwargs)
                model = replace_with_matgptq_linear(model, quant_weights_path, quantized_config_path, quant_master_level,quant_uniform_bitwidth, group_size)                
                return model
        elif flag == Mode.FAKE_QUANTIZED:
            def from_pretrained_overriden(*args, **kwargs):
                model = from_pretrained_orig(*args, **kwargs)
                model = load_mat_gptq_weights(model, quant_weights_path, quantized_config_path, quant_master_level, quant_uniform_bitwidth)
                return model
    
        AutoModelForCausalLM.from_pretrained = staticmethod(from_pretrained_overriden)

        self.config = AutoConfig.from_pretrained(pretrained_model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_path,
            trust_remote_code=True,
            torch_dtype=self.dtype,
            config=self.config,
        )
        self.model = self.model.to(device=self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path, use_fast=False)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()
        print(f"Model loaded on {self.device} | dtype={self.dtype} | compiled={self.compile_forward}")
        if verbose:
            print(self.model)

    def generate(self, input_str) -> Tuple:
        inputs = self.tokenizer(input_str, return_tensors="pt").to(device=self.device)

        input_ids = inputs.input_ids
        seq_len = input_ids.shape[1]

        cache_position = torch.arange(seq_len, dtype=torch.int64, device=self.device)
        generated_ids = torch.zeros(1, seq_len * 2 + 1, dtype=torch.int, device=self.device)
        generated_ids[:, cache_position] = input_ids.to(self.device).to(torch.int)

        past_key_values = StaticCache(
            config = self.model.config,
            max_batch_size=1,
            max_cache_len=seq_len * 2 + 1,
            device=self.device,
            dtype=torch.float16,
        )

        logits = self.model(
            input_ids, cache_position=cache_position, past_key_values=past_key_values, return_dict=False, use_cache=True
        )[0]
        next_token = torch.argmax(logits[:, [-1]], dim=-1).to(torch.int)
        generated_ids[:, [seq_len]] = next_token

        torch._dynamo.config.capture_scalar_outputs = True

        with torch.no_grad():
            # Compile the CUDA graph
            if self.compile_forward:
                decode_one_tokens_compiled = torch.compile(decode_one_tokens, fullgraph=True, dynamic=True, mode="reduce-overhead")
            else:
                decode_one_tokens_compiled = decode_one_tokens
            

            # Generate tokens one by one
            cache_position = torch.tensor([seq_len + 1], device="cuda")
            times = time_prof(lambda: decode_one_tokens_compiled(
                self.model, next_token.clone(), None, cache_position, past_key_values
            ), init=lambda: (), sync=torch.cuda.synchronize, benchmark_runs=16, number_of_runs=16, warmups=16)

            return times


def main():
    import argparse

    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--pretrained_model_path", type=str, required=True,
                        help="HF model id or path for the *base* model (dense weights).")
    parser.add_argument("--execution_mode", choices=[0, 1, 2], required=True, type=int,
                        help="0=DENSE (FP16), 1=FAKE_QUANTIZED (load_compressed_weights), 2=KERNEL_QUANTIZED (replace_linear).")
    parser.add_argument("--input_text", type=str, default="Starwars is ", help="Prompt to prime the model/cache before steady-state benchmarking.")
    parser.add_argument("--no_compile", action="store_true", help="Disable torch.compile on model.forward.")

    # Quantization-related
    parser.add_argument("--quant_weights_path", type=str, default=None, help="Path to quantized weights (required for modes 1 and 2).")
    parser.add_argument("--quant_non_uniform_config_path", type=str, default=None, help="Path to quantization config.")
    parser.add_argument("--quant_master_level", type=int, default=8)
    parser.add_argument("--quant_uniform_bitwidth", type=int, default=8)
    parser.add_argument("--group_size", type=int, default=128)

    args = parser.parse_args()
    mode = Mode(args.execution_mode)

    with torch.inference_mode():
        demo = BenchEnd2End(
            args.pretrained_model_path,
            mode,
            device="cuda",
            compile_forward=not args.no_compile,
            quant_weights_path=args.quant_weights_path,
            quantized_config_path=args.quant_non_uniform_config_path,
            quant_master_level=args.quant_master_level,
            quant_uniform_bitwidth=args.quant_uniform_bitwidth,
            group_size=args.group_size,
        )

        print(f"before: {torch.cuda.max_memory_allocated() / 1024**3:.3f} GB")
        timings = demo.generate(args.input_text)
        print(f"after:  {torch.cuda.max_memory_allocated() / 1024**3:.3f} GB")

        durations = np.array(timings)
        print("\nRaw per-step means (s):")
        print(durations)

        print(f"Mean  = {durations.mean():.6f}s")
        print(f"Median= {np.median(durations):.6f}s")
        print(f"Best  = {np.min(durations):.6f}s")

        tok_per_s = len(durations) / np.sum(durations)
        ms_per_tok = (1.0 / tok_per_s) * 1000.0
        print(f"\n( Generation speed: {tok_per_s:.1f} tok/s | Latency: {ms_per_tok:.2f} ms/tok )\n")

if __name__ == "__main__":
    main()
