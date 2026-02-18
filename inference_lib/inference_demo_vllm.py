import os
import argparse

import torch
from vllm import LLM, SamplingParams

import vllm_matgptq

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--tokenizer", type=str)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--max-model-len", type=int, default=1024 * 40)
    parser.add_argument("--gpu-memory-util", type=float, default=0.9)
    parser.add_argument("--tensor-parallel", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--prompts", type=str, nargs="+")
    args = parser.parse_args()

    llm = LLM(
        model=args.model,
        tokenizer=args.tokenizer,
        gpu_memory_utilization=args.gpu_memory_util,
        tensor_parallel_size=args.tensor_parallel,
        dtype=getattr(torch, args.dtype),
        max_model_len=args.max_model_len,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    outputs = llm.generate(args.prompts, sampling_params)

    for output in outputs:
        print(f"Prompt:    {output.prompt!r}")
        print(f"Generated: {output.outputs[0].text!r}")
        print("-" * 40)


if __name__ == "__main__":
    main()