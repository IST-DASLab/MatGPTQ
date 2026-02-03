import argparse
from functools import partial

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    import wandb

    has_wandb = True
except ModuleNotFoundError:
    has_wandb = False

from src.data_utils import get_data
from src.common_utils import fix_seed
from src.model_utils import load_gptq_weights, load_mat_gptq_weights
from src.metrics import compute_perplexity, compute_perplexity_layer_per_layer

def parse_args():
    parser = argparse.ArgumentParser()
    # Model params
    parser.add_argument("--model_name_or_path", type=str, required=True, help="The name or path to the model being quantized.")
    # Data params
    parser.add_argument("--sequence_length", default=None, type=int, help="Length of sequences.")
    parser.add_argument("--eval_datasets", nargs="+", type=str, default=["wikitext2", "c4", "fineweb_edu"], help="Datasets used for evaluation.")
    parser.add_argument("--eval_batch_size", type=int, default=1, help="Batch size on evaluation.")
    parser.add_argument("--eval_tokens", default=524288, type=int, help="Number of tokens for evaluation.")
    # Quantization params
    parser.add_argument("--method", type=str, default="matgptq", choices=["matgptq", "gptq"], help="Algorithm that was used for quantization.")
    parser.add_argument("--quant_weights_path", type=str, default=None, help="Path to quantized weights.")
    parser.add_argument("--quant_master_bitwidth", type=int, default=8, help="Master quantization level for quantized weights.")
    parser.add_argument("--quant_uniform_bitwidth", type=int, default=8, help="Quantization bitwidth to use for inference.")
    parser.add_argument("--quant_non_uniform_config_path", type=str, default=None, help="Path to quantization config.")
    # Logging params
    parser.add_argument("--log_wandb", default=False, action="store_true", help="Whether to log to W&B")
    # Misc params
    parser.add_argument("--dtype", type=str, default="float16", choices=["auto", "float16", "float32", "bfloat16"], help="dtype to load the model.")
    parser.add_argument("--seed", default=0, type=int, help="Random seed.")
    parser.add_argument("--verbose", action="store_true", help="Whether to log progress.")
    parser.add_argument("--memory_efficient", action="store_true", help="Whether to use memory efficient implementation.")
    parser.add_argument("--attn_implementation", type=str, default=None, choices=["eager", "sdpa", "flash_attention_2"], help="Attention implementation: eager, sdpa, or flash_attention_2")
    parser.add_argument("--use_fast_tokenizer", action="store_true", help="Whether to use fast tokenizer.")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    # Get device and dtype
    assert torch.cuda.is_available()
    device = torch.device("cuda")
    if args.dtype != "auto":
        args.dtype = getattr(torch, args.dtype)
    # Fix seed
    fix_seed(args.seed)
    # Init W&B logger
    if args.log_wandb:
        assert has_wandb, "`wandb` not installed, try pip install `wandb`"
        wandb.init(config=args)
    
    # Load model
    from_pretrained_orig = AutoModelForCausalLM.from_pretrained
    from_pretrained_overriden = from_pretrained_orig
    # Override from_pretrained
    if args.method == "gptq" and args.quant_weights_path:
        quant_weights_path = args.quant_weights_path
        quant_uniform_bitwidth = args.quant_uniform_bitwidth
        quant_non_uniform_config_path = args.quant_non_uniform_config_path
        
        def from_pretrained_overriden(*args, **kwargs):
            model = from_pretrained_orig(*args, **kwargs)
            model = load_gptq_weights(model, quant_weights_path, quant_non_uniform_config_path, quant_uniform_bitwidth)
            return model
    
    elif args.method == "matgptq" and args.quant_weights_path:
        assert args.quant_uniform_bitwidth <= args.quant_master_bitwidth <= 8, f"Slice bitwidth needs to be <= {args.quant_master_bitwidth} <= 8"
        quant_weights_path = args.quant_weights_path
        quant_master_bitwidth = args.quant_master_bitwidth
        quant_uniform_bitwidth = args.quant_uniform_bitwidth
        quant_non_uniform_config_path = args.quant_non_uniform_config_path

        # Define new init
        def from_pretrained_overriden(*args, **kwargs):
            model = from_pretrained_orig(*args, **kwargs)
            model = load_mat_gptq_weights(model, quant_weights_path, quant_non_uniform_config_path, quant_master_bitwidth, quant_uniform_bitwidth)
            return model

    # Override init
    AutoModelForCausalLM.from_pretrained = staticmethod(from_pretrained_overriden)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map=None if args.memory_efficient else "auto",
        low_cpu_mem_usage=True,
        torch_dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    model.config.use_cache = False  # do not use cache

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=args.use_fast_tokenizer)
    args.sequence_length = args.sequence_length or model.config.max_position_embeddings

    eval_datasets = []
    for eval_dataset_name in args.eval_datasets:
        eval_datasets.append(
            get_data(
                eval_dataset_name,
                args.eval_tokens,  # ignored for WikiText2 and C4
                args.sequence_length,
                tokenizer,
                train=False,
            )
        )
        
    if args.memory_efficient:
        compute_ppl_fn = partial(compute_perplexity_layer_per_layer, device=device, batch_size=args.eval_batch_size)
    else:
        compute_ppl_fn = partial(compute_perplexity, batch_size=args.eval_batch_size)

    # evaluate before layer dropping
    print("-" * 10)
    print(f"Perplexities")
    for eval_dataset_name, eval_dataset in zip(args.eval_datasets, eval_datasets):
        ppl_eval = compute_ppl_fn(model, eval_dataset)
        print(f"{eval_dataset_name}: {ppl_eval:.2f}")
    print("-" * 10)


if __name__ == "__main__":
    main()
