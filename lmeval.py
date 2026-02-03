# Copied from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/__main__.py

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Union
import numpy as np
import torch
from transformers import AutoModelForCausalLM

from lm_eval import evaluator, utils
from lm_eval.tasks import eval_logger
from lm_eval.tasks import TaskManager
from lm_eval.utils import make_table

from src.model_utils import load_gptq_weights, load_mat_gptq_weights

try:
    import wandb

    has_wandb = True
except ModuleNotFoundError:
    has_wandb = False

def _handle_non_serializable(o):
    if isinstance(o, np.int64) or isinstance(o, np.int32):
        return int(o)
    elif isinstance(o, set):
        return list(o)
    else:
        return str(o)


def parse_eval_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--model", "-m", default="hf", help="Name of model e.g. `hf`")
    parser.add_argument("--tasks", "-t", default=None, metavar="task1,task2", help="To get full list of tasks, use the command lm-eval --tasks list")
    parser.add_argument("--model_args", "-a", default="", help="Comma separated string arguments for model, e.g. `pretrained=EleutherAI/pythia-160m,dtype=float32`")
    parser.add_argument("--num_fewshot", "-f", type=int, default=None, metavar="N", help="Number of examples in few-shot context")
    parser.add_argument("--batch_size", "-b", type=str, default=1, metavar="auto|auto:N|N", help="Acceptable values are 'auto', 'auto:N' or N, where N is an integer. Default 1.")
    parser.add_argument("--max_batch_size", type=int, default=None, metavar="N", help="Maximal batch size to try with --batch_size auto.")
    parser.add_argument("--device", type=str, default=None, help="Device to use (e.g. cuda, cuda:0, cpu).")
    parser.add_argument("--output_path", "-o", default=None, type=str, metavar="DIR|DIR/file.json", help="The path to the output file where the result metrics will be saved. If the path is a directory and log_samples is true, the results will be saved in the directory. Else the parent directory will be used.")
    parser.add_argument("--limit", "-L", type=float, default=None, metavar="N|0<N<1", help="Limit the number of examples per task. " "If <1, limit is a percentage of the total number of examples.")
    parser.add_argument("--use_cache", "-c", type=str, default=None, metavar="DIR", help="A path to a sqlite db file for caching model responses. `None` if not caching.")
    parser.add_argument("--check_integrity", action="store_true", help="Whether to run the relevant part of the test suite for the tasks.")
    parser.add_argument("--write_out", "-w", action="store_true", default=False, help="Prints the prompt for the first few documents.")
    parser.add_argument("--log_samples", "-s", action="store_true", default=False, help="If True, write out all model outputs and documents for per-sample measurement and post-hoc analysis. Use with --output_path.")
    parser.add_argument("--show_config", action="store_true", default=False, help="If True, shows the the full config of all tasks at the end of the evaluation.")
    parser.add_argument("--include_path", type=str, default=None, metavar="DIR", help="Additional path to include if there are external tasks to include.")
    parser.add_argument("--gen_kwargs", default=None, help=("String arguments for model generation on greedy_until tasks," " e.g. `temperature=0,top_k=0,top_p=0`."))
    parser.add_argument("--verbosity", "-v", type=str.upper, default="INFO", metavar="CRITICAL|ERROR|WARNING|INFO|DEBUG", help="Controls the reported logging error level. Set to DEBUG when testing + adding new task configurations for comprehensive log output.")
    parser.add_argument("--seed", type=int, default=1, help="Seed for reproduction.")
    # Quantization params
    parser.add_argument("--method", type=str, default="matgptq", choices=["matgptq", "gptq"], help="Algorithm that was used for quantization.")
    parser.add_argument("--quant_weights_path", type=str, default=None, help="Path to quantized weights.")
    parser.add_argument("--quant_master_bitwidth", type=int, default=8, help="Master quantization level for quantized weights.")
    parser.add_argument("--quant_uniform_bitwidth", type=int, default=8, help="Quantization bitwidth to use for inference.")
    parser.add_argument("--quant_non_uniform_config_path", type=str, default=None, help="Path to quantization config.")
    # Logging params
    parser.add_argument("--log_wandb", default=False, action="store_true", help="Whether to log to W&B")

    return parser.parse_args()


def cli_evaluate(args: Union[argparse.Namespace, None] = None) -> None:
    if not args:
        # we allow for args to be passed externally, else we parse them ourselves
        args = parse_eval_args()
    print(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

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

    eval_logger.setLevel(getattr(logging, f"{args.verbosity}"))
    eval_logger.info(f"Verbosity set to {args.verbosity}")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Register default tasks + any extra paths
    include_paths = [args.include_path] if args.include_path else []
    task_manager = TaskManager(include_path=include_paths, verbosity=args.verbosity)

    if args.log_wandb:
        assert has_wandb, "`wandb` not installed, try pip install `wandb`"
        wandb.init(config=args)

    if args.limit:
        eval_logger.warning(" --limit SHOULD ONLY BE USED FOR TESTING. REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT.")

    if args.tasks is None:
        task_names = None
    elif args.tasks == "list":
        eval_logger.info("Available Tasks:\n - {}".format("\n - ".join(sorted(task_manager.list_all_tasks()))))
        sys.exit()
    else:
        if os.path.isdir(args.tasks):
            import glob

            task_names = []
            yaml_path = os.path.join(args.tasks, "*.yaml")
            for yaml_file in glob.glob(yaml_path):
                config = utils.load_yaml_config(yaml_file)
                task_names.append(config)
        else:
            task_names = args.tasks.split(",")
            task_missing = [task for task in task_names if task not in task_manager.list_all_tasks() and "*" not in task]

            if task_missing:
                missing = ", ".join(task_missing)
                eval_logger.error(f"Tasks were not found: {missing}\n{utils.SPACING}Try `lm-eval --tasks list` for list of available tasks")
                raise ValueError(f"Tasks not found: {missing}. Try `lm-eval --tasks list` for list of available tasks, or '--verbosity DEBUG' to troubleshoot task registration issues.")


    if args.output_path:
        path = Path(args.output_path)
        if path.is_file() or Path(args.output_path).joinpath(f"results_seed={args.seed}.json").is_file():
            eval_logger.warning(f"File already exists at {path}. Results will be overwritten.")
            output_path_file = path.joinpath(f"results_seed={args.seed}.json")
            assert not path.is_file(), "File already exists"
        elif path.suffix in (".json", ".jsonl"):
            output_path_file = path
            path.parent.mkdir(parents=True, exist_ok=True)
            path = path.parent
        else:
            path.mkdir(parents=True, exist_ok=True)
            output_path_file = path.joinpath(f"results_seed={args.seed}.json")
    elif args.log_samples and not args.output_path:
        assert args.output_path, "Specify --output_path"

    eval_logger.info(f"Selected Tasks: {task_names}")

    results = evaluator.simple_evaluate(
        model=args.model,
        model_args=args.model_args,
        tasks=task_names,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        max_batch_size=args.max_batch_size,
        device=args.device,
        use_cache=args.use_cache,
        limit=args.limit,
        check_integrity=args.check_integrity,
        write_out=args.write_out,
        log_samples=args.log_samples,
        gen_kwargs=args.gen_kwargs,
        task_manager=task_manager,
        random_seed=args.seed,
        numpy_random_seed=args.seed,
        torch_random_seed=args.seed,
        fewshot_random_seed=args.seed
    )

    if results is not None:
        if args.log_samples:
            samples = results.pop("samples")
        dumped = json.dumps(results, indent=2, default=_handle_non_serializable, ensure_ascii=False)
        if args.show_config:
            print(dumped)

        batch_sizes = ",".join(map(str, results["config"]["batch_sizes"]))

        if args.output_path:
            output_path_file.open("w").write(dumped)

            if args.log_samples:
                for task_name, config in results["configs"].items():
                    output_name = f"{re.sub(r"/|=", "__", args.model_args)}_{task_name}"
                    filename = path.joinpath(f"{output_name}.jsonl")
                    samples_dumped = json.dumps(
                        samples[task_name],
                        indent=2,
                        default=_handle_non_serializable,
                        ensure_ascii=False,
                    )
                    filename.open("w").write(samples_dumped)

        print(
            f"{args.model} ({args.model_args}), gen_kwargs: ({args.gen_kwargs}), limit: {args.limit}, num_fewshot: {args.num_fewshot}, "
            f"batch_size: {args.batch_size}{f' ({batch_sizes})' if batch_sizes else ''}"
        )
        print(make_table(results))
        if "groups" in results:
            print(make_table(results, "groups"))


cli_evaluate()
