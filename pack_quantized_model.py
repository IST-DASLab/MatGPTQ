import argparse
import json
import os
from typing import Dict

import torch
from transformers import AutoConfig, AutoTokenizer
from safetensors.torch import load_file, save_file


def load_quantised_layer(layer_dir: str) -> Dict[str, torch.Tensor]:
    data_pt = os.path.join(layer_dir, "data.pt")
    if os.path.exists(data_pt):
        layer_data = torch.load(data_pt, map_location="cpu")
        return {
            "qweight": layer_data["qweight"],
            "scale": layer_data["scale"]
        }
    else:
        raise FileNotFoundError(
            f"No data.pt found in {layer_dir}"
        )


def should_skip_quantisation(module_name: str) -> bool:
    excluded = [
        "lm_head",
        "norm",
    ]
    return any(excl in module_name for excl in excluded)


def pack_model(
    model_name_or_path: str,
    quantised_dir: str,
    output_dir: str,
    group_size: int,
    master_bitwidth: int,
    inference_bitwidth: int,
    quant_dtype: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    index_path = os.path.join(model_name_or_path, "model.safetensors.index.json")
    single_file_path = os.path.join(model_name_or_path, "model.safetensors")

    quantised_layers = {
        name
        for name in os.listdir(quantised_dir)
        if os.path.isdir(os.path.join(quantised_dir, name))
    }

    new_weight_map: Dict[str, str] = {}

    # ---------------------------------------------------------
    # CASE 1: SHARDED MODEL (has index file)
    # ---------------------------------------------------------
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            index_data = json.load(f)

        original_weight_map: Dict[str, str] = index_data["weight_map"]
        shards_seen = set(original_weight_map.values())

        for shard_name in sorted(shards_seen):
            orig_shard_path = os.path.join(model_name_or_path, shard_name)
            orig_tensors = load_file(orig_shard_path)
            new_tensors: Dict[str, torch.Tensor] = {}

            tensor_names_in_shard = [
                t_name
                for t_name, s_name in original_weight_map.items()
                if s_name == shard_name
            ]

            for t_name in tensor_names_in_shard:
                tensor = orig_tensors.get(t_name)
                if tensor is None:
                    continue

                if t_name.endswith(".weight"):
                    layer_name = t_name.rsplit(".", 1)[0]
                    if (layer_name in quantised_layers) and not should_skip_quantisation(layer_name):
                        q_data = load_quantised_layer(os.path.join(quantised_dir, layer_name))
                        new_tensors[f"{layer_name}.qweight"] = q_data["qweight"].cpu()
                        new_tensors[f"{layer_name}.scales"] = q_data["scale"].cpu()

                        new_weight_map[f"{layer_name}.qweight"] = shard_name
                        new_weight_map[f"{layer_name}.scales"] = shard_name
                        continue

                new_tensors[t_name] = tensor
                new_weight_map[t_name] = shard_name

            save_file(new_tensors, os.path.join(output_dir, shard_name))

        new_index = {"metadata": {}, "weight_map": new_weight_map}
        with open(os.path.join(output_dir, "model.safetensors.index.json"), "w") as f:
            json.dump(new_index, f)

    # ---------------------------------------------------------
    # CASE 2: SINGLE model.safetensors
    # ---------------------------------------------------------
    elif os.path.exists(single_file_path):
        shard_name = "model.safetensors"
        orig_tensors = load_file(single_file_path)
        new_tensors: Dict[str, torch.Tensor] = {}

        for t_name, tensor in orig_tensors.items():
            if t_name.endswith(".weight"):
                layer_name = t_name.rsplit(".", 1)[0]
                if (layer_name in quantised_layers) and not should_skip_quantisation(layer_name):
                    q_data = load_quantised_layer(os.path.join(quantised_dir, layer_name))
                    new_tensors[f"{layer_name}.qweight"] = q_data["qweight"].cpu()
                    new_tensors[f"{layer_name}.scales"] = q_data["scale"].cpu()
                    continue

            new_tensors[t_name] = tensor

        save_file(new_tensors, os.path.join(output_dir, shard_name))

    else:
        raise FileNotFoundError(
            "Neither model.safetensors.index.json nor model.safetensors found."
        )

    # ---------------------------------------------------------
    # SAVE CONFIG + TOKENIZER (unchanged)
    # ---------------------------------------------------------
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    if hasattr(config, "quantization_config"):
        delattr(config, "quantization_config")

    config.quantization_config = {
        "group_size": group_size,
        "modules_to_not_convert": ["lm_head", "norm"],
        "master_bitwidth": master_bitwidth,
        "inference_bitwidth": inference_bitwidth,
        "quant_method": "matgptq_quant",
        "quant_dtype": quant_dtype,
    }
    config.save_pretrained(output_dir)

    try:
        gen_config_path = os.path.join(model_name_or_path, "generation_config.json")
        if os.path.exists(gen_config_path):
            from transformers import GenerationConfig
            gen_config = GenerationConfig.from_pretrained(model_name_or_path)
            gen_config.save_pretrained(output_dir)
    except Exception:
        pass

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
        tokenizer.save_pretrained(output_dir)
    except Exception:
        pass

def main() -> None:
    parser = argparse.ArgumentParser(description="Pack a quantised model into new safetensor shards")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to the original pre‑trained model directory")
    parser.add_argument("--quantized_weights_path", type=str, required=True, help="Path to the directory containing quantised layer subdirectories")
    parser.add_argument("--packed_output_path", type=str, required=True, help="Directory where the packed model will be saved")
    parser.add_argument("--inference_bitwidth", type=int, default=4, help="Default inference bitwidth the model should be executed with")
    parser.add_argument("--master_bitwidth", type=int, default=8, help="Master bitwidth of the quantized model")
    parser.add_argument("--group_size", type=int, default=128, help="Groupsize of the quantized model")
    parser.add_argument("--quant_dtype", type=str, default="float16", choices=["float16", "bfloat16"], help="Quantization dtype")

    args = parser.parse_args()
    pack_model(
        model_name_or_path=args.model_name_or_path,
        quantised_dir=args.quantized_weights_path,
        output_dir=args.packed_output_path,
        group_size=args.group_size,
        master_bitwidth=args.master_bitwidth,
        inference_bitwidth=args.inference_bitwidth,
        quant_dtype=args.quant_dtype,
    )



if __name__ == "__main__":
    main()