from contextlib import contextmanager
import re
import os
import gc
from collections import defaultdict
from typing import List, Dict, Optional, Union, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.conv import _ConvNd
from transformers import AutoModelForCausalLM

from src.common_utils import to
from src.quant_utils_matgpqt import FakeMatryoshkaLinear, dequantize


### Layer and activation getters


class CatcherExit(Exception):
    pass


class Catcher(nn.Module):

    def __init__(self, module: nn.Module, offload: bool = False):
        super().__init__()
        self.module = module
        self.inputs = []
        self.input_kwargs = []
        self.offload = offload

    def forward(self, inputs, **kwargs):
        offload_device = "cpu" if self.offload else None
        self.inputs.append(inputs.to(offload_device))
        self.input_kwargs.append(kwargs)
        raise CatcherExit()


def get_layers(model: AutoModelForCausalLM):
    if model.config.model_type in ("llama", "gemma", "gemma2", "phi3", "mistral", "qwen3"):
        return model.model.layers
    if model.config.model_type == "opt":
        return model.model.decoder.layers
    else:
        raise ValueError(f"{model.config.model_type} is not supported.")


def get_lm_head(model: AutoModelForCausalLM):
    lm_head = nn.ModuleList()
    if model.config.model_type in ("llama", "gemma", "gemma2", "phi3", "mistral", "qwen3"):
        if model.model.norm is not None:
            lm_head.append(model.model.norm)
        lm_head.append(model.lm_head)
    elif model.config.model_type == "opt":
        if model.model.decoder.final_layer_norm is not None:
            lm_head.append(model.model.decoder.final_layer_norm)
        if model.model.decoder.project_out is not None:
            lm_head.append(model.model.decoder.project_out)
        lm_head.append(model.lm_head)
    else:
        raise ValueError(f"{model.config.model_type} is not supported.")
    return lm_head


def get_lm_logits(hidden_states: torch.Tensor, model: nn.Module):
    if model.config.model_type in ("llama", "gemma", "gemma2", "phi3", "mistral", "qwen3"):
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
    elif model.config.model_type == "opt":
        if model.model.decoder.final_layer_norm is not None:
            hidden_states = model.model.decoder.final_layer_norm(hidden_states)
        if model.model.decoder.project_out is not None:
            hidden_states = model.model.decoder.project_out(hidden_states)
        lm_logits = model.lm_head(hidden_states)
    else:
        raise ValueError(f"{model.config.model_type} is not supported.")
    return lm_logits

### Calibration utils and modules

LINEAR_LAYERS = (nn.Linear, _ConvNd)


class ForwardInterrupt(Exception):
    pass


class InputCollector(nn.Module):
    def __init__(self, module: nn.Module, cpu_offload: bool = False):
        super().__init__()
        self.module = module
        self.cpu_offload = cpu_offload
        self.input_args = []
        self.input_kwargs = []

    def __getattr__(self, name):
        # avoid infinite recursion for known attrs
        if name in {"module", "cpu_offload", "input_args", "input_kwargs"}:
            return super().__getattr__(name)
        return getattr(self.module, name)

    def forward(self, *input_args, **input_kwargs):
        if self.cpu_offload:
            input_args = to(input_args, device="cpu")
            input_kwargs = to(input_kwargs, device="cpu")
        self.input_args.append(input_args)
        self.input_kwargs.append(input_kwargs)
        raise ForwardInterrupt

def select_layers(
    model: nn.Module,
    layer_prefix: Optional[str] = "",
    layer_regex: str = ".*",
    layer_classes: Union[nn.Module, List[nn.Module]] = nn.Module,
) -> Dict[str, nn.Module]:
    layers = {}
    for layer_name, layer in model.named_modules():
        if (
            isinstance(layer, layer_classes)
            and re.search(layer_regex, layer_name)
            and layer_name.startswith(layer_prefix)
        ):
            layers[layer_name] = layer
    return layers


def get_number_of_rows_and_cols(layer):
    return layer.weight.shape[0], np.prod(layer.weight.shape[1:])


### Quantized model loader
def load_gptq_weights(
    model: AutoModelForCausalLM,
    quantized_weights_path: Union[str, os.PathLike],
    quantized_config_path: Optional[str] = None,
    default_level: int = 0,
):
    # Load weights from configuration if provided
    if quantized_config_path:
        with open(os.path.join(quantized_config_path), "r") as f:
            for line in f:
                layer_name, level = line.split(":")
                layer = model.get_submodule(layer_name.strip(" "))
                orig_dtype = layer.weight.dtype
                layer.weight.data = torch.load(
                    os.path.join(quantized_weights_path, layer_name, f"{int(level)}.pt"),
                    map_location=layer.weight.device,
                ).to(orig_dtype)
    # Otherwise load uniform configuration
    else:
        for layer_name in sorted(os.listdir(quantized_weights_path)):
            if not os.path.isdir(os.path.join(quantized_weights_path, layer_name)):
                continue
            layer = model.get_submodule(layer_name.strip(" "))
            orig_dtype = layer.weight.dtype
            layer.weight.data = torch.load(
                os.path.join(quantized_weights_path, layer_name, f"{default_level}.pt"),
                map_location=layer.weight.device,
            ).to(orig_dtype)
    return model

def load_mat_gptq_weights(
    model: AutoModelForCausalLM,
    quantized_weights_path: Union[str, os.PathLike],
    quantized_config_path: Optional[str] = None,
    master_bitwidth: int = 8,
    slice_bitwidth: int = 8,
):
        # Load weights from configuration if provided
    if quantized_config_path:
        with open(os.path.join(quantized_config_path), "r") as f:
            for line in f:
                layer_name, bitwidth = line.split(":")
                layer = model.get_submodule(layer_name.strip(" "))
                orig_dtype, orig_device = layer.weight.dtype, layer.weight.device
                
                layer_data = torch.load(os.path.join(quantized_weights_path, layer_name, f"data.pt"),)

                qweight = layer_data["qweight"]
                scale = layer_data["scale"]
                zero = layer_data["zero"]
                perm = layer_data.get("perm", None).argsort() if layer_data.get("perm", None) is not None else torch.arange(qweight.shape[1])

                weight = dequantize(
                    layer_data["qweight"].view(qweight.shape[0], scale.shape[1], -1),
                    scale.view(qweight.shape[0], -1, 1),
                    zero.view(qweight.shape[0], -1, 1),
                    master_bitwidth,
                    int(bitwidth)
                ).view_as(qweight)[:, perm]

                layer.weight.data = weight.to(dtype=orig_dtype, device=orig_device)
    # Otherwise load uniform configuration
    else:
        for layer_name in sorted(os.listdir(quantized_weights_path)):
            if not os.path.isdir(os.path.join(quantized_weights_path, layer_name)):
                continue
            layer = model.get_submodule(layer_name.strip(" "))
            orig_dtype, orig_device = layer.weight.dtype, layer.weight.device
                
            layer_data = torch.load(os.path.join(quantized_weights_path, layer_name, f"data.pt"),)

            qweight = layer_data["qweight"]
            scale = layer_data["scale"]
            zero = layer_data["zero"]
            perm = layer_data.get("perm", None).argsort() if layer_data.get("perm", None) is not None else torch.arange(qweight.shape[1])

            weight = dequantize(
                layer_data["qweight"].view(qweight.shape[0], scale.shape[1], -1),
                scale.view(qweight.shape[0], -1, 1),
                zero.view(qweight.shape[0], -1, 1),
                master_bitwidth,
                slice_bitwidth
            ).view_as(qweight)[:, perm]

            layer.weight.data = weight.to(dtype=orig_dtype, device=orig_device)
    return model

def layer_order_fn(layer_name: str):
    split_key = layer_name.split(".")
    block_id = int(split_key[2])
    misc = split_key[3:]
    return (block_id, *misc)

def group_layers(model: nn.Module, layer_names: Sequence[str], group_rule: Optional[str] = None) -> Tuple[Sequence[str]]:
    assert group_rule in ["none", "name", "size"]
    # No grouping
    if group_rule == "none":
        group_key_fn = lambda layer_name: 0
    # Group by last part of the name
    elif group_rule == "name":
        group_key_fn = lambda layer_name: layer_name.split(".")[-1]
    # Group by size
    elif group_rule == "size":
        group_key_fn = lambda layer_name: model.get_submodule(layer_name).weight.numel()
    groups = defaultdict(list)
    for layer_name in layer_names:
        groups[group_key_fn(layer_name)].append(layer_name)
    return tuple(groups.values())


@contextmanager
def suspend_nn_inits():
    def skip(*args, **kwargs):
        pass

    saved_inits = torch.nn.init.kaiming_uniform_, torch.nn.init.uniform_, torch.nn.init.normal_  # saving
    torch.nn.init.kaiming_uniform_ = torch.nn.init.uniform_ = torch.nn.init.normal_ = skip  # replacing
    try:
        yield
    finally:
        torch.nn.init.kaiming_uniform_, torch.nn.init.uniform_, torch.nn.init.normal_ = saved_inits  # restoring
