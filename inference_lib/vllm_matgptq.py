import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from torch.nn.parameter import Parameter
from vllm.model_executor.utils import set_weight_attrs
from vllm.model_executor.parameter import ModelWeightParameter
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig, QuantizeMethodBase
from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase, UnquantizedLinearMethod

from src.quant_utils_matgpqt import matryoshka_slice

import matgptq_cuda
from matgptq_cuda.matgptq_linear import MatGTPQ_FeatureFlags

@register_quantization_config("matgptq_quant")
class MatGPTQConfig(QuantizationConfig):
    """Custom quantization config."""

    def __init__(
        self,
        group_size: int = 128,
        master_bitwidth: int = 8,
        inference_bitwidth: int = 4,
        quant_dtype: str = "float16",
        modules_to_not_convert: list[str] | None = None,
    ) -> None:
        super().__init__()
        assert group_size in [-1, 32, 128]
        assert quant_dtype in ["float16", "bfloat16"]
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }

        self.group_size = group_size
        self.master_bitwidth = master_bitwidth
        self.inference_bitwidth = inference_bitwidth
        self.quant_dtype = dtype_map.get(quant_dtype, torch.float16)
        self.modules_to_not_convert = modules_to_not_convert

        self.flag = MatGTPQ_FeatureFlags.ASYNC + MatGTPQ_FeatureFlags.IS_BF16 * (self.quant_dtype == torch.bfloat16)

    def get_name(self) -> str:
        return "matgptq_quant"

    def get_supported_act_dtypes(self) -> list:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict) -> "MatGPTQConfig":
        group_size = cls.get_from_keys(config, ["group_size"])
        master_bitwidth = cls.get_from_keys(config, ["master_bitwidth"])
        inference_bitwidth = cls.get_from_keys(config, ["inference_bitwidth"])
        modules_to_not_convert = cls.get_from_keys(config, ["modules_to_not_convert"])
        quant_dtype = cls.get_from_keys(config, ['quant_dtype'])
        return cls(
            group_size,
            master_bitwidth,
            inference_bitwidth,
            quant_dtype,
            modules_to_not_convert,
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        # Dispatch based on layer type
        if self.modules_to_not_convert is not None and any(
            prefix.endswith(module) for module in self.modules_to_not_convert
        ):
            return UnquantizedLinearMethod()
        
        if isinstance(layer, LinearBase):
            return MatGPTQLinearMethod(self)
        return None
    

class MatGPTQLinearMethod(LinearMethodBase):
    """Custom quantization method for linear layers."""

    def __init__(self, config: MatGPTQConfig):
        super().__init__()
        self.config = config

    def create_weights(
        self, 
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs
    ):
        if input_size_per_partition % self.config.group_size != 0:  # noqa: E501
            raise ValueError(
                "The input size is not aligned with the quantized "
                "weight shape. This can be caused by too large "
                "tensor parallel size. Or other skill issues."
            )

        self.n_out_features = sum(output_partition_sizes)
        self.n_in_features = input_size_per_partition

        qweight = ModelWeightParameter(
            data=torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=extra_weight_attrs.get("weight_loader"),
        )

        layer.register_parameter("qweight", qweight)

        scales = Parameter(
            torch.empty(
                self.n_out_features,
                input_size_per_partition // self.config.group_size,
                dtype=self.config.quant_dtype
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            scales,
            {
                "input_dim": 1,
                "output_dim": 0,
                "packed_dim": 1,
                "pack_factor": 1,
            }
            | extra_weight_attrs,
        )
        
        layer.register_parameter("scales", scales)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:        
        with torch.no_grad():
            layer.matgptq_weights = Parameter(
                torch.ops.matgptq_cuda.pack_matgptq_qweights(
                    matryoshka_slice(
                        layer.qweight.data, 
                        self.config.master_bitwidth, 
                        self.config.inference_bitwidth
                    ),
                    self.n_out_features, 
                    self.n_in_features, 
                    self.config.inference_bitwidth,
                    self.config.group_size,
                    self.config.flag,
                ), 
                requires_grad=False
            )
            layer.matgptq_scales = Parameter(
                torch.ops.matgptq_cuda.reorder_matgptq_scales(
                    layer.scales, 
                    self.n_out_features, 
                    self.n_in_features, 
                    self.config.group_size
                ), 
                requires_grad=False
            )

            del layer.qweight
            del layer.scales

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None) -> torch.Tensor:
        
        y = torch.ops.matgptq_cuda.matmul_matgptq(
            x,
            layer.matgptq_weights,
            layer.matgptq_scales,
            self.n_out_features,
            self.n_in_features,
            self.config.inference_bitwidth,
            self.config.group_size,
            self.config.flag
        )

        y = y.view(*x.shape[:-1], y.shape[-1])
        if bias is not None:
            y += bias

        return y