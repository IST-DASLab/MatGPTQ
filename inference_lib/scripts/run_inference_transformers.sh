#!/bin/bash

MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"
# QUANT_PATH="/path/to/your/data/folder"  # Modify this path to point to your local folder
MASTER_BITWIDTH=8
UNIFORM_BITWIDTH=4
EXECUTION_MODE=2 # 0: Dense, 1: Fake-Quantized, 2: Kernel-Quantized

python inference_demo_transformers.py \
    --pretrained_model_path $MODEL \
    --execution_mode $EXECUTION_MODE \
    --quant_weights_path $QUANT_PATH \
    --quant_master_level $MASTER_BITWIDTH \
    --quant_uniform_bitwidth $UNIFORM_BITWIDTH
