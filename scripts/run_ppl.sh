#!/bin/bash

MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"
# QUANT_PATH="/path/to/your/data/folder"  # Modify this path to point to your local folder
MASTER_BITWIDTH=8
UNIFORM_BITWIDTH=4

python eval_ppl.py \
    --model_name_or_path $MODEL_ID \
    --sequence_length 2048 \
    --eval_datasets wikitext2 \
    --method "matgptq" \
    --quant_weights_path $QUANT_PATH \
    --quant_master_bitwidth $MASTER_BITWIDTH \
    --quant_uniform_bitwidth $UNIFORM_BITWIDTH