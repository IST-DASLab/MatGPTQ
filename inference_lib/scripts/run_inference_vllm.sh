#!/bin/bash

MODEL_ID="ISTA-DASLab/Llama-3.1-8B-Instruct-MatGPTQ"
TOKENIZER="ISTA-DASLab/Llama-3.1-8B-Instruct-MatGPTQ"
DTYPE="float16"
GPU_MEMORY_UTIL=0.9
TEMPERATURE=0.7
TOP_P=0.95
MAX_TOKENS=512

python inference_demo_vllm.py \
    --model $MODEL_ID \
    --tokenizer $TOKENIZER \
    --dtype $DTYPE \
    --gpu-memory-util $GPU_MEMORY_UTIL \
    --temperature $TEMPERATURE \
    --top-p $TOP_P \
    --max-tokens $MAX_TOKENS \
    --prompts "Star Wars is a Franchise "