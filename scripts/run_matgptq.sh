MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"

SEQUENCE_LENGTH=4096
CALIB_DATA="fineweb_edu" 

NUM_TOKENS=2097152
BITS_LIST="3 4 8"
BITS_WEIGHTS="1 1 1" # Bitwidths for weights, in the same order as BITS_LIST
MASTER_BITWIDTH=8
GROUP_SIZE=128

# SAVE_DIR="/path/to/your/data/folder"  # Modify this path to point to your local folderSAVE_DIR=".weights/matgptq2"

# For Llama models, the pre_block_modules should be "model.embed_tokens model.rotary_emb"
torchrun --nnodes=1 --nproc-per-node=4 --master_port 29501 quant.py \
    --model_name_or_path $MODEL_ID \
    --quantizable_modules '.*layers.*((q|k|v|o|gate|up|down)_proj)$' \
    --pre_block_modules model.embed_tokens\
    --block_modules model.layers \
    --post_block_modules model.norm lm_head \
    \
    --calibration_data $CALIB_DATA \
    --calibration_tokens $NUM_TOKENS \
    --calibration_sequence_length $SEQUENCE_LENGTH \
    \
    --method matgptq \
    --master_bitwidth $MASTER_BITWIDTH \
    --bitwidth_options $BITS_LIST \
    --bitwidth_weights $BITS_WEIGHTS \
    --group_size $GROUP_SIZE \
    --perchannel \
    --sym \
    \
    --verbose \
    \
    --dtype float16 \
    --attn_implementation eager \
    \
    --save_dir $SAVE_DIR