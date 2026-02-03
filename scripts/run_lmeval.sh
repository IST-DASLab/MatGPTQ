export MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"
# QUANT_PATH="/path/to/your/data/folder"  # Modify this path to point to your local folder
export MASTER_BITWIDTH=8
export UNIFORM_BITWIDTH=4

python lmeval.py \
    --model hf \
    --model_args pretrained=$MODEL_ID,dtype=float16,add_bos_token=True \
    --method "matgptq" \
    --quant_master_bitwidth $MASTER_BITWIDTH \
    --quant_uniform_bitwidth $UNIFORM_BITWIDTH \
    --seed 1234 \
    --tasks arc_easy,arc_challenge,hellaswag,winogrande,piqa \
    --batch_size 32 \
    --output_path ./results/