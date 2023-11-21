#!/bin/bash

#SBATCH --job-name=decrypting-crosswords # Job name
#SBATCH --error=/home/abdelrahman.sadallah/mbzuai/decrypting-crosswords/llms/logs/%j%x.err # error file
#SBATCH --output=/home/abdelrahman.sadallah/mbzuai/decrypting-crosswords/logs/%j%x.out # output log file
#SBATCH --time=24:00:00 # 10 hours of wall time
#SBATCH --nodes=1  # 1 GPU node
#SBATCH --mem=46000 # 32 GB of RAM
#SBATCH --nodelist=ws-l6-002


echo "starting......................."
###################### RUN LLM Finetune ######################





# MODEL_NAME="meta-llama/Llama-2-7b-hf"
MODEL_NAME="mistralai/Mistral-7B-v0.1"

# --do_train \
WANDB_PROJECT=decrypting-crosswords

echo $WANDB_PROJECT
python train.py \
--save_steps=1000 \
--eval_steps=1000 \
--report_to="all" \
--logging_steps=500 \
--logging_dir="experiments/$MODEL_NAME" \
--model_name=$MODEL_NAME \
--run_name=$MODEL_NAME \
--per_device_train_batch_size=64 \
--per_device_val_batch_size=16 \
--gradient_accumulation_steps=2 \
--gradient_checkpointing=1 \
--eval_accumulation_steps=2 \
--use_flash_attention_2=1 \
# --checkpoint_path="experiments/Mistral-7B-v0.1/checkpoint-24000" 

# --checkpoint_path="experiments/Llama-2-7b-hf/checkpoint-24000"

# --checkpoint_path="experiments/Mistral-7B-v0.1/checkpoint-24000" 




echo " ending "
#srun python run_clm.py config.json
