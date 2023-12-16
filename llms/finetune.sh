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
WANDB_PROJECT=decrypting-crosswords_mistral_disjoint_half_targets_2

echo $WANDB_PROJECT
python train.py \
--max_steps=3000 \
--save_steps=500 \
--eval_steps=500 \
--logging_steps=500 \
--report_to="all" \
--model_name=$MODEL_NAME \
--run_name=$MODEL_NAME \
--per_device_train_batch_size=64 \
--per_device_val_batch_size=16 \
--gradient_accumulation_steps=2 \
--gradient_checkpointing=1 \
--eval_accumulation_steps=2 \
--save_dir='experiments/mistral_disjoint_half_targets_2' \
--train_dataset_path='data/disjoint_half_targets' \
--test_dataset_path='data/disjoint_half_targets' \
--use_flash_attention_2=1 \
--old_dataset=0 \
# --checkpoint_path="experiments/mistral-7b-v0.1_disjoint_2/checkpoint-1500"
# --checkpoint_path="experiments/Mistral-7B-v0.1/checkpoint-24000" 


# --checkpoint_path="experiments/Mistral-7B-v0.1/checkpoint-24000" 




echo " ending "
#srun python run_clm.py config.json
