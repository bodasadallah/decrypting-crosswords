#!/bin/bash

#SBATCH --job-name=cryptic_crosswords_cryptonite
#SBATCH --error=logs/%j%x.err # error file
#SBATCH --output=logs/%j%x.out # output log file
#SBATCH --nodes=1                   # Run all processes on a single node    
#SBATCH --ntasks=1                  # Run on a single CPU
#SBATCH --mem=40G                   # Total RAM to be used
#SBATCH --cpus-per-task=8          # Number of CPU cores
#SBATCH --gres=gpu:1                # Number of GPUs (per node)
#SBATCH -p gpu                      # Use the gpu partition
#SBATCH --time=12:00:00             # Specify the time needed for your experiment
#SBATCH --qos=gpu-8 








echo "starting......................."
###################### RUN LLM Finetune ######################


# MODEL_NAME="meta-llama/Llama-2-7b-hf"
MODEL_NAME="mistralai/Mistral-7B-v0.1"

# --do_train \
WANDB_PROJECT=decrypting-crosswords_mistral_cryptonite_full

echo $WANDB_PROJECT
python train.py \
--max_steps=3000 \
--save_steps=500 \
--eval_steps=1000 \
--logging_steps=500 \
--report_to="all" \
--model_name=$MODEL_NAME \
--run_name=$MODEL_NAME \
--per_device_train_batch_size=64 \
--per_device_val_batch_size=32 \
--gradient_accumulation_steps=2 \
--gradient_checkpointing=1 \
--use_flash_attention_2=1 \
--eval_accumulation_steps=2 \
--save_dir='experiments/mistral-7b-v0.1_mistral_cryptonite_fulle' \
--train_dataset_path='boda/cryptonite' \
--test_dataset_path='boda/cryptonite' \
--dataset_type='cryptonite' \
--spaces=0 \
--hints=0 
# --checkpoint_path="experiments/mistral-7b-v0.1_naive_random_unique/checkpoint-1000"
# --checkpoint_path="experiments/Mistral-7B-v0.1/checkpoint-24000" 


# 'boda/naive_random_unique'
# --base_prompt="Below is a clue for a cryptic crossword. Replace underscores _ with letters of the answer to the clue." \

# --checkpoint_path="experiments/Mistral-7B-v0.1/checkpoint-24000" 


echo " ending "
#srun python run_clm.py config.json
