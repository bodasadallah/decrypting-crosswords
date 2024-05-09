#!/bin/bash

#SBATCH --job-name=cryptic_crosswords_base_prompt_word_init_disjoint_half # Job name
#SBATCH --error=logs/%j%x.err # error file
#SBATCH --output=logs/%j%x.out # output log file
#SBATCH --nodes=1                   # Run all processes on a single node    
#SBATCH --ntasks=1                  # Run on a single CPU
#SBATCH --mem=40G                   # Total RAM to be used
#SBATCH --cpus-per-task=8          # Number of CPU cores
#SBATCH --gres=gpu:1                # Number of GPUs (per node)
#SBATCH --time=12:00:00             # Specify the time needed for your experiment
#SBATCH --qos=gpu-8 
#SBATCH -p gpu                      # Use the gpu partition
##SBATCH --nodelist=ws-l6-005






###################### RUN LLM Finetune ######################

WANDB_PROJECT=decrypting-crosswords_base_prompt
FULL_MODEL_NAME="meta-llama/Llama-2-7b-hf"
# FULL_MODEL_NAME="mistralai/Mistral-7B-v0.1"
TRAIN_DATASET="boda/word_init_disjoint_half"
TEST_DATASET="boda/word_init_disjoint_half"
DATASET_NAME="word_init_disjoint_half"
DATASET_TYPE="new"
PROMPT_TYPE="base_prompt"
MODEL_NAME=$(echo $FULL_MODEL_NAME | cut -d "/" -f 2 | cut -d "-" -f 1)
SAVE_DIR="/l/users/abdelrahman.sadallah/$PROMPT_TYPE/$MODEL_NAME/$DATASET_NAME"

BASE_PROMPT="The next line is a clue for a cryptic crossword. Solve this clue. The number in the parentheses in the clue represents the number of characters of the answer. Output only the answer."

echo $BASE_PROMPT
echo $SAVE_DIR
echo $WANDB_PROJECT
python train.py \
--max_steps=2000 \
--save_steps=500 \
--eval_steps=500 \
--logging_steps=500 \
--report_to="all" \
--model_name=$FULL_MODEL_NAME \
--run_name=$FULL_MODEL_NAME \
--per_device_train_batch_size=128 \
--per_device_val_batch_size=64 \
--save_dir=$SAVE_DIR \
--train_dataset_path=$TRAIN_DATASET \
--test_dataset_path=$TEST_DATASET \
--dataset_type=$DATASET_TYPE \
--base_prompt="The next line is a clue for a cryptic crossword. Solve this clue. The number in the parentheses in the clue represents the number of characters of the answer. Output only the answer." \
echo "ending "
