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







###################### RUN LLM Finetune ######################

WANDB_PROJECT=decrypting-crosswords
# FULL_MODEL_NAME="meta-llama/Llama-2-7b-hf"
FULL_MODEL_NAME="mistralai/Mistral-7B-v0.1"
TRAIN_DATASET="data/clue_json/guardian/word_initial_disjoint/train.json"
TEST_DATASET="data/clue_json/guardian/word_initial_disjoint/val.json"
DATASET_NAME="word_initial_disjoint"
DATASET_TYPE="old"
MODEL_NAME=$(echo $FULL_MODEL_NAME | cut -d "/" -f 2 | cut -d "-" -f 1)
SAVE_DIR='/l/users/abdelrahman.sadallah/' + $MODEL_NAME + '/' + $DATASET_NAME

echo $WANDB_PROJECT
python train.py \
--max_steps=3000 \
--save_steps=500 \
--eval_steps=500 \
--logging_steps=500 \
--report_to="all" \
--model_name=$FULL_MODEL_NAME \
--run_name=$FULL_MODEL_NAME \
--per_device_train_batch_size=64 \
--per_device_val_batch_size=32 \
--save_dir=$SAVE_DIR \
--train_dataset_path=$TRAIN_DATASET \
--test_dataset_path=$TEST_DATASET \
--dataset_type=$DATASET_TYPE \
echo " ending "
