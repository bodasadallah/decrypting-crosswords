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



echo "starting Evaluation......................."
###################### RUN LLM Eval ######################
FULL_MODEL_NAME="meta-llama/Llama-2-7b-hf"
# FULL_MODEL_NAME="mistralai/Mistral-7B-v0.1"

TEST_DATASET="data/clue_json/guardian/word_initial_disjoint/val.json"
DATASET_NAME="word_initial_disjoint"
DATASET_TYPE="old"
MODEL_NAME=$(echo $FULL_MODEL_NAME | cut -d "/" -f 2 | cut -d "-" -f 1)
SAVE_DIR="new_experiments/$MODEL_NAME/$DATASET_NAME.txt"
CHECKPOINT_PATH="/l/users/abdelrahman.sadallah/$MODEL_NAME/$DATASET_NAME/best_checkpoint"
# --model_name='meta-llama/Llama-2-7b-hf' \

python evaluate_llm.py \
--model_name=$FULL_MODEL_NAME \
--per_device_train_batch_size=128 \
--num_examples=0 \
--n_shots=0 \
--save_file=$SAVE_DIR \
--test_dataset_path=$TEST_DATASET \
--checkpoint_path=$CHECKPOINT_PATH \


# --test_dataset_path='boda/cryptonite_filtered_testset' \
# --cryptonite_quick=1 \
# --dataset_type='cryptonite_filtered' \
# --test_dataset_path='data/clue_json/guardian/naive_random/test.json' \


echo " ending " 