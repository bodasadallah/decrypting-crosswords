#!/bin/bash

#SBATCH --job-name=cryptic_crosswords_eval_cryptonite_quick
#SBATCH --error=logs/%j%x.err # error file
#SBATCH --output=logs/%j%x.out # output log file
#SBATCH --nodes=1                   # Run all processes on a single node    
#SBATCH --ntasks=1                  # Run on a single CPU
#SBATCH --mem=40G                   # Total RAM to be used
#SBATCH --cpus-per-task=8          # Number of CPU cores
#SBATCH --gres=gpu:1                # Number of GPUs (per node)
##SBATCH --qos=gpu-8 
##SBATCH -p gpu                      # Use the gpu partition
#SBATCH --nodelist=ws-l6-017



echo "starting Evaluation......................."
###################### RUN LLM Eval ######################
FULL_MODEL_NAME="meta-llama/Llama-2-7b-hf"
# FULL_MODEL_NAME="mistralai/Mistral-7B-v0.1"

TEST_DATASET="boda/cryptonite_filtered_testset"
DATASET_NAME="word_initial_disjoint"
DATASET_TYPE="cryptonite_filtered"
SHOTS=0
MODEL_NAME=$(echo $FULL_MODEL_NAME | cut -d "/" -f 2 | cut -d "-" -f 1)
SAVE_DIR="new_experiments/$MODEL_NAME/$DATASET_NAME_cryptonite_quick.txt"
# SAVE_DIR="new_experiments/$MODEL_NAME/$DATASET_NAME_$SHOTS-shots.txt"

CHECKPOINT_PATH="/l/users/abdelrahman.sadallah/$MODEL_NAME/$DATASET_NAME/checkpoint-1000"
python evaluate_llm.py \
--model_name=$FULL_MODEL_NAME \
--per_device_train_batch_size=64 \
--num_examples=0 \
--n_shots=0 \
--save_file=$SAVE_DIR \
--test_dataset_path=$TEST_DATASET \
--checkpoint_path=$CHECKPOINT_PATH \
--dataset_type=$DATASET_TYPE \
--cryptonite_quick=1


# --test_dataset_path='boda/cryptonite_filtered_testset' \
# --cryptonite_quick=1 \
# --dataset_type='cryptonite_filtered' \
# --test_dataset_path='data/clue_json/guardian/naive_random/test.json' \


echo " ending " 