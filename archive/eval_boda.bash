#!/bin/bash

<<<<<<< HEAD
#SBATCH --job-name=cryptic_crosswords_eval_cryptonite_quick
=======
#SBATCH --job-name=cryptic_crosswords_EVAL_10_shot_random # Job name
>>>>>>> 485167d5a2e910a9a3b0764687c13b06430fdff2
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
# FULL_MODEL_NAME="meta-llama/Llama-2-7b-hf"
FULL_MODEL_NAME="mistralai/Mistral-7B-v0.1"

<<<<<<< HEAD
TEST_DATASET="boda/cryptonite_filtered_testset"
DATASET_NAME="word_initial_disjoint"
DATASET_TYPE="cryptonite_filtered"
SHOTS=0
MODEL_NAME=$(echo $FULL_MODEL_NAME | cut -d "/" -f 2 | cut -d "-" -f 1)
SAVE_DIR="new_experiments/$MODEL_NAME/$DATASET_NAME_cryptonite_quick.txt"
# SAVE_DIR="new_experiments/$MODEL_NAME/$DATASET_NAME_$SHOTS-shots.txt"
=======
PROMPT_TYPE="base_prompt"
TEST_DATASET="data/clue_json/guardian/naive_random/test.json"
DATASET_NAME="naive_random"
DATASET_TYPE="old"
SHOTS=10
INDOCATOR_TYPE_SHOTS=0
MODEL_NAME=$(echo $FULL_MODEL_NAME | cut -d "/" -f 2 | cut -d "-" -f 1)
SAVE_DIR="new_experiments/$PROMPT_TYPE/$MODEL_NAME/${DATASET_NAME}_${SHOTS}_shot_random.txt"

# CHECKPOINT_PATH="/l/users/abdelrahman.sadallah/$PROMPT_TYPE/$MODEL_NAME/$DATASET_NAME/best"

>>>>>>> 485167d5a2e910a9a3b0764687c13b06430fdff2

CHECKPOINT_PATH="/l/users/abdelrahman.sadallah/$MODEL_NAME/$DATASET_NAME/checkpoint-1000"
python evaluate_llm.py \
--model_name=$FULL_MODEL_NAME \
<<<<<<< HEAD
--per_device_train_batch_size=64 \
=======
--per_device_train_batch_size=32 \
>>>>>>> 485167d5a2e910a9a3b0764687c13b06430fdff2
--num_examples=0 \
--n_shots=$SHOTS \
--save_file=$SAVE_DIR \
--test_dataset_path=$TEST_DATASET \
--dataset_type=$DATASET_TYPE \
--checkpoint_path=$CHECKPOINT_PATH \
<<<<<<< HEAD
--dataset_type=$DATASET_TYPE \
--cryptonite_quick=1

=======
--indicator_type_shots=$INDOCATOR_TYPE_SHOTS \
--base_prompt="The next line is a clue for a cryptic crossword. Solve this clue. The number in the parentheses in the clue represents the number of characters of the answer. Output only the answer." \
>>>>>>> 485167d5a2e910a9a3b0764687c13b06430fdff2

# --test_dataset_path='boda/cryptonite_filtered_testset' \
# --cryptonite_quick=1 \
# --dataset_type='cryptonite_filtered' \
# --test_dataset_path='data/clue_json/guardian/naive_random/test.json' \


echo " ending " 