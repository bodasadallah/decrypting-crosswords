#!/bin/bash

#SBATCH --job-name=decrypt-crosswords
#SBATCH --error=/home/abdelrahman.sadallah/mbzuai/decrypting-crosswords/llms/logs/%j%x.err # error file
#SBATCH --output=/home/abdelrahman.sadallah/mbzuai/decrypting-crosswords/llms/logs/%j%x.out # output log file
#SBATCH --time=24:00:00 # 10 hours of wall time
#SBATCH --nodes=1  # 1 GPU node
#SBATCH --mem=46000 # 32 GB of RAM
#SBATCH --nodelist=ws-l4-006


echo "starting......................."
###################### RUN LLM Eval ######################

# --checkpoint_dir='experiments/Mistral-7B-v0.1/checkpoint-3000' \
# --model_name='meta-llama/Llama-2-7b-hf' \

python evaluate_llm.py \
--checkpoint_path='experiments/mistral-7b-v0.1_stars_only_disjoint/checkpoint-1500' \
--model_name='mistralai/Mistral-7B-v0.1' \
--per_device_train_batch_size=64 \
--per_device_val_batch_size=64 \
--num_examples=0 \
--n_shots=0 \
--base_prompt='Below is a clue for a cryptic crossword. Eash * symbols represents a letter in the final answer. Fill in this with the correct answer: ' \
--save_file='mistral_stars_only_disjoint_1.5k_evaluation_save_file.txt' \
--test_dataset_path='/home/daria.kotova/boda_code/decrypting-crosswords/decrypt/data/clue_json/guardian/word_initial_disjoint/test.json' \
--old_dataset=1 \

# --base_prompt='Below is a clue for a decrypting crossword. Your task is to solve this clue. The number of characters in the answer should be same as the number in the parenthesis. Just output the answer only.' \
# --test_dataset_path='data/disjoint_half_targets' \
# --test_dataset_path='data/unique_targets' \
# --test_dataset_path='data/unique_targets' \


echo " ending " 
#srun python run_clm.py config.json