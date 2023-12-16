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
--checkpoint_path='experiments/mistral_disjoint_half_targets_2/checkpoint-1500' \
--model_name='mistralai/Mistral-7B-v0.1' \
--per_device_train_batch_size=32 \
--num_examples=0 \
--n_shots=0 \
--base_prompt='Below is a clue for a decrypting crossword. Your task is to solve this clue. The number of characters in the answer should be same as the number in the parenthesis. Just output the answer only.' \
--save_file='mistral_disjoint_half_targets_2_new_2k_output_test_disjoint_half_targets.txt' \
--test_dataset_path='data/disjoint_half_targets' \
--old_dataset=0 \

# --test_dataset_path='data/clue_json/guardian/word_initial_disjoint/test.json' \
# --test_dataset_path='data/unique_targets' \
# --test_dataset_path='data/unique_targets' \


echo " ending " 
#srun python run_clm.py config.json