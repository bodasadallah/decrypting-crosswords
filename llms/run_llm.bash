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

# --model_name='meta-llama/Llama-2-7b-hf' \
# --checkpoint_dir='experiments/Mistral-7B-v0.1/checkpoint-3000' \

python evaluate_llm.py \
--checkpoint_path='experiments/mistral-7b-v0.1_spaces/checkpoint-1000' \
--model_name='mistralai/Mistral-7B-v0.1' \
--per_device_train_batch_size=32 \
--per_device_val_batch_size=64 \
--num_examples=0 \
--n_shots=0 \
--base_prompt='Below is a clue for a cryptic crossword. Replace underscores _ with letters of the answer to the clue.' \
--save_file='mistral_spaces_1k_test_disjoint.txt' \
--test_dataset_path='/home/daria.kotova/boda_code/decrypting-crosswords/decrypt/data/clue_json/guardian/word_initial_disjoint/test.json' \
--old_dataset=1 \
--spaces=1
# --dataset_path='data/unique_targets' \
# --test_dataset_path='data/unique_targets' \


echo " ending " 
#srun python run_clm.py config.json