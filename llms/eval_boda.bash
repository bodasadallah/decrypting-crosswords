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
--model_name='meta-llama/Llama-2-7b-hf' \

python evaluate_llm.py \
--model_name='mistralai/Mistral-7B-v0.1' \
--per_device_train_batch_size=32 \
--num_examples=0 \
--n_shots=10 \
--save_file='mistral_indicator_10_few_shot.txt' \
--test_dataset_path='data/clue_json/guardian/naive_random/test.json' \
--old_dataset=1 \
--indicator_type_shots=1 \
--indicators_dict_path='data/indicators_examples.json' \

# --checkpoint_path='experiments/mistral-7b-v0.1_word_init_disjoint_unique/checkpoint-3000' \
# --test_dataset_path='data/clue_json/guardian/word_initial_disjoint/test.json' \
# --test_dataset_path='data/unique_targets' \
# --test_dataset_path='data/unique_targets' \


echo " ending " 
#srun python run_clm.py config.json