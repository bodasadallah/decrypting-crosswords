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




# echo "starting Evaluation......................."
# ###################### RUN LLM Eval ######################
# # --checkpoint_dir='experiments/Mistral-7B-v0.1/checkpoint-3000' \
# --model_name='meta-llama/Llama-2-7b-hf' \

# python evaluate_llm.py \
# --model_name='mistralai/Mistral-7B-v0.1' \
# --per_device_train_batch_size=32 \
# --num_examples=0 \
# --n_shots=0 \
# --test_dataset_path='boda/cryptonite' \
# --save_file='mistral_cryptonite_2k_cryptonite_test.txt' \
# --dataset_type='cryptonite' \
# --indicator_type_shots=0 \
# --indicators_dict_path='data/indicators_examples.json' \
# --checkpoint_path='experiments/mistral-7b-v0.1_mistral_cryptonite_fulle/checkpoint-1500' \


# # --test_dataset_path='data/clue_json/guardian/naive_random/test.json' \
# # --test_dataset_path='data/clue_json/guardian/word_initial_disjoint/test.json' \


# echo " ending " 

################################ Save File Naming Convention:  MODELNAME_TRAIN-DATASET_TRAIN-STEPS_TEST-DATASET.txt

echo "starting Evaluation......................."
###################### RUN LLM Eval ######################
# --checkpoint_dir='experiments/Mistral-7B-v0.1/checkpoint-3000' \

python evaluate_llm.py \
--model_name='mistralai/Mistral-7B-v0.1' \
--per_device_train_batch_size=32 \
--num_examples=0 \
--n_shots=0 \
--test_dataset_path='boda/cryptonite_filtered_testset' \
--save_file='mistral_disjoint_1k_cryptonite_filtered_testset.txt' \
--dataset_type='cryptonite_filtered' \
--cryptonite_quick=0 \
--indicator_type_shots=0 \
--checkpoint_path='experiments/mistral_disjoint_1k' \


# --test_dataset_path='data/clue_json/guardian/naive_random/test.json' \
# --test_dataset_path='data/clue_json/guardian/word_initial_disjoint/test.json' \


echo " ending " 