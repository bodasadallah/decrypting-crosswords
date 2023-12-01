#!/bin/bash

#SBATCH --job-name=decrypt-crosswords
#SBATCH --error=/home/daria.kotova/mbzuai/decrypting-crosswords/llms/logs/%j%x.err # error file
#SBATCH --output=/home/daria.kotova/mbzuai/decrypting-crosswords/llms/logs/%j%x.out # output log file
#SBATCH --time=24:00:00 # 10 hours of wall time
#SBATCH --nodes=1  # 1 GPU node
#SBATCH --mem=46000 # 32 GB of RAM
#SBATCH --nodelist=ws-l4-006


echo "starting......................."
###################### RUN LLM Eval ######################

# --model_name='mistralai/Mistral-7B-v0.1' \
# --model_name='meta-llama/Llama-2-7b-hf' \

python evaluate_llm.py \
--checkpoint_dir='experiments/Mistral-7B-v0.1/checkpoint-5000' \
--model_name='mistralai/Mistral-7B-v0.1' \
--batch_size=16 \
--num_examples=0 \
--n_shots=0 \
--prompt='Below is a clue for a decrypting crossword. Your task is to solve this clue. The number of characters in the answer should be same as the number in the parenthesis. Just output the answer only.' \
--dataset_path='../data/naive_random.json' \
--save_file='mistral_5k_output_pred.txt' \


echo " ending "
#srun python run_clm.py config.json
