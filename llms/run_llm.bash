#!/bin/bash

#SBATCH --job-name=decrypt-crosswords
#SBATCH --error=/home/abdelrahman.sadallah/mbzuai/decrypting-crosswords/llms/logs/%j%x.err # error file
#SBATCH --output=/home/abdelrahman.sadallah/mbzuai/decrypting-crosswords/llms/logs/%j%x.out # output log file
#SBATCH --time=24:00:00 # 10 hours of wall time
#SBATCH --nodes=1  # 1 GPU node
#SBATCH --nodelist=ws-l5-009

# srun python v2.py

echo "starting......................."
###################### RUN LLM Eval ######################

python evaluate_llm.py \
--model_name='mistralai/Mistral-7B-v0.1' \
--batch_size=64 \
--num_examples=0 \
--prompt='Below is a clue for a decrypting crossword. Your task is to solve this clue. The number of charachters in the answer should be same as the number in the parenthesis. Just output the answer only. Do not output any explanitions, just the words in the answer.' \
--num_shots=1 \
--dataset_path='../data/naive_random.json' \
--save_file='pred_output.txt' \


echo " ending "
#srun python run_clm.py config.json
