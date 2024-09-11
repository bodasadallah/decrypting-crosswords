#!/bin/bash

#SBATCH --job-name=cryptic_crosswords_EVAL_gemma-9b # Job name
#SBATCH --error=logs/%j%x.err # error file
#SBATCH --output=logs/%j%x.out # output log file
#SBATCH --nodes=1                   # Run all processes on a single node    
#SBATCH --ntasks=1                  # Run on a single CPU
#SBATCH --mem=40G                   # Total RAM to be used
#SBATCH --cpus-per-task=8          # Number of CPU cores
#SBATCH --gres=gpu:1                # Number of GPUs (per node)
#SBATCH -p it-hpc                      # Use the gpu partition
#SBATCH --time=12:00:00  
# meta-llama/Meta-Llama-3-8B-Instruct
# gpt-3.5-turbo 

# "google/gemma-2-9b-it"
# python eval_definition_and_wordplay.py \
# --model  google/gemma-2-9b-it  \
# --output_file "results/included_results/gemma_1k_all-inclusve_no_def.txt" \
# --data_path "data/georgo_ho_clues_sampled.csv" \
# --prompt ALL_INCLUSIVE_PROMPT \
# --eval_clue 
# # --eval_definition 
# # WORDPLAY_WITH_DEF_PROMPT
# echo " ending "


# # "google/gemma-2-9b-it"
CUDA_VISIBLE_DEVICES=3
python eval_definition_and_wordplay.py \
--model  gpt-3.5-turbo  \
--output_file "results/included_results/chatgpt_WORDPLAY_PROMPT_200.txt" \
--data_path "data/200_clues.csv" \
--prompt WORDPLAY_PROMPT \
--eval_wordplay

# # google/gemma-2-9b-it
# gpt-3.5-turbo
# meta-llama/Meta-Llama-3-8B-Instruct
# WORDPLAY_WITH_DEF_EX_PROMPT_ANS
# WORDPLAY_WITH_DEF_EX_PROMPT
# WORDPLAY_PROMPT
# --eval_definition 
# --eval_clue \
# WORDPLAY_WITH_DEF_PROMPT
echo " ending "     