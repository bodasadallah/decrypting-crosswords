#!/bin/bash

# meta-llama/Meta-Llama-3-8B-Instruct
# gpt-3.5-turbo 
python eval_definition_and_wordplay.py \
--model  meta-llama/Meta-Llama-3-8B-Instruct  \
--output_file "results/definition_wordplay_extraction/llama-instruct_1k_all-inclusve_def.txt" \
--data_path "data/georgo_ho_clues_sampled.csv" \
--eval_clue \
--prompt ALL_INCLUSIVE_DEFINITION
# --eval_definition \
# --data_path "data/georgo_ho_clues_sampled.csv" \
# WORDPLAY_WITH_DEF_PROMPT
echo " ending "     