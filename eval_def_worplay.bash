#!/bin/bash

# meta-llama/Meta-Llama-3-8B-Instruct
# gpt-3.5-turbo 
python eval_definition_and_wordplay.py \
--model  meta-llama/Meta-Llama-3-8B-Instruct  \
--output_file "results/definition_wordplay_extraction/llama_instruct_eval_wordplay_25_dif.txt" \
--data_path "data/short_list_clues.csv" \
--eval_wordplay \
--prompt WORDPLAY_PROMPT
# --eval_definition \
# --data_path "data/georgo_ho_clues_sampled.csv" \
# WORDPLAY_WITH_DEF_PROMPT
echo " ending "     