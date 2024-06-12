#!/bin/bash

# meta-llama/Meta-Llama-3-8B-Instruct
# gpt-3.5-turbo 
python eval_definition_and_wordplay.py \
--model  meta-llama/Meta-Llama-3-8B-Instruct \
--output_file "results/definition_wordplay_extraction/llama_instruct_eval_def_1k_samples.txt" \
--eval_definition \
--data_path "data/georgo_ho_clues_sampled.csv"


echo " ending " 