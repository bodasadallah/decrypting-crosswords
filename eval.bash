#!/bin/bash

#SBATCH --job-name=cryptic_crosswords_EVAL_llama3_word_init_no_samle # Job name
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


# "meta-llama/Meta-Llama-3-8B" \
    # --dataset "boda/guardian_word_initial_disjoint" \

echo "starting Evaluation......................."

    # --results_save_file "results/llama3-${DATASET}_results.txt" \

# DATASET="guardian_naive_random"
DATASET="guardian_word_initial_disjoint"
# MODEL="Meta-Llama-3-8B-Instruct"
MODEL="Meta-Llama-3-8B"
###################### RUN LLM Eval ######################
python evaluate.py \
    --split "test" \
    --dataset "boda/${DATASET}" \
    --model_name_or_path "meta-llama/${MODEL}" \
    --tokenizer_name_or_path "meta-llama/${MODEL}" \
    --wandb_run_name "llama3-base-prompt-${DATASET}-0-shot" \
    --save_model_predicitons "yes" \
    --save_folder "results/${MODEL}-${DATASET}-no-sample" \
    --quantize False \
    --do_sample False \
    --per_device_eval_batch_size 8 \
    --num_examples 0 \
    --wandb_project "cryptic_crosswords" \
    --checkpoint_path "" \
    --use_flash_attention_2 True \
    --top_p 0.9 \
    --temperature 0.6 \
    --n_shots 0 \
    --max_new_tokens 32 \
    --write_outputs_in_results True \
    --logging_dir "logs" \
    --prompt_key "prompt" \
    --prompt_head "LLAMA3_BASE_PROMPT" \
    --bnb_4bit_quant_type "nf4" \
    --bnb_4bit_compute_dtype "bfloat16" \
    --bnb_4bit_use_double_quant True \
    --load_in_4bit False \
    --output_dir "/l/users/abdelrahman.sadallah/cryptic_crosswords_checkpoints/" \


    




echo " ending " 