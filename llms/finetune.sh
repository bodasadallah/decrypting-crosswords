#!/bin/bash

#SBATCH --job-name=cryptic_crosswords_spaces # Job name
#SBATCH --error=/home/daria.kotova/boda_code/decrypting-crosswords/llms/logs/%j%x.err # error file
#SBATCH --output=/home/daria.kotova/boda_code/decrypting-crosswords/llms/logs/%j%x.out # output log file
#SBATCH --time=24:00:00 # 10 hours of wall time
#SBATCH --nodes=1  # 1 GPU node
#SBATCH --mem=45000 # 32 GB of RAM
#SBATCH --nodelist=ws-l6-013


echo "starting......................."
###################### RUN LLM Finetune ######################


# MODEL_NAME="meta-llama/Llama-2-7b-hf"
MODEL_NAME="mistralai/Mistral-7B-v0.1"

# --do_train \
WANDB_PROJECT=cryptic_crosswords_mistral_stars_only_disjoint

echo $WANDB_PROJECT
python train.py \
--max_steps=10000 \
--save_steps=1000 \
--eval_steps=1000 \
--logging_steps=500 \
--report_to="all" \
--model_name=$MODEL_NAME \
--run_name=$MODEL_NAME \
--per_device_train_batch_size=64 \
--per_device_val_batch_size=32 \
--gradient_accumulation_steps=2 \
--gradient_checkpointing=1 \
--use_flash_attention_2=1 \
--eval_accumulation_steps=2 \
--save_dir='experiments/mistral-7b-v0.1_stars_only_disjoint' \
--train_dataset_path='/home/daria.kotova/boda_code/decrypting-crosswords/decrypt/data/clue_json/guardian/word_initial_disjoint/train.json' \
--test_dataset_path='/home/daria.kotova/boda_code/decrypting-crosswords/decrypt/data/clue_json/guardian/word_initial_disjoint/test.json' \
--old_dataset=1 \
--base_prompt="The next line is a clue for a cryptic crossword. Solve this clue. The number in the parenthesis in the clue represents the number of characters of the answer. After the clue, there is a template for the answer, where each * symbol represents a letter. Some letters are already filled in. Replace the * symbols with the correct letters of the answer. Output only the answer." \
--spaces=1 \
--percentage=0.2
# --checkpoint_path="experiments/mistral-7b-v0.1_disjoint_2/checkpoint-1500"
# --checkpoint_path="experiments/Mistral-7B-v0.1/checkpoint-24000" 

# --base_prompt="Below is a clue for a cryptic crossword. Replace underscores _ with letters of the answer to the clue." \

# --checkpoint_path="experiments/Mistral-7B-v0.1/checkpoint-24000" 


echo " ending "
#srun python run_clm.py config.json
