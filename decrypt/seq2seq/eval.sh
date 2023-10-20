#!/bin/bash

#SBATCH --job-name=decrypt-crosswords
#SBATCH --error=../../logs/%j%x.err # error file
#SBATCH --output=../../logs/%j%x.out # output log file
#SBATCH --time=24:00:00 # 10 hours of wall time
#SBATCH --nodes=1  # 1 GPU node

python train_clues.py \
--default_val=base \
--name=naive_word_intial_disjoint_val \
--wandb_dir='../../wandb' \
--project=baseline_naive \
--data_dir='../data/clue_json/guardian/word_initial_disjoint' \
--ckpt_path='../../wandb/wandb/run-20231011_211723-3sdgff0h/files/epoch_12.pth.tar'
