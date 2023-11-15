#!/bin/bash

#SBATCH --job-name=decrypt-crosswords
#SBATCH --error=../../logs/%j%x.err # error file
#SBATCH --output=../../logs/%j%x.out # output log file
#SBATCH --time=24:00:00 # 10 hours of wall time
#SBATCH --nodes=1  # 1 GPU node
#SBATCH --nodelist=ws-l4-009

python train_clues.py \
--default_val=base \
--name=naive_acw_val \
--wandb_dir='../../wandb' \
--project=baseline_naive \
--data_dir='../data/clue_json/guardian/naive_random' \
--ckpt_path='../../wandb/wandb/run-20231028_200336-23m5kqq7/files/epoch_34.pth.tar'
