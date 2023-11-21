#!/bin/bash

#SBATCH --job-name=decrypt-crosswords
#SBATCH --error=../../logs/%j%x.err # error file
#SBATCH --output=../../logs/%j%x.out # output log file
#SBATCH --time=24:00:00 # 10 hours of wall time
#SBATCH --nodes=1  # 1 GPU node
## SBATCH --partition=gpu
## SBATCH --gres=gpu:4
#SBATCH --mem=46000
# SBATCH --nodelist=ws-l4-014

# srun python v2.py

echo "starting......................."
###################### Without Criculm ######################
# python train_clues.py \
# --default_train=base \
# --name=naive_word_intial_disjoint \
# --project=baseline \
# --wandb_dir='../../wandb' \
# --data_dir='../data/clue_json/guardian/word_initial_disjoint' \
# --num_epochs=25 \
# --batch_size=128
# # --ckpt_path=
# # --resume_train

###################### With Criculm ######################

python train_clues.py \
--default_train=base \
--name=naive_random_ACW \
--project=baseline \
--wandb_dir='../../wandb' \
--data_dir='../data/clue_json/guardian/naive_random' \
--num_epochs=30 \
--multitask='ACW' \
--batch_size=128 \
--ckpt_path='../../wandb/wandb/run-20231026_140400-5sv7luib/files/epoch_3.pth.tar' \
--resume_train \
--hacky

echo " ending "
#srun python run_clm.py config.json
