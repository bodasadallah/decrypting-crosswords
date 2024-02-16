#!/bin/bash

#SBATCH --job-name=cryptic_crosswords_text_similarity # Job name
#SBATCH --error=logs/%j%x.err # error file
#SBATCH --output=logs/%j%x.out # output log file
#SBATCH --nodes=1                   # Run all processes on a single node    
#SBATCH --ntasks=1                  # Run on a single CPU
#SBATCH --mem=40G                   # Total RAM to be used
#SBATCH --cpus-per-task=2          # Number of CPU cores
#SBATCH --time=12:00:00             # Specify the time needed for your experiment
#SBATCH -p cpu 
#SBATCH -q cpu-2                   # Use the gpu partition
##SBATCH --qos=gpu-8 
##SBATCH --gres=gpu:1                # Number of GPUs (per node)
##SBATCH --nodelist=ws-l6-005






python text_similarity.py