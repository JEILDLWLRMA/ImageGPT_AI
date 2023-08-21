#!/bin/bash

#SBATCH --job-name=blip-VQA
#SBATCH --nodelist=ariel-v4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=3G
#SBATCH --time=1-0
#SBATCH --partition=batch_ugrad
#SBATCH -o logs/slurm-%A-%x.out

python blip2.py

# Letting SLURM know this code finished without any problem
exit 0
