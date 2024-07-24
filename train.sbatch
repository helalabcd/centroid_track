#! /bin/bash
#SBATCH -c 6
#SBATCH --mem 32G
#SBATCH -p gpu
#SBATCH -t 10080
#SBATCH -G RTX5000:1
source ~/.bashrc
mamba activate torchem2
wandb offline
python train_centroid.py --lr=5e-4 --bs=16

