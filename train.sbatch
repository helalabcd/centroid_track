#! /bin/bash
#SBATCH -c 6
#SBATCH --mem 64G
#SBATCH -p gpu
#SBATCH -t 7200
#SBATCH -G V100:1
source ~/.bashrc
mamba activate torchem2
wandb offline
python train_centroid.py --lr=5e-4 --bs=16

