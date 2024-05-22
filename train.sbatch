#! /bin/bash
#SBATCH -c 6
#SBATCH --mem 32G
#SBATCH -p gpu
#SBATCH -t 7200
#SBATCH -G rtx5000:1
source ~/.bashrc
mamba activate torchem2
wandb offline
python train.py --lr=1e-2 --bs=16
