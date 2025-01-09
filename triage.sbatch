#! /bin/bash
#SBATCH -c 24
#SBATCH --mem 64G
#SBATCH -p gpu
#SBATCH -t 7200
#SBATCH -G RTX5000:1
source ~/.bashrc
mamba activate torchem2
wandb offline
python train_centroid.py --lr=3e-4 --bs=16 --eval_mode=first --experiment=triage
