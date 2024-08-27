from datasets import HeLaCentroidDataset
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch_em
from torch_em.model import UNet2d
from wrapper import WrapUnet
import torch
import wandb
import argparse
from torch.optim.lr_scheduler import ReduceLROnPlateau

from evaluate_aogm import calculate_aogm, calculate_edit_distance
from augmentation import FixedTransform
from helpers import plot_sequence
import os

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default=None)
args = parser.parse_args()

print(args.model)

if args.model is None:
    for f in os.listdir("models"):
        print(f)
        content = f"""#! /bin/bash
#SBATCH -c 6
#SBATCH --mem 64G
#SBATCH -p gpu
#SBATCH -t 7200
#SBATCH -G RTX5000:1
source ~/.bashrc
mamba activate torchem2
wandb offline
python run_multi_eval.py --model="""
        content += f
        print(content, file=open(f"tmp_slurm.sbatch",'w'))
        os.system("sbatch -q7d tmp_slurm.sbatch")
        

    import sys
    sys.exit()
device = "cuda"


model = torch.load(f"models/{args.model}")


print(model)

aogm = calculate_aogm(model, mode="full")

print(aogm, file=open(f"models/{args.model}.txt",'w'))
