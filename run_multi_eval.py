from datasets import HeLaCentroidDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from aim import Run
from aim import Image


import torch_em
from torch_em.model import UNet2d
from unetr_wrapper import WrapUnetr
import torch
import wandb
import argparse
from torch.optim.lr_scheduler import ReduceLROnPlateau

from evaluate_aogm import calculate_aogm, calculate_edit_distance
from augmentation import FixedTransform
from helpers import plot_sequence
import os
import sys
parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default=None)
args = parser.parse_args()

print(args.model)

if args.model is None:
    for f in os.listdir("models"):
        if not f.endswith(".pt"):
            # not a model file
            continue
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
        

    sys.exit()
device = "cuda"

#aim_hash = args.model.split("_")[0]
#aim_hash = aim_hash.split("/")[-1]
#run = Run(aim_hash)

#print(run)

model = torch.load(f"{args.model}")


print(model)

aogm = calculate_aogm(model, mode="first", filename_prefix=args.model)

print(aogm, file=open(f"{args.model}.txt",'w'))
