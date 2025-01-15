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
from torch.utils.data import DataLoader

from evaluate_aogm import calculate_aogm, calculate_edit_distance
from augmentation import FixedTransform
from helpers import plot_sequence
import os
import sys
parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default=None)
parser.add_argument('--eval_mode', type=str, default="first")
args = parser.parse_args()

HELAPATH = os.getenv('helapath')
os.system("mkdir models_done")
print(args.model)

device = "cuda"

aim_hash = args.model.split("_")[0]
aim_hash = aim_hash.split("/")[-1]
print("Aim hash", aim_hash)
run = Run(aim_hash)

print("Loaded run", run)

models = [x for x in os.listdir("models") if x.startswith(args.model)]
sorted_models = sorted(models, key=lambda x: int(x.split('_')[-1].split('.')[0]))
#sorted_models.reverse()

for modelname in sorted_models:
    print(modelname)

    model = torch.load(f"models/{modelname}")
    model = torch.compile(model)
    
    epoch = modelname.split("_")[1]
    epoch = epoch.split(".pt")[0]
    epoch = int(epoch)

    # TEMPORARY HACK! TMP
    if epoch < 76:
        print("Epoch too small, continuing!")
        continue

    print(model)
    print("Calculating aogm in mode ", args.eval_mode)
    aogm_dict = calculate_aogm(model, mode=args.eval_mode, filename_prefix=modelname)

    for burst, aogm in aogm_dict.items():
        #print(aogm, file=open(f"{args.model}-{burst}.txt",'w'))
        print(burst, aogm)
        run.track(float(aogm), name=f'NEW_aogm_{burst}', epoch=epoch)

    with open(f"models_done/{modelname}.DONE",'w') as file:
        pass
