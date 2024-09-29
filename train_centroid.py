from datasets import HeLaCentroidDataset
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch_em
from torch_em.model import UNet2d
from wrapper import WrapUnet
import torch
import argparse
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import re

from evaluate_aogm import calculate_aogm, calculate_edit_distance
from augmentation import FixedTransform
from helpers import plot_sequence

parser = argparse.ArgumentParser()

parser.add_argument('--lr', type=float)
parser.add_argument('--bs', type=int)
parser.add_argument('--eval_each_n_epochs', type=int, default=100)
args = parser.parse_args()

from aim import Run
from aim import Image

run = Run()

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

device = "cuda"

train_dataset = HeLaCentroidDataset("HeLa_dataset/train")
validation_dataset = HeLaCentroidDataset("HeLa_dataset/test")

model = WrapUnet(in_channels=1, out_channels=1)
model.to(device)

optim = torch.optim.Adam(model.parameters(), lr=args.lr)
criterion = torch.nn.MSELoss()

from torch.utils.data import DataLoader

train_dataloader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=args.bs, shuffle=True)

step = 0
for epoch in range(250000000):
    train_losses = []

    if epoch%args.eval_each_n_epochs == 0:
        for fname in os.listdir("models"):
            if not fname.endswith(".txt"):
                continue

            if not fname.startswith(run.hash):
                continue
            print("Reading", fname)
            epoch = fname.split("_")
            match = re.search(r'_(\d+)\.pt', fname)
            if match:
                epoch = int(match.group(1))
            else:
                epoch = -1
            with open('models/' + fname, 'r') as file:
                aogm_score = file.read().rstrip()
            os.system(f"rm models/{fname}")
            run.track(float(aogm_score), name='aogm_score', epoch=epoch)

        print("Saving model")
        torch.save(model, f"models/{run.hash}_{epoch}.pt")

        content = f"""#! /bin/bash
#SBATCH -c 6
#SBATCH --mem 16G
#SBATCH -p gpu
#SBATCH -t 2880
#SBATCH -G RTX5000:1
source ~/.bashrc
mamba activate torchem2
wandb offline
python run_multi_eval.py --model="""
        content += f"models/{run.hash}_{epoch}.pt"
        print(content, file=open(f"tmp_slurm.sbatch",'w'))
        os.system("sbatch -q7d tmp_slurm.sbatch")



    for X, y in tqdm(train_dataloader):

        #print("y shape before transform", y.shape)
        fixed_transformation = FixedTransform(min_angle=0, max_angle=359, crop_height=128, crop_width=128)
        X = torch.permute(X, (0,3,2,1))
        y = y[:, :, :, None]
        y = torch.permute(y, (0,3,2,1))
        
        X = fixed_transformation(X)
        y = fixed_transformation(y)

        optim.zero_grad()

        X = X[:, :1]
        X = X.to(device)
        y = y.to(device)
        y_pred = model(X)

        loss = criterion(y_pred, y)
        loss.backward()
        train_losses.append(loss.item())
        optim.step()
        step += 1

        run.track(loss.item(), name='train_loss', step=step, epoch=epoch)
        run.track(get_lr(optim), name='learning_rate', step=step, epoch=epoch)
        #scheduler.step(loss.item())

    fig, ax = plt.subplots(1,3)
    ax.flat[0].imshow(X[0].T.detach().cpu())
    ax.flat[1].imshow(y[0].T.detach().cpu())
    ax.flat[2].imshow(y_pred[0].T.detach().cpu())
    #plt.savefig(f"epoch_{epoch}.png")
    plt.suptitle(f"Epoch {epoch}")
    #wandb.log({"train_heatmap": wandb.Image(fig)}, step=step)
    run.track(Image(fig), step=step, epoch=epoch, name="train_heatmap")
    plt.close(fig)
    

    val_losses = []
    with torch.no_grad():
        for X, y in tqdm(validation_dataloader):
            fixed_transformation = FixedTransform(min_angle=0, max_angle=359, crop_height=128, crop_width=128)
            X = torch.permute(X, (0,3,2,1))
            y = y[:, :, :, None]
            y = torch.permute(y, (0,3,2,1))

            X = fixed_transformation(X)
            y = fixed_transformation(y)
            X = X[:, :1]
            X = X.to(device)
            y = y.to(device)
            y_pred = model(X)

            loss = criterion(y_pred, y)
            val_losses.append(loss.item())
        #wandb.log({"val_loss": sum(val_losses) / len(val_losses), "learning_rate": get_lr(optim)}, step=step)
        run.track(sum(val_losses) / len(val_losses), name='val_loss', step=step, epoch=epoch)
        fig, ax = plt.subplots(1,3)
        ax.flat[0].imshow(X[0].T.detach().cpu())
        ax.flat[1].imshow(y[0].T.detach().cpu())
        ax.flat[2].imshow(y_pred[0].T.detach().cpu())
        #plt.savefig(f"epoch_{epoch}.png")
        plt.suptitle(f"Epoch {epoch}")
        run.track(Image(fig), step=step, epoch=epoch, name="validation_heatmap") 
        plt.close(fig)
