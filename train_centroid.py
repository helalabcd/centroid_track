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

from evaluate_aogm import calculate_aogm
from augmentation import FixedTransform
from helpers import plot_sequence

parser = argparse.ArgumentParser()

parser.add_argument('--lr', type=float)
parser.add_argument('--bs', type=int)
args = parser.parse_args()

wandb.init(project='helalab5.0', mode='offline')

wandb.config = {
  "lr": args.lr,
  "bs": args.bs
}

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

device = "cuda"

train_dataset = HeLaCentroidDataset("HeLa_dataset/train")
validation_dataset = HeLaCentroidDataset("HeLa_dataset/test")

#model = UNet2d(in_channels=1, out_channels=1)
model = WrapUnet(in_channels=1, out_channels=1)
model.to(device)

optim = torch.optim.Adam(model.parameters(), lr=wandb.config["lr"])
#scheduler = ReduceLROnPlateau(optim, min_lr=4e-3, mode='min', factor=0.1, patience=5000, verbose=True)
criterion = torch.nn.MSELoss()

from torch.utils.data import DataLoader

train_dataloader = DataLoader(train_dataset, batch_size=wandb.config["bs"], shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=wandb.config["bs"], shuffle=True)
step = 0
for epoch in range(2500):
    train_losses = []

    if True or epoch%250 == 0:
        print("Starting AOGM calculation!")
        aogm = calculate_aogm(model, mode="first")
        wandb.log({"full_aogm": aogm}, step=step)

    for X, y in tqdm(train_dataloader):

        print("y shape before transform", y.shape)
        fixed_transformation = FixedTransform(min_angle=0, max_angle=359, crop_height=128, crop_width=128)
        X = torch.permute(X, (0,3,2,1))
        y = y[:, :, :, None]
        y = torch.permute(y, (0,3,2,1))
        
        X = fixed_transformation(X)
        y = fixed_transformation(y)

        #X = torch.permute(X, (0,3,2,1))
        #y = torch.permute(y, (0,3,2,1))
        optim.zero_grad()

        X = X[:, :1]
        print(X.shape, y.shape)
        #X = X.moveaxis(3, 1)[:, :1]
        #y = y[:, None]

        #print(X.shape, y.shape)
        #X = X.to(device)[:, :, :336, :464]
        #y = y.to(device)[:, :, :336, :464]
        print("X shape", X.shape)
        print("y shape", y.shape)
        X = X.to(device)
        y = y.to(device)
        #print(X.shape, y.shape)
        y_pred = model(X)
        print(y_pred.shape)

        loss = criterion(y_pred, y)
        loss.backward()
        #print(loss.item())
        train_losses.append(loss.item())
        optim.step()
        step += 1

        wandb.log({"train_loss": loss.item(), "learning_rate": get_lr(optim)}, step=step)

        #scheduler.step(loss.item())

    fig, ax = plt.subplots(1,3)
    ax.flat[0].imshow(X[0].T.detach().cpu())
    ax.flat[1].imshow(y[0].T.detach().cpu())
    ax.flat[2].imshow(y_pred[0].T.detach().cpu())
    #plt.savefig(f"epoch_{epoch}.png")
    plt.suptitle(f"Epoch {epoch}")
    wandb.log({"train_heatmap": wandb.Image(fig)}, step=step)
    plt.close(fig)
    
    print("tracking sequence 1")
    imgpath = plot_sequence(model)
    wandb.log({"tracking_sequence": wandb.Image(imgpath)}, step=step)
    print("tracking sequence 2")

    val_losses = []
    with torch.no_grad():
        for X, y in tqdm(validation_dataloader):
            X = X.moveaxis(3, 1)[:, :1]
            y = y[:, None]
            X = X.to(device)[:, :, :336, :464]
            y = y.to(device)[:, :, :336, :464]
            y_pred = model(X)

            loss = criterion(y_pred, y)
            val_losses.append(loss.item())
        wandb.log({"val_loss": sum(val_losses) / len(val_losses), "learning_rate": get_lr(optim)}, step=step)
        fig, ax = plt.subplots(1,3)
        ax.flat[0].imshow(X[0].T.detach().cpu())
        ax.flat[1].imshow(y[0].T.detach().cpu())
        ax.flat[2].imshow(y_pred[0].T.detach().cpu())
        #plt.savefig(f"epoch_{epoch}.png")
        plt.suptitle(f"Epoch {epoch}")
        wandb.log({"validation_heatmap": wandb.Image(fig)}, step=step)
        plt.close(fig)
