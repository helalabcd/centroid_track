from datasets import HeLaCentroidDataset
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch_em
from torch_em.model import UNet2d
import torch
import wandb
import argparse
from torch.optim.lr_scheduler import ReduceLROnPlateau

parser = argparse.ArgumentParser()

parser.add_argument('--lr', type=float)
parser.add_argument('--bs', type=int)
args = parser.parse_args()

wandb.init(project='helalab4', mode='offline')

wandb.config = {
  "lr": args.lr,
  "bs": args.bs
}

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

device = "cuda"

train_dataset = HeLaCentroidDataset("HeLa_dataset/train")

model = UNet2d(in_channels=1, out_channels=1)
model.to(device)

optim = torch.optim.Adam(model.parameters(), lr=wandb.config["lr"])
#scheduler = ReduceLROnPlateau(optim, min_lr=4e-3, mode='min', factor=0.1, patience=5000, verbose=True)
criterion = torch.nn.MSELoss()

from torch.utils.data import DataLoader

train_dataloader = DataLoader(train_dataset, batch_size=wandb.config["bs"], shuffle=True)
step = 0
for epoch in range(2500):
    train_losses = []
    for X, y in tqdm(train_dataloader):

        optim.zero_grad()


        #print(X.shape, y.shape)
        X = X.moveaxis(3, 1)[:, :1]
        y = y[:, None]

        #print(X.shape, y.shape)
        X = X.to(device)[:, :, :336, :464]
        y = y.to(device)[:, :, :336, :464]


        #print(X.shape, y.shape)
        y_pred = model(X)

        loss = criterion(y_pred, y)
        loss.backward()
        print(loss.item())
        train_losses.append(loss.item())
        optim.step()
        step += 1

        wandb.log({"loss": loss.item(), "learning_rate": get_lr(optim)}, step=step)
        #scheduler.step(loss.item())    
    fig, ax = plt.subplots(1,3)
    ax.flat[0].imshow(X[0].T.detach().cpu())
    ax.flat[1].imshow(y[0].T.detach().cpu())
    ax.flat[2].imshow(y_pred[0].T.detach().cpu())
    #plt.savefig(f"epoch_{epoch}.png")
    plt.suptitle(f"Epoch {epoch}")
    wandb.log({"heatmap": wandb.Image(fig)}, step=step)
    plt.close(fig)

