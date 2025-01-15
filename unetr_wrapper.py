from datasets import HeLaCentroidDataset
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch_em
from torch_em.model import UNet2d, UNETR
import torch
import wandb
import argparse
from torch.optim.lr_scheduler import ReduceLROnPlateau

from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from skimage import data, img_as_float
from torch_em.model import UNet2d

import os
import networkx as nx
from tqdm import tqdm
import numpy as np

from skimage.color import rgb2gray
from skimage.feature import blob_log

import uuid

class WrapUnetr(UNETR):
    def __init__(self, *args, **kwargs):
        # Call the __init__ method of the UNet2d class
        super(WrapUnetr, self).__init__(*args, **kwargs)


    def configure_inference(self):
        print("configuring inference (not doing anything)")


    def forward_inference(self, images, device="cuda"):
        # Takes PIL images and returns a DiGraph
        def expand(image):
            x,y = image.shape
            canvas = torch.zeros([x*3 for x in image.shape])
            #print(canvas.shape, image.shape)

            canvas[0:x, 0:y] = torch.flip(image, dims=(0,1))
            canvas[x:2*x, 0:y] = torch.flip(image, dims=(1,))
            canvas[2*x:3*x, 0:y] = torch.flip(image, dims=(0,1))

            canvas[0:x, y:2*y] = torch.flip(image, dims=(0,))
            canvas[x:2*x, y:2*y] = image
            canvas[2*x:3*x, y:2*y] = torch.flip(image, dims=(0,))

            canvas[0:x, 2*y:3*y] = torch.flip(image, dims=(1, 0))
            canvas[x:2*x, 2*y:3*y] = torch.flip(image, dims=(1,))
            canvas[2*x:3*x, 2*y:3*y] = torch.flip(image, dims=(0,1))

            return canvas

        graph = nx.DiGraph()

        node_id = 0
        for idx, i in tqdm(enumerate(images), total=len(images)):

            arr = np.array(i)

            shift_size = 8
            crop_size = 128
            canvi = []

            for ix, x in enumerate(range(0, arr.shape[0], shift_size)):
                #print("missing the last ", arr.shape[0] % shift_size)
                #print("Modified X", _x, arr.shape)
                for iy, y in enumerate(range(0, arr.shape[1], shift_size)):
                    #print("missing the last ", arr.shape[1] % shift_size)

                    crop = arr[ix*shift_size:ix*shift_size+crop_size, iy*shift_size:iy*shift_size+crop_size]
                    if crop.shape != (crop_size,crop_size,3):
                        #print("First pass shape mismatch, continuing", crop.shape)
                        continue

                    with torch.no_grad():
                        prediction = self.forward(torch.Tensor(crop).moveaxis(2, 0)[0][None, None].to(device))

                    canvas = torch.zeros(arr.shape[:2])
                    canvas[ix*shift_size:ix*shift_size+crop_size, iy*shift_size:iy*shift_size+crop_size] = prediction
                    canvi.append(canvas)

            for ix, x in enumerate(range(0, arr.shape[0], shift_size)):
                #print("missing the last ", arr.shape[0] % shift_size)
                x_shift = arr.shape[0] % shift_size
                #print("Modified X", _x, arr.shape)
                for iy, y in enumerate(range(0, arr.shape[1], shift_size)):
                    #print("missing the last ", arr.shape[1] % shift_size)
                    y_shift = arr.shape[1] % shift_size

                    crop = arr[ix*shift_size + x_shift:ix*shift_size+crop_size+x_shift, iy*shift_size + y_shift:iy*shift_size+crop_size+y_shift]
                    if crop.shape != (crop_size,crop_size,3):
                        #print("First pass shape mismatch, continuing", crop.shape)
                        continue

                    with torch.no_grad():
                        prediction = self.forward(torch.Tensor(crop).moveaxis(2, 0)[0][None, None].to(device))

                    canvas = torch.zeros(arr.shape[:2])
                    canvas[ix*shift_size + x_shift:ix*shift_size+crop_size+x_shift, iy*shift_size + y_shift:iy*shift_size+crop_size+y_shift] = prediction
                    canvi.append(canvas)


            stack = torch.stack(canvi, axis=0)

            xd = torch.sum(stack, axis=0) / torch.count_nonzero(stack, axis=0)

            np_array = xd.detach().cpu().numpy()

            # Save to a .npy file
            np.save(f"tmp/{str(uuid.uuid4())}.npy", np_array)

            plt.imshow(xd)
            plt.savefig(f"tmp/{str(uuid.uuid4())}.png")
            plt.close()

            """ old
            _xd = expand(xd)
            image_max = ndi.maximum_filter(_xd, size=20, mode='constant')
            coordinates = peak_local_max(image_max, min_distance=30)
            #print("Coordinates shape", coordinates.shape)
            coordinates = coordinates - xd.shape
            """

            expanded_img = expand(xd)
            image = torch.nan_to_num(torch.stack([expanded_img,expanded_img,expanded_img], axis=2))
            image_gray = rgb2gray(image)
            image_gray = image_gray / image_gray.max()

            blobs_log = blob_log(image_gray, max_sigma=10, num_sigma=10, overlap=0.1)
            coordinates = blobs_log[:, :2].astype(int)

            for x, y in coordinates:
                xmax, ymax = xd.shape
        
                x = x-xmax
                y = y-ymax
    
                if (0 <= x <= xmax) and (0 <= y <= ymax):
                    attributes = {'t': idx, 'x': y, 'y': x}
                    graph.add_node(node_id, **attributes)
                    node_id += 1

            """
            for x,y in coordinates:
                if (0 <= x <= xd.shape[0]) and (0 <= y <= xd.shape[1]):
                    #print(x,y)
                    attributes = {'t': idx, 'x': x, 'y': y}
                    print(attributes)
                    graph.add_node(node_id, **attributes)
                    node_id += 1
            """
        return graph
