import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import random

from torch.utils.data import Dataset
import os
from PIL import Image
from helpers import gaussian_on_canvas, get_cell_centroids, get_centroid_map

import os.path
import pickle

import warnings
warnings.filterwarnings("ignore")

class HeLaCentroidDataset(Dataset):

    def __init__(self, base_path, simulate_smaller_dataset=None):
        # simulate_smaller_dataset must be integer between 0 and 1, indicating percentage value

        self.bursts = os.listdir(base_path)
        random.shuffle(self.bursts)

        if simulate_smaller_dataset is not None:
            limit = int(len(self.bursts) * simulate_smaller_dataset)
            self.bursts = self.bursts[:limit]
            print(f"DATASET: SIMULATING SMALLER DATASET WITH size {simulate_smaller_dataset}")
        print(f"DATASET: NUMBER OF BURSTS IN USE: {len(self.bursts)}")

        self.cached_bursts = []
        self.burst_start_index = []
        current_img_index_counter = 0
        
        for burst in tqdm(self.bursts):
            
            cache_base = "CACHE"
            if not os.path.exists(cache_base):
                os.makedirs(cache_base)


            cache_path = f"{cache_base}/{burst}.cache.p"
            if not os.path.isfile(cache_path):
                print("Processing uncached burst", burst)
                this_burst = []
                self.burst_start_index.append(current_img_index_counter)
                
                images = os.listdir(f"{base_path}/{burst}/img1/")
                head = ["frame", "cellid", "a", "b", "c", "d", "xy","xz","xu","x"]
                df = pd.read_csv(f"{base_path}/{burst}/gt/gt.txt", sep=",", names=head)

                for img in sorted(images):
                    frame_index = int(img.split(".")[0])
                    img = Image.open(f"{base_path}/{burst}/img1/{img}")

                    centroid_map = get_cell_centroids(img, df, frame_index=frame_index)

                    mask = get_centroid_map(centroid_map)
                    img = torch.Tensor(np.array(img)) / 255
                    mask = torch.Tensor(mask)

                    this_burst.append((img, mask))
                    current_img_index_counter += 1
                    
                self.cached_bursts.append(this_burst)
                pickle.dump( this_burst, open( cache_path, "wb" ) )

            else:
                print("Loading cached burst", burst)
                this_burst = pickle.load( open( cache_path, "rb" ) )
                self.cached_bursts.append(this_burst)
                self.burst_start_index.append(current_img_index_counter)
                current_img_index_counter += len(this_burst)

        self.length = current_img_index_counter

    def __getitem__(self, idx):
        lst = [x for x in self.burst_start_index if x <= idx]
        start_value = max(lst)
        start_index = lst.index(start_value)

        this_val = self.cached_bursts[start_index][idx-start_value]
        return this_val

    def __len__(self):
        return self.length
        
