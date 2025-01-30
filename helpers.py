import numpy as np
import pandas as pd
import os
import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image
import math
import uuid

HELAPATH = os.getenv('helapath')

def plot_sequence(model):
    #datapath = "HeLa_dataset/test/Burst4_A2_1_VesselID-29_1-0/img1/"
    datapath = os.path.join(HELAPATH, "test/Burst4_A2_1_VesselID-29_1-0/img1/")
    frames = os.listdir(datapath)
    frames = [x for x in frames if x.endswith(".tiff")]
    frames = sorted(frames)
    frames = [datapath + x for x in frames]

    inframes = [Image.open(x) for x in frames]
    g = model.forward_inference(inframes)

    side = int(math.sqrt(len(frames))) + 1
    fig, ax = plt.subplots(side, side, figsize=(50,50))

    for idx, frame in enumerate(frames):
        idx += 1
        ax.flat[idx-1].imshow(Image.open(frame))

    for n in g.nodes:
        t = g.nodes[n]["t"]
        x = g.nodes[n]["x"]
        y = g.nodes[n]["y"]
        ax.flat[t-1].scatter([x], [y])
    
    filename = "/tmp/" + str(uuid.uuid4()) + ".png"
    plt.savefig(filename)
    return filename


def gaussian_on_canvas(canvas_size, mue=(10,10), sigma=(1,1)):    
    # Parameters
    mu_x, mu_y = mue  # Center of the distribution
    sigma_x, sigma_y = sigma  # Standard deviations
    size = 100  # Size of the 2D array
    
    # Create a grid of (x,y) coordinates
    x = np.linspace(0, canvas_size[0], canvas_size[0])
    y = np.linspace(0, canvas_size[1], canvas_size[1])
    X, Y = np.meshgrid(x, y)
    
    # Gaussian function
    canvas = (1/(2 * np.pi * sigma_x * sigma_y)) * np.exp(-((X - mu_x)**2 / (2 * sigma_x**2) + (Y - mu_y)**2 / (2 * sigma_y**2)))

    return canvas / canvas.max()

def get_cell_centroids(img, df, frame_index):
    mask_canvas = np.zeros(np.array(img).shape)[:, :, 0]
    
    for cell_id in df[df["frame"] == frame_index]["cellid"].tolist():
    
        row = df[df["frame"] == frame_index][df["cellid"] == cell_id]
        a = row.a.values[0]
        b = row.b.values[0]
        c = row.c.values[0]
        d = row.d.values[0]
        a = int(a)
        b = int(b)
        c = int(c)
        d = int(d)
    
        #img[b:b+d, a:a+c] = 255
        try:
            mask_canvas[b+(d//2), a+(c//2)] = cell_id
        except:
            print("mask canvas out of bounds, skipping this edge case")
    return mask_canvas

def get_centroid_map(mask_img):
    gaussians = []
    
    for cell_id in np.unique(np.array(mask_img)):
        
        if cell_id == 0:
            # Background
            continue
        
        match_mask = np.array(mask_img) == cell_id
    
        # Find the indices where the array is True
        y_indices, x_indices = np.where(match_mask)
        
        # Calculate the average of the indices
        centroid_y = np.mean(y_indices)
        centroid_x = np.mean(x_indices)
    
        #print(cell_id, centroid_y, centroid_x)
        gaussians.append(gaussian_on_canvas(mask_img.shape, mue=(centroid_y, centroid_x), sigma=(2,2)))


    return np.max(np.array(gaussians), axis=0).T
