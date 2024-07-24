import torch
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import numpy as np


class FixedTransform():
    def __init__(self, min_angle, max_angle, crop_height, crop_width):
        self.angle = torch.FloatTensor(1).uniform_(min_angle, max_angle).item()
        self.position = None
        self.crop_height = crop_height
        self.crop_width = crop_width

    def __call__(self, img):
        if self.position is not None:
            return self.inner__call__(img)
        
        out_of_bounds = True
        while out_of_bounds:
            self.position = None
            frame = self.inner__call__(img)

            ex = frame[0]

            zero = torch.Tensor([0,0,0])
            out_of_bounds = torch.allclose(ex[:, 0, 0], zero) or \
                torch.allclose(ex[:, 0, -1], zero) or \
                torch.allclose(ex[:, -1, 0], zero) or \
                torch.allclose(ex[:, -1, -1], zero)
        return frame


    def inner__call__(self, img):
        # Check if the input is a PIL Image or a torch Tensor
        if isinstance(img, torch.Tensor):
            # For torch Tensors, the channel dimension is typically first
            _, _, h, w = img.size()
        else:
            # For PIL images, use the .size attribute
            w, h = img.size

        if self.position is None:
            th, tw = self.crop_height, self.crop_width
            if w == tw and h == th:
                self.position = (0, 0)
            else:
                i = torch.randint(0, h - th + 1, size=(1,)).item()
                j = torch.randint(0, w - tw + 1, size=(1,)).item()
                self.position = (i, j)

        img = TF.rotate(img, self.angle)
        img = TF.crop(img, *self.position, self.crop_height, self.crop_width)
        #img = TF.resize(img, (224, 224))
        if not isinstance(img, torch.Tensor):
            img = TF.to_tensor(img)

        return img
