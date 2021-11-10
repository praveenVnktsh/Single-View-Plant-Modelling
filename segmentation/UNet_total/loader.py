
from torch.utils.data.dataset import Dataset 
import torch
import numpy as np
import glob
from skimage import io
import pandas as pd
from torchvision.transforms import transforms
from skimage.transform import resize
import cv2
from transforms import *


# dist = LensDistortion()
# vig = Vignetting()
# noise = GaussianNoise(std= 20)
# contrast = Contrast((0.5, 2))
# bri = Brightness((-25, 25))
# cut = Cutout([0.01, 0.01], [0.1, 0.1])
# persp = Perspective(max_rotation=(10, 10, 45))
# flip = Flip()
# def randomizeDomain(image, mask):
#     image, mask = noise(image, mask)
#     image, mask = contrast(image, mask)
#     image, mask = bri(image, mask)
    
#     image, mask = dist(image, mask)
#     image, mask = persp(image, mask)
#     image, mask = flip(image, mask)
        
#     return image, mask


dist = LensDistortion()
vig = Vignetting()
noise = GaussianNoise(std= 20)
contrast = Contrast((0.5, 2))
bri = Brightness((-25, 25))
cut = Cutout([0.01, 0.01], [0.1, 0.1])
persp = Perspective(max_rotation=(10, 10, 45))
flip = Flip()
backgroundRandom = BackgroundRandom('bgs.pt')


def randomizeDomain(image, mask, fullmask):

    # print(mask.dtype, fullmask.dtype, image.dtype)
    image = cv2.resize(image, (640, 360), cv2.INTER_NEAREST)
    mask = cv2.resize(mask, (640, 360), cv2.INTER_NEAREST)
    fullmask = cv2.resize(fullmask, (640, 360), cv2.INTER_NEAREST)
    image, mask, fullmask = backgroundRandom(image, mask, fullmask)

    image, mask = noise(image, mask)
    image, mask = contrast(image, mask)
    image, mask = bri(image, mask)
    image, mask = dist(image, mask)
    image, mask = persp(image, mask)
    image, mask = flip(image, mask)

        
    return image, mask, fullmask

    

class CustomDataset(Dataset):
    def __init__(self, items, randomize = True):   # initial logic  happens like transform
        self.items = items
        self.randomize = randomize
        self.length = len(items)
        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((360, 640)),
            # transforms.Resize((640, 360)),
            transforms.ToTensor(),
            ])

    def __getitem__(self, index):
        img, mask, fullmask = self.items[index]
        if self.randomize:
            img, mask, fullmask = randomizeDomain(img, mask, fullmask)

        img = self.transforms(img).float()
        mask = self.transforms(mask).float()

        return img, mask

    def __len__(self): 
        return self.length


if __name__ == "__main__":
    cd = CustomDataset('data/airsimDataset/')
    print(cd[0]['target'])