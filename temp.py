import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch.nn as nn
import cv2
transt = transforms.ToTensor()
transp = transforms.ToPILImage()


im = cv2.imread(r'E:\Google Drive\Acads\research\Single-View-Plant-Modelling\Dataset\TrainData\images\00041.png')
height, width = im.shape[:2]
step = 16

if min(height, width) == height:
    height = int(width*(640/height))
    height -= height % step
    im = cv2.resize(im, (height, 640))
else:
    width = int(height*(640/width))
    width -= width % step
    im = cv2.resize(im, (640, width))

height, width = im.shape[:2]
im = torch.tensor(im)
#torch.Tensor.unfold(dimension, size, step)

patches = im.data.unfold(0, step, step).unfold(1, step, step)
nh, nw = patches.shape[0:2]
patches = patches.flatten(start_dim=0, end_dim=1)
patches = torch.transpose(patches, 1, 3)
patches = torch.transpose(patches, 1, 2)

# for i in range(patches.shape[0]):
#     imm = cv2.resize(patches[i].numpy(), (0, 0), fx = 4, fy = 4)
#     cv2.imshow('a', imm)
#     if cv2.waitKey(10) == ord('q'):
#         exit()

finalimg = np.zeros((height, width, 3),dtype=  np.uint8)
print(patches.numel(), finalimg.size)
print(nh, nw)
for i in range(nh):
    for j in range(nw):
        im = patches[i * nw + j].numpy()
        # imm = cv2.resize(patches[i].numpy(), (0, 0), fx = 4, fy = 4)
        finalimg[i*step:i*step+step, j*step:j*step+step] = im
        cv2.imshow('a', finalimg)
        if cv2.waitKey(1) == ord('q'):
            exit()

    

cv2.imshow('a', finalimg)
cv2.waitKey(0)