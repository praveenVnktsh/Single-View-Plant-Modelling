import torch
import cv2

data = torch.load('fullImgDataset.pt')

images = data[0]
masks = data[1]

dataset = list(zip(images, masks))
for img, mask in dataset:
    cv2.imshow('a', img)
    cv2.imshow('b', mask)
    cv2.waitKey(0)