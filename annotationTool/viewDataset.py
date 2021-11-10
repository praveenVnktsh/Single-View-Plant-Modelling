import torch
import cv2

from functions import scaleAndShow
dataset = torch.load('dataset.pt')

for im in dataset:
    a, b, c = im    
    scaleAndShow(a, 'a')
    scaleAndShow(b, 'b')
    scaleAndShow(c, 'c', waitkey=0)