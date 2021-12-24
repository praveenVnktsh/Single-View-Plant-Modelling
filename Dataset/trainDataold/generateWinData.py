import torch
from functions import *
import glob
import cv2
import numpy as np
from tqdm import tqdm
import random
import os

os.makedirs('winData/leaf/mask', exist_ok=True)
os.makedirs('winData/leaf/images', exist_ok=True)

os.makedirs('winData/stem/mask', exist_ok=True)
os.makedirs('winData/stem/images', exist_ok=True)

os.makedirs('winData/bg/mask', exist_ok=True)
os.makedirs('winData/bg/images', exist_ok=True)


hWinSize = 50
drawing = False


i = 0
maindic = torch.load(r'E:\Google Drive\Acads\research\Single-View-Plant-Modelling\trainData/dict_tripartite.pt')
nLeaves, nStems = 0, 0
leafbuffer = []
stembuffer = []
counters = [
    0, 0, 0
]
viz = False
for i in range(maindic['length']):
    image = maindic['images'][i]
    stem = maindic['stems'][i]
    leaves = maindic['leaves'][i]
    
    background = 255 - (stem.astype(bool) + leaves.astype(bool)).astype(np.uint8)*255
    
    x, y, indices = sampleGrid(stem, step = 15, viz = viz)

    image = cv2.copyMakeBorder(image, hWinSize, hWinSize, hWinSize, hWinSize, cv2.BORDER_CONSTANT)
    stem = cv2.copyMakeBorder(stem, hWinSize, hWinSize, hWinSize, hWinSize, cv2.BORDER_CONSTANT)
    for j, point in (enumerate(list(zip(y, x)))):
        yy, xx = point
        window = image[yy : yy + hWinSize * 2, xx : xx + hWinSize * 2]
        cv2.imwrite(f'winData/stem/images/stem_{str(counters[0] + j).zfill(5)}.png', window)        
        mask = stem[yy : yy + hWinSize * 2, xx : xx + hWinSize * 2]
        cv2.imwrite(f'winData/stem/mask/stem_{str(counters[0] + j).zfill(5)}.png', mask)
        # cv2.imshow('stem', window)
        # cv2.waitKey(1)
        
    counters[0] += j
    x, y, indices = sampleGrid(leaves, step = 15, viz = viz)
    lis =  list(zip(y, x))

    random.shuffle(lis)
    leaves = cv2.copyMakeBorder(leaves, hWinSize, hWinSize, hWinSize, hWinSize, cv2.BORDER_CONSTANT)
    for leafindex, point in (enumerate((lis))):
        yy, xx = point
        window = image[yy : yy + hWinSize * 2, xx : xx + hWinSize * 2]
        if counters[1] + leafindex > counters[0]:
            break
        cv2.imwrite(f'winData/leaf/images/leaf_{str(counters[1] + leafindex).zfill(5)}.png', window)      
        mask = leaves[yy : yy + hWinSize * 2, xx : xx + hWinSize * 2]
        cv2.imwrite(f'winData/leaf/mask/leaf_{str(counters[1] + leafindex).zfill(5)}.png', mask)
  
        # cv2.imshow('stem', window)
        # cv2.waitKey(1)
    
    x, y, indices = sampleGrid(background, step = 25, viz = viz)
    lis =  list(zip(y, x))

    counters[1] += leafindex
    for j, point in (enumerate((lis))):
        yy, xx = point
        window = image[yy : yy + hWinSize * 2, xx : xx + hWinSize * 2]
        if counters[2] + j > counters[1]:
            break
        cv2.imwrite(f'winData/bg/images/bg_{str(counters[2] + j).zfill(5)}.png', window)  
        cv2.imwrite(f'winData/bg/mask/bg_{str(counters[2] + j).zfill(5)}.png', np.zeros_like(window))  
        # cv2.imshow('stem', window)
        # cv2.waitKey(1)
    
    counters[2] += j

    print(counters)


