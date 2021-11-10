import torch
from functions import *
import glob
import cv2
import numpy as np
from tqdm import tqdm
import random
hWinSize = 16
drawing = False


i = 0
maindic = torch.load(r'E:\Google Drive\Acads\research\Single-View-Plant-Modelling\trainData/dict_tripartite.pt')
nLeaves, nStems = 0, 0
leafbuffer = []
stembuffer = []
for i in range(maindic['length']):
    image = maindic['images'][i]
    stem = maindic['stems'][i]
    leaves = maindic['leaves'][i]
    x, y, indices = sampleGrid(stem, step = 5, viz = False)

    image = cv2.copyMakeBorder(image, hWinSize, hWinSize, hWinSize, hWinSize, cv2.BORDER_CONSTANT)
    for point in tqdm(list(zip(y, x))):
        yy, xx = point
        window = image[yy - hWinSize : yy + hWinSize, xx - hWinSize : xx + hWinSize]
        try:
            H = hog(window)
            dic = {
                'feature' : H,
                'label' : 1,
            }
            nLeaves += 1
            stembuffer.append(dic)
        except:
            continue
        

    x, y, indices = sampleGrid(leaves, step = 12, viz = False)
    lis=  list(zip(y, x))
    random.shuffle(lis)
    for point in tqdm(lis):
        yy, xx = point
        window = image[yy - hWinSize : yy + hWinSize, xx - hWinSize : xx + hWinSize]
        
        try:
            H = hog(window)
            dic = {
                'feature' : H,
                'label' : 0,
            }
            nStems += 1
            leafbuffer.append(dic)
        except:
            continue
    print(f'\n{nStems}, {nLeaves}')
    
torch.save(leafbuffer, 'leafFeatures.pt')
torch.save(stembuffer, 'stemFeatures.pt')
    



