import glob
from typing import List
from utils import *
import cv2
import numpy as np

from sklearn.linear_model import Ridge
from numpy.core.arrayprint import DatetimeFormat
import os
import time
import random
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from functions import *



def predict(image):
    import torch
    model = torch.load('svm.pt')

    enhanced = enhanceContrast(image)
    mask = threshold(enhanced, lg= np.array([25, 40, 40]), ug = np.array([86, 255, 255]))
    mask = grabcut(image, mask)
    hWinSize = 16
    mask = cv2.copyMakeBorder(mask, hWinSize, hWinSize, hWinSize, hWinSize, cv2.BORDER_CONSTANT)
    enhanced = cv2.copyMakeBorder(enhanced, hWinSize, hWinSize, hWinSize, hWinSize, cv2.BORDER_CONSTANT)

    x, y, indices = sampleGrid(mask, step = 5, viz = False)
    mask = np.stack((mask, mask, mask), axis = 2)
    plantSeg = ((mask.astype(float)/255) * enhanced).astype(np.uint8)


    predictions = plantSeg.copy()
    prednodes = np.zeros_like(predictions)
    nodes = []
    leafnodes = []
    stepsize = 3
    for point in list(zip(y, x)):
        yy, xx = point
        window = enhanced[yy - hWinSize : yy + hWinSize, xx - hWinSize : xx + hWinSize]
        H = hog(window)
        label = model.predict(H.reshape(1, 324))
        if label == 1:
            prednodes[yy - stepsize : yy + stepsize, xx - stepsize : xx + stepsize] = 255
            cv2.circle(predictions, (xx, yy), 2, (0, 0, 255), -1)

    
    prednodes = prednodes & mask
    prednodes = prednodes[:, :, 0]


    scaleAndShow(predictions, 'outp', waitkey = 1)
    print('Classification complete---------')
    # return nodes, leafnodes
    return prednodes



if __name__ == '__main__':
    folderpath = r'E:\Google Drive\Acads\Mitacs\dataset\SterioCameraPicsFilms\*.jpg'

    date = datetime.now().strftime("%m%d_%I%M%S")
    i =0 
    path = r'E:\Google Drive\Acads\Mitacs\dataset\Cam 202106\N1/WhatsApp Image 2021-06-09 at 09.11.31.jpeg'
    im = cv2.imread(path)
    im = cv2.resize(im, (352, 480), interpolation= cv2.INTER_NEAREST)
    for i, path in enumerate(glob.glob(r'E:\Google Drive\Acads\Mitacs\dataset\images\indoor/*.jpg', recursive= True)):
        ima = cv2.imread(path)
        
        im = cv2.resize(ima, (0, 0), fx = 0.1, fy = 0.1, interpolation= cv2.INTER_NEAREST)
        predictions = predict(im)
        cv2.imwrite(f'masks/{i}.jpg', predictions)


