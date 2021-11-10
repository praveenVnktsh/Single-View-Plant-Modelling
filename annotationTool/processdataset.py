from collate import scaleAndShow
from functions import getWindow, sampleGrid
import cv2
import numpy as np
import torch
from glob import glob

k = 0
windowsize = 50
hwinsize=  windowsize//2
buffer = []
# image, mask, stemmask

stepsize = 10
for i in range(5):

    im = cv2.imread(f'annotated/image/img_{i}.png')
    if im is None:
        continue
    fullmask = cv2.imread(f'annotated/plantmask/mask_{i}.png', 0)
    if fullmask is None:
        continue

    stemmask = cv2.imread(f'annotated/stemmask/mask_{i}.png', 0)
    if stemmask is None:
        continue
    temp = im.copy()

    temp[:, :, 0][fullmask == 255] = 255
    temp[:, :, 1][stemmask == 255] = 255

    k = scaleAndShow(temp, waitkey=1)
    if k == ord('a'):
        print(i, end = ',')
    
    fullmask = cv2.copyMakeBorder(fullmask, hwinsize, hwinsize, hwinsize, hwinsize, cv2.BORDER_CONSTANT)
    im = cv2.copyMakeBorder(im, hwinsize, hwinsize, hwinsize, hwinsize, cv2.BORDER_REFLECT)
    stemmask = cv2.copyMakeBorder(stemmask, hwinsize, hwinsize, hwinsize, hwinsize, cv2.BORDER_REFLECT)

    x, y, indices = sampleGrid(fullmask, step = stepsize, viz = True)    
    pts = list(zip(y, x))
    for pt in pts:
        imgwindow = getWindow(im, pt, winsize= windowsize)
        plantwindow = getWindow(fullmask, pt, winsize= windowsize)
        stemwindow = getWindow(stemmask, pt, winsize= windowsize)
        # scaleAndShow(imgwindow, 'a', waitkey=1)
        # scaleAndShow(plantwindow, 'b', waitkey=1)
        # scaleAndShow(stemwindow, 'c', waitkey=1)
        t = [imgwindow, plantwindow, stemwindow]
        buffer.append(t)
        k += 1


    i = 0
    while i != len(pts):
        pt = [np.random.randint(windowsize, im.shape[0] - windowsize), np.random.randint(windowsize, im.shape[1] - windowsize)]
        imgwindow = getWindow(im, pt, winsize= windowsize)
        plantwindow = getWindow(fullmask, pt, winsize= windowsize)
        stemwindow = getWindow(stemmask, pt, winsize= windowsize)
        try:
            # cv2.imshow('a', imgwindow)
            # cv2.imshow('b', plantwindow)
            # cv2.imshow('c', stemwindow)

            if imgwindow.shape != (windowsize + 1, windowsize + 1, 3):
                print('continue', imgwindow.shape)
                continue
            if plantwindow.shape != (windowsize + 1, windowsize + 1):
                print('continue plant', plantwindow.shape)
                continue
            if stemwindow.shape != (windowsize + 1, windowsize + 1):
                print('continue stem', stemwindow.shape)
                continue
            t = [imgwindow, plantwindow, stemwindow]
            buffer.append(t)
            i += 1
        except:
            continue
            
    print(len(buffer))
torch.save(buffer, 'dataset.pt')