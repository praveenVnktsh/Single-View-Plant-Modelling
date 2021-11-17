from utils import scaleAndShow
import cv2
import numpy as np
import timeit
from skimage.morphology import skeletonize as sk
from numba import jit
from skimage import feature
from skimage import exposure
from skimage import feature
import scipy.ndimage as ndi



def sampleGrid(mask, step = 5, viz = True):
    h, w = mask.shape
    print('Gridding', mask.shape)

    x = np.arange(0, w - 1, step).astype(int)
    y = np.arange(0, h - 1, step).astype(int)

    indices = np.ix_(y,x)
    temp = mask.copy()
    temp[indices] = 128
    temp[mask == 0] = 0
    indices = temp == 128

    if viz:
        temp = mask.copy()
        temp[indices] = 128
        scaleAndShow(temp, 'ab', height = 600)
    y, x = np.where(indices == True)
    return x, y, indices
