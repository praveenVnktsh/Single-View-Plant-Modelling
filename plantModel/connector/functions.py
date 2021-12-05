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

def removeBranches(skeleton, mask):
    selems = list()
    selems.append(np.array([[0, 1, 0], [1, 1, 1], [0, 0, 0]]))
    selems.append(np.array([[1, 0, 1], [0, 1, 0], [1, 0, 0]]))
    selems.append(np.array([[1, 0, 1], [0, 1, 0], [0, 1, 0]]))
    selems.append(np.array([[0, 1, 0], [1, 1, 0], [0, 0, 1]]))
    selems.append(np.array([[0, 0, 1], [1, 1, 1], [0, 1, 0]]))
    selems += [np.rot90(selems[i], k=j) for i in range(5) for j in range(4)]

    selems.append(np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
    selems.append(np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]]))

    branches = np.zeros_like(skeleton, dtype=bool)
    for selem in selems:
        branches |= ndi.binary_hit_or_miss(skeleton, selem)
    skeleton = skeleton.astype(np.uint8)*255
    branches = branches.astype(np.uint8)*255
    y, x = np.where(branches == 255)
    for p in list(zip(x, y)):
        # cv2.circle(mask, p, 15, 0, -1)
        mask[p[1] - 15 : p[1] + 15, p[0] - 15: p[0] + 15] = 0
    # skeleton[branches == 255] = 0

    return mask


def skeletonize(mask):
    mask[mask == 255] = 1
    skeleton = sk(mask)

    return skeleton


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
