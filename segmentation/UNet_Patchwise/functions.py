# from utils import scaleAndShow
import cv2
import numpy as np
from skimage.morphology import skeletonize as sk
from skimage import feature
from skimage import exposure
from skimage import feature


def hog(image):
    (H, hogImage) = feature.hog(
        image, orientations=9, pixels_per_cell=(8, 8),
        cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1",
        visualize=True
    )
    hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
    hogImage = hogImage.astype("uint8")
    return H



def getWindow(mask, point, winsize):
    window = mask[
        point[0] : point[0] + winsize,
        point[1] : point[1] + winsize
    ]
    return window

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
        # scaleAndShow(temp, 'ab', height = 600)
    y, x = np.where(indices == True)
    return x, y, indices

def enhanceContrast(image) : 
    print('Enhancing contrast')
    lab= cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
    cl = clahe.apply(l)

    limg = cv2.merge((cl,a,b))

    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final
    
def grabcut(image, mask):
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    mask[mask == 0] = cv2.GC_PR_BGD
    mask[mask == 255] = cv2.GC_PR_FGD

    
    grabcutmask, bgdModel, fgdModel = cv2.grabCut(image,  mask, None,bgdModel,fgdModel,1,cv2.GC_INIT_WITH_MASK)

    grabcutmask[grabcutmask == cv2.GC_PR_FGD] = 255
    grabcutmask[grabcutmask == cv2.GC_PR_BGD] = 0


    return grabcutmask


def threshold(image, lg = np.array([ 30, 40, 40]), ug = np.array([ 86, 255,255])):
    print('Thresholding')
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lg, ug)

    return mask


