import cv2
import numpy as np
import timeit
from skimage.morphology import skeletonize as sk
from numba import jit
from skimage import feature
from skimage import exposure
from skimage import feature
import scipy.ndimage as ndi


def scaleAndShow(im, name = 'outdoor', height = None, waitkey = 1):
    def callback(event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(x, y, im[y, x])
    
    cv2.namedWindow(name)
    cv2.setMouseCallback(name,callback)
    if height is not None:
        width = int(im.shape[1]*height/im.shape[0])
        im = cv2.resize(im, (width, height), interpolation= cv2.INTER_NEAREST)
    cv2.imshow(name, im)
    if cv2.waitKey(waitkey) == ord('q'):
        exit()


def getWindow(mask, point, winsize):
    hwins = int(winsize/2)
    window = mask[
        point[0] - hwins: point[0] + hwins + 1,
        point[1] - hwins: point[1] + hwins + 1
    ]
    return window

def sampleGrid(mask, step = 5, viz = True):
    h, w = mask.shape

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
        scaleAndShow(temp, 'a')
    y, x = np.where(indices == True)
    return x, y, indices

def enhanceContrast(image) : 
    lab= cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)

    limg = cv2.merge((cl,a,b))

    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final
    
def grabcut(image, mask):
    print('grabcut')
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    mask[mask == 0] = cv2.GC_PR_BGD
    mask[mask == 255] = cv2.GC_PR_FGD

    
    grabcutmask, bgdModel, fgdModel = cv2.grabCut(image,  mask, None,bgdModel,fgdModel,1,cv2.GC_INIT_WITH_MASK)

    grabcutmask[grabcutmask == cv2.GC_PR_FGD] = 255
    grabcutmask[grabcutmask == cv2.GC_PR_BGD] = 0


    return grabcutmask

def threshold(image, lg = np.array([ 30, 40, 40]), ug = np.array([ 86, 255,255])):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lg, ug)

    return mask
