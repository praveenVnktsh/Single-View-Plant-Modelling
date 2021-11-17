import glob
import cv2
import numpy as np

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

def threshold(image, lg = np.array([ 30, 40, 40]), ug = np.array([ 86, 255,255])):
    print('Thresholding')
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lg, ug)

    return mask


def enhanceContrast(image) : 
    print('Enhancing contrast')
    lab= cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)

    limg = cv2.merge((cl,a,b))

    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final
    
def grabcut(image, mask):
    print('Grabcut')
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    mask[mask == 0] = cv2.GC_PR_BGD
    mask[mask == 255] = cv2.GC_PR_FGD

    
    grabcutmask, bgdModel, fgdModel = cv2.grabCut(image,  mask, None,bgdModel,fgdModel,1,cv2.GC_INIT_WITH_MASK)

    grabcutmask[grabcutmask == cv2.GC_PR_FGD] = 255
    grabcutmask[grabcutmask == cv2.GC_PR_BGD] = 0


    return grabcutmask
i = 0
for path in glob.glob(r'E:\Google Drive\Acads\Mitacs\dataset\images\indoor/*.jpg'):
    img = cv2.imread(path)
    img = cv2.resize(img, (0,0), fx=0.1, fy=0.1)
    mask = threshold(img)
    mask = grabcut(img, mask)
    scaleAndShow(mask)
    i += 1
    cv2.imwrite(f'masks/{i}.jpg', mask)
    cv2.imwrite(f'masks/img_{i}.jpg', img)
    