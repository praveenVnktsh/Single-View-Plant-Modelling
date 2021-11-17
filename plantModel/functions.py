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

def removeBranches(skeleton):
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
        cv2.circle(skeleton, p, 3, 0, -1)
    # skeleton[branches == 255] = 0

    return skeleton, list(zip(x, y))



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
        scaleAndShow(temp, 'ab', height = 600)
    y, x = np.where(indices == True)
    return x, y, indices

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


def sobel(img):
    xKernel = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    yKernel = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    sobelled = np.zeros((img.shape[0]-2, img.shape[1]-2, 3), dtype="uint8")
    @jit(nopython = True)
    def help(img, sobelled, xKernel, yKernel):
        for y in range(1, img.shape[0]-1):
            for x in range(1, img.shape[1]-1):
                gx = np.sum(np.multiply(img[y-1:y+2, x-1:x+2], xKernel))
                gy = np.sum(np.multiply(img[y-1:y+2, x-1:x+2], yKernel))
                g = abs(gx) + abs(gy) #math.sqrt(gx ** 2 + gy ** 2) (Slower)
                g = g if g > 0 and g < 255 else (0 if g < 0 else 255)
                sobelled[y-1][x-2] = g

        
        return sobelled

    sobelled = help(img, sobelled, xKernel, yKernel)
    sobelled = abs(sobelled)
    sobelled -= sobelled.min()
    return sobelled



def cannyfilt(image):
    canny = cv2.Canny(image, 800, 1000)
    return canny

def threshold(image, lg = np.array([ 30, 40, 40]), ug = np.array([ 86, 255,255])):
    print('Thresholding')
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lg, ug)

    return mask


def skeletonize(mask):
    mask[mask == 255] = 1
    skeleton = sk(mask)

    return skeleton