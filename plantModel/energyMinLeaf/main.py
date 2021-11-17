from leafModel import Leaf
# from loader import Loader
from imgloader import Loader
from typing import List
from node import Node
from utils import scaleAndShow
import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import imageio
import time

loader = Loader(r'E:\Google Drive\Acads\Mitacs\dataset\Cam 202106\A1-1/*.jpg')

a = loader.getNextImage(split = True)

img, allmask = a



stemmask = loader.getStem()
stemdist = cv2.distanceTransform(255 - stemmask, cv2.DIST_L2, 3).astype(float) #- cv2.distanceTransform(stemmask, cv2.DIST_L2, 3).astype(float)
stemdist -= stemdist.min()

leafmask = loader.getLeaves()

leafdist =  cv2.distanceTransform(leafmask, cv2.DIST_L2, 3).astype(float)
leafdist -= leafdist.min()

img[:, :, 2][leafmask == 255] = 128
img[:, :, 1][stemmask == 255] = 128

leaf  = Leaf()

while True:
    t = img.copy()
    # t[:, :, 1][mask == 255] += 1
    t = leaf.isConverged(leafdist, stemdist, t)
    time.sleep(0.5)


# dist1 /= dist1.max()
# dist1 *= 255
# dist1 = dist1.astype(np.uint8)
# scaleAndShow(dist1, waitkey=0)
    
    

