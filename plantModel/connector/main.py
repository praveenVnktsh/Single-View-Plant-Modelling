from leafModel import Leaf
# from loader import Loader
from maskLoader import Loader
from typing import List
from node import Node
from utils import scaleAndShow
import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import imageio
import time

loader : Loader = Loader(r'E:\Google Drive\Acads\research\Single-View-Plant-Modelling\segmentation\UNet_Patchwise\toModel\old/')

loader.getNextImage()

stemmask = loader.stem
kernel = np.ones((3,3),np.uint8)
stemmask = cv2.morphologyEx(stemmask,cv2.MORPH_OPEN,kernel, iterations = 2)
# stemmask = cv2.dilate(stemmask,kernel,iterations=3)

stemmask[stemmask > 200] = 255
stemmask[stemmask < 200] = 0
stemdist = cv2.distanceTransform(255 - stemmask, cv2.DIST_L2, 3).astype(float) #- cv2.distanceTransform(stemmask, cv2.DIST_L2, 3).astype(float)
stemdist -= stemdist.min()




leafmask = loader.leaves
kernel = np.ones((3,3),np.uint8)
leafmask = cv2.morphologyEx(leafmask,cv2.MORPH_OPEN,kernel, iterations = 2)
leafmask = cv2.dilate(leafmask,kernel,iterations=3)

leafmask[leafmask > 200] = 255
leafmask[leafmask < 200] = 0
leafdist =  cv2.distanceTransform(leafmask, cv2.DIST_L2, 3).astype(float)
leafdist -= leafdist.min()


contours, hierarchy = cv2.findContours(leafmask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
sure_fg = np.zeros_like(leafmask)

maxArea = 0
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > maxArea:
        maxArea = area
        maxCnt = cnt

for cnt in contours:
    if cv2.contourArea(cnt) < 0.01 * maxArea:
        continue
    tempimg = np.zeros_like(leafmask)
    cv2.drawContours(tempimg, [cnt], -1, 255, -1)
    dist_transform = cv2.distanceTransform(tempimg, cv2.DIST_L2, 5)
    ret, tempimg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
    sure_fg += tempimg.astype(np.uint8)



img = loader.image
# 

# unknown = cv2.subtract(leafmask, sure_fg)
# ret, markers = cv2.connectedComponents(sure_fg)
# markers = markers+1
# markers[unknown==255] = 0
# markers = cv2.watershed(cv2.cvtColor(leafmask* img[:, :, 1], cv2.COLOR_GRAY2BGR) ,markers)
# img[markers == -1] = [255,0,0]

# scaleAndShow(markers/markers.max(), waitkey=0)
# scaleAndShow(img, waitkey=0)


# leaves = [
    
#     Leaf(centroid = (64,343), base = (300, 600)), 
#     Leaf(centroid = (163,66), base = (300, 600)), 
# ]
leaves = []
for i in range(10):
    # for j in range(5):
    leaf = Leaf(centroid = (10 + 50*i,200), base = (100, 600))
    leaves.append(leaf)
        
# for i in range(5):
#     for j in range(5):
#         leaf = Leaf(centroid = (10 + 100*i,10 + 50 * j), base = (100, 600))
#         leaves.append(leaf)
        


i = 0
t = img.copy()

while True:
    leaf = leaves[i]
    # t[:, :, 1][leafmask == 255] = 255
    # t[:, :, 2][stemmask == 255] = 255
    t = leaf.isConverged(leafdist, stemdist, t)
    i += 1
    i %= len(leaves)



    

