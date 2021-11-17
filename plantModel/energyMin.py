# from ptloader import Loader
from imgLoader import Loader
from typing import List
from rope import Rope, getRope
from node import Node
from utils import scaleAndShow
import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import imageio

import time
# loader = Loader('masks/img_*.jpg')

# img, mask = loader.getNextImage()

loader = Loader(r'E:\Google Drive\Acads\Mitacs\dataset\images\outdoor/*.jpg')

a = loader.getNextImage(split = False)
img = a[0]
allmask = a[1]

mask = loader.getStem(allmask, img)
dist1 = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 3)
dist = dist1.copy()
h, w = dist.shape

temp = np.stack((dist, dist, dist), axis = 2)
temp -= temp.min()
temp /= np.average(temp)
temp *= 255
temp[temp > 255] = 255
# temp[temp < 1] = 255
scaleAndShow(allmask.astype(np.uint8), 'allmask', waitkey=1)
temp = img.copy()
temp[:, :, 0][mask == 255] = 255
scaleAndShow(temp.astype(np.uint8), 'mask', waitkey=0)

temp = temp.astype(np.uint8)


markers = np.zeros_like(dist, dtype = np.uint8)
tempdist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
markers[(tempdist) > tempdist.max() * 0.5] = 255


leaves, hierarchy = cv2.findContours(markers, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


ropes : List[Rope]= []
# startpoint = (136, 307)
# startpoint = (810, 705)
# endpts = [
#     [300, 120],
#     [651,300],
#     [615, 33],
#     [900, 33],
#     [1064, 100]
# ]

startpoint = (372, 427)
endpts = [
    [285, 24],
    [545,22],
]

for i in endpts:
    ropes.append(getRope(i ,startpoint, mask))
    # ropes.append(getRope((820, 151), startpoint, mask))
while True:
    t = temp.copy()
    # t = mask.copy()
    
    t[:, :, 1][mask == 255] += 1
    for j, rope in enumerate(ropes):
        # if j == 0:
        t = rope.isConverged(dist, t)
        for k in range(10):
            rope.removeNode()
        time.sleep(3)


    
    

