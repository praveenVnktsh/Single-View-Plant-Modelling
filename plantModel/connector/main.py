from functions import removeBranches, skeletonize
from jointModel import Model
from leafModel import Leaf
# from loader import Loader
from maskLoader import Loader
from typing import List
from utils import scaleAndShow
import cv2
import numpy as np

loader : Loader = Loader(r'E:\Google Drive\Acads\research\Single-View-Plant-Modelling\segmentation\UNet_Patchwise\toModel\old/')

loader.getNextImage()
render =  False


leafmask = loader.leaves

def getDistMask(mask, invert = False):
    kernel = np.ones((3,3),np.uint8)
    
    mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel, iterations = 2)

    mask[mask > 120] = 255
    mask[mask < 120] = 0
    if invert:
        leafdist =  cv2.distanceTransform(255 - mask, cv2.DIST_L2, 3).astype(float)
    else:
        leafdist =  cv2.distanceTransform(mask, cv2.DIST_L2, 3).astype(float)

    leafdist -= leafdist.min()
    return leafdist, mask



stemmask = loader.stem
netStemDist, stemmask = getDistMask(stemmask, invert = True)


contours, hierarchy = cv2.findContours(stemmask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

maxArea = 0
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > maxArea:
        maxArea = area
        maxCnt = cnt


img = loader.image
leaves : List[Leaf] = []


for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 0.01 * maxArea:
        continue
    tempimg = np.zeros_like(leafmask)
    cv2.drawContours(tempimg, [cnt], -1, 255, -1)
    if area > 0.3 * maxArea:
        t = skeletonize(tempimg)
        stemmask = removeBranches(t, stemmask)

if render:
    scaleAndShow(stemmask, waitkey=1)
contours, hierarchy = cv2.findContours(stemmask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

maxArea = 0
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > maxArea:
        maxArea = area
        maxCnt = cnt


vizimg = img.copy()
vizimg[:, :, 1][stemmask == 255] = 255
print('Fitting individual elastic bands')
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 0.01 * maxArea:
        continue
    tempimg = np.zeros_like(leafmask)
    cv2.drawContours(tempimg, [cnt], -1, 255, -1)
    (center, axes, angle) = cv2.fitEllipse(cnt)
    if angle > 90:
        angle = angle - 90
    else:
        angle = angle + 90

    majorAxis = max(axes)
    minorAxis = min(axes)
    slack = 10
    center = list(center)
    top = center.copy()
    top[0] -= np.cos(np.deg2rad(angle)) * (majorAxis + 10)
    top[1] -= np.sin(np.deg2rad(angle)) * (majorAxis + 10)

    bottom = center.copy()
    bottom[0] += np.cos(np.deg2rad(angle)) * (majorAxis + 10)
    bottom[1] += np.sin(np.deg2rad(angle)) * (majorAxis + 10)
    leaf = Leaf(
        centroid = top ,
        base = bottom, 
        slack = int(slack * (area/maxArea))
    )
    stemDist, tempMask = getDistMask(tempimg, invert = True)
    
    vizimg = leaf.isConverged(stemDist, stemDist, vizimg, render = render)

    leaves.append(leaf)

print('Starting attracting phase...')
for leaf in leaves:
    if leaf.stem[-1].vector[1] < 400:

        leaf.attract(leaves, vizimg)



if render:
    scaleAndShow(vizimg, 'leaf', waitkey= 1)
cv2.imwrite(f'outputs/{loader.i}_magnets.png', vizimg)

model = Model(leaves)
vizimg = img.copy()
print('Fitting joint model')
vizimg = model.converge(vizimg, netStemDist, render = render)
cv2.imwrite(f'outputs/{loader.i}_output.png', vizimg)

if render:  
    scaleAndShow(vizimg, 'stemDist', waitkey = 0)




    

