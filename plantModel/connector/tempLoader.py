from utils import scaleAndShow
from functions import enhanceContrast, grabcut, hog, sampleGrid, threshold
import cv2
import glob
import torch
import numpy as np
class Loader():

    def getLeaves(self,a , b):
        im = np.zeros((600, 600), dtype=np.uint8)

        # cv2.circle(im, (300, 300), 100, (255, 255, 255), -1)
        im[250 : 350, 200:400] = 255
        return im

    def getNextImage(self, split = False):  
        a = np.zeros((600, 600, 3), dtype=np.uint8)
        a[250 : 350,  200:400] = 255
        return a, a.copy()[:, :, 0]

    def getStem(self, a):
        b = np.zeros_like(a)
        b[350 : 500 , 295 : 300] = 255
        return b