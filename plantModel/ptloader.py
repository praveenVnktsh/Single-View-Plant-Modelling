from utils import scaleAndShow
from functions import enhanceContrast, grabcut, hog, sampleGrid, threshold
import cv2
import glob
import torch
import numpy as np
class Loader():

    def __init__(self, path):
        self.paths = glob.glob(path)
        self.i = 0
        self.model = torch.load('svm.pt')
        self.images, self.stems = torch.load('fullImgDataset.pt')

    def getStem(self, tmask, image):
        
        return cv2.copyMakeBorder(self.stems[self.i - 1], 30, 30, 30, 30, cv2.BORDER_REFLECT)
        
    def getNextImage(self, split = False):
        if self.i == len(self.paths):
            return None


        im = self.images[self.i]
        mask=  self.stems[self.i]
        im = cv2.copyMakeBorder(im, 30, 30, 30, 30, cv2.BORDER_REFLECT)
        mask = cv2.copyMakeBorder(mask, 30, 30, 30, 30, cv2.BORDER_REFLECT)
        self.i += 1
        return im, mask