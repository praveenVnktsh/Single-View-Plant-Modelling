from utils import scaleAndShow
import cv2
import glob
import torch
import numpy as np


class Loader():

    def __init__(self, base):
        self.base = base
        self.i = 0
        self.stem = None
        self.image = None
        self.mask = None
        self.leaves = None


    def getNextImage(self, split = False):
        if self.i <0:
            return None

        self.image = cv2.imread(self.base + f'{self.i}_img.jpg')
        self.mask = cv2.imread(self.base + f'{self.i}_mask.jpg', 0)
        self.stem = cv2.imread(self.base + f'{self.i}_stem.jpg', 0)


        self.leaves = self.mask - self.stem
        self.i -= 1

        if self.stem is None or self.image is None or self.mask is None:
            return -1
        else:
            return 1
