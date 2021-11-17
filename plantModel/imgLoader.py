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


    def getStem(self):
        hWinSize = 16
        tmask = self.mask
        image = self.image
        h, w = tmask.shape
        mask = cv2.copyMakeBorder(tmask, hWinSize, hWinSize, hWinSize, hWinSize, cv2.BORDER_CONSTANT)
        prednodes = np.zeros_like(mask)
        x, y, indices = sampleGrid(mask, step = 5, viz = False)
        plantseg = cv2.copyMakeBorder(image, hWinSize, hWinSize, hWinSize, hWinSize, cv2.BORDER_CONSTANT)
        plantseg[np.stack((mask, mask, mask), axis = 2) != 255] = 0

        stepsize = 4
        for point in list(zip(y, x)):
            yy, xx = point
            window = plantseg[yy - hWinSize : yy + hWinSize, xx - hWinSize : xx + hWinSize]
            H = hog(window)
            label = self.model.predict(H.reshape(1, 324))
            if label == 1:
                prednodes[yy - stepsize : yy + stepsize, xx - stepsize : xx + stepsize] = 255

        h, w = prednodes.shape

        prednodes = prednodes & mask
        ret = prednodes[hWinSize : h - hWinSize , hWinSize : w - hWinSize ]

        return ret

    def getNextImage(self, split = False):
        if self.i == len(self.paths):
            return None


        im = cv2.imread(self.paths[self.i])
        im = cv2.resize(im, (0, 0), fx = 0.15, fy = 0.15)
        # if split:
        #     im = cv2.resize(im, (0, 0), fx = 0.4, fy = 0.4)
        # resize if needed
        im = enhanceContrast(im)
        mask = threshold(im, lg= np.array([25, 40, 20]), ug = np.array([86, 255, 255]))
        mask = grabcut(im, mask)

        if split:
            h, w, _ = im.shape

            limg = im[:, :w//2]
            rimg = im[:, w//2:]

            maskl = mask[:, :w//2]
            maskr = mask[:, w//2:]
            # scaleAndShow(maskl, waitkey=0)
            return (limg, rimg), (maskl, maskr)
        self.i += 1
        self.image = im
        self.mask = mask
        return im, mask