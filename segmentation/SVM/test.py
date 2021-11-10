import torch
from functions import *

def predict(image):
    model = torch.load('svm.pt')

    mask = threshold(image, lg= np.array([25, 40, 40]), ug = np.array([86, 255, 255]))
    mask = grabcut(image, mask)
    hWinSize = 16
    mask = cv2.copyMakeBorder(mask, hWinSize, hWinSize, hWinSize, hWinSize, cv2.BORDER_CONSTANT)

    x, y, indices = sampleGrid(mask, step = 3, viz = False)
    mask = np.stack((mask, mask, mask), axis = 2)
    plantSeg = ((mask.astype(float)/255) * image).astype(np.uint8)

    leaves = plantSeg.copy()
    stems = np.zeros_like(leaves)
    

    classWindow = 10
    for point in list(zip(y, x)):
        yy, xx = point
        window = plantSeg[yy - hWinSize : yy + hWinSize, xx - hWinSize : xx + hWinSize]
        H = hog(window)
        label = model.predict(H.reshape(1, 324))
        if label == 1:
            stems[yy - classWindow : yy + classWindow, xx - classWindow : xx + classWindow] = 255
            leaves[yy - classWindow : yy + classWindow, xx - classWindow : xx + classWindow] = 0

    stems = stems & mask
    leaves = leaves & mask
    return stems, leaves

    

