import cv2
import numpy as np
import torch
from glob import glob
def scaleAndShow(im, name = 'outdoor', height = None, waitkey = 1):
    def callback(event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(x, y, im[y, x])
    
    cv2.namedWindow(name)
    cv2.setMouseCallback(name,callback)
    if height is not None:
        width = int(im.shape[1]*height/im.shape[0])
        im = cv2.resize(im, (width, height), interpolation= cv2.INTER_NEAREST)
    cv2.imshow(name, im)
    k = cv2.waitKey(waitkey)
    if k == ord('q'):
        exit()
    return k



k = 0
buffer = [[], []]
for i in range(0, 86):


    im = cv2.imread(f'annotated/image/img_{i}.png')
    if im is None:
        continue

    mask = cv2.imread(f'annotated/mask/mask_{i}.png', 0)
    if mask is None:
        continue

    if i > 60:
        im1 = im[0 : 400, :]
        im2 = im[200 : 600, :]
        im3 = im[500 : , :]
        mask1 = mask[0 : 400, :]
        mask2 = mask[200 : 600, :]
        mask3 = mask[500 : , :]
        buffer[0] += [im1, im2, im3]
        buffer[1] += [mask1, mask2, mask3]
        scaleAndShow(im1, 'outdoor', height = 400)
        scaleAndShow(mask1, 'o2utdoor', height = 400)
    else:
        buffer[0].append(im)
        buffer[1].append(mask)

    
    print(i)
    
torch.save(buffer, 'combinedDatasetIndoorOutdoor.pt')