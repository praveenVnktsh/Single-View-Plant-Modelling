import cv2
import numpy as np
import glob
from functions import *


drawing = False
erasing = False



def drawcallback(event,x,y,flags,param):
    global drawmask, drawing, tempshow, tempimg, erasing, drawsize
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

    tempshow = tempimg.copy()
    if erasing:
        color = 0
    else:
        color = 255
    cv2.circle(tempshow, (x, y), drawsize, (0, color, 255), thickness= -1)
    if drawing:
        cv2.circle(drawmask, (x, y), drawsize, (color), thickness=-1)

def showImg(im, name = 'mask', waitkey = 1, callback = None):
    cv2.namedWindow(name)
    if callback is not None:
        cv2.setMouseCallback(name,callback)
    cv2.imshow(name, im)
    if waitkey is not None:
        k = cv2.waitKey(waitkey)
        if k == ord('q'):
            exit()
        return k
    return -1



def grabcutCallback(event,x,y,flags,param):
    global drawing, erasing, mask, drawsize
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

    if erasing:
        color = 0
    else:
        color = 255
    if drawing:
        cv2.circle(mask, (x, y), drawsize, (color), thickness=-1)




def execute(img, i):
    global tempimg, drawmask, erasing, mask, drawsize, finalmask
    tempimg = img.copy()
    print('Draw the mask first')
    erasing = False
    drawsize = 10
    while True:
        finalmask = drawmask.copy()
        finalmask[mask == 0] = 0
        tempimg[:, :, 1][finalmask != 0] = 255
        # tempimg[:, :, 1][finalmask == 0] = img[:, :, 1][finalmask == 0]
        
        k = showImg(tempshow, callback= drawcallback)
        if k == ord('e'):
            erasing = not erasing
        elif k == ord('d'):
            erasing = False
            break
        elif k == ord('s'):
            return
        elif k == ord('r'):
            if drawsize > 5:
                drawsize = 3
            else:
                drawsize = 10
    cv2.imwrite(f'annotated/image/img_{i}.png', img)
    cv2.imwrite(f'annotated/plantmask/mask_{i}.png', mask)
    cv2.imwrite(f'annotated/stemmask/mask_{i}.png', finalmask)


for i in range(34):
    path = f'fulldata/image/img_{i}.png'
    ima = cv2.imread(path)
    if ima == None:
        continue
    h, w, _ = ima.shape

    drawsize = 30

    im = ima
    tempimg = im.copy()
    tempshow = im.copy()
    mask = cv2.imread(f'fulldata/plantmask/mask_{i}.png', 0)
    drawmask = cv2.imread(f'fulldata/stemmask/mask_{i}.png', 0)
    finalmask = drawmask.copy()
    print(mask.shape, finalmask.shape, im.shape)
    execute(im, i )

