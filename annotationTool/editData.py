import cv2


drawing = False
erasing = True
image = None
drawSize = 25
def drawcallback(event,x,y,flags,param):
    global drawing, erasing, image, drawSize

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

    if erasing:
        color = 0
    else:
        color = 255

    if drawing:
        cv2.circle(image, (x, y), drawSize, (color, color, color), thickness=-1)

    cv2.imshow('a', image)
    


base = r'E:\Google Drive\Acads\research\Single-View-Plant-Modelling\trainData\data\leaves'

import glob
cv2.namedWindow('a')
cv2.setMouseCallback('a', drawcallback)
for path in glob.glob(base + '/*.png'):

    image = cv2.imread(path, 0)
    k = 1
    cv2.imshow('a', image)
    while k != ord('d'):
        k = cv2.waitKey(1)

        if k == ord('q'):
            exit()
        elif k == ord('e'):
            erasing = not erasing
    cv2.imwrite(path, image)