import numpy as np
from transforms import *

dist = LensDistortion()
vig = Vignetting()
noise = GaussianNoise(std= 20)
contrast = Contrast((0.5, 2))
bri = Brightness((-25, 25))
cut = Cutout([0.01, 0.01], [0.1, 0.1])
persp = Perspective(max_rotation=(10, 10, 45))
flip = Flip()
backgroundRandom = BackgroundRandom('bgs.pt')


def randomizeDomain(image, mask, fullmask):

    # print(mask.dtype, fullmask.dtype, image.dtype)
    image = cv2.resize(image, (640, 360), cv2.INTER_NEAREST)
    mask = cv2.resize(mask, (640, 360), cv2.INTER_NEAREST)
    fullmask = cv2.resize(fullmask, (640, 360), cv2.INTER_NEAREST)
    cv2.imshow('imga', image)
    image, mask, fullmask = backgroundRandom(image, mask, fullmask)

    image, mask = noise(image, mask)
    image, mask = contrast(image, mask)
    image, mask = bri(image, mask)
    image, mask = dist(image, mask)
    image, mask = persp(image, mask)
    image, mask = flip(image, mask)

        
    return image, mask, fullmask


import torch
import cv2
from functions import *

images, masks, fullmasks  = torch.load('finalDatasetWithFullmask.pt')
print(len(images), len(masks), len(fullmasks))
i = 0
while True:
    i += 1
    print(i)
    img = images[i]
    mask = masks[i]
    fullmask = fullmasks[i]

    img, mask, fullmask = randomizeDomain(img, mask, fullmask)
    
    temp = img.copy()
    temp[:, :, 0][mask == 255] = 255
    
    cv2.imshow('s', temp)
    
    key = cv2.waitKey(0)

    if key == ord('q') or len(images) == i:
        break

print(len(images))


    
