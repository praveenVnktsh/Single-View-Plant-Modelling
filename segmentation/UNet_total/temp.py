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
def randomizeDomain(image, mask):

    # image = image.astype(float)/255.0
    # mask = mask.astype(float)/255.0

    # image, mask = dist(image, mask)
    # image, mask = vig(image, mask)
    image, mask = noise(image, mask)
    image, mask = contrast(image, mask)
    image, mask = bri(image, mask)
    image, mask = dist(image, mask)
    image, mask = persp(image, mask)
    image, mask = flip(image, mask)

        
    return image, mask


import torch
import cv2
from functions import *

images, masks  = torch.load('combinedDatasetIndoorOutdoor.pt')
# for i in range(len(images)):
print(len(images))

i = 0
fullmasks = []
indices = [14,
20,
21,
41,
42,
50,
51,
52,
84]

finals = [[], [], []]
while True:
    while i in indices:
        i += 1
    img = images[i]
    mask = masks[i]
    # img, mask = randomizeDomain(img, mask)
    
    mask[mask >= 128] = 255
    mask[mask < 128] = 0
    # img = enhanceContrast(img)
    fullmask = threshold(img, lg= np.array([25, 30, 15]), ug = np.array([86, 255, 255]))

    fullmask = grabcut(img, fullmask.copy())
    fullmask = grabcut(img, fullmask.copy())
    fullmask = grabcut(img, fullmask.copy())

    temp = img.copy()
    temp[:, :, 1][fullmask == 255] = 255
    temp[:, :, 0][mask == 255] = 255
    
    
    
    



    cv2.imshow('s', temp)
    
    key = cv2.waitKey(1)
    if key == ord('1'):
        images.pop(i)
        masks.pop(i)
    else:
        i += 1

    finals[0].append(img.copy())
    finals[1].append(mask.copy())
    finals[2].append(fullmask.copy())


    if key == ord('q') or len(images) == i:
        break

print(len(images))

torch.save(finals, 'finalDatasetWithFullmask.pt')

    
