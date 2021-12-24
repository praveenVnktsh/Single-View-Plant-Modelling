import cv2
import torch
import glob
import numpy as np

# for i in range(31, 95, 3):
i = 31
while True:
    try:
        im1 = cv2.imread(f'data/leaves/{i}.png')
        im2 = cv2.imread(f'data/leaves/{i+1}.png')
        im3 = cv2.imread(f'data/leaves/{i+2}.png')

        img = np.zeros((922, 691, 3), dtype = np.uint8)
        h = 30
        img[:400,:] = im1
        img[200:600,:] = im2
        img[500:922,:] = im3
        cv2.imwrite('fulldata/leaves/' + str(i).zfill(5) + '.png', img)
        cv2.imshow('img', img)
        if cv2.waitKey(0) == ord('q'):
            exit()
    except:
        i += 1
        if i == 95:
            exit()
        continue

    i += 3

    print(i)