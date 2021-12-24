import cv2
import glob
import os
import numpy as np
base = r'E:\Google Drive\Acads\research\Single-View-Plant-Modelling\Dataset\valdata/stem/'
# modelname = 'Nearest Neighbors_90_30000'
# modelname = f'RBF SVM_91_20000'
ious = []

for path in glob.glob(base + '*.png'):
    if not os.path.basename(path).startswith('0'):
        continue
    pred = cv2.imread(f'test/{os.path.basename(path)}', 0)
    gt = cv2.imread(path, 0)
    
    if cv2.waitKey(0) == ord('q'):
        exit()
    pred[pred > 128] = 255
    pred[pred < 128] = 0
    iou =  cv2.bitwise_and(pred, gt).sum() / (pred.sum() + gt.sum() -  cv2.bitwise_and(pred, gt).sum())

    ious.append(iou)


print(np.average(ious))