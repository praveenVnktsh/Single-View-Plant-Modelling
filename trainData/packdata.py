import cv2
import torch



dic = {'images': [], 'leaves': [], 'stems': [], 'length' : 0}

base = r'E:\Google Drive\Acads\research\Single-View-Plant-Modelling\trainData\data'
for i in range(94):
    dic['images'].append(cv2.imread(f'{base}/image/{i}.png'))
    dic['leaves'].append(cv2.imread(f'{base}/leaves/{i}.png', 0))
    dic['stems'].append(cv2.imread(f'{base}/stem/{i}.png', 0))
    dic['length'] += 1
    


torch.save(dic, r'E:\Google Drive\Acads\research\Single-View-Plant-Modelling\trainData/dict_tripartite.pt')