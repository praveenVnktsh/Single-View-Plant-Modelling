import cv2
import torch


images, stems, mask  = torch.load(r'E:\Google Drive\Acads\research\Single-View-Plant-Modelling\trainData/stem_leaves_separate.pt')

dic = {'images': [], 'leaves': [], 'stems': []}

base = r'E:\Google Drive\Acads\research\Single-View-Plant-Modelling\trainData\data'
for i in range(len(images)):
    img = images[i].copy()
    img[:, :, 0][mask[i] == 255] = 255
    img[:, :, 1][stems[i] == 255] = 255

    leaves = mask[i] - stems[i]

    cv2.imshow('a', img)
    k = cv2.waitKey(1)

    if k != ord('d'):
        dic['images'].append(images[i])
        dic['stems'].append(stems[i])
        dic['leaves'].append(leaves)

        ret = cv2.imwrite(f'{base}/image/{i}.png', images[i])
        cv2.imwrite(f'{base}/stem/{i}.png', stems[i])
        cv2.imwrite(f'{base}/leaves/{i}.png', leaves)
        
    if k == ord('q'):
        exit()
    


# torch.save(r'E:\Google Drive\Acads\research\Single-View-Plant-Modelling\trainData/dict_tripartite.pt')