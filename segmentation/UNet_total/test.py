import torch
from torch.utils.data.dataset import random_split
from loader import CustomDataset
from unet import Model
import cv2
import numpy as np
import glob
if __name__ == '__main__':

    model = Model.load_from_checkpoint(r'version_15\checkpoints\epoch=423-step=13567.ckpt')
    # dataset =  torch.load('fullImgDataset.pt')
    # images = dataset[0]
    # masks = dataset[1]
    dataset = []

    # for path in glob.glob(r'E:\Google Drive\Acads\Mitacs\dataset\images\outdoor/*.jpg'):
    #     d = cv2.imread(path)
    #     d = cv2.resize(d, (0, 0), fx = 0.1, fy = 0.1)
    #     # d =  cv2.cvtColor(d, cv2.COLOR_BGR2RGB)
    #     d = d[:360, :640]
    #     dataset.append(d)

    for i, path in enumerate( glob.glob(r'E:\Google Drive\Acads\Mitacs\dataset\images\indoor/*.jpg')):
        d = cv2.imread(path)
        d = cv2.resize(d, (0, 0), fx = 0.15, fy = 0.15)
        # d = d[:360, :640]
        n = np.zeros_like(d)
        d = cv2.resize(d, (0, 0), fx = 0.4, fy = 0.4)
        n[:d.shape[0], :d.shape[1]] = d
        dataset.append(n)
        if i == 4:
            break


    images = dataset
    masks = dataset
    dataset = list(zip(images, masks, masks))
    n_val = 0#int(len(dataset) * 0.1)
    n_train = len(dataset)
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    data = CustomDataset(train_ds, randomize = False)
    for i in range(len(data)):
        inp = data[i][0].unsqueeze(0)
        output = model(inp).squeeze()
        out = output.detach().numpy()
        inp = np.transpose(inp.squeeze().detach().numpy(), (1, 2, 0))
        out = (out * 255).astype(np.uint8)
        inp = (inp * 255).astype(np.uint8)
        # d = np.vstack((inp, cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)))
        d = inp.copy()
        indices = out != 0
        d[:, :, 0][indices]  = out[indices]
        
        cv2.imshow('data', d)
        cv2.imwrite(f'test/{i}.png', d)
        if cv2.waitKey(0) == ord('q'):
            exit()
    