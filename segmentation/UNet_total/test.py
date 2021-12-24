import torch
from torch.utils.data.dataset import Subset
from loader import CustomDataset
from unet import Model
import cv2
import numpy as np
import os
from torchvision.transforms import transforms
import glob
if __name__ == '__main__':

    model = Model.load_from_checkpoint(r'version_28\checkpoints\epoch=116-step=9928.ckpt')
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
    transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((360, 640)),
                # transforms.Resize((640, 360)),
                transforms.ToTensor(),
                ])

    for i, path in enumerate( glob.glob(r'E:\Google Drive\Acads\research\Single-View-Plant-Modelling\Dataset\valdata/images/*.png')):
        d = cv2.imread(path)
        if os.path.basename(path).startswith('0'):
            continue
            
            d1 = d[:360, :640]
            d2 = d[202:562, :640]
            d3 = d[562:, :640]
            preds = []
            for data in [d1, d2, d3]:
                inp = transform(data).unsqueeze(0)
                output = model(inp).squeeze()
                out = output.detach().numpy()
                inp = np.transpose(inp.squeeze().detach().numpy(), (1, 2, 0))
                print(out.min(), out.max())
                out[out < 0.5] = 0        
                out[out > 0.5] = 1
                out = (out * 255).astype(np.uint8)
                preds.append(out)

            img = np.zeros(d.shape[:2])
            img[:360, :640] = preds[0]
            img[202:562, :640] = preds[1]
            img[562:, :640] = preds[2]
            
            # cv2.imshow('data', img)
            # if cv2.waitKey(1) == ord('q'):
            #     exit()
        else:
            inp = transform(d).unsqueeze(0)
            output = model(inp).squeeze()
            out = output.detach().numpy()
            inp = np.transpose(inp.squeeze().detach().numpy(), (1, 2, 0))
            out[out < 0.5] = 0        
            out[out > 0.5] = 1
            img = (out * 255).astype(np.uint8)

        cv2.imwrite(f'test/{os.path.basename(path)}', img)



    
