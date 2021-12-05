import cv2
from torch import functional
from torch.utils.data.dataset import Dataset
import torch
from torchvision.transforms import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torch.optim.rmsprop import RMSprop
import numpy as np
import pytorch_lightning as pl
from torchvision.transforms.functional import hflip
import albumentations as A
import glob
import random
import albumentations.pytorch

class CustomDataset(Dataset):

    def __init__(self, paths):
        self.paths = paths
        self.length = len(self.paths)
        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((100, 100)),
            transforms.ToTensor(),
        ])
        

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):

        inp = cv2.imread(self.paths[index][0])
        
        
        if 'leaf' in self.paths[index][1]:
            target = np.zeros(inp.shape[:2], dtype = np.uint8)
        else:
            target = cv2.imread(self.paths[index][1], 0)
            
        target = self.transforms(target).squeeze()
        inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
        inp = self.transforms(inp).float()

        return {
            "input": inp,
            "target": target,
        } 

    def __len__(self):
        return self.length


class LitCustomData(pl.LightningDataModule):

    def start(self, path):
        self.cpu = 0
        self.pin = True
        self.trainBatchSize=  60
        self.valBatchSize = 60


        self.bgPaths = sorted(glob.glob(path + 'bg/images/*.png'))
        self.bgMaskPaths = sorted(glob.glob(path + 'bg/mask/*.png'))

        self.stempaths = sorted(glob.glob(path + 'stem/images/*.png'))
        self.stemMaskpaths = sorted(glob.glob(path + 'stem/mask/*.png'))

        self.leafPaths = sorted(glob.glob(path + 'leaf/images/*.png'))
        self.leafMaskPaths = sorted(glob.glob(path + 'leaf/mask/*.png'))

        self.bgPaths =   zip(self.bgPaths  , self.bgMaskPaths)
        self.stempaths = zip(self.stempaths, self.stemMaskpaths)
        self.leafPaths = zip(self.leafPaths, self.leafMaskPaths)

        data = list(self.bgPaths) + list(self.stempaths)+ list(self.leafPaths)

        random.shuffle(data)

        train_size = int(0.8 * len(data))
        self.train_dataset, self.test_dataset = data[:train_size], data[train_size:]


    def train_dataloader(self):
        dataset = CustomDataset(self.train_dataset)
        return DataLoader(dataset, batch_size=self.trainBatchSize,
                          num_workers=self.cpu, pin_memory=self.pin)

    def val_dataloader(self):
        dataset = CustomDataset(self.test_dataset)
        return DataLoader(dataset, batch_size=self.valBatchSize,
                          num_workers=self.cpu, pin_memory=self.pin)

if __name__ == '__main__':
    path = 'e:/Google Drive/Acads/research/Single-View-Plant-Modelling/data/'

    data = LitCustomData()