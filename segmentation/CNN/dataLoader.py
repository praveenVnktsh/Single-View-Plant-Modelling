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
        self.transforms = A.Compose(
            [
                A.Resize(30, 30),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
                A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                albumentations.pytorch.transforms.ToTensorV2(),
            ]
        )

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):

        inp = cv2.imread(self.paths[index][0])
        inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
        inp = self.transforms(image = inp)['image'].float()

        target = torch.tensor(self.paths[index][1]).float()

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
        bgs = path + 'bg/images/*.png'
        stems = path + 'stem/images/*.png'
        leaves = path + 'leaf/images/*.png'


        self.bgPaths = sorted(glob.glob(bgs))
        self.stempaths = sorted(glob.glob(stems))
        self.leafPaths = sorted(glob.glob(leaves))

        self.bgPaths =   zip(self.bgPaths  , [[1, 0, 0] for i in range(len(self.bgPaths))])
        self.stempaths = zip(self.stempaths, [[0, 1, 0] for i in range(len(self.stempaths))])
        self.leafPaths = zip(self.leafPaths, [[0, 0, 1] for i in range(len(self.leafPaths))])

        data = list(self.bgPaths) + list(self.stempaths) + list(self.leafPaths)
        random.shuffle(data)

        train_size = int(0.8 * len(data))
        self.train_dataset, self.test_dataset = data[:train_size], data[train_size:]
        print(train_size)
        print(len(self.train_dataset), len(self.test_dataset))


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