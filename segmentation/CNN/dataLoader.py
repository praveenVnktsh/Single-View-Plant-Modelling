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



class CustomDataset(Dataset):

    def __init__(self, loc = 'dataset.pt'):
        self.inp, self.outp = torch.load(loc)
        self.length = len(self.inp)
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):

        inp = self.inp[index]
        outp = self.outp[index]
        inp = self.transforms(inp)

        return {
            "input": inp,
            "target": outp,
        } 

    def __len__(self):
        return self.length


class LitCustomData(pl.LightningDataModule):

    def setup(self, stage = ''):
        self.cpu = 0
        self.pin = True
        self.stage =  stage
        self.trainBatchSize=  60
        self.valBatchSize = 60
        print('Loading dataset')
        

    def train_dataloader(self):
        dataset = CustomDataset(torch.load(self.stage + 'traindataset.pt'))
        return DataLoader(dataset, batch_size=self.trainBatchSize,
                          num_workers=self.cpu, pin_memory=self.pin)

    def val_dataloader(self):
        dataset = CustomDataset(torch.load(self.stage + 'valdataset.pt'))
        return DataLoader(dataset, batch_size=self.valBatchSize,
                          num_workers=self.cpu, pin_memory=self.pin)