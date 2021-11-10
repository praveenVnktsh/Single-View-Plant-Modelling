from model import UNet
import os
import logging
from argparse import ArgumentParser
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from loader import CustomDataset
import pytorch_lightning as pl



class Model(pl.LightningModule):
    def __init__(self, hparams):
        super(Model, self).__init__()
        self.hparams = hparams
        self.n_classes = 1
        self.net = UNet(n_channels = 3, n_classes = self.n_classes, bilinear= True)
        self.data = None
        self.valdatalength = -1
        

    def forward(self, x):
        out = self.net(x)
        return out

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y) if self.n_classes > 1 else \
            F.binary_cross_entropy(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y) if self.n_classes > 1 else \
            F.binary_cross_entropy(y_hat, y)

        if self.valdatalength == -1:
            y = y[:10]
            x = x[:10]
            yhat = y_hat[:10]
            y = torch.stack((y, y, y), dim=1).float().squeeze()
            
            yhat = torch.stack((yhat,yhat,yhat), dim=1).float().squeeze()
            self.data = torch.cat((x.squeeze(), y.squeeze(), yhat), dim = 1)
            self.valdatalength = 1
        elif self.valdatalength == 7:
            self.logger.experiment.add_images('validateImagesIndex0', self.data.unsqueeze(0), self.current_epoch)
            self.data = None
            self.valdatalength = - 1
        else:
            y = y[:10]
            x = x[:10]
            yhat = y_hat[:10]
            y = torch.stack((y, y, y), dim=1).float().squeeze()
            
            yhat = torch.stack((yhat,yhat,yhat), dim=1).float().squeeze()
            
            data = torch.cat((x.squeeze(), y.squeeze(), yhat), dim = 1)
            self.data = torch.cat((self.data, data), dim = 2)
            self.valdatalength += 1

        print(self.valdatalength)
        
        return {'val_loss': loss}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.RMSprop(self.parameters(), lr=0.1, weight_decay=1e-8)

    def __dataloader(self):
        dataset = torch.load('finalDatasetWithFullmask.pt')
        # images = dataset['image']
        # masks = dataset['mask']``
        images = dataset[0]
        masks = dataset[1]
        fullmasks = dataset[2]
        dataset = list(zip(images, masks, fullmasks))

        n_val = int(len(dataset) * 0.1)
        n_train = len(dataset) - n_val
        train_ds, val_ds = random_split(dataset, [n_train, n_val])
        traindataset = CustomDataset(train_ds)
        valdataset = CustomDataset(val_ds)
        

        train_loader = DataLoader(traindataset, batch_size=1, pin_memory=True, shuffle=True)
        val_loader = DataLoader(valdataset, batch_size=1, pin_memory=True, shuffle=True)

        return {
            'train': train_loader,
            'val': val_loader,
        }

    def train_dataloader(self):
        return self.__dataloader()['train']

    def val_dataloader(self):
        return self.__dataloader()['val']

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser])

        parser.add_argument('--n_channels', type=int, default=3)
        parser.add_argument('--n_classes', type=int, default=1)
        return parser