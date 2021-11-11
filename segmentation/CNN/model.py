import pytorch_lightning as pl
from torch import nn
import numpy as np
import torch

class Model(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.nn = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(6, 6, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((3, 3)),
            nn.Conv2d(6, 6, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(6, 6, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 2 )),
            nn.Flatten(),
            nn.Linear(150, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.lossfunc = nn.CrossEntropyLoss()

    def forward(self,x):
        y = self.nn(x)
        return y
    
    def configure_optimizers(self):
        LR = 1e-3
        optimizer = torch.optim.AdamW(self.parameters(),lr=LR)
        return optimizer

    def runBatch(self, batch, batch_idx):
        x = batch['input']
        y = batch['target']

        out = self(x)
        loss = self.lossfunc(out, y)
        return {
            'x' : x,
            'y' : y,
            'pred' : out,
            'loss' : loss
        }


    def training_step(self, batch, batch_idx):
        dic = self.runBatch(batch, batch_idx)
        loss = dic['loss']
        self.log('train_loss', loss)

        return {'loss': loss}

    def validation_step(self,batch,batch_idx):
        dic = self.runBatch(batch, batch_idx)
        loss = dic['loss']

        self.log('val_loss', loss)

        return {'loss': loss}



if __name__ == "__main__":
    model = Model({})
    output = model(torch.randn(1, 3, 30, 30))
    print(output.shape)