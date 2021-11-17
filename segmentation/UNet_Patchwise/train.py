
from model import Model

from dataLoader import CustomDataset, LitCustomData
import pytorch_lightning as pl

if __name__ == '__main__':
    hparams = {
        'lr': 0.0019054607179632484
    }
    dataset = LitCustomData()
    model = Model(hparams)
    dataset.start(r'E:\Google Drive\Acads\research\Single-View-Plant-Modelling\trainData/winData/')
    trainer = pl.Trainer(gpus=1, max_epochs=100)
    trainer.fit(model, dataset)