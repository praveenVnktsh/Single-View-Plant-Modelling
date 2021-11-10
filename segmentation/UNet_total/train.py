import os
from argparse import ArgumentParser

import numpy as np
import torch

from unet import Model
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


def main(hparams):
    model = Model(hparams)

    os.makedirs(hparams.log_dir, exist_ok=True)
    try:
        log_dir = sorted(os.listdir(hparams.log_dir))[-1]
    except IndexError:
        log_dir = os.path.join(hparams.log_dir, 'version_0')
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(log_dir, 'checkpoints'),
        # save_best_only=False,
        verbose=True,
    )
    stop_callback = EarlyStopping(
        monitor='val_loss',
        mode='auto',
        patience=5,
        verbose=True,
    )
    trainer = Trainer(
        gpus=1,
        checkpoint_callback=checkpoint_callback,
    )

    trainer.fit(model)


if __name__ == '__main__':
    parent_parser = ArgumentParser(add_help=False)
    parent_parser.add_argument('--log_dir', default='lightning_logs')

    parser = Model.load_from_checkpoint(r'version_24\checkpoints\epoch=18-step=1614.ckpt').add_model_specific_args(parent_parser)
    hparams = parser.parse_args()

    main(hparams)