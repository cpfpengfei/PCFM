# Modifications for PCFM © 2025 Pengfei Cai (Learning Matter @ MIT) and Utkarsh (Julia Lab @ MIT), licensed under the MIT License.
# Original portions © Amazon.com, Inc. or its affiliates, licensed under the Apache License 2.0.
# PytorchLightning module for training FFM (used for Navier Stokes)

import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only

from . import get_flow_model
from scripts.training.utils import get_optimizer, get_scheduler
from scripts.training.vis_utils import draw


class FunctionalModule(pl.LightningModule):
    def __init__(self, **config):
        super().__init__()
        self.save_hyperparameters(logger=True)
        self.model = get_flow_model(self.hparams.model, self.hparams.encoder)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        loss = self.model.get_loss(batch)
        self.log('train_loss', loss, prog_bar=True, logger=True)
        self.log('lr', self.optimizers().param_groups[0]['lr'], logger=True)
        return loss

    @rank_zero_only
    @torch.no_grad()
    def visualize(self):
        gen = self.model.sample(self.hparams.n_sample, self.hparams.n_eval, self.hparams.sample_dims, self.device)
        for i in range(self.hparams.n_sample):
            self.logger.experiment.add_image(f'sample/{i}', draw(gen[i], **self.hparams.vis), self.trainer.global_step)

    def on_train_epoch_end(self) -> None:
        self.log('train_loss_epoch', self.trainer.callback_metrics['train_loss'], sync_dist=True)

    def validation_step(self, batch, batch_idx):
        loss = self.model.get_loss(batch)
        self.log('val_loss', loss, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def on_validation_epoch_end(self) -> None:
        self.log('val_loss_epoch', self.trainer.callback_metrics['val_loss'], logger=True, sync_dist=True)
        self.visualize()

    def optimizer_step(self, *args, **kwargs):
        optimizer = kwargs.get('optimizer', args[2])
        if self.trainer.global_step < self.hparams.train.lr_warmup_steps:
            lr_scale = min(1., float(self.trainer.global_step + 1) / self.hparams.train.lr_warmup_steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_scale * self.hparams.train.optimizer.lr
        super().optimizer_step(*args, **kwargs)

    def configure_optimizers(self):
        optimizer = get_optimizer(self.hparams.train.optimizer, self.model)
        scheduler = get_scheduler(self.hparams.train.scheduler, optimizer)
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                return {
                    'optimizer': optimizer,
                    'lr_scheduler': {
                        'scheduler': scheduler,
                        'monitor': 'val_loss',
                        'interval': 'step',
                        'frequency': self.hparams.train.val_check_interval,
                    }
                }
            else:
                return {
                    'optimizer': optimizer,
                    'lr_scheduler': {
                        'scheduler': scheduler,
                        'interval': 'step',
                        'frequency': 1,
                    }
                }
        return {'optimizer': optimizer}
