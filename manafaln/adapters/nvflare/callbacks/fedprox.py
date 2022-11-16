from collections import OrderedDict

import torch
from pytorch_lightning import Callback, LightningModule, Trainer

class FedProxLoss(Callback):
    def __init__(self, mu: float = 1.0):
        super().__init__()

        self.mu = mu
        self.global_state = OrderedDict()

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule):
        self.global_state = OrderedDict(
            (name, param.clone().detach()) for name, param in pl_module.model.named_parameters()
        )

    def on_before_backward(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        loss: torch.Tensor
    ):
        fedprox_loss = 0.0
        for name, param in pl_module.model.named_parameters():
            fedprox_loss += torch.sum((param - self.global_state[name]) ** 2)

        loss.assign(loss + self.mu * fedprox_loss)

