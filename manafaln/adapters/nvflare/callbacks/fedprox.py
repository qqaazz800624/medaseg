import logging
from collections import OrderedDict

import torch
from pytorch_lightning import Callback, LightningModule, Trainer

class FedProxLoss(Callback):
    """
    Callback class for Federated Proximal (FedProx) loss.

    Args:
        mu (float): The coefficient for the FedProx loss. Default is 1.0.
    """

    def __init__(self, mu: float = 1.0):
        super().__init__()

        self.mu = mu
        self.global_state = OrderedDict()
        self.logger = logging.getLogger(__name__)

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule):
        """
        Called when the training starts.

        Args:
            trainer (Trainer): The trainer object.
            pl_module (LightningModule): The LightningModule object.
        """
        self.global_state = OrderedDict(
            (name, param.clone().detach()) for name, param in pl_module.model.named_parameters()
        )

        weight_norm = torch.norm(
            torch.stack([torch.norm(p, 2.0) for p in self.global_state.values()]),
            2.0
        )
        for logger in trainer.loggers:
            logger.log_metrics({"global_weight_norm": weight_norm}, step=trainer.global_step)

        self.logger.info(f"FedProx global model updated.")

    def on_before_backward(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        loss: torch.Tensor
    ):
        """
        Called before the backward pass.

        Args:
            trainer (Trainer): The trainer object.
            pl_module (LightningModule): The LightningModule object.
            loss (torch.Tensor): The loss tensor.
        """
        fedprox_loss = 0.0
        for name, param in pl_module.model.named_parameters():
            fedprox_loss += torch.sum((param - self.global_state[name]) ** 2)
        fedprox_loss *= self.mu / 2.0

        loss += fedprox_loss

        for logger in trainer.loggers:
            logger.log_metrics({
                "fedprox_loss": fedprox_loss,
                "fedprox_total_loss": loss
            },
            step=trainer.global_step
        )

