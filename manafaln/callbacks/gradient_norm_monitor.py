from typing import Literal

import torch
from torch.optim import Optimizer
from pytorch_lightning import Callback, LightningModule, Trainer

class GradientNormMonitor(Callback):
    def __init__(
        self,
        norm_type: float = 2.0
    ) -> None:
        super().__init__()
        self.norm_type = norm_type

    def on_before_optimizer_step(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        optimizer: Optimizer,
        opt_idx: int
    ):
        # Extract gradients
        grads = [p.grad for p in pl_module.parameters() if p.grad is not None]
        device = grads[0].device

        # Calculate norm (same as torch.nn.utils.clip_grad_norm_)
        norm_type = float(self.norm_type)
        total_norm = torch.norm(
            torch.stack([torch.norm(g.detach(), norm_type).to(device) for g in grads]),
            norm_type
        )

        # Log gradient norm
        gradient_norm_metrics = {"gradient_norm": total_norm}
        for logger in trainer.loggers:
            logger.log_metrics(
                gradient_norm_metrics,
                step=trainer.fit_loop.epoch_loop._batches_that_stepped
            )

