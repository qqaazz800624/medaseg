import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class LinearWarmupCosineAnnealingLR(_LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float = 0.0,
        eta_min: float = 0.0,
        last_epoch: int = -1
    ) -> None:
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min

        self.cosine_cycle_length = max_epochs - warmup_epochs

        super(LinearWarmupCosineAnnealingLR, self).__init__(
            optimizer, last_epoch
        )

    def get_lr(self):
        if self.last_epoch == 0:
            return [self.warmup_start_lr for _ in self.base_lrs]
        elif self.last_epoch < self.warmup_epochs:
            return [
                group["lr"] + (base_lr - self.warmup_start_lr) / self.warmup_epochs
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        elif self.last_epoch == self.warmup_epochs:
            return self.base_lrs

        epoch_offset = self.last_epoch - 1 - self.max_epochs
        is_restart = epoch_offset % (2 * self.cosine_cycle_length) == 0

        if is_restart:
            return [
                group["lr"] + (base_lr - self.eta_min) *
                (1 - math.cos(math.pi / self.cosine_cycle_length)) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]

        # Regular cosine annealing
        progress_curr = (self.last_epoch - self.warmup_epochs) / self.cosine_cycle_length
        progress_prev = (self.last_epoch - self.warmup_epochs - 1) / self.cosine_cycle_length

        return [
            ((1 + math.cos(math.pi * progress_curr)) /
             (1 + math.cos(math.pi * progress_prev))) *
            (group["lr"] - self.eta_min) + self.eta_min
            for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self) -> list[float]:
        if self.last_epoch < self.warmup_epochs:
            return [
                self.warmup_start_lr + self.last_epoch * (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr in self.base_lrs
            ]

        progress = (self.last_epoch - self.warmup_epochs) / self.cosine_cycle_length

        return [
            self.eta_min + 0.5 * (base_lr - self.eta_min) * (1 + math.cos(math.pi * progress))
            for base_lr in self.base_lrs
        ]
