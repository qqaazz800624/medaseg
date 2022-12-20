from copy import deepcopy
from typing import Optional
from logging import getLogger

import torch
from pytorch_lightning import Callback

class RestoreLR(Callback):
    def __init__(self, from_checkpoint: Optional[str] = None):
        super().__init__()

        if from_checkpoint:
            ckpt = torch.load(from_checkpoint)
        else:
            ckpt = {}

        self.optimizer_states = ckpt.get("optimizer_states", [])
        self.lr_schedulers = ckpt.get("lr_schedulers", [])

        self.logger = getLogger(__name__)

    def on_fit_start(self, trainer, pl_module):
        if len(self.optimizer_states) > 0:
            trainer.strategy.load_optimizer_state_dict({
                "optimizer_states": self.optimizer_states
            })
            self.logger.info("optimizer states restored")
        else:
            return

        if len(self.lr_schedulers) > 0:
            for config, lrs_state in zip(trainer.lr_scheduler_configs, self.lr_schedulers):
                config.scheduler.load_state_dict(lrs_state)
            self.logger.info("LR scheduler state restored")

    def on_fit_end(self, trainer, pl_module):
        opts = trainer.optimizers
        schs = trainer.lr_scheduler_configs

        self.optimizer_states = [deepcopy(opt.state_dict()) for opt in opts]
        self.lr_schedulers = [deepcopy(config.scheduler.state_dict()) for config in schs]

