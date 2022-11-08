from copy import deepcopy

import torch
from pytorch_lightning import Callback

class RestoreLR(Callback):
    def __init__(self):
        self.optimizer_states = []
        self.lr_schedulers = []

    def on_fit_start(self, trainer, pl_module):
        if len(self.optimizer_states) > 0:
            trainer.strategy.load_optimizer_state_dict({
                "optimizer_states": self.optimizer_states
            })
            print("optimizer states restored")
        else:
            return

        if len(self.lr_schedulers) > 0:
            for config, lrs_state in zip(trainer.lr_scheduler_configs, self.lr_schedulers):
                config.scheduler.load_state_dict(lrs_state)
            print("LR scheduler state restored")

    def on_fit_end(self, trainer, pl_module):
        opts = trainer.optimizers
        schs = trainer.lr_scheduler_configs

        self.optimizer_states = [deepcopy(opt.state_dict()) for opt in opts]
        self.lr_schedulers = [deepcopy(config.scheduler.state_dict()) for config in schs]
