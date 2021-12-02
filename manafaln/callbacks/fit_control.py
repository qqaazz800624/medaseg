from typing import Any

from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.utilities.types import STEP_OUTPUT

class PauseTraining(Callback):
    def __init__(
        self,
        every_n_iter: int = 0,
        every_n_epoch: int = 0
    ):
        self.every_n_iter = every_n_iter
        self.every_n_epoch = every_n_epoch

    def on_fit_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule
    ) -> None:
        self.iter_counter = 0
        self.epoch_counter = 0
        # Force reset stop flag, may have some side effect
        trainer.should_stop = False

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        unused: int = 0,
    ) -> None:
        self.iter_counter += 1

        if self.every_n_iter > 0:
            # Evaluate criteria
            should_stop = (self.iter_counter >= self.every_n_iter)
            # Make sure all ddp processes stops
            should_stop = trainer.training_type_plugin.reduce_boolean_decision(should_stop)
            # Set stop flag to trainer
            trainer.should_stop = should_stop

    def on_train_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule
    ) -> None:
        self.epoch_counter += 1

        if self.every_n_epoch > 0:
            # Evaluate criteria
            should_stop = (self.epoch_counter >= self.every_n_epoch)
            # Make sure all ddp processes stops
            should_stop = trainer.training_type_plugin.reduce_boolean_decision(should_stop)
            # Set stop flag to trainer
            trainer.should_stop = should_stop
            # Increase epoch number for restart
            trainer.fit_loop.current_epoch += 1
