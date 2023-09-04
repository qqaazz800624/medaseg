from copy import deepcopy
from typing import Any

from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.utilities.types import STEP_OUTPUT

class PauseTraining(Callback):
    """
    Callback to pause training after a certain number of iterations or epochs.

    Args:
        every_n_iter (int): Number of iterations after which to pause training. Default is 0.
        every_n_epoch (int): Number of epochs after which to pause training. Default is 0.
    """

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
        """
        Called when the fit begins.

        Args:
            trainer (Trainer): The trainer object.
            pl_module (LightningModule): The current LightningModule.
        """
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
        """
        Called when a training batch ends.

        Args:
            trainer (Trainer): The trainer object.
            pl_module (LightningModule): The current LightningModule.
            outputs (STEP_OUTPUT): The outputs of the training step.
            batch (Any): The current batch.
            batch_idx (int): The index of the current batch.
            unused (int): Unused argument. Default is 0.
        """
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
        """
        Called when a training epoch ends.

        Args:
            trainer (Trainer): The trainer object.
            pl_module (LightningModule): The current LightningModule.
        """
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

# class AbortFitWithSignal(Callback):
#     def __init__(self):
#         super(AbortTraining).__init__()
#
#         self.signal_attached = False
#
#     def attach_signal(self, signal: Signal):
#         self.signal = signal
#         self.signal_attached = True
#
#     def detach_signal(self):
#         self.signal_attached = False
#
#     def _handle_signal(self, trainer):
#         if self.signal_attached and self.signal.triggered:
#             trainer.fit_loop.should_stop = True
#
#     def on_sanity_check_end(self, trainer, pl_module):
#         self._handle_signal(trainer)
#
#     def on_batch_end(self, trainer, pl_module):
#         self._handle_signal(trainer)

class RestoreFitLR(Callback):
    """
    Callback to restore optimizer and learning rate scheduler states at the start and end of training.

    """

    def __init__(self):
        self.optimizer_states = []
        self.lr_schedulers = []

    def on_fit_start(self, trainer, pl_module):
        """
        Restore states when the fit begins.

        Args:
            trainer (Trainer): The trainer object.
            pl_module (LightningModule): The current LightningModule.
        """
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
        """
        Save states when the fit ends.

        Args:
            trainer (Trainer): The trainer object.
            pl_module (LightningModule): The current LightningModule.
        """
        opts = trainer.optimizers
        schs = trainer.lr_scheduler_configs

        self.optimizer_states = [deepcopy(opt.state_dict()) for opt in opts]
        self.lr_schedulers = [deepcopy(config.scheduler.state_dict()) for config in schs]

