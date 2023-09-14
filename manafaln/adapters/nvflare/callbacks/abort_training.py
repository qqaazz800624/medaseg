from pytorch_lightning import Callback
from nvflare.apis.signal import Signal

class AbortTraining(Callback):
    """
    Callback class to abort training based on a signal trigger.

    Args:
        None

    Attributes:
        signal_attached (bool): Flag indicating whether a signal is attached.
        signal (Signal): The signal object attached to the callback.

    Methods:
        attach_signal(signal: Signal): Attaches a signal object to the callback.
        detach_signal(): Detaches the signal object from the callback.
        _handle_signal(trainer): Handles the signal trigger and stops the training loop if triggered.
        on_sanity_check_end(trainer, pl_module): Callback function called at the end of the sanity check step.
        on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx): Callback function called at the end of each training batch.
        on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx): Callback function called at the end of each validation batch.
    """

    def __init__(self):
        super(AbortTraining).__init__()

        self.signal_attached = False

    def attach_signal(self, signal: Signal):
        """
        Attaches a signal object to the callback.

        Args:
            signal (Signal): The signal object to attach.

        Returns:
            None
        """
        self.signal = signal
        self.signal_attached = True

    def detach_signal(self):
        """
        Detaches the signal object from the callback.

        Args:
            None

        Returns:
            None
        """
        self.signal_attached = False

    def _handle_signal(self, trainer):
        """
        Handles the signal trigger and stops the training loop if triggered.

        Args:
            trainer: The trainer object.

        Returns:
            None
        """
        if self.signal_attached and self.signal.triggered:
            trainer.fit_loop.should_stop = True

    def on_sanity_check_end(self, trainer, pl_module):
        """
        Callback function called at the end of the sanity check step.

        Args:
            trainer: The trainer object.
            pl_module: The LightningModule object.

        Returns:
            None
        """
        self._handle_signal(trainer)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """
        Callback function called at the end of each training batch.

        Args:
            trainer: The trainer object.
            pl_module: The LightningModule object.
            outputs: The outputs of the training batch.
            batch: The current batch.
            batch_idx: The index of the current batch.

        Returns:
            None
        """
        self._handle_signal(trainer)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """
        Callback function called at the end of each validation batch.

        Args:
            trainer: The trainer object.
            pl_module: The LightningModule object.
            outputs: The outputs of the validation batch.
            batch: The current batch.
            batch_idx: The index of the current batch.
            dataloader_idx: The index of the current dataloader.

        Returns:
            None
        """
        self._handle_signal(trainer)
