from pytorch_lightning import Callback
from nvflare.apis.signal import Signal

class AbortTraining(Callback):
    def __init__(self):
        super(AbortTraining).__init__()

        self.signal_attached = False

    def attach_signal(self, signal: Signal):
        self.signal = signal
        self.signal_attached = True

    def detach_signal(self):
        self.signal_attached = False

    def _handle_signal(self, trainer):
        if self.signal_attached and self.signal.triggered:
            trainer.fit_loop.should_stop = True

    def on_sanity_check_end(self, trainer, pl_module):
        self._handle_signal(trainer)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._handle_signal(trainer)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self._handle_signal(trainer)

