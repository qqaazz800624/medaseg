from typing import Dict, List

from pytorch_lightning import Callback, LightningModule, Trainer

class MetricsAverager(Callback):
    """
    Average multiple metrics to a new metric.

    Args:
        training (Dict[str, List[str]]):
            A dictionary where each key is a new metric to log during training.
            Its value is a list for metrics to average.
        validation (Dict[str, List[str]]):
            A dictionary where each key is a new metric to log during validation.
            Its value is a list for metrics to average.
        train_prog_bar (bool):
            Whether to show training metrics in progress bar.
            Defaults to False.
        validation_prog_bar (bool):
            Whether to show validation metrics in progress bar.
            Defaults to False.
    """
    def __init__(
        self,
        training: Dict[str, List[str]] = {},
        validation: Dict[str, List[str]] = {},
        train_prog_bar: bool = False,
        validation_prog_bar: bool = False,
    ):
        super().__init__()
        self.training = training
        self.validation = validation
        self.train_prog_bar = train_prog_bar
        self.validation_prog_bar = validation_prog_bar

    def _log_ave_metrics(
        self,
        metrics: Dict[str, List[str]],
        trainer: Trainer,
        pl_module: LightningModule,
        prog_bar: bool=False
    ) -> None:
        for ave_metric_name, sub_metrics_name in metrics.items():
            total_metric = 0
            for sub_metric_name in sub_metrics_name:
                total_metric += trainer.logged_metrics[sub_metric_name]
            ave_metric = total_metric / len(sub_metrics_name)
            pl_module.log(ave_metric_name, ave_metric, prog_bar=prog_bar)

    def on_train_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule
    ) -> None:
        self._log_ave_metrics(self.training, trainer, pl_module, self.train_prog_bar)

    def on_validation_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule
    ) -> None:
        self._log_ave_metrics(self.validation, trainer, pl_module, self.validation_prog_bar)
