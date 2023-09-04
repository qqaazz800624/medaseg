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
    """
    def __init__(
        self,
        training: Dict[str, List[str]] = {},
        validation: Dict[str, List[str]] = {},
    ):
        super().__init__()
        self.training = training
        self.validation = validation

    def _log_avg_metrics(
        self,
        metrics: Dict[str, List[str]],
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        """
        Log the average metrics.

        Args:
            metrics (Dict[str, List[str]]):
                A dictionary where each key is a new metric to log.
                Its value is a list of metrics to average.
            trainer (Trainer):
                The PyTorch Lightning Trainer object.
            pl_module (LightningModule):
                The PyTorch Lightning LightningModule object.
        """
        for avg_metric_name, sub_metrics_name in metrics.items():
            total_metric = 0
            for sub_metric_name in sub_metrics_name:
                total_metric += trainer.logged_metrics[sub_metric_name]
            avg_metric = total_metric / len(sub_metrics_name)
            pl_module.log(avg_metric_name, avg_metric, prog_bar=False)

    def on_train_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule
    ) -> None:
        """
        Callback function called at the end of each training epoch.

        Args:
            trainer (Trainer):
                The PyTorch Lightning Trainer object.
            pl_module (LightningModule):
                The PyTorch Lightning LightningModule object.
        """
        self._log_ave_metrics(self.training, trainer, pl_module)

    def on_validation_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule
    ) -> None:
        """
        Callback function called at the end of each validation epoch.

        Args:
            trainer (Trainer):
                The PyTorch Lightning Trainer object.
            pl_module (LightningModule):
                The PyTorch Lightning LightningModule object.
        """
        self._log_avg_metrics(self.validation, trainer, pl_module)
