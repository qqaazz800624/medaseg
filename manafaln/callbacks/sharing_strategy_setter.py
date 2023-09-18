import logging
from typing import Literal

import torch
from lightning import Callback, LightningModule, Trainer

logger = logging.getLogger(__name__)


class SharingStrategySetter(Callback):
    """
    Callback to set the sharing strategy for PyTorch multiprocessing.

    Args:
        strategy (Literal["file_system", "file_descriptor"], optional): The sharing strategy to set. Defaults to "file_system".

    Methods:
        setup(trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
            Sets the sharing strategy to the specified value before training starts.

        teardown(trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
            Resets the sharing strategy to its original value after training ends.
    """

    def __init__(
        self,
        strategy: Literal["file_system", "file_descriptor"] = "file_system",
    ):
        self.strategy = strategy

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        """
        Sets the sharing strategy to the specified value before training starts.

        Args:
            trainer (Trainer): The PyTorch Lightning Trainer instance.
            pl_module (LightningModule): The PyTorch Lightning LightningModule instance.
            stage (str): The current training stage.

        Returns:
            None
        """
        self.ori_strategy = torch.multiprocessing.get_sharing_strategy()
        if self.ori_strategy != self.strategy:
            torch.multiprocessing.set_sharing_strategy(self.strategy)
            logger.info(
                f"Sharing strategy changed from {self.ori_strategy} to {self.strategy}"
            )

    def teardown(
        self, trainer: Trainer, pl_module: LightningModule, stage: str
    ) -> None:
        """
        Resets the sharing strategy to its original value after training ends.

        Args:
            trainer (Trainer): The PyTorch Lightning Trainer instance.
            pl_module (LightningModule): The PyTorch Lightning LightningModule instance.
            stage (str): The current training stage.

        Returns:
            None
        """
        current_strategy = torch.multiprocessing.get_sharing_strategy()
        if current_strategy != self.ori_strategy:
            torch.multiprocessing.set_sharing_strategy(self.ori_strategy)
            logger.info(
                f"Sharing strategy changed from {current_strategy} to {self.ori_strategy}"
            )
