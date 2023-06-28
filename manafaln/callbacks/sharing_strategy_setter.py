import logging
from typing import Literal

import torch
from pytorch_lightning import Callback, LightningModule, Trainer

logger = logging.getLogger(__name__)


class SharingStrategySetter(Callback):
    def __init__(
        self,
        strategy: Literal["file_system", "file_descriptor"] = "file_system",
    ):
        self.strategy = strategy

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        self.ori_strategy = torch.multiprocessing.get_sharing_strategy()
        if self.ori_strategy != self.strategy:
            torch.multiprocessing.set_sharing_strategy(self.strategy)
            logger.info(
                f"Sharing strategy changed from {self.ori_strategy} to {self.strategy}"
            )

    def teardown(
        self, trainer: Trainer, pl_module: LightningModule, stage: str
    ) -> None:
        current_strategy = torch.multiprocessing.get_sharing_strategy()
        if current_strategy != self.ori_strategy:
            torch.multiprocessing.set_sharing_strategy(self.ori_strategy)
            logger.info(
                f"Sharing strategy changed from {current_strategy} to {self.ori_strategy}"
            )
