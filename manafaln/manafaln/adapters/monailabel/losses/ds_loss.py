import logging
from typing import Sequence

import torch
from monai.losses import DeepSupervisionLoss, DiceCELoss

logger = logging.getLogger(__name__)


class DsDiceCELoss(DeepSupervisionLoss):
    def __init__(self, **kwargs) -> None:
        loss = DiceCELoss(**kwargs)
        super().__init__(loss, weight_mode="exp")

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        pred = input
        if pred.dim() == target.dim() + 1:
            pred = [p.squeeze(dim=1) for p in torch.split(input, 1, dim=1)]
        loss = super().forward(pred, target)
        return loss
