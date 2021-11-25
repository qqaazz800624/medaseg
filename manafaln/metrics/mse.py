from typing import Union

import torch
import numpy as np
from monai.metrics import CumulativeIterationMetric
from monai.metrics.utils import do_metric_reduction, ignore_background
from monai.utils import MetricReduction

class MeanSquareError(CumulativeIterationMetric):
    def __init__(
        self,
        include_background: bool = True,
        reduction: Union[MetricReduction, str] = MetricReduction.MEAN
    ) -> None:
        super().__init__()

        self.include_background = include_background
        self.reduction = reduction

    def _compute_tensor(self, y_pred: torch.Tensor, y: torch.Tensor):
        se = torch.square(y_pred - y)

        ndim = len(se.shape)
        if ndim > 2:
            return torch.mean(se, dim=tuple(range(2, ndim)))
        else:
            return torch.mean(se, dim=1, keepdim=True)

    def aggregate(self):
        data = self.get_buffer()
        if not isinstance(data, torch.Tensor):
            raise ValueError("The data to aggregate must be a torch tensor")
        value, _ = do_metric_reduction(data, self.reduction)
        return value
