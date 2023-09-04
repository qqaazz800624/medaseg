from typing import Union

import torch
import numpy as np
from monai.metrics import CumulativeIterationMetric
from monai.metrics.utils import do_metric_reduction, ignore_background
from monai.utils import MetricReduction

class MeanSquareError(CumulativeIterationMetric):
    """
    Computes the mean square error (MSE) between predicted and target tensors.

    Args:
        include_background (bool): Whether to include the background class in the calculation. Defaults to True.
        reduction (MetricReduction or str): The method used to reduce the computed metric values. Defaults to MetricReduction.MEAN.

    """

    def __init__(
        self,
        include_background: bool = True,
        reduction: Union[MetricReduction, str] = MetricReduction.MEAN
    ) -> None:
        """
        Initializes a new instance of the MeanSquareError class.

        Args:
            include_background (bool): Whether to include the background class in the calculation. Defaults to True.
            reduction (MetricReduction or str): The method used to reduce the computed metric values. Defaults to MetricReduction.MEAN.

        """
        super().__init__()

        self.include_background = include_background
        self.reduction = reduction

    def _compute_tensor(self, y_pred: torch.Tensor, y: torch.Tensor):
        """
        Computes the mean square error (MSE) between predicted and target tensors.

        Args:
            y_pred (torch.Tensor): The predicted tensor.
            y (torch.Tensor): The target tensor.

        Returns:
            torch.Tensor: The computed mean square error (MSE) tensor.

        """
        se = torch.square(y_pred - y)

        ndim = len(se.shape)
        if ndim > 2:
            return torch.mean(se, dim=tuple(range(2, ndim)))
        else:
            return torch.mean(se, dim=1, keepdim=True)

    def aggregate(self):
        """
        Aggregates the computed metric values.

        Returns:
            float: The aggregated metric value.

        Raises:
            ValueError: If the data to aggregate is not a torch tensor.

        """
        data = self.get_buffer()
        if not isinstance(data, torch.Tensor):
            raise ValueError("The data to aggregate must be a torch tensor")
        value, _ = do_metric_reduction(data, self.reduction)
        return value
