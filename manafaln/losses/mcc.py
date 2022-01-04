import warnings
from typing import Callable, List, Optional, Union, Sequence

import torch
from torch.nn.modules.loss import _Loss

from monai.networks import one_hot
from monai.utils import LossReduction
from monai.losses.focal_loss import FocalLoss

class MCCLoss(_Loss):
    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        reduction: Union[LossReduction, str] = LossReduction.MEAN,
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        batch: bool = False
    ):
        super().__init__(reduction=LossReduction(reduction).value)

        if sigmoid and softmax:
            raise ValueError("Only one of sigmoid or softmax can be used.")

        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.smooth_nr = smooth_nr
        self.smooth_dr = smooth_dr
        self.batch = batch

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        if self.sigmoid:
            input = torch.sigmoid(input)

        n_pred_ch = input.shape[1]
        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn("Skipping softmax for single channel prediction.")
            else:
                input = torch.softmax(input, 1)

        if self.to_onehot_y and n_pred_ch > 1:
            target = one_hot(target, num_classes=n_pred_ch)

        if not self.include_background and n_pred_ch > 1:
            input = input[:, 1:]
            target = target[:, 1:]

        if input.shape != target.shape:
            raise AssertionError(
                f"Input shape ({input.shape}) does not meet the target shape ({target.shape})."
            )

        reduce_axis: List[int] = torch.arange(2, len(input.shape)).tolist()
        if self.batch:
            reduce_axis = [0] + reduce_axis

        # Preprare for MCC value
        tp = torch.sum(input * target, reduce_axis)
        tn = torch.sum((1.0 - input) * (1.0 - target), reduce_axis)
        fp = torch.sum(input * (1.0 - target), reduce_axis)
        fn = torch.sum((1.0 - input) * target, reduce_axis)

        numerator = tp * tn - fp * fn + self.smooth_nr
        denominator = torch.sqrt(
            (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + self.smooth_dr
        )

        mcc_loss = 1.0 - numerator / denominator

        if self.reduction == LossReduction.SUM.value:
            return torch.sum(mcc_loss)
        if self.reduction == LossReduction.NONE.value:
            return mcc_loss
        if self.reduction == LossReduction.MEAN.value:
            return torch.mean(mcc_loss)
        raise ValueError(f"Unsupported reduction: {self.reduction}")

class MCCFocalLoss(_Loss):
    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        reduction: Union[LossReduction, str] = LossReduction.MEAN,
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        batch: bool = False,
        gamma: float = 2.0,
        focal_weight: Optional[Union[Sequence[float], float, int, torch.Tensor]] = None,
        lambda_mcc: float = 1.0,
        lambda_focal: float = 1.0
    ):
        super().__init__()

        assert lambda_mcc > 0.0
        assert lambda_focal > 0.0

        self.mcc = MCCLoss(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            sigmoid=sigmoid,
            softmax=softmax,
            reduction=reduction,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
            batch=batch
        )
        self.focal = FocalLoss(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            gamma=gamma,
            weight=focal_weight,
            reduction=reduction
        )

        self.lambda_mcc = lambda_mcc
        self.lambda_focal = lambda_focal

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        mcc_loss = self.mcc(input, target)
        focal_loss = self.focal(input, target)
        total_loss = self.lambda_mcc * mcc_loss + self.lambda_focal * focal_loss

        return total_loss

