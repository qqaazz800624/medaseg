import warnings
from typing import Callable, List, Optional, Union, Sequence

import torch
from torch.nn.modules.loss import _Loss

from monai.networks import one_hot
from monai.utils import LossReduction
from monai.losses.focal_loss import FocalLoss

class MCCLoss(_Loss):
    """
    Compute the Matthews Correlation Coefficient (MCC) loss.

    Args:
        include_background: Whether to include the background class in the loss calculation. Defaults to True.
        to_onehot_y: Whether to convert the target tensor to one-hot encoding. Defaults to False.
        sigmoid: Whether to apply sigmoid activation to the input tensor. Defaults to False.
        softmax: Whether to apply softmax activation to the input tensor. Defaults to False.
        reduction: The reduction method to apply to the loss. Can be one of LossReduction.MEAN, LossReduction.SUM, or LossReduction.NONE. Defaults to LossReduction.MEAN.
        smooth_nr: The smoothing factor for the numerator in the MCC calculation. Defaults to 1e-5.
        smooth_dr: The smoothing factor for the denominator in the MCC calculation. Defaults to 1e-5.
        batch: Whether the input tensor has a batch dimension. Defaults to False.

    Raises:
        ValueError: If both sigmoid and softmax are set to True.

    """

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
        """
        Compute the forward pass of the MCC loss.

        Args:
            input: The input tensor.
            target: The target tensor.

        Returns:
            The computed MCC loss.

        Raises:
            AssertionError: If the input and target tensors have different shapes.

        """

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
    """
    Compute the combined loss of MCC loss and Focal loss.

    Args:
        include_background: Whether to include the background class in the loss calculation. Defaults to True.
        to_onehot_y: Whether to convert the target tensor to one-hot encoding. Defaults to False.
        sigmoid: Whether to apply sigmoid activation to the input tensor. Defaults to False.
        softmax: Whether to apply softmax activation to the input tensor. Defaults to False.
        reduction: The reduction method to apply to the loss. Can be one of LossReduction.MEAN, LossReduction.SUM, or LossReduction.NONE. Defaults to LossReduction.MEAN.
        smooth_nr: The smoothing factor for the numerator in the MCC calculation. Defaults to 1e-5.
        smooth_dr: The smoothing factor for the denominator in the MCC calculation. Defaults to 1e-5.
        batch: Whether the input tensor has a batch dimension. Defaults to False.
        gamma: The gamma parameter for the Focal loss. Defaults to 2.0.
        focal_weight: The weight parameter for the Focal loss. Defaults to None.
        lambda_mcc: The weight parameter for the MCC loss. Defaults to 1.0.
        lambda_focal: The weight parameter for the Focal loss. Defaults to 1.0.

    Raises:
        AssertionError: If lambda_mcc or lambda_focal is less than or equal to 0.

    """

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
        """
        Compute the forward pass of the combined MCC and Focal loss.

        Args:
            input: The input tensor.
            target: The target tensor.

        Returns:
            The computed combined loss.

        """

        mcc_loss = self.mcc(input, target)
        focal_loss = self.focal(input, target)
        total_loss = self.lambda_mcc * mcc_loss + self.lambda_focal * focal_loss

        return total_loss

