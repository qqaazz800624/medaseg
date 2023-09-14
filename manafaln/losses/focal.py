from typing import Optional, Sequence, Union

import torch
import torch.nn.functional as F
from monai.utils import LossReduction
from torch.nn.modules.loss import _Loss


def _label_smoothing(label: torch.Tensor, smooth: Optional[float]=None, binary: bool=False):
    """
    Label smoothing with

    $$ (1-smooth)*label + smooth/num_classes $$

    References:
    Rafael Müller et al. (2019). When Does Label Smoothing Help? https://arxiv.org/abs/1906.02629

    Args:
        label (torch.Tensor): tensor to apply label smoothing, with last channel as label.
        smooth (float): smoothing factor, requires 0 <= smooth < 1. Default: 0.0
        binary (bool): whether the label is binary. Default: False
    """

    if smooth is None:
        return label

    assert 0 <= smooth < 1, "smoothing factor must be 0 <= smooth < 1"

    if binary:
        # shape: (..., 1) or (..., )
        label_smooth = (1-smooth)*label + smooth/2
    else:
        # shape: (..., C)
        label_smooth = (1-smooth)*label + smooth/label.size(-1)

    return label_smooth


class MulticlassFocalLoss(_Loss):
    """
    This is an extension of CrossEntropyLoss that down-weights loss from
    high confidence correct predictions.
    This is a reimplementation of Focal Loss [1] that supports multiclass.
    Label Smoothing [2] is also supported.

    The data `output` (B, C, ...) is compared with ground truth `target` (B, C, ...).
    Note that `output` is expected to be logits.
    Additional dimensions will be treated as instances.

    $ LS-Focal(\hat{y}, y) = -\sum_{i=1}^{K}\alpha_i(1-\hat{y}_i)^{\gamma}(y_i(1-\delta)+\delta/K)\cdot\log \hat{y}_i $

    References:
    [1] Lin et al. (2017). Focal Loss for Dense Object Detection. https://arxiv.org/abs/1708.02002
    [2] Rafael Müller et al. (2019). When Does Label Smoothing Help? https://arxiv.org/abs/1906.02629
    """

    def __init__(
            self,
            gamma: Optional[float] = 2.0,
            alpha: Optional[Sequence[float]] = None,
            smooth: Optional[float] = None,
            reduction: Union[LossReduction, str] = LossReduction.MEAN,
        ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.smooth = smooth
        self.reduction = reduction

    def convert_instance(self, x: torch.Tensor):
        """
        Convert the input tensor to the instance format.

        Args:
            x (torch.Tensor): The input tensor to convert.

        Returns:
            torch.Tensor: The converted tensor.
        """
        if x.dim()>2:
            x = x.transpose(1, -1)      # (B, C, ...) => (B, ..., C)
            x = x.reshape(-1, x.size(-1))  # (B, ..., C) => (N, C)
        return x

    def forward(self,
            input: torch.Tensor,    # logit, shape: (B, C, ...)
            target: torch.Tensor,   # probability, shape: (B, C, ...)
        ) -> torch.Tensor:
        """
        Compute the multiclass focal loss.

        Args:
            input (torch.Tensor): The logit tensor, with shape (B, C, ...).
            target (torch.Tensor): The probability tensor, with shape (B, C, ...).

        Returns:
            torch.Tensor: The computed loss tensor.
        """

        # Convert other dimensions to instance, shape: (N, C)
        input = self.convert_instance(input)    # (B, C, ...) => (N, C)
        target = self.convert_instance(target)  # (B, C, ...) => (N, C)

        # Label smoothing, shape: (N, C)
        target = _label_smoothing(target, self.smooth)

        # Compute log-probability, shape: (N, C)
        logpt = F.log_softmax(input, dim=-1)

        # Compute crosss entropy, shape: (N, C)
        loss = -1 * logpt * target

        # Down-weight loss from high confidence correct predictions, shape: (N, C)
        # Fallback to Cross Entropy if gamma is None
        if self.gamma is not None:
            # Probability, shape: (N, C)
            pt = F.softmax(input, dim=-1)
            loss *= (1-pt)**self.gamma

        # Weight loss by class with alpha, shape: (N, C)
        # Fallback to unweighted loss if alpha not given
        if self.alpha is not None:
            # Alpha weighting, shape: (C, )
            alpha = torch.tensor(self.alpha, device=loss.device)
            loss *= alpha

        # Sums of loss over instance, shape: (N, )
        loss = loss.sum(-1)

        if self.reduction == LossReduction.SUM.value:
            return loss.sum()
        if self.reduction == LossReduction.NONE.value:
            return loss
        if self.reduction == LossReduction.MEAN.value:
            return loss.mean()
        raise ValueError(f"Unsupported reduction: {self.reduction}")
