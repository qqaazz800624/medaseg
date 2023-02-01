from typing import List, Union, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from monai.losses import DiceLoss, FocalLoss
from monai.networks import one_hot
from monai.utils import DiceCEReduction, Weight, look_up_option

from manafaln.utils import SpatialWeightedMixin
from .mcc import MCCLoss

class MultipleBackgroundDiceFocalLoss(_Loss):
    def __init__(
        self,
        background_channels: List[int],
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        squared_pred: bool = False,
        jaccard: bool = False,
        reduction: str = "mean",
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        batch: bool = False,
        gamma: float = 1.0,
        focal_weight: Optional[torch.Tensor] = None,
        lambda_dice: float = 1.0,
        lambda_focal: float = 1.0
    ) -> None:
        super().__init__()
        reduction = look_up_option(reduction, DiceCEReduction).value

        # These options should be handled here
        self.background_channels = background_channels
        self.to_onehot_y = to_onehot_y
        self.sigmoid = sigmoid
        self.softmax = softmax

        self.dice = DiceLoss(
            include_background=True,
            to_onehot_y=False,
            sigmoid=False,
            softmax=False,
            other_act=None,
            squared_pred=squared_pred,
            jaccard=jaccard,
            reduction=reduction,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
            batch=batch
        )
        self.focal = FocalLoss(
            include_background=True,
            to_onehot_y=False,
            gamma=gamma,
            weight=focal_weight,
            reduction=reduction
        )

        if lambda_dice < 0.0:
            raise ValueError("lambda_dice should be no less than 0.0.")
        if lambda_focal < 0.0:
            raise ValueError("lambda_ce should be no less than 0.0.")
        self.lambda_dice = lambda_dice
        self.lambda_focal = lambda_focal

    def reduce_background_channels(self, tensor: torch.Tensor) -> torch.Tensor:
        n_chs = tensor.shape[1]
        slices = torch.split(tensor, 1, dim=1)

        bg = sum([slices[i] for i in self.background_channels])
        fg = [slices[i] for i in range(n_chs) if i not in self.background_channels]

        output = torch.cat([bg] + fg, dim=1)
        return output

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Sigmoid
        if self.sigmoid:
            input = torch.sigmoid(input)

        # Softmax
        n_pred_ch = input.shape[1]
        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `softmax=True` ignored.")
            else:
                input = torch.softmax(input, 1)

        # One hot encoding
        if self.to_onehot_y:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                target = one_hot(target, num_classes=n_pred_ch)

        # Sum background channels
        input = self.reduce_background_channels(input)
        target = self.reduce_background_channels(target)

        dice_loss = self.dice(input, target)
        focal_loss = self.focal(input, target)
        total_loss: torch.Tensor = self.lambda_dice * dice_loss + self.lambda_focal * focal_loss

        return total_loss

class MultipleBackgroundDiceCELoss(_Loss):
    def __init__(
        self,
        background_channels: List[int],
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        squared_pred: bool = False,
        jaccard: bool = False,
        reduction: str = "mean",
        smooth_nr: float = 1e-5,
        smooth_dr: float = 1e-5,
        batch: bool = False,
        ce_weight: Optional[torch.Tensor] = None,
        lambda_dice: float = 1.0,
        lambda_ce: float = 1.0
    ) -> None:
        super().__init__()

        self.loss = MultipleBackgroundDiceFocalLoss(
            background_channels=background_channels,
            to_onehot_y=to_onehot_y,
            sigmoid=sigmoid,
            softmax=softmax,
            squared_pred=squared_pred,
            jaccard=jaccard,
            reduction=reduction,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
            batch=batch,
            gamma=1.0,
            focal_weight=ce_weight,
            lambda_dice=lambda_dice,
            lambda_focal=lambda_ce
        )

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss(input, target)

class SWDiceLoss(_Loss, SpatialWeightedMixin):
    """
    Compute Spatial-weighted Dice loss between two tensors.
    Spatial-weight is designed such that the weight is higher if closer to POI.
    It only supports multi-labels tasks of 2D inputs.
    The data `output` (B, C, H, W) is compared with ground truth `target` (B, C, H, W).
    Note that `output` is expected to be logits.

    $$
    \sum_{n=1}^{W\times H} w_n = W\times H \nonumber
    SW-Dice(\hat{y}, y) = 1 - \frac{2\sum_{n=1}^{W\times H} w_n \cdot y_n \cdot\hat{y}_n}
        {\sum_{n=1}^{W\times H} w_n \cdot y_n +\sum_{n=1}^{W\times H} w_n \cdot \hat{y}_n}
    $$
    """
    def __init__(self,
            sigmoid: bool=True,
            weight: Sequence[float]=None,
            poi: Optional[Sequence[int]]=None,
            sigma: float = 1/12,
            gamma: Optional[float]=None,
            log: bool=False,
            smooth: float = 1e-5,
        ):
        _Loss.__init__(self)
        SpatialWeightedMixin.__init__(self, poi=poi, sigma=sigma)
        self.weight = weight
        self.sigmoid = sigmoid
        self.smooth = smooth
        self.log = log
        self.gamma = gamma

    def forward(self,
            input: torch.Tensor,    # Input logit or probability mask, shape: (B, C, H, W)
            target: torch.Tensor    # Target probability mask, shape: (B, C, H, W)
        ) -> torch.Tensor:

        if input.size() != target.size():
            raise ValueError(
                f"input and target needs to be have shape, got input {input.size()} and target {target.size()}"
            )

        # If input is logit
        if self.sigmoid:
            # Read output logit mask and apply sigmoid to probability, shape: (B, C, H, W)
            input = torch.sigmoid(input)

        # Get shape of output, shape: ()
        B, C, H, W = input.size()

        # Get spatial weights, shape: (B, C, H, W)
        spatial_weight = self.get_spatial_weights(target)

        # Reshape to spatial dims over instances and channels, shape: (B*C, H*W)
        input = input.view(B*C, H*W)
        target = target.view(B*C, H*W)
        spatial_weight = spatial_weight.view(B*C, H*W)

        # Compute dice score, shape: (B*C, )
        numerator = 2 * (spatial_weight * input * target).sum(dim=1) + self.smooth
        denominator = (spatial_weight * input).sum(dim=1) + (spatial_weight * target).sum(dim=1) + self.smooth
        dice = numerator / denominator

        # Compute dice loss, shape: (B*C, )
        # Use log variation if self.log is True
        if self.log:
            loss = -1 * dice.log()
        # Else use standard dice loss
        else:
            loss = 1 - dice

        # Gamma for focal weight, shape: (B*C, )
        if self.gamma is not None:
            loss = loss**self.gamma

        # Reshape loss (B*C, ) => (B, C)
        loss = loss.view(B, C)

        # Compute loss over channels for each instance, shape: (B, )
        # If channel weights are given
        if self.weight is not None:
            # Get channel weights, shape: (C, )
            weight = torch.tensor(self.weight).to(loss.device)
            # Normalize to sum 1, shape: (C,)
            weight = F.normalize(weight, p=1, dim=0)
            # Compute the weighted average over channels, shape: (B, )
            loss = (weight*loss).sum(-1)
        # Else if channel weights are not given
        else:
            # Simply average over channelsm shape: (B, )
            loss = loss.mean(-1)

        # Mean of loss over instances, shape: ()
        loss = loss.mean()

        if loss < 0:
            raise RuntimeError(f"Invalid Dice loss value: {loss}")

        # Return loss, shape: ()
        return loss

