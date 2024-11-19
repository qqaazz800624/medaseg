import torch
from typing import Callable, Optional
from monai.losses import DeepSupervisionLoss, DiceCELoss

def DsDiceCELoss(
    include_background: bool = True,
    to_onehot_y: bool = False,
    sigmoid: bool = False,
    softmax: bool = False,
    other_act: Optional[Callable] = None,
    squared_pred: bool = False,
    jaccard: bool = False,
    reduction: str = "mean",
    smooth_nr: float = 1e-5,
    smooth_dr: float = 1e-5,
    batch: bool = False,
    ce_weight: Optional[torch.Tensor] = None,
    lambda_dice: float = 1.0,
    lambda_ce: float = 1.0
):
    dice = DiceCELoss(
        include_background=include_background,
        to_onehot_y=to_onehot_y,
        sigmoid=sigmoid,
        softmax=softmax,
        other_act=other_act,
        squared_pred=squared_pred,
        jaccard=jaccard,
        reduction=reduction,
        smooth_nr=smooth_nr,
        smooth_dr=smooth_dr,
        batch=batch,
        ce_weight=ce_weight,
        lambda_dice=lambda_dice,
        lambda_ce=lambda_ce
    )
    return DeepSupervisionLoss(dice)

