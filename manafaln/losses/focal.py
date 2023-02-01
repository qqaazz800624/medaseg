from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from manafaln.utils import label_smoothing

class ClassificationFocalLoss(_Loss):
    """
    This is a implementation of Focal Loss [1] with Label Smoothing [2].
    It supports multi-class classification with other dimensions.
    The data `output` (B, C, ...) is compared with ground truth `target` (B, C, ...).
    Note that `output` is expected to be logits.

    $ LS-Focal(\hat{y}, y) = -\sum_{i=1}^{K}\alpha_i(1-\hat{y}_i)^{\gamma}(y_i(1-\delta)+\delta/K)\cdot\log \hat{y}_i $

    References:
    [1] Lin et al. (2017). Focal Loss for Dense Object Detection. https://arxiv.org/abs/1708.02002
    [2] Rafael Müller et al. (2019). When Does Label Smoothing Help? https://arxiv.org/abs/1906.02629
    """
    def __init__(
            self,
            gamma   : Optional[float]           = None,
            alpha   : Optional[Sequence[float]] = None,
            smooth  : Optional[float]           = None,
        ):
        super().__init__()
        self.gamma  = gamma
        self.alpha  = alpha
        self.smooth = smooth

    def convert_instance(self, x: torch.Tensor):
        if x.dim()>2:
            x = x.transpose(1, -1)      # (B, C, ...) => (B, ..., C)
            x = x.reshape(-1, x.size(-1))  # (B, ..., C) => (N, C)
        return x

    def forward(self,
            input: torch.Tensor,    # logit, shape: (B, C, ...)
            target: torch.Tensor,   # probability, shape: (B, C, ...)
        ) -> torch.Tensor:

        # Convert other dimensions to instance, shape: (N, C)
        input = self.convert_instance(input)    # (B, C, ...) => (N, C)
        target = self.convert_instance(target)  # (B, C, ...) => (N, C)

        # Label smoothing, shape: (N, C)
        target = label_smoothing(target, self.smooth)

        # Compute log-probability, shape: (N, C)
        logpt = F.log_softmax(input, dim=-1)

        # Compute crosss entropy, shape: (N, C)
        loss = -1 * logpt * target

        # Compute focal loss, shape: (N, C)
        # Fallback to Cross Entropy if gamma is None
        if self.gamma is not None:
            # Probability, shape: (N, C)
            pt = F.softmax(input, dim=-1)
            loss *= (1-pt)**self.gamma

        # Compute loss with alpha, shape: (N, C)
        # Fallback to unweighted focal if alpha not given
        if self.alpha is not None:
            # Alpha weighting, shape: (C, )
            alpha = torch.tensor(self.alpha, device=input.device)
            loss *= alpha

        # Sums of loss over instance, shape: (N, )
        loss = loss.sum(-1)

        # Mean of loss over batch, shape: ()
        loss = loss.mean()

        # Return loss
        return loss
