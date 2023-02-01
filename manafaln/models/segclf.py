from typing import Literal, Dict

import torch

from manafaln.core.builders import ModelBuilder

class SegClfModel(torch.nn.Module):
    def __init__(
        self,
        seg_model: Dict,
        clf_model: Dict,
        seg_activation: Literal["sigmoid", "softmax"]=None,
    ):
        super().__init__()

        builder = ModelBuilder()
        self.segmentation_model: torch.nn.Module = builder(seg_model)
        self.classification_model: torch.nn.Module = builder(clf_model)

        self.seg_activation = seg_activation

    def forward(self, img: torch.Tensor):
        """Sequentially pass `x` through segmentation and classification model."""

        pred_seg: torch.Tensor = self.segmentation_model(img)

        if self.seg_activation == "sigmoid":
            seg_for_clf = pred_seg.sigmoid()
        elif self.seg_activation == "softmax":
            seg_for_clf = pred_seg.softmax(1)
        else:
            seg_for_clf = pred_seg

        clf_input = torch.cat([img, seg_for_clf], dim=1)
        pred_clf = self.classification_model(clf_input)

        return pred_seg, pred_clf
