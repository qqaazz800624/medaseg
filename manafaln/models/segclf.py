from typing import Literal, Dict

import torch

from manafaln.core.builders import ModelBuilder

class SegClfModel(torch.nn.Module):
    """
    A PyTorch module that combines a segmentation model and a classification model,
    where the input of classification model is the concatenation of segmentation output and input image.

    Args:
        seg_model (Dict): A dictionary that contains the configuration for the segmentation model.
        clf_model (Dict): A dictionary that contains the configuration for the classification model.
        seg_activation (Literal["sigmoid", "softmax"], optional): The activation function to use
            on the segmentation input for classification model.
            Can be either "sigmoid" or "softmax". If None, the segmentation output is used as is.
    """
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
        """
        Sequentially pass `img` through segmentation and classification model.

        Args:
            img (torch.Tensor): The input image tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The segmentation output and classification output.
        """
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
