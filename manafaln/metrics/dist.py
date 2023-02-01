from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torchmetrics import Metric
from kornia.geometry.subpix import spatial_soft_argmax2d
from monai.utils import MetaKeys

SPACING_KEY = "spacing"

class CentroidDistance(Metric):
    full_state_update: bool = True
    def __init__(self, temperature=100):
        super().__init__()
        self.add_state("dists", default=torch.tensor(0, dtype=float), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.temperature = torch.tensor(temperature)

    def update(self, preds: torch.Tensor, target: torch.Tensor, meta_data: Optional[Dict]=None):
        """
        Expects shape (B, C, H, W)
        """

        assert preds.dim() == target.dim() == 4, f"preds and target expected to have shape (B, C, H, W), got {preds.shape} and {target.shape}"
        assert preds.shape == target.shape, f"preds and target expected to have same shape, got {preds.shape} and {target.shape}"

        B, C, H, W = preds.shape
        spatial_shape = torch.tensor([H, W], device=preds.device)

        preds = preds.view(B*C, 1, H, W)
        target = target.view(B*C, 1, H, W)

        non_empty_indices = (preds.sum(dim=(-1, -2, -3)) > 0) and (target.sum(dim=(-1, -2, -3)) > 0)
        total_non_empty = non_empty_indices.sum()

        if total_non_empty == 0:
            return

        preds = preds[non_empty_indices, ...]
        target = target[non_empty_indices, ...].float()

        preds_centroid = spatial_soft_argmax2d(preds, self.temperature, normalized_coordinates=False).flip(-1)
        target_centroid = spatial_soft_argmax2d(target, self.temperature, normalized_coordinates=False).flip(-1)

        if meta_data is not None:
            ori_spacing = torch.tensor(list(meta_data[SPACING_KEY]), device=preds_centroid.device)
            ori_shape   = torch.tensor(list(meta_data[MetaKeys.SPATIAL_SHAPE]), device=preds_centroid.device)
            # Pixel loc => relative loc => ori pixel loc => real loc in mm
            preds_centroid  = (preds_centroid / spatial_shape) * ori_shape * ori_spacing
            target_centroid = (target_centroid / spatial_shape) * ori_shape * ori_spacing

        dists = F.pairwise_distance(preds_centroid, target_centroid)

        self.dists += dists.sum()
        self.total += total_non_empty

    def compute(self):
        return self.dists / self.total

if __name__ == '__main__':
    preds = torch.zeros((1, 1, 512, 512))
    preds[0,0,200-10:200+10,130-5:130+5] = 1

    target = torch.zeros((1, 1, 512, 512))
    target[0,0,150-10:150+10,100-10:100+10] = 1

    ori_spacing = torch.tensor([0.1, 0.1])
    ori_shape = torch.tensor([5120, 5120])

    metric = CentroidDistance()
    metric.update(preds, target, {SPACING_KEY: ori_spacing, MetaKeys.SPATIAL_SHAPE: ori_shape})
    dist = metric.compute()

    import matplotlib.pyplot as plt
    visual = torch.zeros((512, 512, 3), dtype=int)
    visual[preds.squeeze().bool()] = torch.tensor([255, 0, 0])
    visual[target.squeeze().bool()] = torch.tensor([0, 255, 0])
    plt.title(f"Distance: {dist:.0f} (mm)")
    plt.imshow(visual)
    # plt.savefig("dist.png")
