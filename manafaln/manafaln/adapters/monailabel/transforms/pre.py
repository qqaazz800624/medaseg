import logging
from typing import Any, Hashable, Sequence

import torch
from monai.data import MetaTensor
from monai.transforms import MapTransform

logger = logging.getLogger(__name__)


class NormalizeLabelsd(MapTransform):
    def __init__(self, keys: Sequence[str], label_names: dict[str, int]) -> None:
        super().__init__(keys, allow_missing_keys=False)
        self.labels = label_names

    def __call__(self, data: dict[Hashable, Any]) -> dict:
        d: dict = dict(data)
        for key in self.key_iterator(d):
            label = torch.zeros_like(d[key])
            for _, val in self.labels.items():
                label[d[key] == val] = val

            if isinstance(d[key], MetaTensor):
                d[key].array = label
            else:
                d[key] = label
        return d
