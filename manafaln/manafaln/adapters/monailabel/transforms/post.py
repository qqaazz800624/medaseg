import logging
from typing import Any, Hashable, Sequence

import torch
from monai.data import MetaTensor
from monai.transforms import MapTransform

logger = logging.getLogger(__name__)


class RemoveDsHeadsd(MapTransform):
    def __init__(self, keys: Sequence[str], dim: int = 0) -> None:
        super().__init__(keys, allow_missing_keys=False)

        self.dim = dim

    def __call__(self, data: dict[Hashable, Any]) -> dict: 
        d = dict(data)
        for key in self.key_iterator(d):
            heads = torch.split(d[key], 1, dim=self.dim)
            if isinstance(d[key], MetaTensor):
                d[key].array = heads[0].squeeze(dim=self.dim)
            else:
                d[key] = heads[0].squeeze(dim=self.dim)
        return d
