from typing import Optional, Sequence

import numpy as np
from monai.config import KeysCollection
from monai.transforms import MapTransform, Transform
from scipy.ndimage import binary_fill_holes
from skimage.morphology import binary_closing, disk


class Fill(Transform):
    def __init__(
        self,
        mask_idx: Optional[Sequence[int]]=None,
        channel_dim: int=0,
        footprint: Optional[int]=10,
    ):
        self.mask_idx = mask_idx
        self.channel_dim = channel_dim
        self.footprint = disk(footprint) if footprint is not None else None

    def fill(self, mask):
        mask = binary_fill_holes(mask)
        return mask

    def __call__(self, masks):
        masks = np.swapaxes(masks, self.channel_dim, 0)
        mask_indices = self.mask_idx if self.mask_idx is not None else range(masks.shape[0])
        for mask_i in mask_indices:
            mask = masks[mask_i]
            if self.footprint is not None:
                mask = binary_closing(mask, self.footprint)
            mask = self.fill(mask)
            masks[mask_i] = mask
        masks = np.swapaxes(masks, 0, self.channel_dim)
        return masks

class Filld(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        mask_idx: Optional[Sequence[int]]=None,
        channel_dim: int=0,
        footprint: Optional[int]=10,
        allow_missing_keys: bool=False
    ):
        super().__init__(keys, allow_missing_keys)
        self.t = Fill(mask_idx, channel_dim, footprint)

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.t(d[key])
        return d
