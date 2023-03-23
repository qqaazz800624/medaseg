from typing import Optional, Sequence

import numpy as np
from monai.transforms import MapTransform, Transform
from monai.utils import convert_to_numpy, ensure_tuple
from monai.utils.enums import TransformBackends
from PIL import Image


class OverlayMask(Transform):
    backend = [TransformBackends.NUMPY]
    def __init__(
        self,
        colors: Sequence[str]=["#ff0000"],
        overlay_order: Optional[Sequence[int]]=None
    ):
        self.colors = colors
        self.overlay_order = range(len(colors)) if overlay_order is None else overlay_order

    def __call__(self, image, masks):
        image = convert_to_numpy(image * 255, dtype=np.uint8)
        image = np.transpose(image, (2, 1, 0)) # (C, W, H) => (H, W ,C)
        if image.shape[-1] == 1:
            image = np.concatenate((image, )*3, axis=-1)
        image = Image.fromarray(image, mode="RGB")

        masks = convert_to_numpy(masks, dtype=bool)
        masks = np.transpose(masks, (2, 1, 0)) # (C, W, H) => (H, W ,C)

        for idx in self.overlay_order:
            mask = Image.fromarray(np.take(masks, idx, axis=-1))
            color = Image.new('RGB', image.size, self.colors[idx])
            image.paste(color, mask=mask)

        image = np.array(image)
        image = image / 255
        image = np.transpose(image, (2, 1, 0)) # (H, W, C) => (C, W, H)
        return image

class OverlayMaskd(MapTransform):
    backend = OverlayMask.backend
    def __init__(
        self,
        image_keys,
        mask_keys,
        names=None,
        colors=["#ff0000"],
        overlay_order: Optional[Sequence[int]]=None
    ):
        self.image_keys = ensure_tuple(image_keys)
        self.mask_keys = ensure_tuple(mask_keys)
        self.names = ensure_tuple(names) if names is not None else self.image_keys
        self.t = OverlayMask(colors=colors, overlay_order=overlay_order)

    def __call__(self, data):
        for image_key, mask_key, overlayed_key in zip(self.image_keys, self.mask_keys, self.names):
            data[overlayed_key] = self.t(data[image_key], data[mask_key])
        return data
