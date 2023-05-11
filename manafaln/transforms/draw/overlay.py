"""
Module: overlay.py
This module defines two classes - OverlayMask and OverlayMaskd - that can be used to overlay a mask on an image.
"""

from typing import Hashable, Mapping, Optional, Sequence, Union

import numpy as np
from monai.config import KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms import MapTransform, Transform
from monai.utils import convert_to_numpy, ensure_tuple, ensure_tuple_rep
from monai.utils.enums import TransformBackends
from PIL import Image


class OverlayMask(Transform):
    """
    Transform to overlay a mask on an image.
    Args:
        colors (Union[str, Sequence[str]]): The colors to use for the overlay mask. Default: "#ff0000".
        alpha (Optional[Union[int, Sequence[int]]]): Alpha value to use for the overlay mask.
            If None, no alpha value is applied. Default: None.
        indices (Optional[Sequence[int]]): Indices of the masks to use for the overlay.
            If None, all masks are used. Default: None.
    """

    backend = [TransformBackends.NUMPY]

    def __init__(
        self,
        colors: Union[str, Sequence[str]] = "#ff0000",
        alpha: Optional[Union[int, Sequence[int]]] = None,
        indices: Optional[Sequence[int]] = None,
    ):
        self.colors = ensure_tuple(colors)
        self.alpha = ensure_tuple_rep(alpha, len(self.colors))
        self.indices = (
            ensure_tuple_rep(indices, len(self.colors)) if indices is not None else None
        )

    def __call__(self, image: NdarrayOrTensor, masks: NdarrayOrTensor):
        """
        Applies the overlay mask transform on the given image and masks.

        Args:
            image (NdarrayOrTensor): The image to apply the transform on. Shape: (C, W, H).
            masks (NdarrayOrTensor): The masks to use for the overlay. Shape: (C, W, H).

        Returns:
            The overlaid image. Shape: (C, W, H).
        """
        image = convert_to_numpy(image * 255, dtype=np.uint8)
        image = np.transpose(image, (2, 1, 0))  # (C, W, H) => (H, W ,C)
        if image.shape[-1] == 1:
            image = np.concatenate((image,) * 3, axis=-1)
        image = Image.fromarray(image, mode="RGB")

        masks = convert_to_numpy(masks, dtype=bool)
        masks = np.transpose(masks, (2, 1, 0))  # (C, W, H) => (H, W ,C)

        indices = self.indices if self.indices is not None else range(masks.shape[-1])

        if len(self.colors) != len(indices):
            raise ValueError(
                f"Number of colors ({len(self.colors)}) must match number of indices ({len(indices)})."
            )

        for color, alpha, idx in zip(self.colors, self.alpha, indices):
            mask = np.take(masks, idx, axis=-1)
            if alpha is not None:
                mask = (mask * alpha * 255).astype(np.uint8)
            mask = Image.fromarray(mask)
            image.paste(color, mask=mask)

        image = np.array(image)
        image = image / 255
        image = np.transpose(image, (2, 1, 0))  # (H, W, C) => (C, W, H)
        return image


class OverlayMaskd(MapTransform):
    """
    Dictionary version of :py:class:`OverlayMask`.
    Args:
        image_keys (KeysCollection): Keys of the images to apply the transform on.
        mask_keys (KeysCollection): Keys of the masks to use for the overlay.
        names (Optional[KeysCollection]): Keys to use for the overlaid images.
            If None, the keys from `image_keys` are used. Default: None.
        colors (Union[str, Sequence[str]]): The colors to use for the overlay mask. Default: "#ff0000".
        alpha (Optional[Union[int, Sequence[int]]]): Alpha value to use for the overlay mask.
            If None, no alpha value is applied. Default: None.
        indices (Optional[Sequence[int]]): Indices of the masks to use for the overlay.
            If None, all masks are used. Default: None.
        allow_missing_keys (bool): don't raise exception if key is missing. Default: False.
    """

    backend = OverlayMask.backend

    def __init__(
        self,
        image_keys: KeysCollection,
        mask_keys: KeysCollection,
        names: Optional[KeysCollection] = None,
        colors: Union[str, Sequence[str]] = "#ff0000",
        alpha: Optional[Union[int, Sequence[int]]] = None,
        indices: Optional[Sequence[int]] = None,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys=image_keys, allow_missing_keys=allow_missing_keys)
        self.mask_keys = ensure_tuple(mask_keys)
        self.names = ensure_tuple(names) if names is not None else self.keys
        self.overlay = OverlayMask(colors=colors, alpha=alpha, indices=indices)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]):
        d = dict(data)
        for image_key, mask_key, name in self.key_iterator(
            d, self.mask_keys, self.names
        ):
            d[name] = self.overlay(d[image_key], d[mask_key])
        return d
