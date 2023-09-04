from typing import Dict, Hashable, Mapping, Optional, Union

import torch
import numpy as np
from monai.config import DtypeLike, KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.data.meta_obj import get_track_meta
from monai.data.meta_tensor import MetaTensor
from monai.transforms import Transform, MapTransform
from monai.transforms.utils_pytorch_numpy_unification import clip
from monai.utils.enums import TransformBackends
from monai.utils.type_conversion import convert_data_type, convert_to_tensor

class NormalizeIntensityRange(Transform):
    """
    Normalize the intensity range of the input image.

    Args:
        a_min (float): The minimum value of the intensity range.
        a_max (float): The maximum value of the intensity range.
        subtrahend (float): The value to subtract from the image.
        divisor (float): The value to divide the image by.
        dtype (DtypeLike, optional): The data type of the output image. Defaults to np.float32.

    Returns:
        NdarrayOrTensor: The normalized image.

    Raises:
        ValueError: If a_min is greater than a_max.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
        self,
        a_min: float,
        a_max: float,
        subtrahend: float,
        divisor: float,
        dtype: DtypeLike = np.float32
    ):
        if a_min > a_max:
            raise ValueError("a_min must be lesser than a_max.")

        self.a_min = a_min
        self.a_max = a_max

        self.subtrahend = subtrahend
        self.divisor = divisor

        self.dtype = dtype

    def __call__(
        self,
        img: NdarrayOrTensor,
        subtrahend: Optional[float] = None,
        divisor: Optional[float] = None,
        dtype: Optional[DtypeLike] = None
    ) -> NdarrayOrTensor:
        """
        Apply the normalization to the input image.

        Args:
            img (NdarrayOrTensor): The input image to be normalized.
            subtrahend (float, optional): The value to subtract from the image. Defaults to None.
            divisor (float, optional): The value to divide the image by. Defaults to None.
            dtype (DtypeLike, optional): The data type of the output image. Defaults to None.

        Returns:
            NdarrayOrTensor: The normalized image.
        """
        if subtrahend is None:
            subtrahend = self.subtrahend
        if divisor is None:
            divisor = self.divisor
        if dtype is None:
            dtype = self.dtype

        img = convert_to_tensor(img, track_meta=get_track_meta())

        img = clip(img, self.a_min, self.a_max)
        img = (img - subtrahend) / divisor

        ret: NdarrayOrTensor = convert_data_type(img, dtype=dtype)[0]
        return ret


class NormalizeIntensityRanged(MapTransform):
    """
    Apply NormalizeIntensityRange transform to a collection of images.

    Args:
        keys (KeysCollection): The keys corresponding to the images to be normalized.
        a_min (float): The minimum value of the intensity range.
        a_max (float): The maximum value of the intensity range.
        subtrahend (float): The value to subtract from the images.
        divisor (float): The value to divide the images by.
        dtype (DtypeLike, optional): The data type of the output images. Defaults to np.float32.
        allow_missing_keys (bool, optional): Whether to allow missing keys. Defaults to False.

    Returns:
        Dict[Hashable, NdarrayOrTensor]: The dictionary containing the normalized images.

    """

    backend = NormalizeIntensityRange.backend

    def __init__(
        self,
        keys: KeysCollection,
        a_min: float,
        a_max: float,
        subtrahend: float,
        divisor: float,
        dtype: Optional[DtypeLike] = np.float32,
        allow_missing_keys: bool=False
    ):
        super().__init__(keys, allow_missing_keys)
        self.t = NormalizeIntensityRange(
            a_min, a_max,
            subtrahend, divisor,
            dtype=dtype
        )

    def __call__(
        self,
        data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        """
        Apply the NormalizeIntensityRange transform to the input data.

        Args:
            data (Mapping[Hashable, NdarrayOrTensor]): The input data to be transformed.

        Returns:
            Dict[Hashable, NdarrayOrTensor]: The dictionary containing the transformed data.
        """
        d = dict(data)
        for key in self.keys:
            d[key] = self.t(d[key])
        return d

