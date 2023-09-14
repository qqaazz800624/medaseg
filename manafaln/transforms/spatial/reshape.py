from typing import Dict, Hashable, Mapping, Sequence, Union

import numpy as np
import torch
from monai.config import KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms import MapTransform, Transform
from monai.utils.enums import TransformBackends


class Reshape(Transform):
    """
    Reshape the input data to the specified shape.

    Args:
        shape (Sequence[int]): The desired shape of the data. Default is (-1,).

    Returns:
        NdarrayOrTensor: The reshaped data.

    Raises:
        NotImplementedError: If the data type is not supported for reshaping.
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
        self,
        shape: Sequence[int] = (-1,),
    ):
        self.shape = shape

    def __call__(self, data: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Reshape the input data.

        Args:
            data (NdarrayOrTensor): The input data to be reshaped.

        Returns:
            NdarrayOrTensor: The reshaped data.
        """
        if isinstance(data, torch.Tensor):
            data = data.reshape(*self.shape)
        elif isinstance(data, np.ndarray):
            data = data.reshape(self.shape)
        else:
            raise NotImplementedError(f"Data type {type(data)} cannot be reshaped.")

        return data


class Reshaped(MapTransform):
    """
    Apply the Reshape transform to a dictionary of data.

    Args:
        keys (KeysCollection): The keys to be transformed.
        shape (Union[Sequence[int], Sequence[Sequence[int]]]): The desired shape of the data. Default is (-1,).
        allow_missing_keys (bool): Whether to allow missing keys in the input data. Default is False.

    Returns:
        Dict[Hashable, NdarrayOrTensor]: The dictionary of transformed data.

    Raises:
        NotImplementedError: If the data type is not supported for reshaping.
    """

    backend = Reshape.backend

    def __init__(
        self,
        keys: KeysCollection,
        shape: Union[Sequence[int], Sequence[Sequence[int]]] = (-1,),
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.t = Reshape(shape)

    def __call__(
        self, data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        """
        Apply the Reshape transform to the input data.

        Args:
            data (Mapping[Hashable, NdarrayOrTensor]): The input data to be transformed.

        Returns:
            Dict[Hashable, NdarrayOrTensor]: The dictionary of transformed data.
        """
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.t(d[key])
        return d
