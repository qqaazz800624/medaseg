from typing import Dict, Hashable, Mapping, Optional, Sequence, Union

import numpy as np
import torch
from monai.config import KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms import MapTransform, Transform

class Reshape(Transform):
    def __init__(
        self,
        shape: Sequence[int] = (-1, ),
    ):
        self.shape = shape

    def __call__(
        self,
        data: NdarrayOrTensor,
        shape: Optional[Sequence[int]] = None
    ) -> NdarrayOrTensor:
        shape = self.shape if shape is None else shape

        if isinstance(data, torch.Tensor):
            data = data.reshape(*shape)
        elif isinstance(data, np.ndarray):
            data = data.reshape(shape)
        else:
            raise NotImplementedError(f"Data type {type(data)} cannot be resized.")

        return data

class Reshaped(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        shape: Union[Sequence[int], Sequence[Sequence[int]]] = (-1, ),
        allow_missing_keys: bool = False
    ):
        super().__init__(keys, allow_missing_keys)

        n = len(keys)
        if isinstance(shape[0], int):
            self.shape = (shape, )*n
        else:
            self.shape = shape

        self.converter = Reshape()

    def __call__(
        self,
        data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key, shape in self.key_iterator(d, self.shape):
            d[key] = self.converter(d[key], shape)
        return d
