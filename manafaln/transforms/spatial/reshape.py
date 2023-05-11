from typing import Dict, Hashable, Mapping, Sequence, Union

import numpy as np
import torch
from monai.config import KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms import MapTransform, Transform
from monai.utils.enums import TransformBackends


class Reshape(Transform):
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
        self,
        shape: Sequence[int] = (-1,),
    ):
        self.shape = shape

    def __call__(self, data: NdarrayOrTensor) -> NdarrayOrTensor:
        if isinstance(data, torch.Tensor):
            data = data.reshape(*self.shape)
        elif isinstance(data, np.ndarray):
            data = data.reshape(self.shape)
        else:
            raise NotImplementedError(f"Data type {type(data)} cannot be reshaped.")

        return data


class Reshaped(MapTransform):
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
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.t(d[key])
        return d
