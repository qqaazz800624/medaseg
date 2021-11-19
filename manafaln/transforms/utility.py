from typing import Dict, Hashable, Mapping, Optional, Sequence, Union

import torch
from monai.config import KeysCollection
from monai.transforms import Transform, MapTransform
from monai.config.type_definitions import NdarrayOrTensor
from monai.utils import ensure_tuple_rep

class Unsqueeze(Transform):
    def __init__(self, dim: int = 0):
        self.dim = dim

    def __call__(
        self,
        data: NdarrayOrTensor,
        dim: Optional[int] = None
    ) -> NdarrayOrTensor:
        dim = self.dim if dim is None else dim
        return torch.unsqueeze(data, dim)

class Unsqueezed(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        dim: Union[Sequence[int], int],
        allow_missing_keys: bool = False
    ):
        super().__init__(keys, allow_missing_keys)

        n = len(keys)
        self.dim = ensure_tuple_rep(dim, n)

        self.converter = Unsqueeze()

    def __call__(
        self,
        data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key, dim in self.key_iterator(d, self.dim):
            d[key] = self.converter(d[key])
        return d
