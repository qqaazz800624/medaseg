from typing import Dict, Hashable, Mapping, Optional, Sequence, Union

import numpy as np
import torch
from monai.config import KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms import MapTransform, Transform
from monai.utils import ensure_tuple_rep

class AnyDim(Transform):
    """
    Apply the `any` operation along a specified dimension of the input data.

    Args:
        dim (int): The dimension along which to apply the `any` operation. Default is 0.
        keep_dim (bool): Whether to keep the dimension after applying the `any` operation. Default is False.

    Returns:
        NdarrayOrTensor: The input data after applying the `any` operation.

    Raises:
        NotImplementedError: If the input data type is not supported.

    """

    def __init__(
        self,
        dim: int = 0,
        keep_dim: bool = False,
    ):
        self.dim = dim
        self.keep_dim = keep_dim

    def __call__(
        self,
        data: NdarrayOrTensor,
        dim: Optional[int] = None,
        keep_dim: Optional[bool] = None,
    ) -> NdarrayOrTensor:
        """
        Apply the `any` operation along a specified dimension of the input data.

        Args:
            data (NdarrayOrTensor): The input data to be transformed.
            dim (Optional[int]): The dimension along which to apply the `any` operation. Default is None.
            keep_dim (Optional[bool]): Whether to keep the dimension after applying the `any` operation. Default is None.

        Returns:
            NdarrayOrTensor: The input data after applying the `any` operation.

        """
        dim = self.dim if dim is None else dim
        keep_dim = self.keep_dim if keep_dim is None else keep_dim

        if isinstance(data, torch.Tensor):
            data = data.any(dim=dim, keepdim=keep_dim)
        elif isinstance(data, np.ndarray):
            data = data.any(axis=dim, keepdims=keep_dim)
        else:
            raise NotImplementedError(f"Any cannot be applied on type {type(data)}.")

        return data


class AnyDimd(MapTransform):
    """
    Apply the `AnyDim` transform to a collection of data specified by keys.

    Args:
        keys (KeysCollection): The keys specifying the data to be transformed.
        dim (Union[int, Sequence[int]]): The dimension(s) along which to apply the `any` operation. Default is 0.
        keep_dim (Union[bool, Sequence[bool]]): Whether to keep the dimension(s) after applying the `any` operation. Default is False.
        allow_missing_keys (bool): Whether to allow missing keys in the input data. Default is False.

    Returns:
        Dict[Hashable, NdarrayOrTensor]: The transformed data with the same keys.

    """

    def __init__(
        self,
        keys: KeysCollection,
        dim: Union[int, Sequence[int]] = 0,
        keep_dim: Union[bool, Sequence[bool]] = False,
        allow_missing_keys: bool = False
    ):
        super().__init__(keys, allow_missing_keys)
        self.dim = ensure_tuple_rep(dim, len(self.keys))
        self.keep_dim = ensure_tuple_rep(keep_dim, len(self.keys))
        self.t = AnyDim()

    def __call__(
        self,
        data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        """
        Apply the `AnyDim` transform to the input data.

        Args:
            data (Mapping[Hashable, NdarrayOrTensor]): The input data to be transformed.

        Returns:
            Dict[Hashable, NdarrayOrTensor]: The transformed data with the same keys.

        """
        d = dict(data)
        for key, dim, keep_dim in self.key_iterator(d, self.dim, self.keep_dim):
            d[key] = self.t(d[key], dim, keep_dim)
        return d
