from typing import Dict, Hashable, Mapping, Optional, Sequence, Union
from numbers import Real

import torch
import numpy as np
from monai.config import KeysCollection
from monai.transforms import Transform, MapTransform
from monai.config.type_definitions import NdarrayOrTensor
from monai.utils import ensure_tuple_rep

class Unsqueeze(Transform):
    """
    Apply the unsqueeze operation to the input data along the specified dimension.

    Args:
        dim (int): The dimension along which to unsqueeze the input data. Default is 0.

    Returns:
        NdarrayOrTensor: The unsqueezed input data.
    """

    def __init__(self, dim: int = 0):
        self.dim = dim

    def __call__(
        self,
        data: NdarrayOrTensor,
        dim: Optional[int] = None
    ) -> NdarrayOrTensor:
        """
        Apply the unsqueeze operation to the input data along the specified dimension.

        Args:
            data (NdarrayOrTensor): The input data to be unsqueezed.
            dim (Optional[int]): The dimension along which to unsqueeze the input data. Default is None.

        Returns:
            NdarrayOrTensor: The unsqueezed input data.
        """
        dim = self.dim if dim is None else dim
        return torch.unsqueeze(data, dim)

class Unsqueezed(MapTransform):
    """
    Apply the unsqueeze operation to the input data along the specified dimension for multiple keys.

    Args:
        keys (KeysCollection): The keys corresponding to the data to be unsqueezed.
        dim (Union[Sequence[int], int]): The dimension(s) along which to unsqueeze the input data.
        allow_missing_keys (bool): Whether to allow missing keys in the input data. Default is False.

    Returns:
        Dict[Hashable, NdarrayOrTensor]: The unsqueezed input data for the specified keys.
    """

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
        """
        Apply the unsqueeze operation to the input data along the specified dimension for the specified keys.

        Args:
            data (Mapping[Hashable, NdarrayOrTensor]): The input data to be unsqueezed.

        Returns:
            Dict[Hashable, NdarrayOrTensor]: The unsqueezed input data for the specified keys.
        """
        d = dict(data)
        for key, dim in self.key_iterator(d, self.dim):
            d[key] = self.converter(d[key], dim)
        return d

class ScalarToNumpyArrayd(MapTransform):
    """
    Convert scalar values in the input data to numpy arrays.

    Args:
        keys (KeysCollection): The keys corresponding to the data to be converted.

    Returns:
        Dict[Hashable, NdarrayOrTensor]: The input data with scalar values converted to numpy arrays.
    """

    def __init__(
        self,
        keys: KeysCollection
    ):
        super().__init__(keys, False)

    def __call__(
        self,
        data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        """
        Convert scalar values in the input data to numpy arrays.

        Args:
            data (Mapping[Hashable, NdarrayOrTensor]): The input data to be converted.

        Returns:
            Dict[Hashable, NdarrayOrTensor]: The input data with scalar values converted to numpy arrays.
        """
        d = dict(data)
        for key in self.keys:
            if isinstance(d[key], Real):
                d[key] = np.array([d[key]])
        return d

class UnpackDict(Transform):
    """
    Unpack a dictionary into a list of values corresponding to the specified keys.

    Args:
        item_keys (Sequence[str]): The keys corresponding to the values to be unpacked.

    Returns:
        list: The unpacked values.
    """

    def __init__(
        self,
        item_keys: Sequence[str]=None,
    ):
        self.item_keys = item_keys

    def __call__(self, data: dict) -> list:
        """
        Unpack a dictionary into a list of values corresponding to the specified keys.

        Args:
            data (dict): The dictionary to be unpacked.

        Returns:
            list: The unpacked values.
        """
        data = [data.get(k) for k in self.item_keys]
        return data

class UnpackDictd(MapTransform):
    """
    Unpack a dictionary into a list of values corresponding to the specified keys for multiple keys.

    Args:
        keys (KeysCollection): The keys corresponding to the dictionaries to be unpacked.
        item_keys (Sequence[str]): The keys corresponding to the values to be unpacked.

    Returns:
        Dict[Hashable, NdarrayOrTensor]: The input data with dictionaries unpacked into lists of values.
    """

    def __init__(
        self,
        keys: KeysCollection,
        item_keys: Sequence[str]=None,
    ):
        super().__init__(keys, False)
        self.t = UnpackDict(item_keys)

    def __call__(
        self,
        data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        """
        Unpack a dictionary into a list of values corresponding to the specified keys for the specified keys.

        Args:
            data (Mapping[Hashable, NdarrayOrTensor]): The input data to be unpacked.

        Returns:
            Dict[Hashable, NdarrayOrTensor]: The input data with dictionaries unpacked into lists of values.
        """
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.t(d[key])
        return d
