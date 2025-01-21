from typing import Dict, Hashable, Mapping, Optional, Sequence, Union, Callable

import torch
from monai.config import KeysCollection
from monai.data.meta_obj import get_track_meta
from monai.transforms import Transform, MapTransform
from monai.config.type_definitions import NdarrayOrTensor
from monai.utils import convert_data_type, convert_to_tensor, ensure_tuple_rep
from monai.utils.type_conversion import convert_to_dst_type
from monai.utils.enums import TransformBackends

class Activations(Transform):
    """
    Add activation operations to the model output, typically `Sigmoid` or `Softmax`.

    Args:
        sigmoid: whether to execute sigmoid function on model output before transform.
            Defaults to ``False``.
        softmax: whether to execute softmax function on model output before transform.
            Defaults to ``False``.
        other: callable function to execute other activation layers, for example:
            `other = lambda x: torch.tanh(x)`. Defaults to ``None``.
        channel_dim: dimension of channel to compute softmax.
            Defaults to 0.

    Raises:
        TypeError: When ``other`` is not an ``Optional[Callable]``.

    """

    backend = [TransformBackends.TORCH]

    def __init__(self, sigmoid: bool = False, softmax: bool = False, other: Optional[Callable] = None, channel_dim: int = 0) -> None:
        self.sigmoid = sigmoid
        self.softmax = softmax
        if other is not None and not callable(other):
            raise TypeError(f"other must be None or callable but is {type(other).__name__}.")
        self.other = other
        self.channel_dim = channel_dim

    def __call__(
        self,
        img: NdarrayOrTensor,
        sigmoid: Optional[bool] = None,
        softmax: Optional[bool] = None,
        other: Optional[Callable] = None,
        channel_dim: int = None,
    ) -> NdarrayOrTensor:
        """
        Args:
            sigmoid: whether to execute sigmoid function on model output before transform.
                Defaults to ``self.sigmoid``.
            softmax: whether to execute softmax function on model output before transform.
                Defaults to ``self.softmax``.
            other: callable function to execute other activation layers, for example:
                `other = torch.tanh`. Defaults to ``self.other``.
            channel_dim: dimension of channel to compute softmax.
                Defaults to ``self.channel_dim``.


        Raises:
            ValueError: When ``sigmoid=True`` and ``softmax=True``. Incompatible values.
            TypeError: When ``other`` is not an ``Optional[Callable]``.
            ValueError: When ``self.other=None`` and ``other=None``. Incompatible values.

        """
        if sigmoid and softmax:
            raise ValueError("Incompatible values: sigmoid=True and softmax=True.")
        if other is not None and not callable(other):
            raise TypeError(f"other must be None or callable but is {type(other).__name__}.")

        # convert to float as activation must operate on float tensor
        img = convert_to_tensor(img, track_meta=get_track_meta())
        img_t, *_ = convert_data_type(img, torch.Tensor, dtype=torch.float)
        if sigmoid or self.sigmoid:
            img_t = torch.sigmoid(img_t)
        if softmax or self.softmax:
            channel_dim = self.channel_dim if channel_dim is None else channel_dim
            img_t = torch.softmax(img_t, dim=channel_dim)

        act_func = self.other if other is None else other
        if act_func is not None:
            img_t = act_func(img_t)
        out, *_ = convert_to_dst_type(img_t, img)
        return out

class Activationsd(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.AddActivations`.
    Add activation layers to the input data specified by `keys`.
    """

    backend = Activations.backend

    def __init__(
        self,
        keys: KeysCollection,
        sigmoid: Union[Sequence[bool], bool] = False,
        softmax: Union[Sequence[bool], bool] = False,
        other: Optional[Union[Sequence[Callable], Callable]] = None,
        channel_dim: Union[Sequence[int], int] = 0,
        allow_missing_keys: bool = False,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to model output and label.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            sigmoid: whether to execute sigmoid function on model output before transform.
                it also can be a sequence of bool, each element corresponds to a key in ``keys``.
            softmax: whether to execute softmax function on model output before transform.
                it also can be a sequence of bool, each element corresponds to a key in ``keys``.
            other: callable function to execute other activation layers,
                for example: `other = torch.tanh`. it also can be a sequence of Callable, each
                element corresponds to a key in ``keys``.
            channel_dim: dimension of channel to compute softmax.
                it also can be a sequence of int, each element corresponds to a key in ``keys``.
            allow_missing_keys: don't raise exception if key is missing.

        """
        super().__init__(keys, allow_missing_keys)
        self.sigmoid = ensure_tuple_rep(sigmoid, len(self.keys))
        self.softmax = ensure_tuple_rep(softmax, len(self.keys))
        self.other = ensure_tuple_rep(other, len(self.keys))
        self.converter = Activations()
        self.channel_dim = ensure_tuple_rep(channel_dim, len(self.keys))


    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key, sigmoid, softmax, other, channel_dim in self.key_iterator(d, self.sigmoid, self.softmax, self.other, self.channel_dim):
            d[key] = self.converter(d[key], sigmoid, softmax, other, channel_dim)
        return d
