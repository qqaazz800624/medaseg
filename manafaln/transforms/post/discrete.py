from typing import Dict, Hashable, Mapping, Optional, Sequence, Union

import torch
from monai.config import KeysCollection
from monai.networks import one_hot
from monai.transforms import Transform, MapTransform
from monai.config.type_definitions import NdarrayOrTensor
from monai.utils import convert_data_type, look_up_option, ensure_tuple_rep
from monai.utils.type_conversion import convert_to_dst_type

class AsDiscrete(Transform):
    def __init__(
        self,
        argmax: bool = False,
        to_onehot: Optional[int] = None,
        threshold: Optional[float] = None,
        rounding: Optional[str] = None,
        n_classes: Optional[int] = None,
        num_classes: Optional[int] = None,
        logit_thresh: float = 0.5,
        threshold_values: Optional[bool] = False,
        channel_dim: Optional[int] = 0
    ):
        self.argmax = argmax
        if isinstance(to_onehot, bool):
            to_onehot = num_classes if to_onehot else None
        self.to_onehot = to_onehot

        if isinstance(threshold, bool):
            threshold = logit_thresh if threshold else None
        self.threshold = threshold
        self.rounding = rounding
        self.channel_dim = channel_dim

    def __call__(
        self,
        img: NdarrayOrTensor,
        argmax: Optional[bool] = None,
        to_onehot: Optional[int] = None,
        threshold: Optional[float] = None,
        rounding: Optional[str] = None,
        n_classes: Optional[int] = None,
        num_classes: Optional[int] = None,
        logit_thresh: Optional[float] = None,
        threshold_values: Optional[bool] = None,
        channel_dim : Optional[int] = None
    ) -> NdarrayOrTensor:
        if isinstance(to_onehot, bool):
            to_onehot = num_classes if to_onehot else None
        if isinstance(threshold, bool):
            threshold = logit_thresh if threshold else None

        img_t, *_ = convert_data_type(img, torch.Tensor)
        if argmax or self.argmax:
            img_t = torch.argmax(img_t, dim=self.channel_dim, keepdim=True)

        to_onehot = self.to_onehot if to_onehot is None else to_onehot
        if to_onehot is not None:
            if not isinstance(to_onehot, int):
                raise AssertionError("the number of classes for One-Hot must be an integer.")
            img_t = one_hot(img_t, num_classes=to_onehot, dim=self.channel_dim)

        threshold = self.threshold if threshold is None else threshold
        if threshold is not None:
            img_t = img_t >= threshold

        rounding = self.rounding if rounding is None else rounding
        if rounding is not None:
            look_up_option(rounding, ["torchrounding"])
            img_t = torch.round(img_t)

        img, *_ = convert_to_dst_type(img_t, img, dtype=torch.float)
        return img


class AsDiscreted(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        argmax: Union[Sequence[bool], bool] = False,
        to_onehot: Union[Sequence[Optional[int]], Optional[int]] = None,
        threshold: Union[Sequence[Optional[float]], Optional[float]] = None,
        rounding: Union[Sequence[Optional[str]], Optional[str]] = None,
        allow_missing_keys: bool = False,
        n_classes: Optional[Union[Sequence[int], int]] = None,
        num_classes: Optional[Union[Sequence[int], int]] = None,
        logit_thresh: Union[Sequence[float], float] = 0.5,
        threshold_values: Union[Sequence[bool], bool] = False,
        channel_dim: int = 0
    ):
        super().__init__(keys, allow_missing_keys)
        self.argmax = ensure_tuple_rep(argmax, len(self.keys))
        to_onehot_ = ensure_tuple_rep(to_onehot, len(self.keys))
        num_classes = ensure_tuple_rep(num_classes, len(self.keys))
        self.to_onehot = []
        for flag, val in zip(to_onehot_, num_classes):
            if isinstance(flag, bool):
                self.to_onehot.append(val if flag else None)
            else:
                self.to_onehot.append(flag)

        threshold_ = ensure_tuple_rep(threshold, len(self.keys))
        logit_thresh = ensure_tuple_rep(logit_thresh, len(self.keys))
        self.threshold = []
        for flag, val in zip(threshold_, logit_thresh):
            if isinstance(flag, bool):
                self.threshold.append(val if flag else None)
            else:
                self.threshold.append(flag)

        self.rounding = ensure_tuple_rep(rounding, len(self.keys))
        self.converter = AsDiscrete(channel_dim=channel_dim)

    def __call__(
        self,
        data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key, argmax, to_onehot, threshold, rounding in self.key_iterator(
            d, self.argmax, self.to_onehot, self.threshold, self.rounding
        ):
            d[key] = self.converter(d[key], argmax, to_onehot, threshold, rounding)
        return d
