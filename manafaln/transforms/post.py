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
        to_onehot: bool = False,
        num_classes: Optional[int] = None,
        threshold_values: bool = False,
        logit_thresh: float = 0.5,
        rounding: Optional[str] = None,
        n_classes: Optional[int] = None,
        channel_dim: Optional[int] = 0
    ):
        self.argmax = argmax
        self.to_onehot = to_onehot
        self.num_classes = num_classes
        self.threshold_values = threshold_values
        self.logit_thresh = logit_thresh
        self.rounding = rounding
        self.n_classes = n_classes
        self.channel_dim = channel_dim

    def __call__(
        self,
        img: NdarrayOrTensor,
        argmax: Optional[bool] = False,
        to_onehot: Optional[bool] = False,
        num_classes: Optional[int] = None,
        threshold_values: Optional[bool] = False,
        logit_thresh: Optional[float] = 0.5,
        rounding: Optional[str] = None,
        n_classes: Optional[int] = None,
        channel_dim : Optional[int] = None
    ) -> NdarrayOrTensor:

        img_t: torch.Tensor
        img_t, *_ = convert_data_type(img, torch.Tensor)
        if argmax or self.argmax:
            img_t = torch.argmax(img_t, dim=self.channel_dim, keepdim=True)

        if to_onehot or self.to_onehot:
            nc = self.num_classes if num_classes is None else num_classes
            if not isinstance(nc, int):
                raise AssertionError("num_classes must be an int")
            img_t = one_hot(img_t, num_classes=nc, dim=self.channel_dim)

        if threshold_values or self.threshold_values:
            th = self.logit_thresh if logit_thresh is None else logit_thresh
            img_t = img_t >= th

        rounding = self.rounding if rounding is None else rounding
        if rounding is not None:
            look_up_option(rounding, ["torchrounding"])
            img_t = torch.round(img_t)

        img, *_, = convert_to_dst_type(img_t, img, dtype=torch.float)
        return img

class AsDiscreted(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        argmax: Union[Sequence[bool], bool] = False,
        to_onehot: Union[Sequence[bool], bool] = False,
        num_classes: Optional[Union[Sequence[int], int]] = None,
        threshold_values: Union[Sequence[bool], bool] = False,
        logit_thresh: Union[Sequence[float], float] = 0.5,
        rounding: Union[Sequence[Optional[str]], Optional[str]] = None,
        allow_missing_keys: bool = False,
        channel_dim: int = 0
    ):
        super().__init__(keys, allow_missing_keys)

        n = len(keys)
        self.argmax = ensure_tuple_rep(argmax, n)
        self.to_onehot = ensure_tuple_rep(to_onehot, n)
        self.num_classes = ensure_tuple_rep(num_classes, n)
        self.threshold_values = ensure_tuple_rep(threshold_values, n)
        self.logit_thresh = ensure_tuple_rep(logit_thresh, n)
        self.rounding = ensure_tuple_rep(rounding, n)

        self.converter = AsDiscrete(channel_dim=channel_dim)

    def __call__(
        self,
        data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        iterator = self.key_iterator(
            d,
            self.argmax,
            self.to_onehot,
            self.num_classes,
            self.threshold_values,
            self.logit_thresh,
            self.rounding
        )
        for key, *args in iterator:
            d[key] = self.converter(d[key], *args)
        return d
