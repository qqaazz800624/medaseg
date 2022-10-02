from typing import Dict, Hashable, List, Mapping, Optional, Union

import torch
import numpy as np
from monai.config import DtypeLike, KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.data.meta_obj import get_track_meta
from monai.transforms import Transform, MapTransform, RandomizableTransform
from monai.transforms.utils_pytorch_numpy_unification import clip, max, min
from monai.utils.enums import TransformBackends
from monai.utils.misc import ensure_tuple_rep
from monai.utils.type_conversion import convert_data_type, convert_to_tensor

class RandAdjustBrightnessAndContrast(RandomizableTransform):
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
        self,
        probs: Union[float, List[float]] = [0.15, 0.15],
        brightness_range: Optional[List[float]] = None,
        contrast_range: Optional[List[float]] = None,
        dtype: DtypeLike = np.float32
    ):
        probs = ensure_tuple_rep(probs, 2)

        if brightness_range is None:
            p = 0.0
        else:
            p = probs[0]
            if len(brightness_range) == 2:
                self.brightness = sorted(brightness_range)
            else:
                raise ValueError(
                    "Brightness range must be None or a list with length 2."
                )

        if contrast_range is None:
            q = 0.0
        else:
            q = probs[1]
            if len(contrast_range) == 2:
                self.contrast = sorted(contrast_range)
            else:
                raise ValueError(
                    "Contrast range must be None or a list with length 2."
                )

        prob = (p + q) - p * q
        RandomizableTransform.__init__(self, prob)

        if prob != 0.0:
            self.prob_b = p / prob
            self.prob_c = q / prob
        else:
            self.prob_b = 0.0
            self.prob_c = 0.0
        self._brightness = None
        self._contrast = None

        self.dtype = dtype

    def randomize(self) -> None:
        super().randomize(None)
        if not self._do_transform:
            return

        if self.R.rand() < self.prob_b:
            self._brightness = self.R.uniform(
                low=self.brightness[0],
                high=self.brightness[1]
            )
        else:
            self._brightness = None

        if self.R.rand() < self.prob_c:
            self._contrast = self.R.uniform(
                low=self.contrast[0],
                high=self.contrast[1]
            )
        else:
            self._contrast = None

    def __call__(
        self,
        img: NdarrayOrTensor,
        randomize: bool = True
    ) -> NdarrayOrTensor:
        if randomize:
            self.randomize()
        if not self._do_transform:
            return img

        img = convert_to_tensor(img, track_meta=get_track_meta())
        min_intensity = min(img)
        max_intensity = max(img)
        scale = 1.0

        if self._brightness:
            scale *= self._brightness
            min_intensity *= self._brightness
            max_intensity *= self._brightness

        if self._contrast:
            scale *= self._contrast

        img *= scale
        img = clip(img, min_intensity, max_intensity)

        ret: NdarrayOrTensor = convert_data_type(img, dtype=self.dtype)[0]
        return ret

class RandAdjustBrightnessAndContrastd(MapTransform, RandomizableTransform):
    backend = RandAdjustBrightnessAndContrast.backend

    def __init__(
        self,
        keys: KeysCollection,
        probs: Union[float, List[float]] = [0.15, 0.15],
        brightness_range: Optional[List[float]] = None,
        contrast_range: Optional[List[float]] = None,
        dtype: DtypeLike = np.float32
    ):
        MapTransform.__init__(self, keys)
        RandomizableTransform.__init__(self, 1.0)

        self.t = RandAdjustBrightnessAndContrast(
            probs, brightness_range, contrast_range, dtype
        )

    def randomize(self) -> None:
        self.t.randomize()

    def __call__(
        self,
        data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        self.randomize()

        if not self.t._do_transform:
            for key in self.keys:
                d[key] = convert_to_tensor(d[key], track_meta=get_track_meta())
            return d

        for key in self.keys:
            d[key] = self.t(d[key], randomize=False)
        return d

