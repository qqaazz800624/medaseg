from typing import Dict, Hashable, Mapping, Sequence

import numpy as np
from monai.config import DtypeLike, KeysCollection, NdarrayOrTensor
from monai.transforms import MapTransform, RandomizableTransform

from manafaln.transforms.signal.array import (
    ImputeEmptySignal,
    ResampleSignal,
    WienerFiltering,
    NormalizeSignal,
    MedianNormalizeSignal,
    BaselineWanderRemoval,
    RandButterworth,
    RandSignalGaussianNoise,
    RandZeroOut,
    RandShuffle,
    RandJitter,
    RandScalingSignal,
    RandNegateSignal,
    RandResampleSignal,
    RandTimeWarping,
    RandIFFTPhaseShift
)

__all__ = [
    "ImputeEmptySignald",
    "ResampleSignald",
    "WienerFilteringd",
    "NormalizeSignald",
    "MedianNormalizeSignald",
    "BaselineWanderRemovald",
    "RandButterworthd",
    "RandSignalGaussianNoised",
    "RandZeroOutd",
    "RandShuffled",
    "RandJitterd",
    "RandScalingSignald",
    "RandNegateSignald",
    "RandResampleSignald",
    "RandTimeWarpingd",
    "RandIFFTPhaseShiftd"
]

class ImputeEmptySignald(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        ch_axis: int = 0,
        dtype: DtypeLike = np.float32,
        allow_missing_keys: bool = False
    ):
        MapTransform.__init__(self, keys, allow_missing_keys)

        self.transform = ImputeEmptySignal(ch_axis=ch_axis, dtype=dtype)

    def __call__(
        self,
        data: Mapping[Hashable, NdarrayOrTensor]
    ) -> dict[Hashable, NdarrayOrTensor]:
        d: dict = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.transform(d[key])
        return d

class ResampleSignald(MapTransform):
    """ Dictionary version of ResampleSignal """
    def __init__(
        self,
        keys: KeysCollection,
        samples: int,
        axis: int = 1,
        domain: str = "time",
        dtype: DtypeLike = np.float32,
        allow_missing_keys: bool = False
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        self.transform = ResampleSignal(
            samples, axis=axis, domain=domain, dtype=dtype
        )

    def __call__(
        self,
        data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.transform(d[key])
        return d


class WienerFilteringd(MapTransform):
    """ Dictionary version of WienerFiltering """
    def __init__(
        self, keys: KeysCollection,
        ch_axis: int = 0,
        mysize: int = 5,
        noise: float = None,
        dtype: DtypeLike = np.float32,
        allow_missing_keys: bool = False
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)

        self.transform = WienerFiltering(
            ch_axis=ch_axis, mysize=mysize, noise=noise, dtype=dtype
        )

    def __call__(
        self,
        data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        d: dict = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.transform(d[key])
        return d


class NormalizeSignald(MapTransform):
    """ Dictionary version of NormalizeSignal """

    def __init__(
        self,
        keys: KeysCollection,
        ch_axis: int = 0,
        percentile: Sequence[int] = (5, 95),
        dtype: DtypeLike = np.float32,
        allow_missing_keys: bool = False
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)

        self.transform = NormalizeSignal(
            ch_axis=ch_axis,
            percentile=percentile,
            dtype=dtype
        )

    def __call__(
        self,
        data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        d: dict = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.transform(d[key])
        return d


class MedianNormalizeSignald(MapTransform):
    """ Dictionary version of MedianNormalizeSignal """

    def __init__(
        self,
        keys: KeysCollection,
        ch_axis: int = 0,
        dtype: DtypeLike = np.float32,
        allow_missing_keys: bool = False
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)

        self.transform = MedianNormalizeSignal(ch_axis=ch_axis, dtype=dtype)

    def __call__(
        self,
        data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        d: dict = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.transform(d[key])
        return d


class BaselineWanderRemovald(MapTransform):
    """ Dictionary version of BaselineWanderRemoval """

    def __init__(
        self,
        keys: KeysCollection,
        freq: int,
        ch_axis: int = 0,
        dtype: DtypeLike = np.float32,
        allow_missing_keys: bool = False
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)

        self.transform = BaselineWanderRemoval(
            freq=freq, ch_axis=ch_axis, dtype=dtype
        )

    def __call__(
        self,
        data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        d: dict = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.transform(d[key])
        return d


class RandButterworthd(MapTransform, RandomizableTransform):
    """ Dictionary version of RandButterworth """

    def __init__(
        self,
        keys: KeysCollection,
        freq: float,
        ch_axis: int = 0,
        prob: float = 0.3,
        dtype: DtypeLike = np.float32,
        allow_missing_keys: bool = False
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)

        self.transform = RandButterworth(
            freq, ch_axis=ch_axis, prob=1.0, dtype=dtype
        )

    def __call__(
        self,
        data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        d: dict = dict(data)
        self.randomize(None)
        if self._do_transform:
            for key in self.key_iterator(d):
                d[key] = self.transform(d[key])
        return d


class RandSignalGaussianNoised(MapTransform, RandomizableTransform):
    """ Dictionary version of `RandSignalGaussianNoise` """

    def __init__(
        self,
        keys: KeysCollection,
        boundaries: Sequence[float] = (0.001, 0.02),
        prob: float = 1.0,
        allow_missing_keys: bool = False
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)

        self.transform = RandSignalGaussianNoise(
            boundaries=boundaries,
            prob=1.0
        )

    def __call__(
        self,
        data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        d: dict = dict(data)
        self.randomize(None)
        if self._do_transform:
            for key in self.key_iterator(d):
                d[key] = self.transform(d[key])
        return d


class RandZeroOutd(MapTransform, RandomizableTransform):
    """ Dictionary version of RandZeroOut """

    def __init__(
        self,
        keys: KeysCollection,
        max_length: int,
        prob: float = 0.1,
        allow_missing_keys: bool = False
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)

        self.transform = RandZeroOut(max_length, prob=1.0)

    def __call__(
        self,
        data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        d: dict = dict(data)
        self.randomize(None)
        if self._do_transform:
            for key in self.key_iterator(d):
                d[key] = self.transform(d[key])
        return d


class RandShuffled(MapTransform, RandomizableTransform):
    """ Dictionary version of RandShuffle """

    def __init__(
        self,
        keys: KeysCollection,
        prob: float = 0.1,
        allow_missing_keys: bool = False
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)

        self.transform = RandShuffle(prob=1.0)

    def __call__(
        self,
        data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        d: dict = dict(data)
        self.randomize(None)
        if self._do_transform:
            for key in self.key_iterator(d):
                d[key] = self.transform(d[key])
        return d


class RandJitterd(MapTransform, RandomizableTransform):
    """ Dictionary version of RandJitter """

    def __init__(
        self,
        keys: KeysCollection,
        sigma: float = 0.3,
        prob: float = 0.1,
        allow_missing_keys: bool = False
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)

        self.transform = RandJitter(sigma=sigma, prob=1.0)
    def __call__(
        self,
        data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        d: dict = dict(data)
        self.randomize(None)
        if self._do_transform:
            for key in self.key_iterator(d):
                d[key] = self.transform(d[key])
        return d


class RandScalingSignald(MapTransform, RandomizableTransform):
    """ Dictionary version of RandScalingSignal """

    def __init__(
        self,
        keys: KeysCollection,
        sigma: float = 1.1,
        prob: float = 0.1,
        allow_missing_keys: bool = False
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)

        self.transform = RandScalingSignal(sigma=sigma, prob=1.0)

    def __call__(
            self,
            data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        d: dict = dict(data)
        self.randomize(None)
        if self._do_transform:
            for key in self.key_iterator(d):
                d[key] = self.transform(d[key])
        return d

class RandNegateSignald(MapTransform, RandomizableTransform):
    """ Dictionary version of RandNegateSignal """
    def __init__(
        self,
        keys: KeysCollection,
        prob: float = 0.5,
        allow_missing_keys: bool = False
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)

        self.transform = RandNegateSignal(prob=1.0)

    def __call__(
        self,
        data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        d: dict = dict(data)
        self.randomize(None)
        if self._do_transform:
            for key in self.key_iterator(d):
                d[key] = self.transform(d[key])
        return d

class RandResampleSignald(MapTransform, RandomizableTransform):
    """ Dictionary version of RandResampleSignal """
    def __init__(
        self,
        keys: KeysCollection,
        prob: float = 0.1,
        allow_missing_keys: bool = False
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)

        self.transform = RandResampleSignal(prob=1.0)

    def __call__(
        self,
        data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        d: dict = dict(data)
        self.randomize(None)
        if self._do_transform:
            for key in self.key_iterator(d):
                d[key] = self.transform(d[key])
        return d

class RandTimeWarpingd(MapTransform, RandomizableTransform):
    """ Dictionary version of RandTimeWarping """
    def __init__(
        self,
        keys: KeysCollection,
        sigma: float = 0.2,
        num_knots: int = 4,
        prob: float = 0.3,
        allow_missing_keys: bool = False
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)

        self.transform = RandTimeWarping(
            sigma=sigma, num_knots=num_knots, prob=1.0
        )

    def __call__(
        self,
        data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        d: dict = dict(data)
        self.randomize(None)
        if self._do_transform:
            for key in self.key_iterator(d):
                d[key] = self.transform(d[key])
        return d

class RandIFFTPhaseShiftd(MapTransform, RandomizableTransform):
    """ Dictionary version of RandIFFTPhaseShift """
    def __init__(
        self,
        keys: KeysCollection,
        angle: float = np.pi,
        scale: float = 0.8,
        prob: float = 0.1,
        allow_missing_keys: bool = False
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)

        self.transform = RandIFFTPhaseShift(
            angle=angle, scale=scale, prob=1.0
        )

    def __call__(
        self,
        data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        d: dict = dict(data)
        self.randomize(None)
        if self._do_transform:
            for key in self.key_iterator(d):
                d[key] = self.transform(d[key])
        return d
