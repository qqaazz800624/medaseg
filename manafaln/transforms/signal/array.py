from typing import Sequence

import torch
import numpy as np
import scipy.signal as S
from monai.config import DtypeLike, NdarrayOrTensor
from monai.transforms.transform import RandomizableTransform, Transform
from monai.utils.enums import TransformBackends
from monai.utils.type_conversion import (
    convert_to_numpy,
    convert_to_tensor
)
from scipy.interpolate import CubicSpline, interp1d


class ImputeEmptySignal(Transform):
    def __init__(self, ch_axis: int = 0, dtype: DtypeLike = np.float32):
        super().__init__()

        self.ch_axis = ch_axis
        self.dtype = dtype

    @staticmethod
    def impute(x: NdarrayOrTensor) -> NdarrayOrTensor:
        mask = np.isnan(x)
        x[mask] = np.interp(
            np.flatnonzero(mask),
            np.flatnonzero(~mask),
            x[~mask]
        )
        return x

    def __call__(self, x: NdarrayOrTensor) -> NdarrayOrTensor:
        signal = convert_to_numpy(x)
        signal = np.apply_along_axis(self.impute, self.ch_axis, signal)
        return signal.astype(self.dtype)


class ResampleSignal(Transform):
    backend = [TransformBackends.NUMPY, TransformBackends.TORCH]

    def __init__(
        self,
        samples: int,
        axis: int = 1,
        domain: str = "time",
        dtype: DtypeLike = np.float32
    ):
        super().__init__()

        self.samples = samples
        self.axis = axis
        self.domain = domain
        self.dtype = dtype

    def __call__(self, x: NdarrayOrTensor) -> NdarrayOrTensor:
        signal = convert_to_numpy(x)
        signal = S.resample(
            signal, num=self.samples, axis=self.axis, domain=self.domain
        )
        return signal.astype(self.dtype)


class WienerFiltering(Transform):
    backend = [TransformBackends.NUMPY, TransformBackends.TORCH]

    def __init__(
        self,
        ch_axis: int = 0,
        mysize: int = 5,
        noise: float = None,
        dtype: DtypeLike = np.float32
    ):
        super().__init__()

        self.ch_axis = ch_axis
        self.mysize = mysize
        self.noise = noise
        self.dtype = dtype

    def __call__(self, x: NdarrayOrTensor) -> NdarrayOrTensor:
        signal = convert_to_numpy(x)
        with np.errstate(all="ignore"):
            signal = np.apply_along_axis(
                S.wiener,
                self.ch_axis,
                signal,
                mysize=self.mysize,
                noise=self.noise
            )
        return signal.astype(self.dtype)


class NormalizeSignal(Transform):
    backend = [TransformBackends.NUMPY, TransformBackends.TORCH]

    def __init__(
        self,
        ch_axis: int = 0,
        percentile: Sequence[int] = (5, 95),
        dtype: DtypeLike = np.float32
    ):
        super().__init__()

        self.ch_axis = ch_axis
        self.q = percentile
        self.dtype = dtype

    def normalize(self, signal: np.array) -> np.array:
        p_lower, p_upper = np.percentile(signal, self.q)
        signal = np.clip(signal, p_lower, p_upper)
        mean = np.mean(signal)
        std = max(np.std(signal), 0.1)
        return (signal - mean) / std

    def __call__(self, x: NdarrayOrTensor) -> NdarrayOrTensor:
        signal = convert_to_numpy(x)
        signal = np.apply_along_axis(self.normalize, self.ch_axis, signal)
        return signal.astype(self.dtype)


class MedianNormalizeSignal(Transform):
    backend = [TransformBackends.NUMPY, TransformBackends.TORCH]

    def __init__(
        self,
        ch_axis: int = 0,
        dtype: DtypeLike = np.float32
    ):
        super().__init__()

        self.ch_axis = ch_axis
        self.dtype = dtype

    @staticmethod
    def normalize(signal: np.array) -> np.array:
        m = np.median(signal)
        return signal - m

    def __call__(self, x: NdarrayOrTensor) -> NdarrayOrTensor:
        signal = convert_to_numpy(x)
        signal = np.apply_along_axis(self.normalize, self.ch_axis, signal)
        return signal.astype(self.dtype)


class BaselineWanderRemoval(Transform):
    backend = [TransformBackends.NUMPY, TransformBackends.TORCH]

    def __init__(
        self,
        freq: int,
        ch_axis: int = 0,
        dtype: DtypeLike = np.float32
    ):
        super().__init__()

        self.freq = freq
        self.ch_axis = ch_axis
        self.dtype = dtype

    def __call__(self, x: NdarrayOrTensor) -> NdarrayOrTensor:
        signal = convert_to_numpy(x)
        # Apply high-pass Butterworth filter
        butter_sos = S.butter(N=1, Wn=1, btype="hp", fs=self.freq, output="sos")
        signal = S.sosfilt(butter_sos, signal, axis=self.ch_axis)
        return signal.astype(self.dtype)


class RandButterworth(RandomizableTransform):
    backend = [TransformBackends.NUMPY, TransformBackends.TORCH]

    def __init__(
        self,
        freq: float,
        ch_axis: int = 0,
        prob: float = 0.1,
        dtype: DtypeLike = np.float32
    ):
        RandomizableTransform.__init__(self, prob)

        self.freq = freq
        self.ch_axis = ch_axis
        self.dtype = dtype

    def __call__(self, x: NdarrayOrTensor) -> NdarrayOrTensor:
        self.randomize(None)
        if self._do_transform:
            x = convert_to_numpy(x)
            butter_sos = S.butter(N=2, Wn=1, btype="hp", fs=self.freq, output="sos")
            x = S.sosfilt(butter_sos, x, axis=self.ch_axis)
        return x.astype(self.dtype)


class RandSignalGaussianNoise(RandomizableTransform):
    backend = [TransformBackends.NUMPY, TransformBackends.TORCH]

    def __init__(
        self,
        boundaries: Sequence[float] = (0.001, 0.02),
        prob: float = 1.0
    ):
        RandomizableTransform.__init__(self, prob)

        if len(boundaries) != 2:
            raise ValueError("boundaries must be a sequence of two elements.")
        self.low = min(boundaries)
        self.high = max(boundaries)

    def __call__(self, x: NdarrayOrTensor) -> NdarrayOrTensor:
        self.randomize(None)
        if self._do_transform:
            x = convert_to_tensor(x)
            ch, length = x.shape
            for c in range(ch):
                magnitude = self.R.uniform(low=self.low, high=self.high)
                noise = magnitude * torch.randn(length)
                x[c, :] += noise
        return x


class RandZeroOut(RandomizableTransform):
    backend = [TransformBackends.NUMPY, TransformBackends.TORCH]

    def __init__(self, max_length: int, prob: float = 0.1):
        RandomizableTransform.__init__(self, prob)

        self.max_length = max_length

    def __call__(self, x: NdarrayOrTensor) -> NdarrayOrTensor:
        self.randomize(None)
        if self._do_transform:
            ch, length = x.shape
            max_length = min(length // 10, self.max_length)
            start = self.R.randint(0, length - 1)
            end = min(start + self.R.randint(1, max_length), length - 1)
            x[:, start:end] = 0
        return x


class RandShuffle(RandomizableTransform):
    backend = [TransformBackends.NUMPY, TransformBackends.TORCH]

    def __init__(self, prob: float = 0.1):
        RandomizableTransform.__init__(self, prob)

    def __call__(self, x: NdarrayOrTensor) -> NdarrayOrTensor:
        self.randomize(None)
        if self._do_transform:
            p = self.R.permutation(x.shape[0])
            x = x[p, :]
        return x


class RandJitter(RandomizableTransform):
    backend = [TransformBackends.NUMPY, TransformBackends.TORCH]

    # Reference: Data Augmentation of Wearable Sensor Data for Parkinson’s
    #            Disease Monitoring using Convolutional Neural Networks∗
    # Link to paper: https://arxiv.org/pdf/1706.00527.pdf
    def __init__(self, sigma: float = 0.3, prob: float = 0.1):
        RandomizableTransform.__init__(self, prob)

        self.sigma = sigma

    def __call__(self, x: NdarrayOrTensor) -> NdarrayOrTensor:
        self.randomize(None)
        if self._do_transform:
            x += self.R.normal(loc=0.0, scale=self.sigma, size=x.shape)
        return x


class RandScalingSignal(RandomizableTransform):
    backend = [TransformBackends.NUMPY, TransformBackends.TORCH]

    def __init__(self, sigma: float = 1.1, prob: float = 0.1):
        RandomizableTransform.__init__(self, prob)

        self.sigma = sigma

    def __call__(self, x: NdarrayOrTensor) -> NdarrayOrTensor:
        self.randomize(None)
        if self._do_transform:
            ch, length = x.shape
            factor = self.R.normal(loc=2.0, scale=self.sigma, size=(ch, 1))
            x *= factor
        return x


class RandNegateSignal(RandomizableTransform):
    backend = [TransformBackends.NUMPY, TransformBackends.TORCH]

    def __init__(self, prob: float = 0.5):
        RandomizableTransform.__init__(self, prob)

    def __call__(self, x: NdarrayOrTensor) -> NdarrayOrTensor:
        self.randomize(None)
        if self._do_transform:
            x *= -1.0
        return x


class RandResampleSignal(RandomizableTransform):
    backend = [TransformBackends.NUMPY, TransformBackends.TORCH]

    def __init__(self, prob: float = 0.3):
        RandomizableTransform.__init__(self, prob)

    def __call__(self, x: NdarrayOrTensor) -> NdarrayOrTensor:
        self.randomize(None)
        if self._do_transform:
            x = convert_to_numpy(x)

            orig_steps = np.arange(x.shape[1])
            intp_steps = np.arange(0, orig_steps[-1] + 0.001, 1 / 3)

            intp = interp1d(orig_steps, x, axis=1)
            intp_vals = intp(intp_steps)

            start = self.R.choice(orig_steps)
            resample_index = np.arange(start, 3 * x.shape[1], 2)[:x.shape[1]]
            x = intp_vals[:, resample_index, :]
        return x


class RandTimeWarping(RandomizableTransform):
    backend = [TransformBackends.NUMPY, TransformBackends.TORCH]

    def __init__(
        self,
        sigma: float = 0.2,
        num_knots: int = 4,
        prob: float = 0.3
    ):
        RandomizableTransform.__init__(self, prob)

        self.sigma = sigma
        self.num_knots = num_knots

    @staticmethod
    def generate_spline(
        x: np.ndarray,
        x_data: np.ndarray,
        y_data: np.array
    ) -> np.ndarray:
        cubic_spline = CubicSpline(x_data, y_data, axis=1)
        return cubic_spline(x)

    def warp(self, x: np.array) -> np.array:
        ch, length = x.shape

        time_stamps = np.arange(length)
        knot_xs = np.arange(0, self.num_knots + 2, dtype=np.float32) * (length - 1) / (self.num_knots + 1)
        spline_ys = np.random.normal(loc=1.0, scale=self.sigma, size=(ch, self.num_knots + 2))

        spline_vals = self.generate_spline(time_stamps, knot_xs, spline_ys)
        cumulative_sum = np.cumsum(spline_vals, axis=1)
        distorted_time_stamps = cumulative_sum / cumulative_sum[:, -1][:, np.newaxis] * (length - 1)

        x_out = np.empty_like(x)
        for c in range(ch):
            x_out[c] = np.interp(distorted_time_stamps[c], time_stamps, x_out[c])
        return x

    def __call__(self, x: NdarrayOrTensor) -> NdarrayOrTensor:
        self.randomize(None)
        if self._do_transform:
            x = convert_to_numpy(x)
            x = self.warp(x)
        return x


class RandIFFTPhaseShift(RandomizableTransform):
    """
    A transform that applies a random phase shift to the input signal using the Inverse Fast Fourier Transform (IFFT).

    Args:
        prob (float): The probability of applying the transform. Default is 0.5.

    Returns:
        NdarrayOrTensor: The transformed input signal.
    """

    backend = [TransformBackends.NUMPY, TransformBackends.TORCH]

    def __init__(self, prob: float = 0.5):
        RandomizableTransform.__init__(self, prob)

    def __call__(self, x: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Apply a random phase shift to the input signal using the Inverse Fast Fourier Transform (IFFT).

        Args:
            x (NdarrayOrTensor): The input signal to be transformed.

        Returns:
            NdarrayOrTensor: The transformed input signal.
        """
        self.randomize(None)
        if self._do_transform:
            x = convert_to_tensor(x)
            ch, length = x.shape

            fft = torch.fft.fftn(x, dim=1)
            fd = torch.fft.fftshift(fft)

            amp = fd.abs()
            phase = fd.angle()

            # Generate random angles for each channel, then apply to original phase
            rand_angle = self.R.uniform(low=-np.pi, high=np.pi, size=(ch, 1))
            rand_angle = np.repeat(rand_angle, length, axis=1)
            phase += rand_angle

            # Apply perturbation to amp
            amp += self.R.normal(loc=0.0, scale=0.8, size=amp.shape)

            cmp = amp * torch.exp(1j * phase)
            ifft = torch.fft.ifftn(
                torch.fft.ifftshift(cmp),
                dim=1
            )
            x = torch.real(ifft)
        return x
