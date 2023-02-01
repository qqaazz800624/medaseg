import itertools
from typing import Dict, Hashable, Mapping, Sequence

import numpy as np
from skimage.exposure import equalize_adapthist
from monai.config import KeysCollection
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms import Transform, MapTransform
from monai.utils.enums import TransformBackends

class CLAHE(Transform):
    """
    Perform CLAHE on image with params kernel_size * clip_limit, then stack into
    array with shape (|kernel_size| * |clip_limit|, H, W), depends which dim to stack
    """
    backend = [TransformBackends.NUMPY]

    def __init__(self,
        kernel_sizes: Sequence[float]=[0.02, 0.05, 0.10],
        clip_limits:  Sequence[float]=[0.05, 0.1 , 0.5 ],
        dim: int=0,
    ):
        """
        Args:
            kernel_sizes (Sequence[float]): Kernel sizes given in ratio of img. Defaults to [0.02, 0.05, 0.10].
            clip_limits (Sequence[float]): Clipping limits. Defaults to [0.05, 0.1, 0.5].
            dim (int): Dimension to stack transformed imgs. Defaults to 0.
        """
        self.params = list(itertools.product(kernel_sizes, clip_limits))
        self.dim = dim

    def __call__(self,
            img: np.ndarray,    # shape: (H, W)
        ) -> np.ndarray:        # shape: (C, H, W), depends on which dim to stack
        result = []
        img = img.astype(float)
        for params in self.params:
            tmp = self.clahe(img, params[0], params[1])
            result.append(tmp)
        # Stack with channel self.dim
        result =  np.stack(result, axis=self.dim)
        return result

    @staticmethod
    def clahe(img: np.ndarray, kernel_size: float=0.05, clip_limit: float=0.5):
        """
        Contrast Limited Adaptive Histogram Equalization (CLAHE).

        An algorithm for local contrast enhancement, that uses histograms computed over
        different tile regions of the image. Local details can therefore be enhanced even
        in regions that are darker or lighter than most of the image.

        Args:
            img (np.ndarray): Input 2D image with shape (H, W) or (1, H, W)
            kernel_size (float): Defines the shape of contextual regions used in the algorithm,
                given in ratio of img.
            clip_limit (float): Clipping limit, higher values give more contrast.

        Returns:
            img (np.ndarray): Equalized image, normalized between 0 to 1, shape (H, W)

        References:
        [1] K. Zuiderveld. (1994). Contrast Limited Adaptive Histogram Equalization.
        """

        assert (len(img.shape) == 2) or (len(img.shape)==3 and (img.shape[0]==1)), \
            f"Input img needs to be 2D and have no channel or single channel at first dimension, got {img.shape}"

        img = img.squeeze(0)

        img = (img - img.min()) / (img.max() - img.min())
        kernel_size = np.array(img.shape) * kernel_size // 1
        img = equalize_adapthist(
            image=img,
            kernel_size=kernel_size,
            clip_limit=clip_limit
            )
        img = (img - img.min()) / (img.max() - img.min())
        return img

class CLAHEd(MapTransform):
    backend = CLAHE.backend

    def __init__(
        self,
        keys: KeysCollection,
        kernel_sizes: Sequence[float]=[0.02, 0.05, 0.10],
        clip_limits:  Sequence[float]=[0.05, 0.1 , 0.5 ],
        dim: int=0,
        allow_missing_keys: bool=False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.t = CLAHE(kernel_sizes, clip_limits, dim)

    def __call__(
        self,
        data: Mapping[Hashable, NdarrayOrTensor]
    ) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.t(d[key])
        return d
