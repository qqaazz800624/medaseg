from typing import Literal, Tuple

import torch
from monai.transforms import Transform, MapTransform
from monai.config import KeysCollection
from monai.utils.enums import TransformBackends

class Dilation(Transform):
    """
    Apply dilation to 2D input.

    Args:
        dilate (Tuple[int, int]): The size of the dilation kernel.
        mode (Literal["avg", "max"], optional): The type of dilation operation to apply. Defaults to "avg".
        batched (bool, optional): Whether to apply dilation to a batch of images. Defaults to False.

    Raises:
        NotImplementedError: If an unsupported mode is provided.

    """

    backend = [TransformBackends.TORCH]

    def __init__(self,
        dilate: Tuple[int, int],
        mode: Literal["avg", "max"]="avg",
        batched: bool=False
    ):
        self.batched = batched

        padding = [dilate[0] // 2, dilate[1] // 2]
        if mode == "avg":
            self.dilate = torch.nn.AvgPool2d(
                kernel_size=dilate,
                stride=[1, 1],
                padding=padding,
                ceil_mode=True,
            )
        elif mode == "max":
            self.dilate = torch.nn.MaxPool2d(
                kernel_size=dilate,
                stride=[1, 1],
                padding=padding,
                ceil_mode=True,
            )
        else:
            raise NotImplementedError

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """
        Apply dilation to the input tensor.

        Args:
            data (torch.Tensor): The input tensor to apply dilation to.

        Returns:
            torch.Tensor: The dilated tensor.

        """
        if not self.batched:
            data = data.unsqueeze(0)
        data = self.dilate(data)
        if not self.batched:
            data = data.squeeze(0)
        return data


class Dilationd(MapTransform):
    backend = Dilation.backend

    def __init__(
        self,
        keys: KeysCollection,
        dilate: Tuple[int, int],
        mode: Literal["avg", "max"]="avg",
        batched: bool=False,
        allow_missing_keys: bool = False
    ):
        super().__init__(keys, allow_missing_keys)
        self.t = Dilation(dilate, mode, batched)

    def __call__(
        self,
        data
    ):
        """
        Apply dilation to the input data.

        Args:
            data: The input data to apply dilation to.

        Returns:
            dict: The dilated data.

        """
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.t(d[key])
        return d
