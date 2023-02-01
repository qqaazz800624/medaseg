
from typing import Optional, Sequence, Dict, Union

import torch
import numpy as np
from monai.config import KeysCollection
from monai.utils.enums import PostFix
from monai.utils import ensure_tuple_rep
from monai.transforms import Transform, MapTransform, SaveImage, SplitDim

DEFAULT_POST_FIX = PostFix.meta()

class SaveSplitImage(Transform):
    """
    A wrapper transform of SplitDim and SaveImage
    to support multiple output_postfixes for different channel.

    Args:
        output_postfixes (Sequence[str]): Strings to be appended to output filenames.
        channel_dim (int): Dimension on which to split. Defaults to 0.
        reverse_indexing (bool): Whether to swap the first two spatial axes. Default to False.
        **kwargs: Other keyword arguments to be passed into SaveImage.
    """
    def __init__(self,
        output_postfixes: Sequence[str],
        channel_dim: int=0,
        reverse_indexing: bool=False,
        **kwargs
    ):
        self.splitter = SplitDim(dim=channel_dim, keepdim=False, update_meta=True)
        self.savers = [
            SaveImage(output_postfix=output_postfix, channel_dim=None, **kwargs)
            for output_postfix in output_postfixes
        ]
        self.set_options(
            write_kwargs={
                "reverse_indexing": reverse_indexing
                }
            )

    def set_options(self, **kwargs):
        for s in self.savers:
            s.set_options(**kwargs)

    def __call__(self, img: Union[torch.Tensor, np.ndarray], meta_data: Optional[Dict] = None):
        imgs = self.splitter(img)
        for saver, img_i in zip(self.savers, imgs):
            saver(img_i, meta_data)
        return img

class SaveSplitImaged(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        output_postfixes: Sequence[str],
        channel_dim: int=0,
        reverse_indexing: bool=False,
        meta_keys: Optional[KeysCollection] = None,
        meta_key_postfix: str = DEFAULT_POST_FIX,
        allow_missing_keys: bool = False,
        **kwargs
    ):
        super().__init__(keys, allow_missing_keys)
        self.meta_keys = ensure_tuple_rep(meta_keys, len(self.keys))
        self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))
        self.saver = SaveSplitImage(output_postfixes=output_postfixes, channel_dim=channel_dim, reverse_indexing=reverse_indexing, **kwargs)

    def __call__(self, data):
        d = dict(data)
        for key, meta_key, meta_key_postfix in self.key_iterator(d, self.meta_keys, self.meta_key_postfix):
            if meta_key is None and meta_key_postfix is not None:
                meta_key = f"{key}_{meta_key_postfix}"
            meta_data = d.get(meta_key) if meta_key is not None else None
            self.saver(img=d[key], meta_data=meta_data)
        return d
