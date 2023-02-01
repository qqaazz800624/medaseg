from typing import Sequence, Optional, Tuple, Literal, Union

import numpy as np
from skimage.measure import label
from monai.config import KeysCollection
from monai.utils.enums import PostFix

from .draw import DrawPoints, DrawPointsd

DEFAULT_POST_FIX = PostFix.meta()

class DrawLowest(DrawPoints):
    def __init__(
        self,
        spatial_size: Sequence[int],
        apply_index: Optional[Sequence[int]]=None,
        channel_dim: int = 0,
        footprint_shape: Optional[Literal['rectangle', 'disk']]='disk',
        footprint_size: Union[int, Tuple[int, int]] = 10,
        mask_only: bool=False,
    ):
        super().__init__(
            spatial_size=spatial_size,
            apply_index=apply_index,
            channel_dim=channel_dim,
            footprint_shape=footprint_shape,
            footprint_size=footprint_size,
            mask_only=mask_only
        )

    def extract_points(
        self,
        pts: Sequence[Tuple[float, float]]
    ) -> Sequence[Tuple[int, int]]:
        if len(pts)==0:
            return []
        lowest_pt = max(pts, key=lambda x: x[0])
        return [lowest_pt]

class DrawLowestd(DrawPointsd):
    def __init__(
        self,
        keys: KeysCollection,
        spatial_size: Sequence[int],
        apply_index: Optional[Sequence[int]]=None,
        channel_dim: int = 0,
        footprint_shape: Optional[Literal['rectangle', 'disk']]='disk',
        footprint_size: Union[int, Tuple[int, int]] = 10,
        from_meta: bool=True,
        stack: bool=True,
        meta_key_postfix: str=DEFAULT_POST_FIX,
        allow_missing_keys: bool=False
    ):
        super().__init__(
            keys=keys,
            spatial_size=spatial_size,
            apply_index=apply_index,
            channel_dim=channel_dim,
            from_meta=from_meta,
            stack=stack,
            meta_key_postfix=meta_key_postfix,
            allow_missing_keys=allow_missing_keys
        )
        self.t = DrawLowest(
            spatial_size=spatial_size,
            apply_index=apply_index,
            channel_dim=channel_dim,
            footprint_shape=footprint_shape,
            footprint_size=footprint_size,
            mask_only=False
        )

class DrawLast(DrawPoints):
    def __init__(
        self,
        spatial_size: Sequence[int],
        apply_index: Optional[Sequence[int]]=None,
        channel_dim: int = 0,
        footprint_shape: Optional[Literal['rectangle', 'disk']]='disk',
        footprint_size: Union[int, Tuple[int, int]] = 10,
        mask_only: bool=False,
    ):
        super().__init__(
            spatial_size=spatial_size,
            apply_index=apply_index,
            channel_dim=channel_dim,
            footprint_shape=footprint_shape,
            footprint_size=footprint_size,
            mask_only=mask_only
        )

    def extract_points(
        self,
        pts: Sequence[Tuple[float, float]]
    ) -> Sequence[Tuple[int, int]]:
        if len(pts)==0:
            return []
        last_pt = pts[-1]
        return [last_pt]

class DrawLastd(DrawPointsd):
    def __init__(
        self,
        keys: KeysCollection,
        spatial_size: Sequence[int],
        apply_index: Optional[Sequence[int]]=None,
        channel_dim: int = 0,
        footprint_shape: Optional[Literal['rectangle', 'disk']]='disk',
        footprint_size: Union[int, Tuple[int, int]] = 10,
        from_meta: bool=True,
        stack: bool=True,
        meta_key_postfix: str=DEFAULT_POST_FIX,
        allow_missing_keys: bool=False
    ):
        super().__init__(
            keys=keys,
            spatial_size=spatial_size,
            apply_index=apply_index,
            channel_dim=channel_dim,
            from_meta=from_meta,
            stack=stack,
            meta_key_postfix=meta_key_postfix,
            allow_missing_keys=allow_missing_keys
        )
        self.t = DrawLast(
            spatial_size=spatial_size,
            apply_index=apply_index,
            channel_dim=channel_dim,
            footprint_shape=footprint_shape,
            footprint_size=footprint_size,
            mask_only=False
        )

class DrawBottom(DrawPoints):
    def __init__(
        self,
        spatial_size: Sequence[int],
        apply_index: Optional[Sequence[int]]=None,
        channel_dim: int = 0,
        footprint_shape: Optional[Literal['rectangle', 'disk']]='disk',
        footprint_size: Union[int, Tuple[int, int]] = 10,
        mask_only: bool=False,
    ):
        super().__init__(
            spatial_size=spatial_size,
            apply_index=apply_index,
            channel_dim=channel_dim,
            footprint_shape=footprint_shape,
            footprint_size=footprint_size,
            mask_only=mask_only
        )

    def extract_points(
        self,
        mask: np.ndarray
    ) -> Sequence[Tuple[int, int]]:
        # Find bottom edge
        edge = np.zeros(mask.shape)
        for h in range(mask.shape[0]-1):
            for w in range(mask.shape[1]):
                if mask[h+1, w] == 0 and mask[h, w] == 1:
                    edge[h, w] = 1

        # Find largest connected component
        labels = label(edge)
        edge = labels == np.argmax(np.bincount(labels.flat,  weights=edge.flat))

        # Find horizontal midpoint of edge
        w = int(edge.sum(0).nonzero()[0].mean())

        # Find the corresponding point in mask
        for h in range(mask.shape[0]-1, -1, -1):
            if mask[h, w] == 1:
                return [(h, w)]

        # If not found
        return []

class DrawBottomd(DrawPointsd):
    def __init__(
        self,
        keys: KeysCollection,
        spatial_size: Sequence[int],
        apply_index: Optional[Sequence[int]]=None,
        channel_dim: int = 0,
        footprint_shape: Optional[Literal['rectangle', 'disk']]='disk',
        footprint_size: Union[int, Tuple[int, int]] = 10,
        from_meta: bool=False,
        stack: bool=True,
        meta_key_postfix: str=DEFAULT_POST_FIX,
        allow_missing_keys: bool=False
    ):
        super().__init__(
            keys=keys,
            spatial_size=spatial_size,
            apply_index=apply_index,
            channel_dim=channel_dim,
            from_meta=from_meta,
            stack=stack,
            meta_key_postfix=meta_key_postfix,
            allow_missing_keys=allow_missing_keys
        )
        self.t = DrawBottom(
            spatial_size=spatial_size,
            apply_index=apply_index,
            channel_dim=channel_dim,
            footprint_shape=footprint_shape,
            footprint_size=footprint_size,
            mask_only=False
        )
