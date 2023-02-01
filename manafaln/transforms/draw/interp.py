from typing import Sequence, Optional, Tuple, Literal, Union

import numpy as np
from scipy.interpolate import interp1d
from monai.config import KeysCollection
from monai.utils.enums import PostFix

from .draw import DrawPoints, DrawPointsd

DEFAULT_POST_FIX = PostFix.meta()

class Interpolate(DrawPoints):
    """
    Interpolate a sequence of (h, w) to a 2D mask,
    dilated with square(footprint).
    """
    def __init__(
        self,
        spatial_size: Sequence[int],
        apply_index: Optional[Sequence[int]]=None,
        channel_dim: int=0,
        footprint_shape: Optional[Literal['rectangle', 'disk']]='rectangle',
        footprint_size: Union[int, Tuple[int, int]] = 5,
        mask_only: bool=False,
    ):
        """
        Args:
            mask_keys (Sequence[str]): Keys from dict of pts to interpolate, and stack in order.
            spatial_size (Sequence[int], optional): Mask shape to be interpolated. Defaults to (512, 512).
            footprint (Optional[int], optional): Side length of square to dilate mask. Set None to skip dilation. Defaults to 5.
            channel_dim (int, optional): Dimension to stack masks. Defaults to 0.
        """
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
        ptss: Sequence[Sequence[Tuple[float, float]]]
    ) -> Sequence[Tuple[int, int]]:
        """
        Args:
            pts: A sequence of sequence of float points, [[(h, w), ...], ...]
        Returns:
            pts: A sequence of integer points, [(h, w), ...]
        """
        flatten_pts = []
        for pts in ptss:
            pts = self.rescale_size(pts)
            pts = self.remove_duplicate(pts)
            pts = self.interpolate(pts)
            flatten_pts.extend(pts)
        return flatten_pts

    def rescale_size(
        self,
        pts: Sequence[Tuple[float, float]]
    ) -> Sequence[Tuple[int, int]]:
        pts = [(int(p[0]*self.spatial_size[0]), int(p[1]*self.spatial_size[1])) for p in pts]
        return pts

    def remove_duplicate(
        self,
        pts: Sequence[Tuple[int, int]]
    ) -> Sequence[Tuple[int, int]]:
        res = [pts[0]]
        for i in range(1, len(pts)):
            if pts[i] != pts[i - 1]:
                res.append(pts[i])
        return res

    def interpolate(
        self,
        pts: Sequence[Tuple[int, int]]
    ) -> Sequence[Tuple[int, int]]:
        """
        Interpolate points to continuous coordinates.
        """
        pts = np.array(pts)
        l = np.linalg.norm(pts[1:] - pts[:-1], axis=-1)
        t = np.cumsum(l)
        t = np.concatenate([[0], t])
        t2 = np.arange(t[-1])
        h, w = pts[:, 0], pts[:, 1]
        h = interp1d(t, h, kind="linear" if len(pts) < 4 else "cubic",
                    bounds_error=False, fill_value="extrapolate")(t2)
        w = interp1d(t, w, kind="linear" if len(pts) < 4 else "cubic",
                    bounds_error=False, fill_value="extrapolate")(t2)
        pts = np.stack([h, w], axis=-1).astype(int)
        pts = np.clip(pts, 0, self.spatial_size-1)
        pts = pts.tolist()
        return pts

class Interpolated(DrawPointsd):
    def __init__(
        self,
        keys: KeysCollection,
        spatial_size: Sequence[int],
        apply_index: Optional[Sequence[int]]=None,
        channel_dim: int=0,
        footprint_shape: Optional[Literal['rectangle', 'disk']]='rectangle',
        footprint_size: Union[int, Tuple[int, int]] = 5,
        from_meta: bool=False,
        stack: bool=False,
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

        self.t = Interpolate(
            spatial_size=spatial_size,
            channel_dim=channel_dim,
            footprint_shape=footprint_shape,
            footprint_size=footprint_size,
            mask_only=False
            )
