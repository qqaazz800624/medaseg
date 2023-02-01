from typing import Any, Dict, Hashable, Mapping, Tuple

import numpy as np
import pydicom
from monai.config import KeysCollection, PathLike
from monai.transforms import MapTransform, Transform
from monai.utils import MetaKeys, PostFix, SpaceKeys
from monai.utils.misc import ImageMetaKey
from skimage import color

DEFAULT_POST_FIX = PostFix.meta()
SPACING_KEY = "spacing"

def read_dicom_2d(
        dcm_path: PathLike,
        voi_lut: bool = False,
        ensure_gray: bool = True,
        range_correct: bool = False,
        normalization: bool = True,
        apply_window: bool = False,
        fix_monochrome: bool = True,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:

    dicom = pydicom.read_file(dcm_path)
    # For ignoring the UserWarning: "Bits Stored" value (14-bit)...
    # dicom.add_new((0x0028, 0x0101), "US", 16)
    elem = dicom[0x0028, 0x0101]
    elem.value = 16

    # VOI LUT (if available by DICOM device) is used to transform raw DICOM
    # data to "human-friendly" view
    if voi_lut:
        data = pydicom.pixel_data_handlers.util.apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array

    if ensure_gray:
        if len(data.shape)==3:
            data = color.rgb2gray(data)

    if range_correct:
        median = np.median(data)
        if dicom.PhotometricInterpretation == "MONOCHROME1":
            data = np.where(data == data.min(), median, data)
        else:
            data = np.where(data == data.max(), median, data)

    if normalization:
        if apply_window and "WindowCenter" in dicom and "WindowWidth" in dicom:
            window_center = float(dicom.WindowCenter)
            window_width = float(dicom.WindowWidth)
            y_min = (window_center - 0.5 * window_width)
            y_max = (window_center + 0.5 * window_width)
        else:
            y_min = data.min()
            y_max = data.max()
        data = (data - y_min) / (y_max - y_min)
        data = np.clip(data, 0, 1)

    # depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data

    # Meta data
    meta_data = {}
    meta_data[MetaKeys.SPATIAL_SHAPE] = np.asarray(data.shape)
    meta_data[MetaKeys.SPACE] = SpaceKeys.RAS
    meta_data[MetaKeys.ORIGINAL_AFFINE] = np.eye(3, dtype=float)
    meta_data[MetaKeys.AFFINE] = np.eye(3, dtype=float)
    meta_data[MetaKeys.ORIGINAL_CHANNEL_DIM] = 'no_channel'
    meta_data[SPACING_KEY] = np.asarray(getattr(dicom, 'PixelSpacing', [1., 1.]))
    meta_data[ImageMetaKey.FILENAME_OR_OBJ] = dcm_path

    return data, meta_data


class LoadDicom2D(Transform):
    def __init__(
        self,
        image_only: bool = False,
        voi_lut: bool = False,
        ensure_gray: bool = True,
        range_correct: bool = False,
        normalization: bool = True,
        apply_window: bool = False,
        fix_monochrome: bool = True,
    ):
        self.image_only     = image_only

        self.voi_lut        = voi_lut
        self.ensure_gray    = ensure_gray
        self.range_correct  = range_correct
        self.normalization  = normalization
        self.apply_window   = apply_window
        self.fix_monochrome = fix_monochrome

    def __call__(self, path: PathLike):
        img, meta = read_dicom_2d(
            dcm_path=path,
            voi_lut = self.voi_lut,
            ensure_gray = self.ensure_gray,
            range_correct = self.range_correct,
            normalization = self.normalization,
            apply_window = self.apply_window,
            fix_monochrome = self.fix_monochrome,
        )
        if self.image_only:
            return img

        return img, meta

class LoadDicom2Dd(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        meta_key_postfix: str = DEFAULT_POST_FIX,
        voi_lut: bool = False,
        ensure_gray: bool = True,
        range_correct: bool = False,
        normalization: bool = True,
        apply_window: bool = False,
        fix_monochrome: bool = True,
        allow_missing_keys: bool=False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.t = LoadDicom2D(
            image_only = False,
            voi_lut = voi_lut,
            ensure_gray = ensure_gray,
            range_correct = range_correct,
            normalization = normalization,
            apply_window = apply_window,
            fix_monochrome = fix_monochrome,
        )
        self.meta_key_postfix = meta_key_postfix

    def __call__(
        self,
        data: Mapping[Hashable, PathLike]
    ):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key], meta_data = self.t(d[key])
            meta_key = PostFix._get_str(key, self.meta_key_postfix)
            d[meta_key] = {**d.get(meta_key, {}), **meta_data}
        return d
