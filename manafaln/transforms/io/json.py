
import json
from typing import Hashable, Mapping, Optional, Sequence

from monai.config import KeysCollection, PathLike
from monai.transforms import MapTransform, Transform
from monai.utils import PostFix
from monai.utils.misc import ImageMetaKey

DEFAULT_POST_FIX = PostFix.meta()
ITEM_KEYS = "item_keys"

# TODO: Read raw JSON from EBM directly, especially penWidth

class LoadJSON(Transform):
    """
    Load json file.
    """
    def __init__(
        self,
        json_keys: Optional[Sequence[str]]=None,
        json_only: bool=False
    ):
        self.json_keys = json_keys
        self.json_only = json_only

    def __call__(self, path: PathLike):
        # Read json file
        with open(path, "r") as f:
            data = json.load(f)
        meta_data = {ImageMetaKey.FILENAME_OR_OBJ: path}

        # Is json_keys is given
        if self.json_keys is not None:
            # Unpack data with json_keys
            data = [data.get(k) for k in self.json_keys]
            meta_data[ITEM_KEYS]=self.json_keys

        if self.json_only:
            return data

        return data, meta_data

class LoadJSONd(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        json_keys: Optional[Sequence[str]]=None,
        meta_key_postfix: str=DEFAULT_POST_FIX,
        allow_missing_keys: bool=False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.t = LoadJSON(json_keys=json_keys, json_only=False)
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
