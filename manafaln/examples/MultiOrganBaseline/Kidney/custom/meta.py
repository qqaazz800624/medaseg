from typing import List

import torch
from monai.data.meta_tensor import MetaTensor
from monai.transforms import MapTransform

def _ensure_list(data):
    if isinstance(data, List):
        return data
    else:
        return [data]

class SaveMeta(MapTransform):
    def __init__(self, keys, meta_keys):
        super().__init__(keys)
        self.meta_keys = _ensure_list(meta_keys)

    def __call__(self, data):
        d = dict(data)
        for key, meta_key in zip(self.keys, self.meta_keys):
            meta = d[key].meta
            d[meta_key] = dict(meta)
        return d

class RestoreMeta(MapTransform):
    def __init__(self, keys, meta_keys):
        super().__init__(keys)
        self.meta_keys = _ensure_list(meta_keys)

    def __call__(self, data):
        d = dict(data)
        for key, meta_key in zip(self.keys, self.meta_keys):
            d[key] = MetaTensor(d[key], meta=d[meta_key])
        return d

