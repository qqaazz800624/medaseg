from typing import Any, Dict, List, Sequence, Tuple

import torch
from monai.utils import ensure_tuple

def ensure_list(data: Any) -> List:
    if not isinstance(data, List):
        return [data]
    return data

def ensure_numpy(data):
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return data

def ensure_python_value(value: Any):
    if isinstance(value, torch.Tensor):
        try:
            return value.item()
        except ValueError:
            return value.tolist()
    return value

def get_attr(obj: object, name: str) -> Any:
    """
    Recursively get attribute from object.
    """
    names = name.split(".")
    def _get_attr(obj, names):
        if len(names) > 1:
            return _get_attr(getattr(obj, names[0]), names[1:])
        return getattr(obj, names[0])
    return _get_attr(obj, names)

def get_item(d: Dict, key: str, sep: str=".") -> Any:
    """
    Recursively get item from dictionary given keys seperated by sep.
    """
    keys = key.split(sep)
    def _get_item(d, keys):
        if len(keys) > 1:
            return _get_item(d[keys[0]], keys[1:])
        return d[keys[0]]
    return _get_item(d, keys)

def get_items(d: Dict, keys: Sequence[str], sep: str=".") -> Tuple[Any]:
    """
    Get items from dictionary with keys.
    """
    keys = ensure_tuple(keys)
    items = (get_item(d, k, sep) for k in keys)
    items = ensure_tuple(items, wrap_array=True)
    return items

def update_items(d: Dict, keys: Sequence[str], items: Sequence[Any]) -> Dict[str, Any]:
    """
    Update items to dictionary with given keys and items.
    """
    keys = ensure_tuple(keys)
    items = ensure_tuple(items, wrap_array=True)
    d.update(zip(keys, items))
    return d
