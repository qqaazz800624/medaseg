from typing import Any, Dict, List, Sequence, Tuple

import torch
from monai.utils import ensure_tuple

def ensure_list(data: Any) -> List:
    """
    Ensure that the input data is a list. If the input data is not a list, it will be wrapped in a list and returned.

    Args:
        data (Any): The input data.

    Returns:
        List: The input data as a list.
    """
    if not isinstance(data, List):
        return [data]
    return data

def ensure_numpy(data):
    """
    Ensure that the input data is a numpy array. If the input data is a torch tensor, it will be converted to a numpy array and returned. Otherwise, the input data will be returned as is.

    Args:
        data: The input data.

    Returns:
        numpy.ndarray or Any: The input data as a numpy array, if it is a torch tensor. Otherwise, the input data as is.
    """
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return data

def ensure_python_value(value: Any):
    """
    Ensure that the input value is a Python value. If the input value is a torch tensor, it will be converted to a Python value and returned. Otherwise, the input value will be returned as is.

    Args:
        value (Any): The input value.

    Returns:
        Any: The input value as a Python value, if it is a torch tensor. Otherwise, the input value as is.
    """
    if isinstance(value, torch.Tensor):
        try:
            return value.item()
        except ValueError:
            return value.tolist()
    return value

def get_attr(obj: object, name: str) -> Any:
    """
    Recursively get attribute from object.

    Args:
        obj (object): The object from which to get the attribute.
        name (str): The name of the attribute to get.

    Returns:
        Any: The value of the attribute.
    """
    names = name.split(".")
    def _get_attr(obj, names):
        if len(names) > 1:
            return _get_attr(getattr(obj, names[0]), names[1:])
        return getattr(obj, names[0])
    return _get_attr(obj, names)

def get_item(d: Dict, key: str, sep: str=".") -> Any:
    """
    Recursively get item from dictionary given keys separated by sep.

    Args:
        d (Dict): The dictionary from which to get the item.
        key (str): The key of the item to get.
        sep (str): The separator used to separate the keys.

    Returns:
        Any: The value of the item.
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

    Args:
        d (Dict): The dictionary from which to get the items.
        keys (Sequence[str]): The keys of the items to get.
        sep (str): The separator used to separate the keys.

    Returns:
        Tuple[Any]: The values of the items as a tuple.
    """
    keys = ensure_tuple(keys)
    items = (get_item(d, k, sep) for k in keys)
    items = ensure_tuple(items, wrap_array=True)
    return items

def update_items(d: Dict, keys: Sequence[str], items: Sequence[Any]) -> Dict[str, Any]:
    """
    Update items to dictionary with given keys and items.

    Args:
        d (Dict): The dictionary to update.
        keys (Sequence[str]): The keys of the items to update.
        items (Sequence[Any]): The values of the items to update.

    Returns:
        Dict[str, Any]: The updated dictionary.
    """
    keys = ensure_tuple(keys)
    items = ensure_tuple(items, wrap_array=True)
    d.update(zip(keys, items))
    return d
