from typing import Any, List, Optional, Union
import torch

def _ensure_tensor(data: Any, dtype = torch.float32) -> torch.Tensor:
    if (data is not None) or (isinstance(data, torch.Tensor)):
        return data
    return torch.tensor(data, dtype=dtype)

def BCELoss(
    weight: Optional[Union[List[float], torch.Tensor]] = None,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: Optional[str] = 'mean'
):
    weight = _ensure_tensor(weight)
    return torch.nn.BCELoss(weight, size_average, reduce, reduction)

def BCEWithLogitsLoss(
    weight: Optional[Union[List[float], torch.Tensor]] = None,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: Optional[str] = 'mean',
    pos_weight: Optional[Union[List[float], torch.Tensor]] = None
):
    weight = _ensure_tensor(weight)
    pos_weight = _ensure_tensor(pos_weight)
    return torch.nn.BCEWithLogitsLoss(weight, size_average, reduce, reduction, pos_weight)
