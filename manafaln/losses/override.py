from typing import Any, List, Optional, Union
import torch

def _ensure_tensor(data: Any, dtype=torch.float32) -> torch.Tensor:
    """
    Ensure that the input data is a tensor. If the input data is already a tensor, it is returned as is.
    If the input data is None or not a tensor, it is converted to a tensor with the specified data type.

    Args:
        data: The input data to be converted to a tensor.
        dtype: The data type of the tensor. Default is torch.float32.

    Returns:
        A tensor representing the input data.

    """
    if (data is not None) or (isinstance(data, torch.Tensor)):
        return data
    return torch.tensor(data, dtype=dtype)

def BCELoss(
    weight: Optional[Union[List[float], torch.Tensor]] = None,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: Optional[str] = 'mean'
) -> torch.nn.BCELoss:
    """
    Creates a binary cross entropy loss criterion.

    Args:
        weight: A tensor of weights to apply to the input. Default is None.
        size_average: Deprecated argument. Default is None.
        reduce: Deprecated argument. Default is None.
        reduction: Specifies the reduction to apply to the output. Default is 'mean'.

    Returns:
        A BCELoss criterion object.

    """
    weight = _ensure_tensor(weight)
    return torch.nn.BCELoss(weight, size_average, reduce, reduction)

def BCEWithLogitsLoss(
    weight: Optional[Union[List[float], torch.Tensor]] = None,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: Optional[str] = 'mean',
    pos_weight: Optional[Union[List[float], torch.Tensor]] = None
) -> torch.nn.BCEWithLogitsLoss:
    """
    Creates a binary cross entropy loss criterion with logits.

    Args:
        weight: A tensor of weights to apply to the input. Default is None.
        size_average: Deprecated argument. Default is None.
        reduce: Deprecated argument. Default is None.
        reduction: Specifies the reduction to apply to the output. Default is 'mean'.
        pos_weight: A tensor of positive class weights. Default is None.

    Returns:
        A BCEWithLogitsLoss criterion object.

    """
    weight = _ensure_tensor(weight)
    pos_weight = _ensure_tensor(pos_weight)
    return torch.nn.BCEWithLogitsLoss(weight, size_average, reduce, reduction, pos_weight)
