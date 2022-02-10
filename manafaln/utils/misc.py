from typing import Any, Dict, List, Union

import torch
import numpy as np

def ensure_list(data: Any) -> List:
    if not isinstance(data, List):
        return [data]
    return data

def ensure_numpy(data):
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return data
