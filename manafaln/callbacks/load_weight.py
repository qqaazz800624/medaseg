from os import PathLike
from typing import Dict, Optional

import torch
from pytorch_lightning import Callback, LightningModule, Trainer

from manafaln.utils.misc import get_attr

class LoadWeights(Callback):
    def __init__(self,
        weights: Dict[str, PathLike],
        strict: bool = True,
        prefix: Optional[str] = "model",
    ):
        self.weights = weights
        self.strict = strict
        self.prefix = prefix

    def load_weight_from_file(
        self,
        path: PathLike
    ):
        weight = torch.load(path, torch.device('cpu'))
        return weight

    def load_weight_to_module(
        self,
        module: torch.nn.Module,
        weight: Dict[str, torch.Tensor]
    ) -> None:
        """
        Loads a checkpoint into a module.
        If self.strict is False, ignores name and size mismatch.
        """
        if self.strict:
            module.load_state_dict(weight, strict=True)
            return
        else:
            module_weight: Dict[str, torch.Tensor] = module.state_dict()
            for k in weight.keys() & module_weight.keys():
                if module_weight[k].shape == weight[k].shape:
                    module_weight[k] = weight[k]
            module.load_state_dict(module_weight, strict=False)
            return

    def on_fit_start(self, _: Trainer, pl_module: LightningModule):
        for attr, weight_path in self.weights.items():
            attr = self.prefix + "." + attr
            module = get_attr(pl_module, attr)
            weight = self.load_weight_from_file(weight_path)
            self.load_weight_to_module(module, weight)
