from typing import Dict, Union

import torch
from monai.config.type_definitions import PathLike
from pytorch_lightning import Callback, LightningModule, Trainer

from manafaln.utils.misc import get_attr


class LoadWeights(Callback):
    def __init__(
        self,
        weights: Union[PathLike, Dict[str, PathLike]],
        strict: bool = True,
    ):
        self.weights = weights
        self.strict = strict

    def load_weight_to_module(
        self, module: torch.nn.Module, weight: Dict[str, torch.Tensor]
    ) -> None:
        """
        Loads a checkpoint into a module.
        If self.strict is False, ignores name and size mismatch.
        """
        if self.strict:
            module.load_state_dict(weight, strict=True)
            return

        # elif not self.strict:
        module_weight: Dict[str, torch.Tensor] = module.state_dict()
        for k in weight.keys() & module_weight.keys():
            if module_weight[k].shape == weight[k].shape:
                module_weight[k] = weight[k]
        module.load_state_dict(module_weight, strict=False)

    def on_fit_start(self, _: Trainer, pl_module: LightningModule):
        if isinstance(self.weights, PathLike):
            weight = torch.load(self.weights, map_location=pl_module.device)
            self.load_weight_to_module(pl_module, weight)
            return

        # elif isinstance(self.weights, dict):
        for module_name, weight_path in self.weights.items():
            module: torch.nn.Module = get_attr(pl_module, module_name)

            if not isinstance(module, torch.nn.Module):
                raise ValueError(
                    f"LoadWeights requires module to be torch.nn.Module but got type {type(module)} for {module_name}"
                )

            weight = torch.load(weight_path, map_location=module.device)
            self.load_weight_to_module(module, weight)
