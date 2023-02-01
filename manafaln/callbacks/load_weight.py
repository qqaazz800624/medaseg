from os import PathLike
from typing import Any, Dict, Optional

import torch
from pytorch_lightning import Callback, LightningModule, Trainer

class LoadWeights(Callback):
    def __init__(self,
        weights: Dict[str, PathLike],
        strict: bool = True,
        sep: str = ".",
        prefix: Optional[str] = "model",
    ):
        self.weights = weights
        self.strict = strict
        self.sep = sep
        self.prefix = prefix
        
    def get_module(self, attr: str, module: torch.nn.Module) -> torch.nn.Module:
        """
        Recursively get submodule from given attr.
        Attr is seperated with self.sep

        Args:
            attr (str): Attribute to the submodule
            module (torch.nn.Module): The base module to get submodule from

        Returns:
            (torch.nn.Module): The submodule to load weight
            
        Example:
        >>> attr = "block.0"
        >>> module = torch.nn.ModuleDict({
                "block": torch.nn.Sequential(
                    torch.nn.Linear(3, 2),
                    torch.nn.Linear(2, 1)
                )
            })
        >>> get_module(attr, module)
        Linear(in_features=3, out_features=2, bias=True)
        """
        attr, *sub_attr = attr.split(".", 1)
        module = getattr(module, attr)
        if sub_attr == []:
            return module
        return self.get_module(sub_attr[0], module)    
            
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
            attr = self.prefix + self.sep + attr
            module = self.get_module(attr, pl_module)
            weight = self.load_weight_from_file(weight_path)
            self.load_weight_to_module(module, weight)
            