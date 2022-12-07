from copy import deepcopy
from typing import Dict, Optional

import torch

from manafaln.core.builders import ModelBuilder

class DummyTorchModule(object):
    def __init__(
        self,
        config: Optional[Dict] = None,
        module: Optional[torch.nn.Module] = None
    ):
        if (config is None) and (module is None):
            raise ValueError("Must provide either config or module")

        if config and module:
            raise ValueError("Can only provide one of config or module")

        if config:
            builder = ModelBuilder()
            self.module = builder(config)

        if module:
            self.module = deepcopy(module)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out = self.module(x)
        return out

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def state_dict(self) -> Dict:
        return self.module.state_dict()

    def load_state_dict(self, state_dict: Dict):
        return self.module.load_state_dict(state_dict)

    def cpu(self):
        self.module = self.module.cpu()
        return self

    def cuda(self):
        self.module = self.module.cuda()
        return

    def to(self, device):
        self.module = self.module.to(device)
        return self

