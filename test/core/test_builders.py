import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from manafaln.core.builders import (
    ComponentBuilder,
    OptimizerBuilder,
    SchedulerBuilder
)

class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = torch.nn.Conv1d(1, 1, 3)

    def forward(self, x):
        y = self.conv(x)
        return y

def test_component_builder():
    builder = ComponentBuilder()

    config = {
        "name": "Linear",
        "path": "torch.nn",
        "args": {
            "in_features": 10,
            "out_features": 10
        }
    }

    out = builder(config)
    assert isinstance(out, torch.nn.Linear)

def test_optimizer_builder():
    builder = OptimizerBuilder()

    model = ToyModel()
    config = {
        "name": "AdamW",
        "args": {
            "lr": 1e-4
        }
    }

    out = builder(config, model.parameters())
    assert isinstance(out, AdamW)

def test_scheduler_builder():
    builder = SchedulerBuilder()

    model = ToyModel()
    opt = AdamW(model.parameters())
    config = {
        "name": "CosineAnnealingLR",
        "args": {
            "T_max": 100,
            "eta_min": 1e-5
        }
    }

    out = builder(config, opt)
    assert isinstance(out, CosineAnnealingLR)

