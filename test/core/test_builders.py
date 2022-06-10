import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from monai.data import Dataset, DataLoader
from monai.inferers import SimpleInferer
from monai.metrics import DiceMetric
from monai.networks.nets import DenseNet121
from monai.transforms import LoadImaged
from pytorch_lightning.callbacks import ModelCheckpoint

from manafaln.core.builders import (
    ComponentBuilder,
    ModelBuilder,
    LossBuilder,
    InfererBuilder,
    OptimizerBuilder,
    SchedulerBuilder,
    MetricBuilder,
    DatasetBuilder,
    DataLoaderBuilder,
    TransformBuilder,
    DataModuleBuilder,
    WorkflowBuilder,
    CallbackBuilder
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

def test_model_builder():
    builder = ModelBuilder()

    config = {
        "name": "DenseNet121",
        "args": {
            "spatial_dims": 3,
            "in_channels": 1,
            "out_channels": 3
        }
    }
    out = builder(config)
    assert isinstance(out, DenseNet121)

def test_loss_builder():
    builder = LossBuilder()

    config = {
        "name": "CrossEntropyLoss",
        "args": {}
    }
    out = builder(config)
    assert isinstance(out, CrossEntropyLoss)

def test_inferer_builder():
    builder = InfererBuilder()

    config = {
        "name": "SimpleInferer",
        "args": {}
    }
    out = builder(config)
    assert isinstance(out, SimpleInferer)

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

def test_metric_builder():
    builder = MetricBuilder()

    config = {
        "name": "DiceMetric",
        "args": {
            "include_background": False,
            "reduction": "mean"
        }
    }
    out = builder(config)
    assert isinstance(out, DiceMetric)

def test_dataset_builder():
    builder = DatasetBuilder()

    datalist = [1, 2, 3, 4, 5]
    config = {
        "name": "Dataset",
        "args": {}
    }
    out = builder(config, [datalist], {"transform": None})
    assert isinstance(out, Dataset)

def test_dataloader_builder():
    builder = DataLoaderBuilder()

    ds = Dataset([1, 2, 3, 4, 5], transform=None)
    config = {
        "name": "DataLoader",
        "args": {
            "shuffle": True,
            "num_workers": 0
        }
    }
    out = builder(config, ds)
    assert isinstance(out, DataLoader)

def test_transform_builder():
    builder = TransformBuilder()

    config = {
        "name": "LoadImaged",
        "args": {
            "keys": ["image", "label"]
        }
    }
    out = builder(config)
    assert isinstance(out, LoadImaged)

def test_callback_builder():
    builder = CallbackBuilder()

    config = {
        "name": "ModelCheckpoint",
        "args": {
            "filename": "best_model.ckpt"
        }
    }
    out = builder(config)
    assert isinstance(out, ModelCheckpoint)

