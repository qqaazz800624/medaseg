from manafaln.common.constants import ComponentType

class LibSpecMeta(type):
    def __new__(mcls, name, base, attrs):
        # Unable to handle UNKNOWN component type
        attr_keys = [
            t.name for t in ComponentType if t.name != "UNKNOWN"
        ]

        for key in attr_keys:
            attrs[key] = attrs.get(key, None)

        return super().__new__(mcls, name, base, attrs)

class LibSpecNative(metaclass=LibSpecMeta):
    MODEL      = "manafaln.models"
    LOSS       = "manafaln.losses"
    METRIC     = "manafaln.metrics"
    METRICV2   = "manafaln.metrics"
    TRANSFORM  = "manafaln.transforms"
    DATAMODULE = "manafaln.data"
    WORKFLOW   = "manafaln.workflow"
    CALLBACK   = "manafaln.callbacks"
    LOGGER     = "manafaln.loggers"
    SAMPLER    = "manafaln.data"

class LibSpecMONAI(metaclass=LibSpecMeta):
    MODEL      = "monai.networks.nets"
    LOSS       = "monai.losses"
    INFERER    = "monai.inferers"
    OPTIMIZER  = "monai.optimizers"
    SCHEDULER  = "monai.optimizers"
    METRIC     = "monai.metrics"
    TRANSFORM  = "monai.transforms"
    DATASET    = "monai.data.dataset"
    DATALOADER = "monai.data.dataloader"

class LibSpecPyTorch(metaclass=LibSpecMeta):
    LOSS      = "torch.nn"
    OPTIMIZER = "torch.optim"
    SCHEDULER = "torch.optim.lr_scheduler"
    SAMPLER   = "torch.utils.data"

class LibSpecPyTorchLightning(metaclass=LibSpecMeta):
    CALLBACK = "pytorch_lightning.callbacks"
    LOGGER   = "pytorch_lightining.loggers"

class LibSpecTorchVision(metaclass=LibSpecMeta):
    MODEL   = "torchvision.models"
    DATASET = "torchvision.datasets"

class LibSpecTorchMetrics(metaclass=LibSpecMeta):
    METRICV2 = "torchmetrics"
