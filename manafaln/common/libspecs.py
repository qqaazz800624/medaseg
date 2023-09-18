from manafaln.common.constants import ComponentType

class LibSpecMeta(type):
    """
    Metaclass for library specification classes.
    """

    def __new__(mcls, name, base, attrs):
        """
        Create a new instance of the class.

        Args:
            mcls (type): The metaclass.
            name (str): The name of the class.
            base (tuple): The base classes.
            attrs (dict): The attributes of the class.

        Returns:
            object: The new instance of the class.
        """
        # Unable to handle UNKNOWN component type
        attr_keys = [
            t.name for t in ComponentType if t.name != "UNKNOWN"
        ]

        for key in attr_keys:
            attrs[key] = attrs.get(key, None)

        return super().__new__(mcls, name, base, attrs)


class LibSpecNative(metaclass=LibSpecMeta):
    """
    Library specification for native components.
    """

    MODEL      = "manafaln.models"
    LOSS       = "manafaln.losses"
    INFERER    = "manafaln.inferers"
    METRIC     = "manafaln.metrics"
    METRICV2   = "manafaln.metrics"
    TRANSFORM  = "manafaln.transforms"
    DATAMODULE = "manafaln.data"
    WORKFLOW   = "manafaln.workflow"
    CALLBACK   = "manafaln.callbacks"
    LOGGER     = "manafaln.loggers"
    SAMPLER    = "manafaln.data"


class LibSpecMONAI(metaclass=LibSpecMeta):
    """
    Library specification for MONAI components.
    """

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
    """
    Library specification for PyTorch components.
    """

    LOSS      = "torch.nn"
    OPTIMIZER = "torch.optim"
    SCHEDULER = "torch.optim.lr_scheduler"
    SAMPLER   = "torch.utils.data"


class LibSpecPyTorchLightning(metaclass=LibSpecMeta):
    """
    Library specification for PyTorch Lightning components.
    """

    CALLBACK = "lightning.pytorch.callbacks"
    LOGGER   = "lightning.pytorch.loggers"


class LibSpecTorchVision(metaclass=LibSpecMeta):
    """
    Library specification for TorchVision components.
    """

    MODEL   = "torchvision.models"
    DATASET = "torchvision.datasets"


class LibSpecTorchMetrics(metaclass=LibSpecMeta):
    """
    Library specification for TorchMetrics components.
    """

    METRICV2 = "torchmetrics"
