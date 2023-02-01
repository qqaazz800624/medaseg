from enum import Enum

from monai.utils.enums import StrEnum


class ComponentType(Enum):
    """
    Enumerate types of components
    """
    UNKNOWN    = 0
    MODEL      = 1
    LOSS       = 2
    INFERER    = 3
    OPTIMIZER  = 4
    SCHEDULER  = 5
    METRIC     = 6
    DATASET    = 7
    DATALOADER = 8
    TRANSFORM  = 9
    DATAMODULE = 10
    WORKFLOW   = 11
    CALLBACK   = 12
    LOGGER     = 13
    SAMPLER    = 14
    METRICV2   = 15

class ComponentPaths(StrEnum):
    """
    The default paths for each component are for the dynamic module loader
    """
    DEFAULT_MODEL_PATH      = "monai.networks.nets"
    DEFAULT_LOSS_PATH       = "monai.losses"
    DEFAULT_INFERER_PATH    = "monai.inferers"
    DEFAULT_OPTIMIZER_PATH  = "torch.optim"
    DEFAULT_SCHEDULER_PATH  = "torch.optim.lr_scheduler"
    DEFAULT_METRIC_PATH     = "monai.metrics"
    DEFAULT_DATASET_PATH    = "monai.data.dataset"
    DEFAULT_DATALOADER_PATH = "monai.data.dataloader"
    DEFAULT_TRANSFORM_PATH  = "monai.transforms"
    DEFAULT_DATAMODULE_PATH = "manafaln.data"
    DEFAULT_WORKFLOW_PATH   = "manafaln.workflow"
    DEFAULT_CALLBACK_PATH   = "pytorch_lightning.callbacks"
    DEFAULT_LOGGER_PATH     = "pytorch_lightining.loggers"
    DEFAULT_SAMPLER_PATH    = "torch.utils.data.sampler"
    DEFAULT_METRICV2_PATH   = "torchmetrics"

class DefaultKeys(StrEnum):
    """
    The default keys for passing data in workflow
    """
    INPUT_KEY = "image"
    OUTPUT_KEY = "preds"
    LABEL_KEY = "label"
