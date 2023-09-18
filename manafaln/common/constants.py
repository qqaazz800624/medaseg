from enum import Enum

from monai.utils.enums import StrEnum


class ComponentType(Enum):
    """
    Enumerate types of components

    Attributes:
        UNKNOWN (int): Unknown component type.
        MODEL (int): Model component type.
        LOSS (int): Loss component type.
        INFERER (int): Inferer component type.
        OPTIMIZER (int): Optimizer component type.
        SCHEDULER (int): Scheduler component type.
        METRIC (int): Metric component type.
        DATASET (int): Dataset component type.
        DATALOADER (int): Dataloader component type.
        TRANSFORM (int): Transform component type.
        DATAMODULE (int): Datamodule component type.
        WORKFLOW (int): Workflow component type.
        CALLBACK (int): Callback component type.
        LOGGER (int): Logger component type.
        SAMPLER (int): Sampler component type.
        METRICV2 (int): MetricV2 component type.
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

    Attributes:
        DEFAULT_MODEL_PATH (str): Default model path.
        DEFAULT_LOSS_PATH (str): Default loss path.
        DEFAULT_INFERER_PATH (str): Default inferer path.
        DEFAULT_OPTIMIZER_PATH (str): Default optimizer path.
        DEFAULT_SCHEDULER_PATH (str): Default scheduler path.
        DEFAULT_METRIC_PATH (str): Default metric path.
        DEFAULT_DATASET_PATH (str): Default dataset path.
        DEFAULT_DATALOADER_PATH (str): Default dataloader path.
        DEFAULT_TRANSFORM_PATH (str): Default transform path.
        DEFAULT_DATAMODULE_PATH (str): Default datamodule path.
        DEFAULT_WORKFLOW_PATH (str): Default workflow path.
        DEFAULT_CALLBACK_PATH (str): Default callback path.
        DEFAULT_LOGGER_PATH (str): Default logger path.
        DEFAULT_SAMPLER_PATH (str): Default sampler path.
        DEFAULT_METRICV2_PATH (str): Default metricV2 path.
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
    DEFAULT_CALLBACK_PATH   = "lightning.pytorch.callbacks"
    DEFAULT_LOGGER_PATH     = "lightning.pytorch.loggers"
    DEFAULT_SAMPLER_PATH    = "torch.utils.data.sampler"
    DEFAULT_METRICV2_PATH   = "torchmetrics"

class DefaultKeys(StrEnum):
    """
    The default keys for passing data in workflow

    Attributes:
        INPUT_KEY (str): Input key for passing data in workflow.
        OUTPUT_KEY (str): Output key for passing data in workflow.
        LABEL_KEY (str): Label key for passing data in workflow.
    """
    INPUT_KEY = "image"
    OUTPUT_KEY = "preds"
    LABEL_KEY = "label"
