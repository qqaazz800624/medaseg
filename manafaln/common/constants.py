from enum import Enum

# This enumerate types of componets
class ComponentType(Enum):
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

# The default paths for each component are for the dynamic modeue loader
class ComponentPaths:
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

