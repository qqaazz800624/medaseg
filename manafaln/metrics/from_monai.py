from monai.utils import optional_import
from torchmetrics import Metric

class MONAI(Metric):
    full_state_update: bool = True
    def __init__(self, name, *args, **kwargs):
        """
        Args:
            name: The metric name in MONAI package.
            args: parameters for the MONAI metric.
            kwargs: parameters for the MONAI metric.
        """
        super().__init__()
        self.name = name
        metric, _ = optional_import("monai.metrics", name=name)
        self.metric = metric(*args, **kwargs)

    def update(self, *args):
        return self.metric(*args)

    def compute(self):
        return self.metric.aggregate()

    def reset(self):
        self.metric.reset()
