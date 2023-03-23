import importlib

from monai.metrics import CumulativeIterationMetric
from torchmetrics import Metric


class MONAI(Metric):
    full_state_update: bool = True
    def __init__(self, name, path="monai.metrics", *args, **kwargs):
        """
        A wrapper for `monai.metrics.CumulativeIterationMetric`.
        Args:
            name: The metric name.
            path: Path to import metric.
            args: arguments for the metric.
            kwargs: keyword arguments for the metric.
        """
        super().__init__()
        self.name = name
        M = importlib.import_module(path)
        metric: CumulativeIterationMetric = getattr(M, name)
        self.metric = metric(*args, **kwargs)

    def update(self, *args):
        return self.metric(*args)

    def compute(self):
        return self.metric.aggregate()

    def reset(self):
        self.metric.reset()
