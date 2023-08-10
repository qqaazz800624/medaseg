import importlib
from warnings import warn

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
            *args: arguments for the metric.
            **kwargs: keyword arguments for the metric.
        """
        warn(
            f"{self.__class__.__name__} metric is deprecated, please use manafaln.metrics.adapter.MONAIAdapter",
            DeprecationWarning,
            stacklevel=2,
        )

        super().__init__()
        self.name = name
        M = importlib.import_module(path)
        metric: CumulativeIterationMetric = getattr(M, name)
        self.metric = metric(*args, **kwargs)

    def update(self, *args):
        """
        Update the metric with input arguments.

        Args:
            *args: Input arguments for the metric.

        Returns:
            The result of the metric update.
        """
        return self.metric(*args)

    def compute(self):
        """
        Compute the aggregated metric value.

        Returns:
            The aggregated metric value.
        """
        return self.metric.aggregate()

    def reset(self):
        """
        Reset the metric to its initial state.
        """
        self.metric.reset()
