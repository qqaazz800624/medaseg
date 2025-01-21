from typing import Dict, Optional

from monai.metrics import CumulativeIterationMetric
from torchmetrics import Metric

from manafaln.core.builders import ComponentBuilder


class MONAIAdapter(Metric):
    """
    Adapter that allows using `monai.metrics.CumulativeIterationMetric` as a `torchmetrics.Metric`.
    """

    full_state_update: bool = True

    def __init__(
        self, name: str, path: str = "monai.metrics", args: Optional[Dict] = None
    ):
        """
        Initializes the MONAIAdapter class.

        Args:
            name (str): The name of the metric.
            path (str, optional): The path to the metric module. Defaults to "monai.metrics".
            args (Dict, optional): Additional arguments for the metric. Defaults to None.
        """
        super().__init__()
        metric_config = {
            "name": name,
            "path": path,
            "args": args if args is not None else {},
        }
        self.metric: CumulativeIterationMetric = ComponentBuilder()(metric_config)

    def update(self, *args):
        """
        Updates the metric with the given arguments.

        Args:
            *args: Variable length argument list.
        """
        return self.metric(*args)

    def compute(self):
        """
        Computes the metric value.

        Returns:
            The computed metric value.
        """
        return self.metric.aggregate()

    def reset(self):
        """
        Resets the metric.
        """
        self.metric.reset()
