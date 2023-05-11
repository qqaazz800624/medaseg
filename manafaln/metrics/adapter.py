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
        super().__init__()
        metric_config = {
            "name": name,
            "path": path,
            "args": args if args is not None else {},
        }
        self.metric: CumulativeIterationMetric = ComponentBuilder()(metric_config)

    def update(self, *args):
        return self.metric(*args)

    def compute(self):
        return self.metric.aggregate()

    def reset(self):
        self.metric.reset()
