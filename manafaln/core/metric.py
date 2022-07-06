from typing import Callable, Dict, List, Union, Optional

import torch
from monai.metrics import Cumulative, IterationMetric

from manafaln.core.builders import MetricBuilder

def _ensure_list(name: Union[str, List[str]]) -> List[str]:
    if isinstance(name, str):
        return [name]
    return name

def _unpack_fn(lambda_fn: Optional[str] = None) -> Callable:
    if lambda_fn is not None:
        return eval(lambda_fn)
    return lambda *args: args

class MetricHelper:
    def __init__(self, config: Dict, builder: MetricBuilder):
        self.metric_name = _ensure_list(config["log_label"])
        self.output_transform = _unpack_fn(config.get("output_transform", None))
        self.metric_instance = builder(config)

    @property
    def is_cumulative(self) -> bool:
        return isinstance(self.metric_instance, Cumulative)

    @property
    def is_iterative(self) -> bool:
        return isinstance(self.metric_instance, IterationMetric)

    def _update_output(self, values, output):
        for name, value in zip(self.metric_name, values):
            output[name] = value
        return output

    def apply(self, data) -> Dict:
        out = {}
        args = self.output_transform(data)
        if self.is_iterative:
            val = self.metric_instance(*args)
            out = self._update_output(val, out)
        else:
            self.metric_instance(*args)
        return out

    def aggregate(self) -> Dict:
        out = {}
        if self.is_cumulative:
            val = self.metric_instance.aggregate()
            out = self._update_output(val, out)
        return out

    def reset(self):
        if self.is_cumulative:
            self.metric_instance.reset()

class MetricCollection:
    def __init__(self, metric_list: List[Dict]):
        builder = MetricBuilder()
        self.metrics = [MetricHelper(config, builder) for config in metric_list]

    def apply(self, data) -> Dict[str, torch.Tensor]:
        metrics = {}
        for m in self.metrics:
            out = m.apply(data)
            metrics.update(out)
        return metrics

    def aggregate(self) -> Dict:
        metrics = {}
        for m in self.metrics:
            out = m.aggregate()
            metrics.update(out)
        return metrics

    def reset(self) -> None:
        for m in self.metrics:
            m.reset()

