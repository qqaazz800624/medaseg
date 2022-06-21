from typing import Any, Callable, Dict, List, Union
from collections.abc import Iterable

import torch
from monai.metrics import Cumulative, IterationMetric

from manafaln.utils.builders import build_metric

def ensure_label_list(data):
    if isinstance(data, str):
        return [data]
    return list(data)

def unpack_fn(unpack_fn: str = None):
    if unpack_fn is None:
        return lambda *args: args
    else:
        return eval(unpack_fn)

class MetricCollection:
    def __init__(
        self,
        metric_list: Union[Dict, List]
    ):
        self.metrics = []
        for metric_config in metric_list:
            metric = {
                "name": ensure_label_list(metric_config["log_label"]),
                "unpack": unpack_fn(metric_config.get("output_transform", None)),
                "instance": build_metric(metric_config)
            }
            self.metrics.append(metric)

    def apply(self, data) -> Dict[str, torch.Tensor]:
        batch_metrics = {}
        for m in self.metrics:
            out = m["unpack"](data)
            if isinstance(m["instance"], IterationMetric):
                for name, value in zip(m["name"], m["instance"](*out)):
                    batch_metrics[name] = value
            else:
                m["instance"].append(*out)
        return batch_metrics

    def aggregate(self) -> Dict:
        metrics = {}
        for m in self.metrics:
            if isinstance(m["instance"], Cumulative):
                results = m["instance"].aggregate()
                # Some metrics will contain more than one results
                for name, value in zip(m["name"], results):
                    metrics[name] = value
        return metrics

    def reset(self) -> None:
        for m in self.metrics:
            if isinstance(m["instance"], Cumulative):
                m["instance"].reset()

