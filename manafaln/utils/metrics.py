from typing import Any, Callable, Dict, List, Union
from collections.abc import Iterable

from monai.metrics import Cumulative

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
    def __init__(self, metric_list: Union[Dict, List]):
        self.metrics = []
        for metric_config in metric_list:
            metric = {
                "name": ensure_label_list(metric_config["log_label"]),
                "unpack": unpack_fn(metric_config.get("output_transform", None)),
                "instance": build_metric(metric_config)
            }
            self.metrics.append(metric)

    def apply(self, data) -> None:
        for m in self.metrics:
            out = m["unpack"](data)
            m["instance"](*out)

    def aggregate(self) -> Dict:
        metrics = {}
        for m in self.metrics:
            results = m["instance"].aggregate()
            for name, value in zip(m["name"], results):
                metrics[name] = value
        return metrics

    def reset(self) -> None:
        for m in self.metrics:
            if isinstance(m["instance"], Cumulative):
                m["instance"].reset()
