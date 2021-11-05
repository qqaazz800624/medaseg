from typing import Any, Callable, Dict, List, Union
from collections.abc import Iterable

from manafaln.utils.builders import build_metric

def ensure_list(data: Any) -> List:
    if not isinstance(data, Iterable):
        return [data]
    return list(data)

def unpack(x: Any, unpack_fn: str = None):
    if unpack_fn is None:
        return data
    else:
        return eval(unpack_fn)

class MetricCollection:
    def __init__(self, metric_list: Union[Dict, List]):
        self.metrics = []
        for metric_config in metric_list:
            metric = {
                "name": ensure_list(metric_config["log_label"]),
                "unpack": metric_config.get("output_transform", None),
                "instance": build_metric(metric_config)
            }
            self.metrics.append(metric)

    def apply(self, data) -> None:
        for m in self.metrics:
            inputs = unpack(data, unpack_fn=m["unpack"])
            m["instance"](*inputs)

    def aggregate(self) -> Dict:
        metrics = {}
        for m in self.metrics:
            results = m["instance"].aggregate()
            for name, value in zip(m.name, results):
                metrics[name] = value
        return metrics
