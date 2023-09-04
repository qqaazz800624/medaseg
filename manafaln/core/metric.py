from typing import Callable, Dict, List, Union, Optional

import torch
from monai.metrics import Cumulative, IterationMetric

from manafaln.core.builders import MetricBuilder

def _ensure_list(name: Union[str, List[str]]) -> List[str]:
    """
    Ensure that the input is a list of strings.
    
    Args:
        name: A string or a list of strings.
        
    Returns:
        A list of strings.
    """
    if isinstance(name, str):
        return [name]
    return name

def _unpack_fn(lambda_fn: Optional[str] = None) -> Callable:
    """
    Unpacks a lambda function from a string representation.
    
    Args:
        lambda_fn: A string representation of a lambda function.
        
    Returns:
        A callable function.
    """
    if lambda_fn is not None:
        return eval(lambda_fn)
    return lambda *args: args

class MetricHelper:
    """
    Helper class for handling metrics.
    """
    def __init__(self, config: Dict, builder: MetricBuilder):
        """
        Initialize the MetricHelper instance.
        
        Args:
            config: A dictionary containing the configuration for the metric.
            builder: An instance of MetricBuilder.
        """
        self.metric_name = _ensure_list(config["log_label"])
        self.output_transform = _unpack_fn(config.get("output_transform", None))
        self.metric_instance = builder(config)

    @property
    def is_cumulative(self) -> bool:
        """
        Check if the metric is cumulative.
        
        Returns:
            True if the metric is cumulative, False otherwise.
        """
        return isinstance(self.metric_instance, Cumulative)

    @property
    def is_iterative(self) -> bool:
        """
        Check if the metric is iterative.
        
        Returns:
            True if the metric is iterative, False otherwise.
        """
        return isinstance(self.metric_instance, IterationMetric)

    def _update_output(self, values, output):
        """
        Update the output dictionary with metric values.
        
        Args:
            values: The metric values.
            output: The output dictionary.
            
        Returns:
            The updated output dictionary.
        """
        if (len(self.metric_name) == 1) and (not isinstance(values, list)):
            values = [values]

        for name, value in zip(self.metric_name, values):
            output[name] = value

        return output

    def apply(self, data) -> Dict:
        """
        Apply the metric to the given data.
        
        Args:
            data: The input data.
            
        Returns:
            A dictionary containing the metric values.
        """
        out = {}
        args = self.output_transform(data)
        if self.is_iterative:
            val = self.metric_instance(*args)
            out = self._update_output(val, out)
        else:
            self.metric_instance(*args)
        return out

    def aggregate(self) -> Dict:
        """
        Aggregate the metric values.
        
        Returns:
            A dictionary containing the aggregated metric values.
        """
        out = {}
        if self.is_cumulative:
            val = self.metric_instance.aggregate()
            out = self._update_output(val, out)
        return out

    def reset(self):
        """
        Reset the metric.
        """
        if self.is_cumulative:
            self.metric_instance.reset()

class MetricCollection:
    """
    Collection of metrics.
    """
    def __init__(self, metric_list: List[Dict]):
        """
        Initialize the MetricCollection instance.
        
        Args:
            metric_list: A list of metric configurations.
        """
        builder = MetricBuilder()
        self.metrics = [MetricHelper(config, builder) for config in metric_list]

    def apply(self, data) -> Dict[str, torch.Tensor]:
        """
        Apply the metrics to the given data.
        
        Args:
            data: The input data.
            
        Returns:
            A dictionary containing the metric values.
        """
        metrics = {}
        for m in self.metrics:
            out = m.apply(data)
            metrics.update(out)
        return metrics

    def aggregate(self) -> Dict:
        """
        Aggregate the metric values.
        
        Returns:
            A dictionary containing the aggregated metric values.
        """
        metrics = {}
        for m in self.metrics:
            out = m.aggregate()
            metrics.update(out)
        return metrics

    def reset(self) -> None:
        """
        Reset the metrics.
        """
        for m in self.metrics:
            m.reset()
