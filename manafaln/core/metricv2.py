from typing import Any, Dict, List

import torchmetrics
from monai.utils import ensure_tuple

from manafaln.common.constants import DefaultKeys
from manafaln.core.builders import MetricV2Builder as MetricBuilder

# Default keys for metric input
DEFAULT_METRIC_INPUT_KEYS = [DefaultKeys.OUTPUT_KEY, DefaultKeys.LABEL_KEY]

class MetricCollection(torchmetrics.MetricCollection):
    """
    A collection to compute different metrics with specified input keys.

    Returns a dictionary with computed metrics for each metric module.

    Args:
        config (List[Dict[str, Any]]): List of dicts to configurate metrics.
            Parameters:
                log_label (str): The label of the metric shown on Tensorboard. Defaults to class name.
                input_keys (str, Sequence[str]): Keys of data that will be passed into metric input.
                    Defaults to DEFAULT_METRIC_INPUT_KEYS.
                **kwargs: Other parameters used to build the metric.
        **kwargs: Other parameters passed into torchmetrics.MetricCollection

    Raises:
        ValueError: log_label is not unique
    """
    def __init__(self, config: List[Dict[str, Any]], **kwargs):
        # Validate the log labels, ensure uniqueness
        self.validate_log_labels(config)

        # The builder for metric
        builder = MetricBuilder()

        # Initialize dict to hold input keys of each metric
        self.input_keys = {}

        # Initialize dict to hold each metric module
        metrics = {}

        # For each cfg for metric in config
        for cfg in config:

            # Get log_label of this metric, defaults to class name
            log_label = cfg.pop("log_label", cfg["name"])

            # Get keys of data that will be passed into metric input, defaults to DEFAULT_METRIC_INPUT_KEYS
            self.input_keys[log_label] = ensure_tuple(cfg.pop("input_keys", DEFAULT_METRIC_INPUT_KEYS))

            # Build the metric module
            metric = builder(cfg)

            # Add the metric into dict with key log_label
            metrics[log_label] = metric

        # Initialize MetricCollection with the dict of metric modules
        super().__init__(metrics, **kwargs)

    def update(self, **kwargs: Any):
        """
        Iteratively call update for each metric.
        Keyword arguments (kwargs) will be passed into metric based on the specified input keys.
        """
        # Use compute groups if already initialized and checked
        if self._groups_checked:
            for _, cg in self._groups.items():
                # only update the first member
                log_label = cg[0]
                m0 = getattr(self, log_label)
                # Get the input with specified input keys
                input = (kwargs[input_key] for input_key in self.input_keys[log_label])
                m0.update(*input)
            if self._state_is_copy:
                # If we have deep copied state inbetween updates, reestablish link
                self._compute_groups_create_state_ref()
                self._state_is_copy = False

        else:  # the first update always do per metric to form compute groups
            for log_label, m in self.items(keep_base=True, copy_state=False):
                # Get the input with specified input keys
                input = (kwargs[input_key] for input_key in self.input_keys[log_label])
                m.update(*input)

            if self._enable_compute_groups:
                self._merge_compute_groups()
                # create reference between states
                self._compute_groups_create_state_ref()
                self._groups_checked = True

    def compute(self) -> Dict[str, Any]:
        """
        Compute the result for each metric in the collection.

        Returns:
            Dict[str, Any]: Computed metrics for each metric module.
        """
        return super().compute()

    def reset(self) -> None:
        """Iteratively call reset for each metric."""
        super().reset()

    def validate_log_labels(self, config: List[Dict[str, Any]]):
        """
        Validate whether given log_labels is valid.

        Args:
            config (List[Dict[str, Any]]): List of configs for metrics.

        Raises:
            ValueError: log_label is not unique
        """
        # Create a set to hold log_labels for uniqueness checking
        log_labels = set()

        # For each cfg for metric in config
        for cfg in config:

            # Get log_label of this metric, defaults to class name
            log_label = cfg.get("log_label", cfg["name"])

            # Raise ValueError if log_label is not unique
            if log_label in log_labels:
                raise ValueError(f"log_label {log_label} in not unique.")

            # Add log_label to the set for uniqueness checking
            log_labels.add(log_label)
