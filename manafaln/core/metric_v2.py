from typing import Any, Dict, List

import torchmetrics
from monai.utils import ensure_tuple

from manafaln.common.constants import DefaultKeys
from manafaln.core.builders import MetricV2Builder as MetricBuilder
from manafaln.utils import get_items


class MetricCollection(torchmetrics.MetricCollection):
    DEFAULT_METRIC_INPUT_KEYS = [DefaultKeys.OUTPUT_KEY, DefaultKeys.LABEL_KEY]

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
        self.validate_log_labels(config)

        builder = MetricBuilder()

        self.input_keys = {}

        metrics = {}
        for cfg in config:
            log_label = cfg.get("log_label", cfg["name"])

            self.input_keys[log_label] = ensure_tuple(
                cfg.get("input_keys", MetricCollection.DEFAULT_METRIC_INPUT_KEYS)
            )

            metrics[log_label] = builder(cfg)

        # Initialize MetricCollection with the dict of metric modules
        super().__init__(metrics, compute_groups=False, **kwargs)

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
                input = get_items(kwargs, self.input_keys[log_label])
                m0.update(*input)
            if self._state_is_copy:
                # If we have deep copied state inbetween updates, reestablish link
                self._compute_groups_create_state_ref()
                self._state_is_copy = False

        else:  # the first update always do per metric to form compute groups
            for log_label, m in self.items(keep_base=True, copy_state=False):
                input = get_items(kwargs, self.input_keys[log_label])
                m.update(*input)

            if self._enable_compute_groups:
                self._merge_compute_groups()
                # create reference between states
                self._compute_groups_create_state_ref()
                self._groups_checked = True

    def compute(self) -> Dict[str, Any]:
        """
        Compute the result for each metric in the collection.
        Flatten result dict by combining keys,
        instead of flatten naively like torchmetrics.MetricCollection.

        Returns:
            Dict[str, Any]: Computed metrics for each metric module.
        """
        res = {k: m.compute() for k, m in self.items(keep_base=True, copy_state=False)}
        res = self._flatten_dict(res)
        res = {self._set_name(k): v for k, v in res.items()}
        return res

    def reset(self) -> None:
        """Iteratively call reset for each metric."""
        super().reset()

    def _flatten_dict(self, res: dict) -> dict:
        flatten_res = {}
        for key, value in res.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    flatten_key = key if k is None else f"{key}_{k}"
                    flatten_res[flatten_key] = v
            else:
                flatten_res[key] = value
        return flatten_res

    def validate_log_labels(self, config: List[Dict[str, Any]]):
        """
        Validate whether given log_labels is valid.

        Args:
            config (List[Dict[str, Any]]): List of configs for metrics.

        Raises:
            ValueError: log_label is not unique
        """

        log_labels = set()

        for cfg in config:
            log_label = cfg.get("log_label", cfg["name"])
            if log_label in log_labels:
                raise ValueError(f"log_label {log_label} in not unique.")
            log_labels.add(log_label)
