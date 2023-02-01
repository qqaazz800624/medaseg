from typing import Any, Dict, List

import torch
from monai.utils import ensure_tuple
from torch.nn import ModuleDict
from torch.nn.modules.loss import _Loss

from manafaln.common.constants import DefaultKeys
from manafaln.core.builders import LossBuilder

# The key for loss required by Pytorch Lightning
PYTORCH_LIGHTNING_LOSS_KEY = "loss"

# Default keys for loss input
DEFAULT_LOSS_INPUT_KEYS = [DefaultKeys.OUTPUT_KEY, DefaultKeys.LABEL_KEY]

class LossHelper(ModuleDict, _Loss):
    """
    A collection to compute different losses with specified input keys,
    then add them together with given factors.

    Returns a dictionary with computed losses for each loss module.
    Item with key `PYTORCH_LIGHTNING_LOSS_KEY` is the total loss.

    Args:
        config (List[Dict[str, Any]]): List of dicts to configurate losses.
            Parameters:
                factor (float): The weighting factor for the loss. Defaults to 1.
                log_label (str): The label of the loss shown on Tensorboard. Defaults to class name.
                input_keys (str, Sequence[str]): Keys of data that will be passed into loss input.
                    Defaults to `DEFAULT_LOSS_INPUT_KEYS`.
                **kwargs: Other parameters used to build the loss.

    Raises:
        ValueError: log_label is not unique
        ValueError: log_label is same as `PYTORCH_LIGHTNING_LOSS_KEY`
    """
    def __init__(self,
        config: List[Dict[str, Any]],
    ):
        super().__init__()

        # Validate the log labels, ensure uniqueness
        # and different with PYTORCH_LIGHTNING_LOSS_KEY
        self.validate_log_labels(config)

        # The builder for loss
        builder = LossBuilder()

        # Initialize dict to hold factor of each loss
        self.factors = {}

        # Initialize dict to hold input keys of each loss
        self.input_keys = {}

        # For each cfg for loss in config
        for cfg in config:

            # Get log_label of this loss, defaults to class name
            log_label = cfg.pop("log_label", cfg["name"])

            # Get the factor of this loss, defaults to 1
            self.factors[log_label] = cfg.pop("factor", 1)

            # Get keys of data that will be passed into loss input, defaults to DEFAULT_LOSS_INPUT_KEYS
            self.input_keys[log_label] = ensure_tuple(cfg.pop("input_keys", DEFAULT_LOSS_INPUT_KEYS))

            # Build the loss module
            loss =  builder(cfg)

            # Add the loss into ModuleDict with key log_label
            self.update({log_label: loss})

    def forward(self, **kwargs: Any) -> Dict[str, torch.Tensor]:
        """
        Compute different losses with specified input keys,
        then add them together with given factors.

        Returns:
            Dict[str, torch.Tensor]: Computed losses for each loss module.
                Item with key `PYTORCH_LIGHTNING_LOSS_KEY` is the total loss.
        """

        # Initialize dict to hold all losses
        loss_logs = {PYTORCH_LIGHTNING_LOSS_KEY: 0}

        # For each loss module
        for log_label, loss_fn in self.items():
            # Get the input with specified input keys
            input = (kwargs[input_key] for input_key in self.input_keys[log_label])

            # Compute the loss
            loss = loss_fn(*input)

            # Add the loss to total loss weighted by given factor
            loss_logs[PYTORCH_LIGHTNING_LOSS_KEY] += self.factors[log_label] * loss

            # Add the loss to dict for logging
            loss_logs[log_label] = loss.detach() if isinstance(loss, torch.Tensor) else loss

        # Return computed losses
        return loss_logs

    def validate_log_labels(self, config: List[Dict[str, Any]]):
        """
        Validate whether given log_labels is valid.

        Args:
            config (List[Dict[str, Any]]): List of configs for losses.

        Raises:
            ValueError: log_label is not unique
            ValueError: log_label is the same as `PYTORCH_LIGHTNING_LOSS_KEY`
        """

        # Create a set to hold log_labels for uniqueness checking
        log_labels = set()

        # For each cfg for loss in config
        for cfg in config:

            # Get log_label of this loss, defaults to class name
            log_label = cfg.get("log_label", cfg["name"])

            # Raise ValueError if log_label is not unique
            if log_label in log_labels:
                raise ValueError(f"log_label {log_label} in not unique.")

            # Raise ValueError if log_label is already used for total loss
            if log_label == PYTORCH_LIGHTNING_LOSS_KEY:
                raise ValueError(f"log_label cannot be \"{log_label}\".")

            # Add log_label to the set for uniqueness checking
            log_labels.add(log_label)
