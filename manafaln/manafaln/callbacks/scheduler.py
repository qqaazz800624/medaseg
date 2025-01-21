import logging
from operator import attrgetter
from typing import Dict, Sequence

import numpy as np
from lightning import Callback, Trainer
from scipy.interpolate import interp1d

from manafaln.core.loss import LossHelper
from manafaln.workflow import SupervisedLearningV2

logger = logging.getLogger(__name__)

class LossScheduler(Callback):
    """
    Dynamically scale the factor for each loss,
    computed by linearly interpolating between the given scalings.
    Example:
    >>> max_epochs = 5
    >>> scalings = [1, 3, 2]
    [1.0, 2.0, 3.0, 2.5, 2.0]
    """

    def __init__(self, scalings: Dict[str, Sequence[float]], loss_module: str="loss_fn"):
        """
        Initialize the LossScheduler callback.

        Args:
            scalings (Dict[str, Sequence[float]]): A dictionary mapping loss names to a sequence of scaling factors.
            loss_module (str, optional): The name of the loss module in the model. Defaults to "loss_fn".
        """
        self.scalings = dict(scalings)
        self.loss_module_name = loss_module
        logger.info(f"{self.loss_module_name} will be dynamically scaled with {self.scalings}")

    def on_train_start(self,
        trainer: Trainer,
        pl_module: SupervisedLearningV2
    ) -> None:
        """
        Callback function called when the training starts.

        Args:
            trainer (Trainer): The PyTorch Lightning Trainer object.
            pl_module (SupervisedLearningV2): The PyTorch Lightning module.

        Raises:
            AssertionError: If the loss module is not an instance of LossHelper.
            AssertionError: If any of the given scalings is not in the LossHelper.
        """
        loss_module = attrgetter(self.loss_module_name)(pl_module)
        assert isinstance(loss_module, LossHelper), f"loss_module must be LossHelper, got {type(loss_module)}"

        for loss_name in self.scalings:
            assert loss_name in loss_module, f"Loss {loss_name} is not in LossHelper"

        self.factors = {k: loss_module.factors[k] for k in self.scalings}
        self.interpolated_scalings = {}

        for loss_name, scaling in self.scalings.items():
            x = np.linspace(0, trainer.max_epochs-1, num=len(scaling))
            interpolated_scaling = interp1d(x=x, y=scaling)
            self.interpolated_scalings[loss_name] = interpolated_scaling

    def on_train_epoch_start(self,
        trainer: Trainer,
        pl_module: SupervisedLearningV2
    ):
        """
        Callback function called at the start of each training epoch.

        Args:
            trainer (Trainer): The PyTorch Lightning Trainer object.
            pl_module (SupervisedLearningV2): The PyTorch Lightning module.
        """
        loss_module = attrgetter(self.loss_module_name)(pl_module)
        for loss_name, interpolated_scaling in self.interpolated_scalings.items():
            scaling = interpolated_scaling(trainer.current_epoch).item()
            factor = scaling * self.factors[loss_name]
            loss_module.factors[loss_name] = factor
