from copy import deepcopy
from typing import Dict, Optional

import torch

from manafaln.core.builders import ModelBuilder

class DummyTorchModule(object):
    """
    A wrapper class for a PyTorch module.

    Args:
        config (Optional[Dict]): The configuration dictionary for building the module.
        module (Optional[torch.nn.Module]): An existing PyTorch module.

    Raises:
        ValueError: If neither config nor module is provided.
        ValueError: If both config and module are provided.

    Attributes:
        module (torch.nn.Module): The PyTorch module.

    Methods:
        forward: Performs a forward pass through the module.
        __call__: Calls the forward method.
        state_dict: Returns the state dictionary of the module.
        load_state_dict: Loads the provided state dictionary into the module.
        cpu: Moves the module to the CPU.
        cuda: Moves the module to the GPU.
        to: Moves the module to the specified device.
    """

    def __init__(
        self,
        config: Optional[Dict] = None,
        module: Optional[torch.nn.Module] = None
    ):
        """
        Initializes the DummyTorchModule.

        Args:
            config (Optional[Dict]): The configuration dictionary for building the module.
            module (Optional[torch.nn.Module]): An existing PyTorch module.

        Raises:
            ValueError: If neither config nor module is provided.
            ValueError: If both config and module are provided.
        """
        if (config is None) and (module is None):
            raise ValueError("Must provide either config or module")

        if config and module:
            raise ValueError("Can only provide one of config or module")

        if config:
            builder = ModelBuilder()
            self.module = builder(config)

        if module:
            self.module = deepcopy(module)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        with torch.no_grad():
            out = self.module(x)
        return out

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calls the forward method.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        return self.forward(x)

    def state_dict(self) -> Dict:
        """
        Returns the state dictionary of the module.

        Returns:
            Dict: The state dictionary.
        """
        return self.module.state_dict()

    def load_state_dict(self, state_dict: Dict):
        """
        Loads the provided state dictionary into the module.

        Args:
            state_dict (Dict): The state dictionary to be loaded.
        """
        return self.module.load_state_dict(state_dict)

    def cpu(self):
        """
        Moves the module to the CPU.

        Returns:
            DummyTorchModule: The updated DummyTorchModule object.
        """
        self.module = self.module.cpu()
        return self

    def cuda(self):
        """
        Moves the module to the GPU.

        Returns:
            DummyTorchModule: The updated DummyTorchModule object.
        """
        self.module = self.module.cuda()
        return self

    def to(self, device):
        """
        Moves the module to the specified device.

        Args:
            device: The device to move the module to.

        Returns:
            DummyTorchModule: The updated DummyTorchModule object.
        """
        self.module = self.module.to(device)
        return self

