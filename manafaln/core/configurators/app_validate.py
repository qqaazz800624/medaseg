import os
from manafaln.core.configurators import DefaultConfigurator

class ValidateConfigurator(DefaultConfigurator):
    def __init__(self, app_name=None, description=None):
        super().__init__(app_name=app_name, description=description)

    def process_args(self, args) -> None:
        # Use default configurator to setup self.raw_config
        super().process_args(args)

        # Set ckpt_path
        if not os.path.exists(self.ckpt_path):
            raise ValueError(
                f"Checkpoint file {self.ckpt_path} does not exist."
            )

