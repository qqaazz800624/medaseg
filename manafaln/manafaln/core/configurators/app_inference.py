import os
from logging import getLogger
from manafaln.core.configurators import DefaultConfigurator

class InferenceConfigurator(DefaultConfigurator):
    def __init__(self, app_name=None, description=None):
        super().__init__(app_name=app_name, description=description)

        self.logger = getLogger(__name__)

    def process_args(self, args) -> None:
        # Use default configurator to setup self.raw_config
        super().process_args(args)

        # Check ckpt_path if given
        if self.ckpt_path is None:
            self.logger.warn("Start inference without given checkpoint.")
        elif not os.path.exists(self.ckpt_path):
            raise ValueError(
                f"Checkpoint file {self.ckpt_path} does not exist."
            )

