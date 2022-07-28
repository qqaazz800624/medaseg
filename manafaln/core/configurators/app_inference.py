import os
from manafaln.core.configurators import DefaultConfigurator

class InferenceConfigurator(DefaultConfigurator):
    def __init__(self, app_name=None, description=None):
        super().__init__(app_name=app_name, description=description)

        # Add non-optional ckpt_path argument
        self.app_parser.add_argument(
            "--ckpt", "-f", type=str, help="Path to model checkpoint file."
        )

        self.ckpt_path = None

    def process_args(self, args) -> None:
        # Use default configurator to setup self.raw_config
        super().process_args(args)

        # Set ckpt_path
        if not os.path.exists(args.ckpt):
            raise ValueError(f"Checkpoint file {args.ckpt} does not exist.")
        self.ckpt_path = args.ckpt

    def get_ckpt_path(self):
        return self.ckpt_path

