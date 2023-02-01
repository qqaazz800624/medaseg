from manafaln.core.configurators import DefaultConfigurator

class TrainConfigurator(DefaultConfigurator):
    def __init__(self, app_name=None, description=None):
        super().__init__(app_name=app_name, description=description)

        self.app_parser.add_argument(
            "--seed", "-s", type=int, help="Random seed for deterministic training."
        )

        self.random_seed = None

    def process_args(self, args) -> None:
        # Use default configurator to setup self.raw_config
        super().process_args(args)
        self.random_seed = args.seed

    def get_random_seed(self):
        return self.random_seed

