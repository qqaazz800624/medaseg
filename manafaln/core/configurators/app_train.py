from collections import OrderedDict
from manafaln.core.configurators import DefaultConfigurator

class TrainConfigurator(DefaultConfigurator):
    def __init__(self, app_name=None, description=None):
        super().__init__(app_name=app_name, description=description)

        self.app_parser.add_argument(
            "--seed", "-s", type=int, default=None, help="Random seed"
        )

    def process_args(self, args):
        super().process_args(args)

        # Get random seed if set
        self.seed = args.seed

    def get_random_seed(self):
        return self.seed

