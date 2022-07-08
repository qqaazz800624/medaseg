from manafaln.core.configurators import Configurator, load_config

class ValidateConfigurator(Configurator):
    def __init__(self, app_name=None, description=None):
        super().__init__(app_name=app_name, description=description)

        self.app_parser.add_argument(
            "--config", "-c", type=str, default=None, help="Path to config file."
        )
        self.app_parser.add_argument(
            "--trainer", "-t", type=str, default=None, help="Path to trainer config file."
        )
        self.app_parser.add_argument(
            "--data", "-d", type=str, default=None, help="Path to data config file."
        )
        self.app_parser.add_argument(
            "--workflow", "-w", type=str, default=None, help="Path to workflow config file."
        )
        self.app_parser.add_argument(
            "--ckpt", "-f", type=str, help="Path to model checkpoint file."
        )

        self.ckpt_path = None

    def process_args(self, args) -> None:
        if args.config:
            config = load_config(args.config)
            self.logger.info(f"Load global configuration from {args.config}.")
        elif None in [args.trainer, args.data, args.workflow]:
            # At least one configuration is missing
            raise ValueError(
                "Must provide a complete config or all three trainer, data and workflow configs."
            )
        else:
            # Config details from different config files
            config = OrderedDict()

        # Override the configuration by component config file
        for f, c in zip([args.trainer, args.data, args.workflow], ["trainer", "data", "workflow"]):
            if f:
                config_f = load_config(f)
                config[c] = config_f[c]
                self.logger.info(f"Load {c} configuration from {f}.")

        self.raw_config = config

        if not os.path.exists(args.ckpt):
            raise ValueError(f"Checkpoint file {args.ckpt} does not exist.")
        self.ckpt_path = args.ckpt

    def configure_trainer(self) -> Dict:
        return self.raw_config["trainer"]

    def configure_data(self) -> Dict:
        return self.raw_config["data"]

    def configure_workflow(self) -> Dict:
        return self.raw_config["workflow"]

    def validate_config(self) -> None:
        super().validate_config()

    def get_ckpt_path(self):
        return self.cpkt_path

