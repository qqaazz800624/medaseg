import os
import logging
from typing import Any, Optional

from ruamel.yaml import YAML
from monailabel.interfaces.config import TaskConfig
from monailabel.interfaces.tasks.infer_v2 import InferTask
from monailabel.interfaces.tasks.scoring import ScoringMethod
from monailabel.interfaces.tasks.strategy import Strategy
from monailabel.interfaces.tasks.train import TrainTask
from monailabel.tasks.activelearning.epistemic import Epistemic
from monailabel.tasks.scoring.dice import Dice
from monailabel.tasks.scoring.epistemic import EpistemicScoring
from monailabel.tasks.scoring.sum import Sum

from manafaln.core.builders import ModelBuilder, TransformBuilder

# import lib.infers
# import lib.trainers
from manafaln.adapters.monailabel.infers import ManafalnInferTask
from manafaln.adapters.monailabel.trainers import ManafalnTrainTask

logger = logging.getLogger(__name__)


class ManafalnSegmentation(TaskConfig):
    def __init__(self):
        super().__init__()

        self.epistemic_enabled = None
        self.epistemic_samples = None

    def init(
        self,
        name: str,
        model_dir: os.PathLike[str],
        conf: dict,
        planner: Any,
        **kwargs
    ) -> None:
        super().init(name, model_dir, conf, planner, **kwargs)

        config_file = conf.get("mfn_config", None)
        if config_file is None:
            raise ValueError(
                "--conf mfn_config <config_file_path> must be"
                "provided for manafaln segmentation app."
            )

        yaml = YAML()
        with open(config_file, "r") as f:
            config = yaml.load(f)

        self.config = {}
        self.config["app"] = dict(config["app"])
        self.config["infer"] = dict(config["task_infer"])
        self.config["train"] = dict(config["task_train"])

        # Define labels
        self.labels = self.config["app"]["labels"]

        self.path = [
            os.path.join(self.model_dir, self.config["app"]["model_path"]),
            os.path.join(self.model_dir, f"{name}.ckpt")
        ]

        # Use config to control epistemic settings instead of conf
        self.epistemic_enabled = self.config["app"].get("epistemic_enabled", False)
        self.epistemic_samples = self.config["app"].get("epistemic_samples", 5)
        logger.info(f"EPISTEMIC Enabled: {self.epistemic_enabled};"
                    f"Samples: {self.epistemic_samples}"
        )

    def _build_epistemic_scoring(self):
        # Create a new model instance (require dropout) for EpistemicScoring
        model_config = self.config["app"].get("epistemic_model", None)
        if model_config is None:
            raise ValueError(
                "`epistemic_model` must be set in the app configuration"
                "if `epistemic_enabled=True`."
            )
        model_builder = ModelBuilder()
        network = model_builder(model_config)

        # Use infer pre_transforms here
        transform_builder = TransformBuilder()
        transforms = [t for t in self.config["infer"]["pre_transforms"]]
        transforms = [transform_builder(t) for t in transforms]

        return EpistemicScoring(
            model=self.path,
            network=network,
            transforms=transforms,
            num_samples=self.epistemic_samples
        )

    def infer(self) -> InferTask | dict[str, InferTask]:
        return ManafalnInferTask(
            self.config["infer"],
            model_path=self.path,
            labels=self.labels,
            model_type=self.config["app"]["model_type"],
            model_state_dict=self.config["app"].get("model_state_dict", "state_dict")
        )

    def trainer(self) -> TrainTask | None:
        model_dir = os.path.join(self.model_dir, self.name)
        load_path = self.path[0] if os.path.exists(self.path[0]) else self.path[1]

        return ManafalnTrainTask(
            self.config["train"],
            labels=self.labels,
            model_dir=model_dir,
            load_path=load_path,
            publish_path=self.path[1],
            model_dict_key=self.config["app"].get("model_dict_key", "state_dict")
        )

    def strategy(self) -> dict[str, Strategy]:
        strategies: dict[str, Strategy] = {}
        if self.epistemic_enabled:
            strategies[f"{self.name}_epistemic"] = Epistemic()
        return strategies

    def scoring_method(self) -> dict[str, ScoringMethod]:
        method: dict[str, ScoringMethod] = {
            "dice": Dice(),
            "sum": Sum()
        }

        if self.epistemic_enabled:
            methods[f"{self.name}_epistemic"] = self._build_epistemic_scoring()

        return methods

