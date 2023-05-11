"""OHEM: Online hard example mining."""

from typing import Callable, Dict, List, Sequence

import torch
from monai.data import DataLoader
from monai.utils import ensure_tuple
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.callbacks.progress.tqdm_progress import Tqdm, TQDMProgressBar
from pytorch_lightning.trainer.supporters import CombinedLoader

from manafaln.common.constants import DefaultKeys
from manafaln.core.builders import ComponentBuilder, DataLoaderBuilder
from manafaln.data import DecathlonDataModule
from manafaln.utils import get_items, update_items
from manafaln.workflow import SupervisedLearningV2


class OnlineHardExampleMining(Callback):
    """
    Callback for online hard example mining.
    On every `epoch_interval` epoch, starting from `starting_epoch`, it will
        1. Inference the training dataset using the current model.
        2. Compute the hardness of every training instances.
        3. Create a weighted random sampler that is weighted by hardness.
        4. Replace the training dataloader with a new one with the sampler from step 3.

    Args:
        hardness_config (Dict): Config to build hardness function. Defaults to `torch.nn.L1Loss`.
        input_keys (Sequence[str]): The keys used for computing hardness.
        starting_epoch (int): The epoch to start mining.
        epoch_interval (int): The interval of mining.
        smooth (float): The smoothing factor that is added to all hardnesses.
    """

    def __init__(
        self,
        hardness_config: Dict = None,
        input_keys: Sequence[str] = (
            DefaultKeys.OUTPUT_KEY,
            DefaultKeys.LABEL_KEY,
        ),
        starting_epoch: int = 1,
        epoch_interval: int = 1,
        smooth: float = 0.0,
    ):
        # Build hardness function. Defaults to `torch.nn.L1Loss`.
        # The output of this function must be castable to float.
        if hardness_config is None:
            hardness_config = {"name": "L1Loss", "path": "torch.nn"}
        self.hardness_function: Callable = ComponentBuilder()(hardness_config)

        # Get the keys used for computing hardness
        self.input_keys = ensure_tuple(input_keys)

        # Smoothing factor
        self.smooth = smooth

        # Get the starting epoch and interval for mining
        if not isinstance(starting_epoch, int):
            raise ValueError("Starting epoch must be integer")
        self.starting_epoch = starting_epoch

        if not isinstance(epoch_interval, int) or epoch_interval < 1:
            raise ValueError("Epoch interval must be positive integer")
        self.epoch_interval = epoch_interval

        # Placeholder for datamodule, used to get the training dataset for mining
        self.datamodule: DecathlonDataModule = None

        # Placeholder for progress bar callback, used to get
        # `process_position` and `is_disabled` attribute to show mining progress
        self.progress_bar: TQDMProgressBar = None

    def on_fit_start(self, trainer: Trainer, pl_module: SupervisedLearningV2) -> None:
        # Get the datamodule from trainer
        self.datamodule = trainer.datamodule

        # Get the progress bar callback from trainer
        self.progress_bar = trainer.progress_bar_callback

    def get_hardnesses(self, pl_module: SupervisedLearningV2) -> List[float]:
        """
        Inference the training dataset using the current model and compute the hardness of every instances.
        """
        # Create a dataloader to inference training data using config of validation dataloader
        config = self.datamodule.hparams.data["validation"]["dataloader"]
        mining_dataloader = DataLoaderBuilder()(
            config={
                "name": config["name"],
                "path": config.get("path"),
                "args": {
                    **config["args"],
                    "shuffle": False,  # Make sure the order of hardness is the same as the training dataset
                    "sampler": None,
                    "drop_last": False,  # Make sure no example are dropped
                },
            },
            dataset=self.datamodule.get_train_dataset(),
        )

        # Get the hardness from prediction and label using a valiation loop without metric
        hardnesses = []
        pl_module.eval()
        with torch.no_grad():
            for batch in Tqdm(
                mining_dataloader,
                desc="Mining",
                position=self.progress_bar.process_position + 1,
                disable=self.progress_bar.is_disabled,
                leave=False,
                dynamic_ncols=True,
            ):
                batch = pl_module.transfer_batch_to_device(batch, pl_module.device, 0)

                model_input = get_items(batch, pl_module.model_input_keys)
                preds = pl_module.forward(*model_input)
                batch = update_items(batch, pl_module.model_output_keys, preds)

                if pl_module.decollate_fn["validation"] is not None:
                    for item in pl_module.decollate_fn["validation"](batch):
                        item = pl_module.post_transforms["validation"](item)
                        hardness_inputs = get_items(item, self.input_keys)
                        hardness = self.hardness_function(*hardness_inputs)
                        hardness = float(hardness)
                        hardness = hardness + self.smooth
                        hardnesses.append(hardness)
                else:
                    batch = pl_module.post_transforms["validation"](batch)
                    for item in batch:
                        hardness_inputs = get_items(item, self.input_keys)
                        hardness = self.hardness_function(*hardness_inputs)
                        hardness = float(hardness)
                        hardness = hardness + self.smooth
                        hardnesses.append(hardness)
        pl_module.train()

        return hardnesses

    def get_train_dataloader(self, pl_module: SupervisedLearningV2) -> DataLoader:
        """
        Create a dataloader with hard example miner as sampler.
        """
        # Get the hardness of every training instances
        hardnesses = self.get_hardnesses(pl_module)

        # Create the config for weighted random sampler with the hardnesses as weights
        miner_config = {
            "name": "WeightedRandomSampler",
            "args": {
                "weights": hardnesses,
                "num_samples": len(hardnesses),
                "replacement": True,
            },
        }

        # Create a dataloader with hard example miner as sampler
        config = self.datamodule.hparams.data["training"]["dataloader"]
        loader = DataLoaderBuilder()(
            config={
                "name": config["name"],
                "path": config.get("path"),
                "args": {
                    **config["args"],
                    "sampler": miner_config,  # override the sampler
                    "shuffle": False,  # disable shuffle as it is mutually exclusive with sampler
                },
            },
            dataset=self.datamodule.get_train_dataset(),
        )
        return loader

    def on_train_epoch_start(
        self, trainer: Trainer, pl_module: SupervisedLearningV2
    ) -> None:
        """
        For every `epoch_interval` epoch, starting from `starting_epoch`,
        replace the training dataloader with a new one with hard example miner as sampler.
        """
        # Skip if it is not the epoch for mining
        if trainer.current_epoch < self.starting_epoch:
            return
        if (trainer.current_epoch - self.starting_epoch) % self.epoch_interval != 0:
            return

        # Get the dataloader with hard example miner as sampler
        dataloader = self.get_train_dataloader(pl_module)

        # Dataloader in trainer is a CombinedLoader, so we need to initialize it before replacing
        trainer.train_dataloader = CombinedLoader(dataloader)
