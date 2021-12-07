import logging
from typing import Union
from itertools import chain
from torch.data import Dataset
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.trainer.supporters import CombinedDataset, CombinedLoader
from monai.data.dataset import SmartCacheDataset

class SmartCacheHandler(Callback):
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.logger = logging.getLogger(self.__class__.__name__)

        self.train_smart_cacher = []
        self.valid_smart_cahcer = []

    def get_dataset(self, dataloader):
        if dataloader is not None:
            return dataloader.dataset

    # Note: pytorch_lightning CombinedDataset is NOT a torch Dataset
    def extract_smart_cacher(self, dataset: Union[Dataset, CombinedDataset]):
        if isinstance(dataset, SmartCacheDataset):
            return [dataset]
        elif isinstance(dataset, CombinedDataset):
            # Get underlying datasets
            combined_datasets = dataset.datasets
            if isinstance(combined_datasets, SmartCacheDataset):
                return [combined_datasets]
            elif isinstance(combined_datasets, Sequence):
                smcs = []
                for ds in combined_datasets:
                    if isinstance(ds, SmartCacheDataset):
                        smcs.append(ds)
                return smcs
        return []

    def on_train_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule
    ) -> None:
        # Get smc dataset from the trainer
        self.train_smart_cacher = self.extract_smart_cacher(
            self.get_dataset(trainer.train_dataloader)
        )
        self.valid_smart_cacher = self.extract_smart_cacher(
            self.get_dataset(trainer.val_dataloader)
        )

        if self.verbose:
            self.logger.info(
                f"Smart Cache Handler startup summary:",
                f"Found {len(self.train_smart_cacher)} smc datasets for training",
                f"Found {len(self.valid_smart_cacher)} smc datasets for validation"
            )

        # Start all smc datasets
        for smc in chain(self.train_smart_cacher, self.valid_smart_cacher):
            smc.start()

    def on_train_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule
    ) -> None:
        if self.verbose:
            self.logger.info(
                f"Updating {len(self.train_smart_cacher)} smart cache dataset(s) for training"
            )

        for smc in self.train_smart_cacher:
            smc.update_cache()

    def on_validate_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule
    ) -> None:
        if self.verbose:
            self.logger.info(
                f"Updating {len(self.valid_smart_cacher)} smart cache dataset(s) for validation"
            )

        for smc in self.valid_smart_cacher:
            smc.update_cache()

    def on_train_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule
    ) -> None:
        if self.verbose:
            self.logger.info("Shutting down all smart cache dataset(s)")

        # Shutdown all smart cache datasets
        for smc in chain(self.train_smart_cacher, self.valid_smart_cacher):
            smc.shutdown()

        # Release references
        self.train_smart_cacher = []
        self.valid_smart_cacher = []
