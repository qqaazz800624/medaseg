import logging
from typing import Union, Sequence
from itertools import chain
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.trainer.supporters import CombinedDataset, CombinedLoader
from monai.data.dataset import SmartCacheDataset

class SmartCacheHandler(Callback):
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.logger = logging.getLogger(self.__class__.__name__)

        self.train_smart_cacher = None
        self.valid_smart_cacher = None

    def get_dataset(self, dataloader):
        if isinstance(dataloader, DataLoader) or isinstance(dataloader, CombinedLoader):
            return dataloader.dataset
        elif isinstance(dataloader, Sequence):
            return [loader.dataset for loader in dataloader]
        else:
            return None

    # Note: pytorch_lightning CombinedDataset is NOT a torch Dataset
    def extract_smart_cacher(self, dataset: Union[Dataset, CombinedDataset, Sequence]):
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
        elif isinstance(dataset, Sequence):
            smcs = []
            for ds in dataset:
                if isinstance(ds, SmartCacheDataset):
                    smcs.append(ds)
                return smcs
        return []

    def on_train_epoch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule
    ) -> None:
        if self.train_smart_cacher is None:
            # Get smc dataset from the trainer
            self.train_smart_cacher = self.extract_smart_cacher(
                self.get_dataset(trainer.train_dataloader)
            )
            # Start all SMC
            for smc in self.train_smart_cacher:
                smc.start()

    def on_train_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule
    ) -> None:
        if self.verbose and len(self.train_smart_cacher) > 0:
            self.logger.info(
                f"Updating {len(self.train_smart_cacher)} smart cache dataset(s) for training"
            )

        for smc in self.train_smart_cacher:
            smc.update_cache()

    def on_validation_epoch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule
    ) -> None:
        if self.valid_smart_cacher is None:
            # Note: the 's' in val_dataloaders is not a typo
            self.valid_smart_cacher = self.extract_smart_cacher(
                self.get_dataset(trainer.val_dataloaders)
            )
            # Start all SMC
            for smc in self.valid_smart_cacher:
                smc.start()


    def on_validate_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule
    ) -> None:
        if self.verbose and len(self.valid_smart_cacher) > 0:
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
        self.train_smart_cacher = None
        self.valid_smart_cacher = None
