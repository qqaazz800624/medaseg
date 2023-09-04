import logging
from typing import Union, Sequence
from itertools import chain
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.trainer.supporters import CombinedDataset, CombinedLoader
from monai.data.dataset import SmartCacheDataset

class SmartCacheHandler(Callback):
    """
    Callback for handling smart cache datasets during training and validation.
    """

    def __init__(self, verbose=False):
        """
        Initialize the SmartCacheHandler.

        Args:
            verbose (bool, optional): Whether to print verbose information. Defaults to False.
        """
        self.verbose = verbose
        self.logger = logging.getLogger(self.__class__.__name__)

        self.train_smart_cacher = None
        self.valid_smart_cacher = None

    def get_dataset(self, dataloader):
        """
        Get the dataset from a dataloader.

        Args:
            dataloader (DataLoader or CombinedLoader or Sequence): The dataloader.

        Returns:
            Dataset or list[Dataset]: The dataset(s) from the dataloader.
        """
        if isinstance(dataloader, DataLoader) or isinstance(dataloader, CombinedLoader):
            return dataloader.dataset
        elif isinstance(dataloader, Sequence):
            return [loader.dataset for loader in dataloader]
        else:
            return None

    def extract_smart_cacher(self, dataset: Union[Dataset, CombinedDataset, Sequence]):
        """
        Extract the SmartCacheDataset(s) from a dataset or CombinedDataset.

        Args:
            dataset (Dataset or CombinedDataset or Sequence): The dataset or CombinedDataset.

        Returns:
            list[SmartCacheDataset]: The extracted SmartCacheDataset(s).
        """
        if isinstance(dataset, SmartCacheDataset):
            return [dataset]
        elif isinstance(dataset, CombinedDataset):
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
        """
        Callback function called at the start of each training epoch.

        Args:
            trainer (Trainer): The trainer object.
            pl_module (LightningModule): The LightningModule.
        """
        if self.train_smart_cacher is None:
            self.train_smart_cacher = self.extract_smart_cacher(
                self.get_dataset(trainer.train_dataloader)
            )
            for smc in self.train_smart_cacher:
                smc.start()

    def on_train_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule
    ) -> None:
        """
        Callback function called at the end of each training epoch.

        Args:
            trainer (Trainer): The trainer object.
            pl_module (LightningModule): The LightningModule.
        """
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
        """
        Callback function called at the start of each validation epoch.

        Args:
            trainer (Trainer): The trainer object.
            pl_module (LightningModule): The LightningModule.
        """
        if self.valid_smart_cacher is None:
            self.valid_smart_cacher = self.extract_smart_cacher(
                self.get_dataset(trainer.val_dataloaders)
            )
            for smc in self.valid_smart_cacher:
                smc.start()

    def on_validate_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule
    ) -> None:
        """
        Callback function called at the end of each validation epoch.

        Args:
            trainer (Trainer): The trainer object.
            pl_module (LightningModule): The LightningModule.
        """
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
        """
        Callback function called at the end of training.

        Args:
            trainer (Trainer): The trainer object.
            pl_module (LightningModule): The LightningModule.
        """
        if self.verbose:
            self.logger.info("Shutting down all smart cache dataset(s)")

        for smc in chain(self.train_smart_cacher, self.valid_smart_cacher):
            smc.shutdown()

        self.train_smart_cacher = None
        self.valid_smart_cacher = None
