import os
from typing import Any
import logging

from lightning.pytorch.loggers import TensorBoardLogger as _TensorBoardLogger
from lightning.pytorch.utilities import rank_zero_only

from manafaln.utils import save_yaml

logger = logging.getLogger(__name__)

class TensorBoardLogger(_TensorBoardLogger):
    """
    Replaces hparams_file saver of pytorch_lightning.loggers.TensorBoardLogger
    with ruamel.yaml.YAML
    """

    def __init__(
        self,
        save_dir: os.PathLike = os.getcwd(),
        **kwargs: Any,
    ):
        """
        Initializes a TensorBoardLogger object.

        Args:
            save_dir (os.PathLike, optional): Directory to save the logs. Defaults to current working directory.
            **kwargs: Additional keyword arguments to be passed to the parent class constructor.
        """
        super().__init__(save_dir, **kwargs)
        logger.info(f"Logs are saved in {self.log_dir}")

    @rank_zero_only
    def save(self) -> None:
        """
        Saves the hparams file using ruamel.yaml.YAML.

        The hparams file is saved in the log directory if it doesn't exist.

        Returns:
            None
        """
        dir_path = self.log_dir
        hparams_file = os.path.join(dir_path, self.NAME_HPARAMS_FILE)

        # save the metatags file if it doesn't exist and the log directory exists
        if self._fs.isdir(dir_path) and not self._fs.isfile(hparams_file):
            save_yaml(hparams_file, self.hparams)
