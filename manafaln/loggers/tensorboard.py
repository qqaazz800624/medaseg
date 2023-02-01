import os
from typing import Any
import logging

from pytorch_lightning.loggers import TensorBoardLogger as _TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_only

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
        super().__init__(save_dir, **kwargs)
        logger.info(f"Logs are saved in {self.log_dir}")

    @rank_zero_only
    def save(self) -> None:
        dir_path = self.log_dir
        hparams_file = os.path.join(dir_path, self.NAME_HPARAMS_FILE)

        # save the metatags file if it doesn't exist and the log directory exists
        if self._fs.isdir(dir_path) and not self._fs.isfile(hparams_file):
            save_yaml(hparams_file, self.hparams)
