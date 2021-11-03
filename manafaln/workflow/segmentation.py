import torch
import monai
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.distributed import rank_zero_only

from manafaln.common.constants import ComponentType
from manafaln.utils.components import instantiate

class SupervisedSegmentation(LightningModule):
    def __init__(self, config: dict):
        super().__init__()

        # Check hyperparameters for conflict

        # Save all hyperparameters
        self.save_hyperparameters(config)

        self.model = instantiate(
            name=config["model"]["name"],
            path=config["model"].get("path", None),
            component_type=ComponentType.MODEL,
            **config["model"].get("args", {})
        )

        self.inferer = instantiate(
            name=config["inferer"]["name"],
            path=config["inferer"].get("path", None),
            component_type=ComponentType.INFERER,
            **config["inferer"].get("args", {})
        )

        self.loss_fn = instantiate(
            name=config["loss"]["name"],
            path=config["loss"].get("path", None),
            component_type=ComponentType.LOSS,
            **config["loss"].get("args", {})
        )

        # TODO: Add training/validation metrics here
        # train_metrics = []
        # valid_metrics = []

    def forward(self, data):
        return self.inferer(data, self.model)

    def training_step(self, batch, batch_idx):
        image = batch["image"]
        label = batch["label"]

        preds = self.model(image)
        loss = self.loss_fn(preds, label)

        # Log loss value
        self.log_dict({ "train_loss": loss })

        return loss

    @rank_zero_only
    def summarize_validation(self, metrics):
        pass

    def validation_step(self, batch, batch_idx):
        image = batch["image"]
        label = batch["label"]

        preds = self.forward(image)

        # Compute metrics here

        return None

    def validation_epoch_end(self, validation_step_outputs):
        return None

    def configure_optimizers(self):
        opt_config = getattr(self.hparams, "optimizer")
        sch_config = getattr(self.hparams, "scheduler", None)

        opt = instantiate(
            name=opt_config["name"],
            path=opt_config.get("path", None),
            component_type=ComponentType.OPTIMIZER,
            params=self.model.parameters(),
            **opt_config.get("args", {})
        )

        if sch_config is not None:
            sch = {
                "scheduler": instantiate(
                    name=sch_config["name"],
                    path=sch_config.get("path", None),
                    component_type=ComponentType.SCHEDULER,
                    optimizer=opt,
                    **sch_config.get("args", {})
                ),
                "interval": self.hparams.settings.get("interval", "epoch"),
                "frequency": self.hparams.settings.get("frequency", 1)
            }
            return [opt], [sch]

        return opt


