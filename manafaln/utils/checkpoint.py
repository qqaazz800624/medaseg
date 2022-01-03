import torch
from pytorch_lightning import LightningModule
from manafaln.utils.builders import get_class

def restore_from_checkpoint(
        ckpt_path: str,
        config: dict = None
    ) -> LightningModule:
    checkpoint = torch.load(ckpt_path)

    if config is None:
        config = checkpoint["hyper_parameters"]["workflow"]

    Workflow = get_class(
        name=config["name"],
        path=config.get("path", None),
        component_type="workflow"
    )
    return Workflow.load_from_checkpoint(ckpt_path, config=config, strict=False)
