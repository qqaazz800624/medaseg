import json
import os
from typing import Any, Dict, List

import torch
from monai.data import DataLoader, Dataset
from monai.utils import PostFix
from monai.utils.misc import ImageMetaKey
from pytorch_lightning import LightningModule, Trainer
from ts.context import Context
from ts.torch_handler.base_handler import BaseHandler

from manafaln.apps.utils import build_callbacks, build_workflow
from manafaln.common.constants import DefaultKeys
from manafaln.core.transforms import build_transforms
from manafaln.utils import load_yaml
from manafaln.utils.misc import ensure_python_value


class SimpleHandler(BaseHandler):
    """
    This is a simple handler used by TorchServe inference server for Manafaln workflow.
    initialize() is invoked once and only once when the model is loaded.
    handle() is invoked for every inference request, which will sequentially call:
        preprocess() to process request
        inference() to run the model
        postprocess() to prepare the response
    """
    trainer: Trainer
    workflow: LightningModule
    def __init__(self):
        self.context: Context = None
        self.initialized: bool = False

        # Set input and output keys
        self.input_key = DefaultKeys.INPUT_KEY    # "image"
        self.output_key = DefaultKeys.OUTPUT_KEY  # "preds"

        # Paths to load and save data
        # Defaults to paths in pytorch/torchserve Docker container
        self.inputs_path = "/home/model-server/inputs"
        self.outputs_path = "/home/model-server/outputs"

    def get_checkpoint_path(self):
        """
        Get checkpoint path with .ckpt extension from archived model directory.
        """
        model_dir = self.context.system_properties.get("model_dir")
        files = [os.path.join(model_dir, f) for f in os.listdir(model_dir)]
        ckpt_paths = [f for f in files if f.endswith(".ckpt")]
        if len(ckpt_paths) != 1:
            raise RuntimeError(f"Model initialization failed, {len(ckpt_paths)} checkpoint file(s) found.")
        self.ckpt_path = ckpt_paths[0]
        return self.ckpt_path

    def get_config(self):
        """
        Get config path with .yaml or .json extension from archived model directory.
        """
        model_dir = self.context.system_properties.get("model_dir")
        files = [os.path.join(model_dir, f) for f in os.listdir(model_dir)]
        config_paths = [f for f in files if f.endswith((".yaml", ".json"))]
        if len(config_paths) != 1:
            raise RuntimeError(f"Model initialization failed, {len(config_paths)} config file(s) found.")
        self.config = load_yaml(config_paths[0])
        return self.config

    def set_device(self):
        """
        Set device from gpu_id system property.
        """
        device = self.context.system_properties.get("gpu_id")
        if device is not None and torch.cuda.is_available():
            self.device = "cuda:"+str(device)
        else:
            self.device = "cpu"

    def set_trainer(self):
        """
        Set Lightning Trainer where predict method will be called for inference.
        """
        if "trainer" not in self.config:
            self.config["trainer"] = {}
        if "settings" not in self.config["trainer"]:
            self.config["trainer"]["settings"] = {}

        # Set device
        if self.device == "cpu":
            self.config["trainer"]["settings"]["accelerator"] = "cpu"
            self.config["trainer"]["settings"]["devices"] = "auto"
        else:
            self.config["trainer"]["settings"]["accelerator"] = "gpu"
            # e.g. "cuda:0,1" => [0, 1]
            self.config["trainer"]["settings"]["devices"] = [int(d) for d in self.device.split(":")[1].split(",")]

        # Set logger and checkpointing
        self.config["trainer"]["settings"]["logger"] = False
        self.config["trainer"]["settings"]["enable_checkpointing"] = False

        # Set callbacks
        blacklist = ["ModelCheckpoint"]
        callbacks = self.config["trainer"].get("callbacks", [])
        callbacks = [c for c in callbacks if c["name"] not in blacklist]
        callbacks = build_callbacks(callbacks)

        # Initialize trainer
        self.trainer = Trainer(
            callbacks=callbacks,
            **self.config["trainer"]["settings"]
        )

    def set_workflow(self):
        """
        Initializes workflow from config.
        """
        self.workflow: LightningModule = build_workflow(self.config["workflow"], ckpt=self.ckpt_path)
        self.workflow.eval()
        self.workflow.to(device=self.device)

    def set_transforms(self):
        """
        Initializes preprocessing transforms for Dataset.
        """
        transforms = self.config.get("data", {}).get("predict", {}).get("transforms", [])
        self.transforms = build_transforms(transforms)

    def initialize(self, context: Context):
        """
        Initializes trainer and workflow.

        Args:
            context (context): It is a JSON Object containing information
            pertaining to the model artifacts parameters.
        """
        self.context = context

        self.get_checkpoint_path()
        self.get_config()

        self.set_device()
        self.set_trainer()
        self.set_workflow()
        self.set_transforms()

        self.initialized = True

    def preprocess(self, _) -> List[Any]:
        """
        Takes the data from the input request and pre-processes it.
        In this handler, we ignore the input and return paths in self.inputs_path.
        """
        input_files = os.listdir(self.inputs_path)
        input_files = [os.path.join(self.inputs_path, f) for f in input_files]
        return input_files

    def get_data_list(self, inputs: List[Any]) -> List[Dict[str, Any]]:
        """
        Get data list that will be passed to Dataset.
        """
        data_list = [{self.input_key: data} for data in inputs]
        return data_list

    def get_dataloader(self, data_list: List[Dict[str, Any]]) -> DataLoader:
        """
        Create Dataloader from data list.
        In this handler, we use simple Dataset and Dataloader from MONAI.
        """
        dataset = Dataset(data=data_list, transform=self.transforms)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.context.system_properties.get("batch_size", 1),
            )
        return dataloader

    def inference(self, inputs: List[Any]) -> List[Dict[str, Any]]:
        """
        Make a prediction call on the given input request.
        In this handler, we generate data list from inputs, create dataloader and run inference.
        """
        data_list = self.get_data_list(inputs)
        dataloader = self.get_dataloader(data_list)
        outputs = self.trainer.predict(self.workflow, dataloader)
        outputs = [output[0] for output in outputs] # Get outputs from the first dataloader
        return outputs

    def postprocess(self, outputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Makes use of the output from the inference and converts into a Torchserve supported response output.
        In this handler, we save outputs to self.outputs_path with filename from meta dict and return outputs.
        """
        _outputs = {}
        for output in outputs:

            # Get filename to save output
            # filename = output["{self.input_key}_meta_dict"]["filename_or_obj"]
            filename = output[PostFix.meta(self.input_key)][ImageMetaKey.FILENAME_OR_OBJ]
            filename = os.path.basename(filename)

            # Convert torch tensor to list
            output = output[self.output_key]
            output = ensure_python_value(output)
            _outputs[filename] = output

            # Save output
            with open(os.path.join(self.outputs_path, filename) + ".json", "w") as f:
                json.dump(output, f)

        # Automatic batching of TorchServe expects batch size of 1
        _outputs = [_outputs]

        # Return outputs that will be responded by the server
        return _outputs
