import os
from typing import Dict, List, Literal

import numpy as np
import torch
from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.shareable import ReturnCode, Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.learner_spec import Learner
from nvflare.app_common.app_constant import AppConstants, ModelName, ValidateType
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from ruamel.yaml import YAML

from manafaln.adapters.nvflare.callbacks import (
    AbortTraining,
    RestoreLR
)
from manafaln.adapters.nvflare.utils import (
    load_weights,
    extract_weights
)
from manafaln.core.builders import (
    CallbackBuilder,
    DataModuleBuilder,
    WorkflowBuilder
)

class SimpleLearner(Learner):
    def __init__(
        self,
        config_train: str,
        config_valid: str,
        aggregation_interval: Literal["step", "epoch"] = "epoch",
        aggregation_frequency: int = 10,
        restore_local_ckpt: str = None,
        train_task_name: str = AppConstants.TASK_TRAIN,
        submit_model_task_name: str = AppConstants.TASK_SUBMIT_MODEL
    ):
        super().__init__()

        self.train_task_name = train_task_name
        self.submit_model_task_name = submit_model_task_name

        self.config_train = config_train
        self.config_valid = config_valid

        if not aggregation_interval in ["step", "epoch"]:
            raise ValueError("Aggregation interval must be one of 'step' or 'epoch'.")
        self.aggregation_interval = aggregation_interval
        self.aggregation_frequency = aggregation_frequency

        self.restore_local_ckpt = restore_local_ckpt

        self.app_root = None
        self.train_datamodule = None
        self.valid_datamodule = None

        self.num_steps_current_round = 0

        self.key_metric = None
        self.signal_handler = AbortTraining()

    def process_train_config(self, config: Dict):
        # Disable training limits
        settings = config["trainer"].get("settings", {})
        settings["max_steps"] = -1
        settings["max_epochs"] = -1

        if self.aggregation_interval == "step":
            settings["val_check_interval"] = self.aggregation_frequency
            settings["check_val_every_n_epoch"] = None
        else:
            settings["val_check_interval"] = None
            settings["check_val_every_n_epoch"] = self.aggregation_frequency

        config["trainer"]["settings"] = settings

        # Overwrite some settings for correct behavior
        config["trainer"]["settings"]["default_root_dir"] = self.app_root

        callbacks = config["trainer"].get("callbacks", [])
        for c in callbacks:
            if c["name"] == "ModelCheckpoint":
                c["args"]["dirpath"] = os.path.join(self.app_root, "models")
                self.key_metric = c["args"].get("monitor", None)
        config["trainer"]["callbacks"] = callbacks

        return config

    def process_valid_config(self, config: Dict):
        # Overwrite some settings for correct behavior
        config["trainer"]["settings"]["default_root_dir"] = self.app_root

        # Disable logging and checkpoints for validation,
        # otherwise there will be a lot of `versions` of logs
        # with only validation result
        config["trainer"]["settings"]["logger"] = False
        config["trainer"]["callbacks"] = [
            c for c in config["trainer"]["callbacks"] if c["name"] != "ModelCheckpoint"
        ]

        return config

    def build_data_module(self, config_data: Dict):
        builder = DataModuleBuilder()
        return builder(config_data)

    def build_workflow(self, config_workflow: Dict):
        builder = WorkflowBuilder()
        return builder(config_workflow)

    def build_callbacks(self, config_callbacks: List[Dict]):
        builder = CallbackBuilder()
        return [builder(c) for c in config_callbacks]

    def initialize(self, parts: dict, fl_ctx: FLContext):
        # Common setups
        self.app_root = fl_ctx.get_prop(FLContextKey.APP_ROOT)

        # Use ruamel.yaml for capability between JSON and YAML
        config_loader = YAML()

        # For training
        with open(self.config_train) as f:
            config_train = config_loader.load(f)
        config_train = self.process_train_config(config_train)

        self.workflow = self.build_workflow(config_train["workflow"])
        self.train_workflow = self.workflow
        self.train_datamodule = self.build_data_module(config_train["data"])

        # Configure training callbacks & insert signal handler
        train_callbacks = self.build_callbacks(config_train["trainer"].get("callbacks", []))
        train_callbacks.append(RestoreLR(from_checkpoint=self.restore_local_ckpt))
        train_callbacks.append(self.signal_handler)
        # Configure logger
        train_logger = TensorBoardLogger(save_dir="logs", name="")
        # Create trainer
        self.trainer = Trainer(
            callbacks=train_callbacks,
            logger=train_logger,
            **config_train["trainer"].get("settings", {})
        )
        self.trainer.num_sanity_val_steps = 0
        self.checkpoint_saver = self.trainer.checkpoint_callback

        # For (cross-site) validation
        with open(self.config_valid) as f:
            config_valid = config_loader.load(f)
        config_valid = self.process_valid_config(config_valid)

        # self.valid_workflow = self.build_workflow(config_valid["workflow"])
        self.valid_workflow = self.train_workflow
        self.valid_datamodule = self.build_data_module(config_valid["data"])

        # Insert signal handler as well
        valid_callbacks = self.build_callbacks(config_valid["trainer"].get("callbacks", []))
        valid_callbacks.append(self.signal_handler)
        # Skip logger for validation
        self.validator = Trainer(
            callbacks=valid_callbacks,
            **config_valid["trainer"].get("settings", {})
        )

    def local_train(self, fl_ctx: FLContext):
        # Manually set training length
        init_steps = self.trainer.global_step
        if self.aggregation_interval == "step":
            self.trainer.fit_loop.epoch_loop.max_steps = (
                self.trainer.global_step + self.aggregation_frequency
            )
        else:
            self.trainer.fit_loop.max_epochs = (
                self.trainer.current_epoch + self.aggregation_frequency
            )

        # Run training
        self.log_info(fl_ctx, "Start Lightning Trainer fit.")
        self.trainer.fit(self.workflow, self.train_datamodule)

        # Get number of steps
        self.num_steps_current_round = self.trainer.global_step - init_steps

    def train(self, data: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        # 1. Prepare datasets (training & validation)
        if self.train_datamodule is None:
            raise RuntimeError(
                "Missing training datamodule, please make sure to call initialize before training."
            )
        self.train_datamodule.setup(stage="fit")

        # 2. Log status
        current_round = data.get_header(AppConstants.CURRENT_ROUND)
        total_rounds = data.get_header(AppConstants.NUM_ROUNDS)
        self.log_info(fl_ctx, f"Current/Total Round: {current_round + 1}/{total_rounds}")
        self.log_info(fl_ctx, f"Client identity: {fl_ctx.get_identity_name()}")

        # Attach signal to handler callback
        self.signal_handler.attach_signal(abort_signal)

        # 5. Run local training
        self.local_train(fl_ctx)
        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)

        # Detach signal from handler callback
        self.signal_handler.detach_signal()

        # 6. Generate Shareable
        local_weights = extract_weights(self.train_workflow.model)
        meta = {
            MetaKey.NUM_STEPS_CURRENT_ROUND: self.num_steps_current_round
        }
        dxo = DXO(
            data_kind=DataKind.WEIGHTS,
            data=local_weights,
            meta=meta
        )
        return dxo.to_shareable()

    def get_model_for_validation(self, model_name: str, fl_ctx: FLContext) -> Shareable:
        if model_name == ModelName.BEST_MODEL:
            model_data = None
            try:
                model_data = torch.load(
                    self.checkpoint_saver.best_model_path,
                    map_location="cpu"
                )
                self.log_info(fl_ctx, f"Load best model from {self.checkpoint_saver.best_model_path}")
            except Exception as e:
                self.log_error(fl_ctx, f"Unable to load best model: {e}")

            if model_data:
                # Convert tensor to numpy and remove duplicated prefix
                data = {}
                for var_name in model_data["state_dict"]:
                    if var_name.startswith("model."):
                        data[var_name[6:]] = model_data["state_dict"][var_name].numpy()
                    else:
                        data[var_name] = model_data["state_dict"][var_name].numpy()
                dxo = DXO(data_kind=DataKind.WEIGHTS, data=data)
                return dxo.to_shareable()
            else:
                # Set return code.
                self.log_error(
                    fl_ctx,
                    f"best local model not found at {self.checkpoint_saver.best_model_path}."
                )
                return make_reply(ReturnCode.EXECUTION_RESULT_ERROR)
        else:
            raise ValueError(f"Unknown model_type {model_name}")

    def validate(self, data: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        # 1. Extract data from shareable
        model_owner = data.get_header(AppConstants.MODEL_OWNER, "?")
        validate_type = data.get_header(AppConstants.VALIDATE_TYPE)

        # 2. Prepare dataset
        if validate_type == ValidateType.BEFORE_TRAIN_VALIDATE:
            self.train_datamodule.setup(stage="fit")
            dm = self.train_datamodule
        elif validate_type == ValidateType.MODEL_VALIDATE:
            self.valid_datamodule.setup(stage="fit")
            dm = self.valid_datamodule

        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)

        # 3. Update model weight
        try:
            dxo = from_shareable(data)
        except:
            self.log_error(fl_ctx, "Error when extracting DXO from shareable")
            return make_reply(ReturnCode.BAD_TASK_DATA)

        if not dxo.data_kind == DataKind.WEIGHTS:
            self.log_exception(
                fl_ctx,
                f"DXO is of type {dxo.data_kind} but expected type WEIGHTS"
            )
            return make_reply(ReturnCode.BAD_TASK_DATA)
        load_weights(self.workflow.model, dxo.data)

        # 4. Run validation
        self.signal_handler.attach_signal(abort_signal)
        # self.validator.validate(
        self.validator.validate(
            self.workflow,
            dm.val_dataloader()
        )
        self.signal_handler.detach_signal()

        if abort_signal.triggered:
            return make_reply(ReturnCode.TASK_ABORTED)

        # Convert metrics from torch Tensor to Python type
        metrics = self.validator.callback_metrics
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                metrics[k] = v.tolist()

        self.log_info(
            fl_ctx,
            f"Validation metrics of {model_owner}'s model on"
            f" {fl_ctx.get_identity_name()}'s data: {metrics}"
        )

        # For validation before training, only key metric is needed
        if validate_type == ValidateType.BEFORE_TRAIN_VALIDATE:
            metrics = { MetaKey.INITIAL_METRICS: metrics[self.key_metric] }

        # 5. Return results
        dxo = DXO(
            data_kind=DataKind.METRICS,
            data=metrics
        )
        return dxo.to_shareable()

    def finalize(self, fl_ctx: FLContext):
        if self.train_datamodule:
            self.train_datamodule.teardown("fit")
        if self.valid_datamodule:
            self.valid_datamodule.teardown("fit")

