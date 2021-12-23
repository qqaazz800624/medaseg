import os
import json
import traceback
from typing import Dict

import torch
import numpy as np
from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable
from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.pt.pt_file_model_persistor import (
    PTModelPersistenceFormatManager
)
from pytorch_lightning import Trainer, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from manafaln.utils.builders import (
    build_callback,
    build_data_module,
    build_workflow
)

class AbortTraining(Callback):
    def __init__(self):
        super(AbortTraining).__init__()

        self.signal_attached = False

    def attach_signal(self, signal: Signal):
        self.signal = signal
        self.signal_attached = True

    def detach_signal(self):
        self.signal_attached = False

    def _handle_signal(self, trainer):
        if self.signal_attached and self.signal.triggered:
            trainer.fit_loop.should_stop = True

    def on_sanity_check_end(self, trainer, pl_module):
        self._handle_signal(trainer)

    def on_batch_end(self, trainer, pl_module):
        self._handle_signal(trainer)

class LightningTrainer(Executor):
    def __init__(
        self,
        config: str = "config/config_train.json",
        aggregation_epochs: int = 1,
        train_task_name=AppConstants.TASK_TRAIN,
        submit_model_task_name=AppConstants.TASK_SUBMIT_MODEL,
        exclude_vars=None
    ):
        super(LightningTrainer, self).__init__()

        self.config_file = config
        self.aggregation_epochs = aggregation_epochs
        self._train_task_name = train_task_name
        self._submit_model_task_name = submit_model_task_name

    def patch_config(self, config: Dict) -> Dict:
        # Remove unwanted settings
        if config["trainer"].get("settings", None):
            config["trainer"]["settings"].pop("max_steps", None)
            config["trainer"]["settings"].pop("max_epochs", None)
        else:
            config["trainer"]["settings"] = {}

        # Overwrite some settings for correct behavior
        config["trainer"]["settings"]["default_root_dir"] = self.app_root
        config["trainer"]["settings"]["strategy"] = "ddp"

        callbacks = config["trainer"].get("callbacks", [])
        for c in callbacks:
            if c["name"] == "ModelCheckpoint":
                c["args"]["dirpath"] = os.path.join(self.app_root, "models")
        config["trainer"]["callbacks"] = callbacks

        return config

    def apply_weights(self, model_weights: Dict[str, np.ndarray]):
        model = self.workflow.model

        local_var_dict = model.state_dict()
        model_keys = model_weights.keys()
        for var_name in local_var_dict:
            if var_name in model_keys:
                weights = model_weights[var_name]
                try:
                    local_var_dict[var_name] = torch.as_tensor(
                        np.reshape(weights, local_var_dict[var_name].shape)
                    )
                except Exception as e:
                    raise ValueError(
                        f"Convert weight from {var_name} failed with error {str(e)}"
                    )

        model.load_state_dict(local_var_dict)

    def extract_weights(self) -> Dict[str, np.ndarray]:
        local_state_dict = self.workflow.model.state_dict()
        local_model_dict = {}
        for var_name in local_state_dict:
            try:
                local_model_dict[var_name] = local_state_dict[var_name].cpu().numpy()
            except Exception as e:
                raise ValueError(
                    f"Convert weight from {var_name} failed with error: {str(e)}"
                )

        return local_model_dict

    def local_train(self):
        # Disable sanity checks
        self.trainer.num_sanity_val_steps = 0

        # Modify max_epochs & continue training
        self.trainer.fit_loop.max_epochs = (
            self.trainer.current_epoch + self.aggregation_epochs
        )

        # In the trainer.fit, avoid passing data module directly, since
        # this will cause trainer to call setup and teardown multiple times
        print("Start Lightning Trainer fit")
        self.trainer.fit(
            self.workflow,
            train_dataloaders=self.data.train_dataloader(),
            val_dataloaders=self.data.val_dataloader()
        )

    def local_validate(self):
        # Run validation manually
        self.trainer.validate(self.workflow, self.data.val_dataloader())

        # Make sure all metrics are on the same device
        if self.checkpoint_saver.current_score is not None:
            device = self.checkpoint_saver.current_score.device
            metrics = {}
            for key, value in self.trainer.callback_metrics.items():
                if isinstance(value, torch.Tensor):
                    metrics[key] = value.to(device)
                else:
                    metrics[key] = value
            self.trainer.logger_connector._callback_metrics = metrics
        # Use saver to save checkpoint
        # The saver will be able to get metrics from the last validation,
        # and handle and compare to the previous results as usual
        # TODO: still have some problem, need fix
        # self.checkpoint_saver.save_checkpoint(self.trainer)

    def generate_shareable(self) -> Shareable:
        if self.achieved_meta is None:
            meta = {MetaKey.NUM_STEPS_CURRENT_ROUND: self.trainer.global_step}
        else:
            meta = self.achieved_meta
            meta[MetaKey.NUM_STEPS_CURRENT_ROUND] = self.trainer.global_step
        return DXO(
            data_kind=DataKind.WEIGHTS,
            data=self.extract_weights(),
            meta=meta
        ).to_shareable()

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        try:
            if event_type == EventType.START_RUN:
                self.app_root = fl_ctx.get_prop(FLContextKey.APP_ROOT)

                # Setup training
                config = os.path.join(self.app_root, self.config_file)
                with open(config) as f:
                    config = json.load(f)

                # Process config for FL & get necessary information
                self.config = self.patch_config(config)

                # Setup lightning data module
                self.data = build_data_module(self.config["data"])

                # Setup model and training process
                self.workflow = build_workflow(self.config["workflow"])

                # Build user-defiend callbacks
                callbacks = self.config["trainer"]["callbacks"]
                callbacks = [build_callback(c) for c in callbacks]

                # Insert necessary callbacks for FL
                self.signal_handler = AbortTraining()
                callbacks.append(self.signal_handler)

                # Create custom logger
                tb_logger = TensorBoardLogger(save_dir="logs", name="")

                # Configure lightning trainer
                self.trainer = Trainer(
                    callbacks=callbacks,
                    logger=tb_logger,
                    **self.config["trainer"]["settings"]
                )
                self.checkpoint_saver = self.trainer.checkpoint_callback

                # Setup persistence manager
                self.default_train_conf = {
                    "train": {"model": type(self.workflow.model).__name__}
                }
                self.persistence_manager = PTModelPersistenceFormatManager(
                    data=self.workflow.model.state_dict(),
                    default_train_conf=self.default_train_conf
                )

                # Manually initialize lightning data module
                self.data.setup()
                self.train_dataloader = self.data.train_dataloader()
                self.val_dataloader = self.data.val_dataloader()
            elif event_type == EventType.ABORT_TASK:
                # Nothing can do here
                pass
            elif event_type == EventType.END_RUN:
                self.data.teardown()
        except Exception as e:
            self.log_exception(f"Exception occured while handling event {e}")
            self.log_exception(traceback.format_exc())

    def execute(
        self,
        task_name: str,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal
    ) -> Shareable:
        try:
            if task_name == self._train_task_name:
                dxo = from_shareable(shareable)
                # Check if dxo valid
                if not isinstance(dxo, DXO):
                    self.log_exception(
                        fl_ctx,
                        f"dxo expects type DXO. Got {type(dxo)} instead"
                    )
                    shareable.set_return_code(ReturnCode.EXECUTION_EXCEPTION)
                    return shareable

                # Ensure data kind in DXO is weight
                if not dxo.data_kind == DataKind.WEIGHTS:
                    self.log_exception(
                        fl_ctx,
                        f"data_kind expected WEIGHTS but got {dxo.data_kind} instead.",
                    )
                    shareable.set_return_code(ReturnCode.EXECUTION_EXCEPTION)
                    return shareable

                # Apply received weights to local model
                self.apply_weights(dxo.data)
                self.achieved_meta = dxo.meta

                # Attach signal handler before and trainer actions
                self.signal_handler.attach_signal(abort_signal)

                # Evaluate local model before training
                # Also save checkpoint if necessary
                self.local_validate()
                # Don't continue if abort triggered
                if abort_signal.triggered:
                    return make_reply(ReturnCode.TASK_ABORTED)

                # Run training
                self.local_train()
                if abort_signal.triggered:
                    return make_reply(ReturnCode.TASK_ABORTED)

                print(self.checkpoint_saver.best_model_path)

                # Run validation before submitting model
                self.local_validate()
                if abort_signal.triggered:
                    return make_reply(ReturnCode.TASK_ABORTED)

                # Reset signal handler
                self.signal_handler.detach_signal()

                # Generate shareable from current model
                return self.generate_shareable()
            elif task_name == self._submit_model_task_name:
                # Get current local model
                model = self.load_local_model(fl_ctx)
                # Create DXO
                dxo = model_learnable_to_dxo(model)
                return dxo.to_shareable()
            else:
                return make_reply(ReturnCode.TASK_UNKNOWN)
        except Exception as e:
            self.log_exception(fl_ctx, f"Exception in LightningTrainer: {str(e)}")
            self.log_exception(fl_ctx, traceback.format_exc())
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

