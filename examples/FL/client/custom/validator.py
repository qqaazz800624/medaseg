import os
import traceback

import torch
from nvflare.apis.dxo import from_shareable, DataKind, DXO
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants

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

class LightningValidator(Executor):
    def __init__(
        self,
        validate_task_name=AppConstants.TASK_VALIDATION,
        config_file="config/config_validation.json"
    ):
        super(LightningValidator, self).__init__()

        self._validate_task_name = validate_task_name
        self.config_file = config_file

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

    def setup(self, fl_ctx: FLContext):
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

        # Manually initialize lightning data module
        self.data.setup()

    def teardown(self, fl_ctx: FLContext):
        self.data.teardown()

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        try:
            if event_type == EventType.START_RUN:
                self.setup(fl_ctx)
            elif event_type == EventType.ABORT_TASK:
                pass
            elif event_type == EventType.END_RUN:
                self.teardown(fl_ctx)
        except Exception as e:
            self.log_exception(traceback.format_exc())

    def apply_weight(self, weights: Dict[str, np.ndarray]):
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

    def run_validation(self):
        return self.trainer.validate(self.workflow, self.data.val_dataloader())

    def execute(
        self,
        task_name: str,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal
    ) -> Shareable:
        if task_name == self._validate_task_name:
            model_owner = "?"
            try:
                try:
                    dxo = from_shareable(shareable)
                except:
                    self.log_error(
                        fl_ctx,
                        "Error in extracting DXO from shareable"
                    )
                    return make_reply(ReturnCode.BAD_TASK_DATA)

                if not dxo.data_kind == DataKind.WEIGHTS:
                    self.log_exception(
                        fl_ctx,
                        f"DXO is of type {dxo.data_kind} but expected type WEIGHTS"
                    )
                    return make_reply(ReturnCode.BAD_TASK_DATA)

                # Get model information & apply weights to model
                model_owner = shareable.get_header(
                    AppConstants.MODEL_OWNER,
                    "?"
                )
                self.apply_weight(dxo.data)

                # Run validation
                metrics = self.run_validation()
                if abort_signal.triggered:
                    return make_reply(ReturnCode.TASK_ABORTED)

                self.log_info(
                    fl_ctx,
                    f"Validation metrics of {model_owner}'s model on"
                    f" {fl_ctx.get_identity_name()}'s data: {metrics}"
                )

                dxo = DXO(data_kind=DataKind.METRICS, data=metrics)
                return dxo.to_shareable()
            except Exception as e:
                self.log_exception(traceback.format_exc())
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)
        else:
            return make_reply(ReturnCode.TASK_UNKNOWN)

