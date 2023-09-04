import os
from typing import Dict, Optional

import numpy as np
from torch.utils.tensorboard import SummaryWriter
from nvflare.apis.dxo import DataKind, MetaKey, from_shareable
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_constant import FLContextKey, ReservedKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.app_event_type import AppEventType
from nvflare.widgets.widget import Widget

class GlobalMetricLogger(Widget):
    """
    A widget for logging global metrics during federated learning.

    Args:
        log_dir (str, optional): The directory to save the log files. Defaults to "logs".
        log_name (str, optional): The name of the log file. Defaults to "key_metric".
        val_metric_name (str, optional): The name of the validation metric to log. Defaults to MetaKey.INITIAL_METRICS.
        aggregation_weights (Dict, optional): The weights for aggregating the metrics from different clients. Defaults to None.
    """

    def __init__(
        self,
        log_dir: str = "logs",
        log_name: str = "key_metric",
        val_metric_name: str = MetaKey.INITIAL_METRICS,
        aggregation_weights: Optional[Dict] = None
    ):
        super().__init__()

        self.log_dir = log_dir
        self.log_name = log_name
        self.val_metric_name = val_metric_name
        self.aggregation_weights = aggregation_weights

        self.writer = None
        self.logger.info(f"metric logger weights control: {aggregation_weights}")

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        """
        Handles the events triggered during federated learning.

        Args:
            event_type (str): The type of the event.
            fl_ctx (FLContext): The context object for federated learning.
        """
        if event_type == EventType.START_RUN:
            self._startup(fl_ctx)
        elif event_type == AppEventType.ROUND_STARTED:
            self._reset_metrics()
        elif event_type == AppEventType.BEFORE_CONTRIBUTION_ACCEPT:
            self._before_accept(fl_ctx)
        elif event_type == AppEventType.BEFORE_AGGREGATION:
            self._before_aggregate(fl_ctx)
        elif event_type == EventType.END_RUN:
            self._shutdown(fl_ctx)

    def _reset_metrics(self):
        """
        Resets the accumulated metrics.
        """
        self.val_metric_sum = 0
        self.val_metric_weights = 0

    def _startup(self, fl_ctx: FLContext):
        """
        Initializes the metric logger at the start of the federated learning run.

        Args:
            fl_ctx (FLContext): The context object for federated learning.
        """
        app_root = fl_ctx.get_prop(FLContextKey.APP_ROOT)
        log_path = os.path.join(app_root, self.log_dir)
        if not os.path.exists(log_path):
            os.makedirs(log_path, exist_ok=True)

        self._reset_metrics()
        self.writer = SummaryWriter(log_dir=log_path)

    def _before_accept(self, fl_ctx: FLContext):
        """
        Handles the BEFORE_CONTRIBUTION_ACCEPT event.

        Args:
            fl_ctx (FLContext): The context object for federated learning.
        """
        peer_ctx = fl_ctx.get_peer_context()
        shareable = peer_ctx.get_prop(FLContextKey.SHAREABLE)

        try:
            dxo = from_shareable(shareable)
        except:
            self.log_exception(fl_ctx, "shareable data is not a valid DXO")
            return

        if dxo.data_kind not in (DataKind.WEIGHT_DIFF, DataKind.WEIGHTS, DataKind.COLLECTION):
            self.log_debug(fl_ctx, "DXO kind is not valid for logging")
            return

        if dxo.data is None:
            self.log_debug(fl_ctx, "No data in DXO")
            return

        current_round = fl_ctx.get_prop(AppConstants.CURRENT_ROUND)
        contribution_round = shareable.get_header(AppConstants.CONTRIBUTION_ROUND)
        client_name = shareable.get_peer_prop(ReservedKey.IDENTITY_NAME, default="?")

        if current_round == 0:
            self.log_debug(fl_ctx, f"Skip the first round.")
            return

        if contribution_round != current_round:
            self.log_warning(
                fl_ctx,
                f"Discard round {contribution_round} metrics from {client_name} at round {current_round}"
            )
            return

        val_metric = dxo.get_meta_prop(self.val_metric_name)
        if val_metric is None:
            self.log_debug(fl_ctx, f"Metric {self.val_metric_name} does not exists.")
            return
        else:
            self.log_info(fl_ctx, f"Received validation metric {val_metric} from {client_name}.")

        client_weight = self.aggregation_weights.get(client_name, 1.0)
        self.val_metric_sum += val_metric * client_weight
        self.val_metric_weights += client_weight

    def _before_aggregate(self, fl_ctx: FLContext):
        """
        Handles the BEFORE_AGGREGATION event.

        Args:
            fl_ctx (FLContext): The context object for federated learning.
        """
        if self.val_metric_weights == 0:
            self.log_debug(fl_ctx, "nothing accumulated")
            return

        current_round = fl_ctx.get_prop(AppConstants.CURRENT_ROUND)
        self.writer.add_scalar(
            self.log_name,
            self.val_metric_sum / self.val_metric_weights,
            current_round
        )
        self.log_info(fl_ctx, f"Write metric summary for round {current_round}.")

        self._reset_metrics()

    def _shutdown(self, fl_ctx: FLContext):
        """
        Shuts down the metric logger at the end of the federated learning run.

        Args:
            fl_ctx (FLContext): The context object for federated learning.
        """
        self.writer.close()

