from typing import Any, Callable, Dict, List, Literal

import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.wrappers.bootstrapping import _bootstrap_sampler
from monai.utils import ensure_tuple
from lightning.pytorch.utilities.seed import isolate_rng

from manafaln.core.builders import MetricV2Builder as MetricBuilder


class CumulativeBootstrapper(Metric):
    """
    Compute the bootstrapped metrics for cumulative metric (e.g. AUC).

    Args:
        metric (Dict): The config to build the Metric object.
        num_bootstraps (int): Number of bootstrapping steps. Defaults to 1000.
        quantile (List[float]): The quantile of bootstrapped metrics to return. Defaults to [0.025, 0.975].
        sampling_strategy (Literal["poisson", "multinomial"]): Sampling strategy. Defaults to "multinomial".
        ignore_nan (bool): If True, bootstrapped metric that is nan will be skipped. Defaults to True.
        seed (int): The random seed for bootstrapping. Defaults to 42.
    """
    full_state_update: bool = True
    def __init__(
        self,
        metric: Dict,
        num_bootstraps: int = 1000,
        quantile: List[float] = [0.025, 0.975],
        sampling_strategy: Literal["poisson", "multinomial"] = "multinomial",
        ignore_nan: bool = True,
        seed: int = 42
        ):
        super().__init__()

        builder = MetricBuilder(check_instance=False)
        self.metric: Callable = builder(metric)
        self.num_bootstraps = num_bootstraps
        self.quantile = quantile
        self.sampling_strategy = sampling_strategy
        self.ignore_nan = ignore_nan
        self.seed = seed

    def update(self, *args: Any) -> None:
        """
        Update the states for the metric to be computed.
        """
        # If no state is present, initialize with the number of given args
        if self._defaults == {}:
            for idx in range(len(args)):
                self.add_state(f"state_{idx}", default=[], dist_reduce_fx='cat')

        # Update the states
        for state, arg in zip(self._defaults, args):
            getattr(self, state).extend(arg)

    def compute(self) -> Dict[str, Tensor]:
        """
        Computes the bootstrapped metric values.

        Returns:
            Dict[Tensor]: Results of bootstrapped metrics, containing the following keys:
                None: The metric as if no bootstrapping is done
                quantile_{q}: The quantile q of bootstrapped metrics

        Raise:
            ValueError: States have different number of length
        """

        states: List[list] = [getattr(self, state) for state in self._defaults]

        if len(set(len(state) for state in states)) != 1:
            raise ValueError("All states must have the same number of non-zero length")
        size = len(states[0])

        states: List[Tensor] = [torch.stack(state, dim=0) for state in states]

        output_dict = {}
        output_dict[None] = self.metric(*states)

        with isolate_rng():
            torch.manual_seed(self.seed)
            bootstrapped_metrics = []
            for _ in range(self.num_bootstraps):
                idx = _bootstrap_sampler(size, sampling_strategy=self.sampling_strategy).to(self.device)
                bootstrapped_states = [state.index_select(index=idx, dim=0) for state in states]
                bootstrapped_metric: Tensor = self.metric(*bootstrapped_states)
                if self.ignore_nan and torch.isnan(bootstrapped_metric):
                    continue
                bootstrapped_metrics.append(bootstrapped_metric)

        bootstrapped_metrics = torch.tensor(bootstrapped_metrics, device=self.device)
        quantile = torch.tensor(self.quantile, device=self.device)
        output_quantiles = torch.quantile(bootstrapped_metrics, quantile)
        for quantile, output_quantile in zip(self.quantile, output_quantiles):
            output_dict[f"quantile_{quantile}"] = output_quantile

        return output_dict

class IterativeBootstrapper(Metric):
    """
    Compute the bootstrapped metrics for iterative metric (e.g. Dice).

    Args:
        metric (Dict): The config to build the Metric object.
        num_bootstraps (int): Number of bootstrapping steps. Defaults to 1000.
        quantile (List[float]): The quantile of bootstrapped metrics to return. Defaults to [0.025, 0.975].
        sampling_strategy (Literal["poisson", "multinomial"]): Sampling strategy. Defaults to "multinomial".
        ignore_nan (bool): If True, bootstrapped metric that is nan will be skipped. Defaults to True.
        seed (int): The random seed for bootstrapping. Defaults to 42.
    """
    full_state_update: bool = False
    def __init__(
        self,
        metric: Dict,
        num_bootstraps: int = 1000,
        quantile: List[float] = [0.025, 0.975],
        sampling_strategy: Literal["poisson", "multinomial"] = "multinomial",
        ignore_nan: bool = True,
        seed: int = 42,
        ):
        super().__init__()

        builder = MetricBuilder(check_instance=False)
        self.metric: Callable = builder(metric)
        self.num_bootstraps = num_bootstraps
        self.quantile = quantile
        self.sampling_strategy = sampling_strategy
        self.ignore_nan = ignore_nan
        self.seed = seed

        self.add_state("scores", default=[], dist_reduce_fx='cat')

    def update(self, *args: Any) -> None:
        """
        Update the computed metrics to state
        """
        score = self.metric(*args)
        score = ensure_tuple(score) # torchmetrics.Metric.compute() will squeeze scalar, need to undo it
        self.scores.extend(score)

    def compute(self) -> Dict[str, Tensor]:
        """
        Computes the bootstrapped metric scores.

        Returns:
            Dict[Tensor]: Results of bootstrapped metrics, containing the following keys:
                None: The metric as if no bootstrapping is done
                quantile_{q}: The quantile q of bootstrapped metrics
        """
        scores = self.scores

        if self.ignore_nan:
            scores = [s for s in scores if not torch.isnan(s)]

        scores = torch.stack(scores)

        output_dict = {}
        output_dict[None] = torch.mean(scores)

        size = scores.size(0)

        with isolate_rng():
            torch.manual_seed(self.seed)
            bootstrapped_scores = []
            for _ in range(self.num_bootstraps):
                idx = _bootstrap_sampler(size, sampling_strategy=self.sampling_strategy).to(self.device)
                bootstrapped_score = scores.index_select(index=idx, dim=0)
                bootstrapped_score = torch.mean(bootstrapped_score)
                bootstrapped_scores.append(bootstrapped_score)

        bootstrapped_scores = torch.tensor(bootstrapped_scores, device=self.device)
        quantile = torch.tensor(self.quantile, device=self.device)
        output_quantiles = torch.quantile(bootstrapped_scores, quantile)
        for quantile, output_quantile in zip(self.quantile, output_quantiles):
            output_dict[f"quantile_{quantile}"] = output_quantile

        return output_dict
