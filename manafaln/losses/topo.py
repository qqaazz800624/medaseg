from typing import Optional, Sequence, Union

import torch
from monai.utils import ensure_tuple_rep
from topologylayer.nn import LevelSetLayer2D, TopKBarcodeLengths
from torch.nn.functional import max_pool2d
from torch.nn.modules.loss import _Loss


class TopologyLoss(_Loss):
    """
    A topological loss function based on Persistent Homology.

    References:
    1.  James R. Clough, Ilkay Öksüz, Nicholas Byrne, Veronika A. Zimmer, Julia A. Schnabel, & Andrew P. King (2019).
        A Topological Loss Function for Deep-Learning based Image Segmentation using Persistent Homology.
        CoRR, abs/1910.01877.

    Args:
        betti (Sequence[int]): Betti number for each channels
        dim (int): The dimension ot topological featrues. Defaults to 0.
        weights (Optional[Sequence[float]], optional): The weight of each channel. Defaults to None.
        grid (Union[int, Sequence[int]]): The size of cubical complex. Defaults to 16.
        ignore_first (bool): Whether to ignore the first bar. Defaults to False.
        max_k (int): The maximum number of bars in the barcode diagram. Defaults to 20.
        sublevel (bool): Whether to use sublevel set or suplevel set. Defaults to False, which is suplevel.
    """
    def __init__(
        self,
        betti: Sequence[int],
        dim: int = 0,
        weights: Optional[Sequence[float]] = None,
        grid: Union[int, Sequence[int]] = 16,
        ignore_first: bool = False,
        max_k: int = 20,
        sublevel: bool = False,
    ):
        super().__init__()
        self.betti = betti
        self.dim = dim
        self.weights = weights
        self.grid = ensure_tuple_rep(grid, 2)
        self.ignore_first = ignore_first
        self.max_k = max_k

        self.level_set = LevelSetLayer2D(size=self.grid, sublevel=sublevel, maxdim=dim)
        self.barcode = TopKBarcodeLengths(dim=dim, k=max_k)

    def forward(self, output: torch.Tensor) -> torch.Tensor:
        output = output.sigmoid()
        device = output.device

        topo_loss = 0
        for channel in range(output.size(1)):

            weight = 1
            if self.weights is not None:
                weight = self.weights[channel]
                if weight == 0: continue

            betti = self.betti[channel]

            channel_output = output.select(dim=1, index=channel)
            channel_output = channel_output.unsqueeze(1)

            channel_output: torch.Tensor = max_pool2d(
                input = channel_output,
                kernel_size=(
                    output.size(-2)//self.grid[0],
                    output.size(-1)//self.grid[1]
                    )
                )
            channel_output = channel_output.squeeze(1) # (B, H, W)
            channel_output = channel_output.cpu()

            channel_topo_loss = []
            for instance in channel_output:
                channel_topo_loss.append(self.compute_topo_loss(instance, betti))

            channel_topo_loss = torch.tensor(channel_topo_loss).mean()

            topo_loss += weight * channel_topo_loss

        topo_loss = topo_loss.to(device)
        return topo_loss

    def fix_dgm(self, dgm_info):
        """
        For all p<0, the entire image is one connected component.
        Hence the persistence bar corresponding to this feature is infinitely long.
        In practice we can cut off the bar at p=0 without affecting any details.
        """
        # dgm_info[0] is the persistence barcode diagram,
        # dgm_info[1] is a boolean incidating if the diagram is sublevel, which is otherwise suplevel
        dgm, issublevel = dgm_info

        # dgm[0] is the persistence barcode of dim 0
        # dgm[0][0] is the longest barcode in dgm[0]
        bar = dgm[0][0]

        # If is using sublevel set
        if issublevel:
            # Clip its death threshold value to 1, which is currently inf
            bar[1] = 1
        # If is using suplevel set
        else:
            # Clip its death threshold value to 0, which is currently -inf
            bar[1] = 0

        return dgm_info

    def compute_topo_loss(self, output: torch.Tensor, betti: int) -> torch.Tensor:
        dgm_info = self.level_set(output)
        dgm_info = self.fix_dgm(dgm_info)
        bars = self.barcode(dgm_info)
        target_bars = torch.zeros(self.max_k)
        target_bars[:betti] = 1

        if self.ignore_first:
            bars = bars[1:]
            target_bars = target_bars[1:]

        loss = ((target_bars-bars)**2).sum()
        return loss
