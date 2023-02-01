from typing import Optional, Sequence

import kornia
import torch
import torch.nn.functional as F

def label_smoothing(label: torch.Tensor, smooth: Optional[float]=None, binary: bool=False):
    """
    Label smoothing with

    $$ (1-smooth)*label + smooth/num_classes $$

    References:
    Rafael Müller et al. (2019). When Does Label Smoothing Help? https://arxiv.org/abs/1906.02629

    Args:
        label (torch.Tensor): tensor to apply label smoothing, with last channel as label.
        smooth (float): smoothing factor, requires 0 <= smooth < 1. Default: 0.0
        binary (bool): whether the label is binary. Default: False
    """

    if smooth is None:
        return label

    assert 0 <= smooth < 1, "smoothing factor must be 0 <= smooth < 1"

    if binary:
        # shape: (..., 1) or (..., )
        label_smooth = (1-smooth)*label + smooth/2
    else:
        # shape: (..., C)
        label_smooth = (1-smooth)*label + smooth/label.size(-1)

    return label_smooth

class SpatialWeightedMixin(object):
    """
    Spatial-weight is designed such that the weight is higher if closer to POI.
    It only supports multi-labels tasks of 2D inputs.
    """
    def __init__(self, poi: Optional[Sequence[int]], sigma: float = 1/12):
        """
        Args:
            poi (Optional[Sequence[int]]): Indices of source, with -1 as non-spatial-weighted. Default: None.
        """
        self.poi = torch.tensor(poi, dtype=torch.long) if poi is not None else None
        self.sigma = sigma

    def get_spatial_weights(self,
            target: torch.Tensor    # shape: (B, C, H, W)
        ) -> torch.Tensor:          # spatial weights, shape: (B, C, H, W)

        # Initialize weights, shape: (B, C, H, W)
        spatial_weight = torch.ones_like(target)

        # If poi is given
        if self.poi is not None:

            # Get channels that requires spatial weights, shape: (C, )
            spatial_weighted_indices = self.poi != -1

            # Get the poi of each spatial weighted channel, shape: (SW, )
            poi = self.poi[spatial_weighted_indices]

            # Get masks of poi, shape: (B, SW, H, W)
            spatial_sources = target[:, poi]

            # Compute spatial weights, shape: (B, SW, H, W)
            computed_spatial_weights = self.compute_spatial_weight(spatial_sources, self.sigma)

            # Assign computed weights to required channels, shape: (B, C, H, W)
            spatial_weight[:, spatial_weighted_indices] = computed_spatial_weights

        # Return spatial weights, shape: (B, C, H, W)
        return spatial_weight

    @staticmethod
    def compute_spatial_weight(
            source: torch.Tensor,      # (B, C, H, W)
            sigma: float = 1/12,
        ) -> torch.Tensor:             # (B, C, H, W)
        """
        Compute spatial weight given mask of point of interest (POI).

        Args:
            source (torch.Tensor): mask of POI with shape (B, C, H, W).

        Returns:
            spatial_weight (torch.Tensor): spatial weight with shape (B, C, H, W).
        """

        # Get shape of source
        B, C, H, W = source.size()

        # Reshape source (B, C, H, W) => (B*C, H, W)
        source = source.view(B*C, H, W)

        # Prepare weight as shape: (B*C, H, W)
        spatial_weight = torch.ones_like(source)

        # Get indices of non zero source over instances and channels, shape: (B*C, )

        NZ = source.sum(dim=(-1, -2)) > 0

        # If some source are non zero
        if any(NZ):
            # Get non zero sources, shape: (NZ, H, W)
            source = source[NZ.tolist()]

            # Size of Gaussian kernel, shape: (2, )
            # Set as double the spatial size of source, so that it dilate throughout the mask
            # Add one since it must be odd
            kernel_size = tuple(2*s+1 for s in (H, W))

            # Standard deviation of Gaussian blur, shape: (2, )
            # Set as a sixth of (kernel size - 1), so that 99.7 percentile is in the kernel
            sigma = tuple((k - 1)*sigma for k in kernel_size)

            # Add channel, shape: (NZ, H, W) => (NZ, 1, H, W)
            gaussian_input  = source.unsqueeze(1)

            # Gaussian blur applied to every instances and channels, shape: (NZ, 1, H, W)
            gaussian_output = kornia.filters.gaussian_blur2d(gaussian_input, kernel_size, sigma, border_type='constant')

            # Reshape to normalize spatial dimensions, shape: (NZ, H*W)
            unnormalized_weight = gaussian_output.view(-1, H*W)

            # Normalize spatial weights to sum H*W, shape: (NZ, H*W)
            normalized_weight = H * W * F.normalize(unnormalized_weight, p=1)

            # Reshape to match spatial weights, shape: (NZ, H, W)
            normalized_weight = normalized_weight.view(-1, H, W)

            # Save the weight back into spatial weight, shape: (B*C, H, W)
            spatial_weight[NZ] = normalized_weight

        # Reshape (B*C, H, W) => (B, C, H, W)
        spatial_weight = spatial_weight.view(B, C, H, W)

        # Return spatial weight, shape (B, C, H, W)
        return spatial_weight
