"""Reusable 3D model blocks."""

import torch
from torch import nn


class SqueezeExcitation3D(nn.Module):
    """3D squeeze-and-excitation block."""

    def __init__(self, channels: int, reduction: int) -> None:
        """Initialize block.

        Args:
        ----
            channels: Number of feature channels.
            reduction: Channel reduction ratio.

        """
        super().__init__()
        reduced_channels = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, reduced_channels),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels),
            nn.Sigmoid(),
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Apply channel attention.

        Args:
        ----
            input_tensor: Input tensor with shape ``(B, C, D, H, W)``.

        Returns:
        -------
            Reweighted tensor with the same shape.

        """
        batch_size, channels = input_tensor.shape[:2]
        pooled = self.pool(input_tensor).view(batch_size, channels)
        weights = self.excitation(pooled).view(batch_size, channels, 1, 1, 1)
        return input_tensor * weights


class ResidualSEBlock3D(nn.Module):
    """Residual 3D convolutional block with squeeze-and-excitation."""

    def __init__(self, in_channels: int, out_channels: int, stride: int, se_reduction: int) -> None:
        """Initialize block.

        Args:
        ----
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            stride: Stride for the first convolution.
            se_reduction: Reduction ratio for SE attention.

        """
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            SqueezeExcitation3D(out_channels, se_reduction),
        )
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()
        self.activation = nn.ReLU(inplace=True)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Run forward pass.

        Args:
        ----
            input_tensor: Input tensor.

        Returns:
        -------
            Block output tensor.

        """
        return self.activation(self.body(input_tensor) + self.shortcut(input_tensor))
