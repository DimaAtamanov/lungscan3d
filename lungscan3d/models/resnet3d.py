"""Multi-scale ResNet-like 3D CNN with SE attention."""

import logging
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as functional

from lungscan3d.models.blocks import ResidualSEBlock3D

LOGGER = logging.getLogger(__name__)


class ResNet3DEncoder(nn.Module):
    """Single-scale 3D ResNet-like feature encoder with SE attention."""

    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        blocks_per_stage: Sequence[int],
        se_reduction: int,
    ) -> None:
        """Initialize encoder.

        Args:
            in_channels: Number of input channels.
            base_channels: Number of stem output channels.
            blocks_per_stage: Number of residual blocks in each stage.
            se_reduction: Reduction ratio for SE blocks.
        """
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
        )
        stages: list[nn.Module] = []
        current_channels = base_channels
        for stage_index, block_count in enumerate(blocks_per_stage):
            out_channels = base_channels * (2**stage_index)
            for block_index in range(int(block_count)):
                stride = 2 if stage_index > 0 and block_index == 0 else 1
                stages.append(
                    ResidualSEBlock3D(
                        current_channels,
                        out_channels,
                        stride=stride,
                        se_reduction=se_reduction,
                    )
                )
                current_channels = out_channels
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.out_channels = current_channels

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Encode one 3D CT patch scale.

        Args:
            input_tensor: Input tensor with shape ``(B, C, D, H, W)``.

        Returns:
            Feature tensor with shape ``(B, out_channels)``.
        """
        features = self.stem(input_tensor)
        features = self.stages(features)
        return self.pool(features).flatten(1)


class MultiScaleResNet3DSE(nn.Module):
    """Multi-scale 3D ResNet-like classifier with residual SE encoders.

    The model uses two branches over the same preprocessed CT candidate patch:
    a context branch over the full patch and a local branch over a center crop.
    Local features and contextual features are concatenated before binary
    classification. This keeps the input dataset simple while implementing the
    multi-scale idea from the project description.
    """

    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        blocks_per_stage: Sequence[int],
        se_reduction: int,
        dropout: float,
        local_crop_fraction: float = 0.5,
    ) -> None:
        """Initialize multi-scale model.

        Args:
            in_channels: Number of input channels.
            base_channels: Number of stem output channels for each branch.
            blocks_per_stage: Number of residual blocks in each stage.
            se_reduction: Reduction ratio for SE blocks.
            dropout: Dropout before classification head.
            local_crop_fraction: Fraction of each spatial dimension used by the local branch.
        """
        super().__init__()
        if not 0.0 < local_crop_fraction <= 1.0:
            raise ValueError("local_crop_fraction must be in (0, 1]")
        self.local_crop_fraction = float(local_crop_fraction)
        self.context_encoder = ResNet3DEncoder(
            in_channels=in_channels,
            base_channels=base_channels,
            blocks_per_stage=blocks_per_stage,
            se_reduction=se_reduction,
        )
        self.local_encoder = ResNet3DEncoder(
            in_channels=in_channels,
            base_channels=base_channels,
            blocks_per_stage=blocks_per_stage,
            se_reduction=se_reduction,
        )
        feature_dim = (
            self.context_encoder.out_channels + self.local_encoder.out_channels
        )
        self.classifier = nn.Sequential(
            nn.Dropout(float(dropout)),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(feature_dim // 2, 1),
        )

    def _center_crop(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Extract and resize the local center crop for the local branch.

        Args:
            input_tensor: Full context patch with shape ``(B, C, D, H, W)``.

        Returns:
            Center-cropped patch resized back to the original spatial shape.
        """
        _, _, depth, height, width = input_tensor.shape
        crop_depth = max(1, int(round(depth * self.local_crop_fraction)))
        crop_height = max(1, int(round(height * self.local_crop_fraction)))
        crop_width = max(1, int(round(width * self.local_crop_fraction)))
        start_depth = (depth - crop_depth) // 2
        start_height = (height - crop_height) // 2
        start_width = (width - crop_width) // 2
        local_patch = input_tensor[
            :,
            :,
            start_depth : start_depth + crop_depth,
            start_height : start_height + crop_height,
            start_width : start_width + crop_width,
        ]
        return functional.interpolate(
            local_patch,
            size=(depth, height, width),
            mode="trilinear",
            align_corners=False,
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Run multi-scale forward pass.

        Args:
            input_tensor: Input tensor with shape ``(B, 1, D, H, W)``.

        Returns:
            Binary logits with shape ``(B, 1)``.
        """
        local_patch = self._center_crop(input_tensor)
        context_features = self.context_encoder(input_tensor)
        local_features = self.local_encoder(local_patch)
        features = torch.cat([local_features, context_features], dim=1)
        return self.classifier(features)


# Backward-compatible alias used by older configs/tests.
ResNet3DSE = MultiScaleResNet3DSE
