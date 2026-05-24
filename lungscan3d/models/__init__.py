"""Model factory."""

from typing import Any

from torch import nn

from lungscan3d.models.baseline3d import LunaModel
from lungscan3d.models.resnet3d import MultiScaleResNet3DSE


def build_model(config: Any) -> nn.Module:
    """Build a model from Hydra configuration.

    Args:
    ----
        config: Hydra configuration with a ``model`` section.

    Returns:
    -------
        Instantiated PyTorch module.

    """
    model_name = str(config.model.name)
    if model_name == "dlwpt_baseline":
        return LunaModel(
            in_channels=int(config.model.in_channels),
            conv_channels=int(config.model.conv_channels),
        )
    if model_name == "resnet3d_se":
        return MultiScaleResNet3DSE(
            in_channels=int(config.model.in_channels),
            base_channels=int(config.model.base_channels),
            blocks_per_stage=[int(value) for value in config.model.blocks_per_stage],
            se_reduction=int(config.model.se_reduction),
            dropout=float(config.model.dropout),
            local_crop_fraction=float(config.model.local_crop_fraction),
        )
    raise ValueError(f"Unknown model: {model_name}")
