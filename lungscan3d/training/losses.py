"""Loss functions for imbalanced binary candidate classification."""

import logging
from typing import Any

import torch
from torch import nn
from torch.nn import functional as functional

LOGGER = logging.getLogger(__name__)


class FocalLossWithLogits(nn.Module):
    """Binary focal loss operating on logits."""

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0) -> None:
        """Initialize focal loss.

        Args:
        ----
            alpha: Positive-class balancing factor.
            gamma: Focusing parameter.

        """
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
        ----
            logits: Binary logits with shape ``(B, 1)``.
            targets: Binary labels with shape ``(B, 1)``.

        Returns:
        -------
            Scalar focal loss.

        """
        bce_loss = functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probabilities = torch.sigmoid(logits)
        p_t = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        loss = alpha_t * torch.pow(1.0 - p_t, self.gamma) * bce_loss
        return loss.mean()


class FocalBCEWithLogitsLoss(nn.Module):
    """Weighted combination of focal loss and binary cross entropy."""

    def __init__(
        self,
        alpha: float,
        gamma: float,
        focal_weight: float,
        bce_weight: float,
        pos_weight: float | None = None,
    ) -> None:
        """Initialize combined loss.

        Args:
        ----
            alpha: Positive-class balancing factor for focal loss.
            gamma: Focusing parameter for focal loss.
            focal_weight: Weight of focal loss component.
            bce_weight: Weight of BCE component.
            pos_weight: Optional positive-class weight for BCE.

        """
        super().__init__()
        if focal_weight < 0 or bce_weight < 0:
            raise ValueError("Loss weights must be non-negative")
        if focal_weight == 0 and bce_weight == 0:
            raise ValueError("At least one loss component must have positive weight")
        self.focal_weight = float(focal_weight)
        self.bce_weight = float(bce_weight)
        self.focal = FocalLossWithLogits(alpha=alpha, gamma=gamma)
        bce_pos_weight = None
        if pos_weight is not None:
            bce_pos_weight = torch.tensor([float(pos_weight)], dtype=torch.float32)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=bce_pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute combined focal/BCE loss.

        Args:
        ----
            logits: Binary logits with shape ``(B, 1)``.
            targets: Binary labels with shape ``(B, 1)``.

        Returns:
        -------
            Scalar combined loss.

        """
        loss = torch.zeros((), dtype=logits.dtype, device=logits.device)
        if self.focal_weight > 0:
            loss = loss + self.focal_weight * self.focal(logits, targets)
        if self.bce_weight > 0:
            loss = loss + self.bce_weight * self.bce(logits, targets)
        return loss


def build_loss(config: Any) -> nn.Module:
    """Build loss function from Hydra config.

    Args:
    ----
        config: Hydra config with a ``loss`` section.

    Returns:
    -------
        PyTorch loss module.

    """
    loss_name = str(config.loss.name)
    if loss_name == "bce":
        pos_weight_value = config.loss.pos_weight
        LOGGER.info("Using BCEWithLogitsLoss, pos_weight=%s", pos_weight_value)
        if pos_weight_value is None:
            return nn.BCEWithLogitsLoss()
        pos_weight = torch.tensor([float(pos_weight_value)], dtype=torch.float32)
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if loss_name == "focal":
        focal_weight = float(getattr(config.loss, "focal_weight", 1.0))
        bce_weight = float(getattr(config.loss, "bce_weight", 0.0))
        pos_weight_value = getattr(config.loss, "pos_weight", None)
        LOGGER.info(
            """Using FocalBCEWithLogitsLoss: alpha=%.3f,
                                             gamma=%.3f,
                                             focal_weight=%.3f,
                                             bce_weight=%.3f,
                                             pos_weight=%s""",
            float(config.loss.alpha),
            float(config.loss.gamma),
            focal_weight,
            bce_weight,
            pos_weight_value,
        )
        return FocalBCEWithLogitsLoss(
            alpha=float(config.loss.alpha),
            gamma=float(config.loss.gamma),
            focal_weight=focal_weight,
            bce_weight=bce_weight,
            pos_weight=None if pos_weight_value is None else float(pos_weight_value),
        )
    raise ValueError(f"Unknown loss: {loss_name}")
