"""Lightning module for training and evaluation."""

from typing import Any

import torch
from torch import nn

try:
    import pytorch_lightning as pl
    from torchmetrics.classification import (
        BinaryAUROC,
        BinaryAveragePrecision,
        BinaryF1Score,
        BinaryPrecision,
        BinaryRecall,
    )
except ImportError:  # pragma: no cover
    pl = None  # type: ignore[assignment]

from lungscan3d.models.outputs import extract_positive_logits
from lungscan3d.training.losses import build_loss


class LungScanLightningModule(pl.LightningModule if pl is not None else nn.Module):
    """Lightning module wrapping a 3D classifier and medical metrics."""

    def __init__(self, model: nn.Module, config: Any) -> None:
        """Initialize Lightning module.

        Args:
            model: PyTorch classifier returning binary logits.
            config: Hydra configuration.
        """
        super().__init__()
        self.model = model
        self.config = config
        self.loss_fn = build_loss(config)
        threshold = float(config.postprocess.threshold)
        self.val_auroc = BinaryAUROC()
        self.val_pr_auc = BinaryAveragePrecision()
        self.val_recall = BinaryRecall(threshold=threshold)
        self.val_precision = BinaryPrecision(threshold=threshold)
        self.val_f1 = BinaryF1Score(threshold=threshold)
        self.test_auroc = BinaryAUROC()
        self.test_pr_auc = BinaryAveragePrecision()
        self.test_recall = BinaryRecall(threshold=threshold)
        self.test_precision = BinaryPrecision(threshold=threshold)
        self.test_f1 = BinaryF1Score(threshold=threshold)
        if hasattr(self, "save_hyperparameters"):
            self.save_hyperparameters(ignore=["model"])

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Run model forward pass.

        Args:
            input_tensor: Input tensor with shape ``(B, C, D, H, W)``.

        Returns:
            Binary logits with shape ``(B, 1)``.
        """
        return extract_positive_logits(self.model(input_tensor))

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Run one training step.

        Args:
            batch: Pair of input volumes and labels.
            batch_idx: Batch index.

        Returns:
            Training loss tensor.
        """
        del batch_idx
        volumes, labels = batch
        logits = self(volumes)
        loss = self.loss_fn(logits, labels)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Run one validation step.

        Args:
            batch: Pair of input volumes and labels.
            batch_idx: Batch index.

        Returns:
            Validation loss tensor.
        """
        del batch_idx
        volumes, labels = batch
        logits = self(volumes)
        loss = self.loss_fn(logits, labels)
        probabilities = torch.sigmoid(logits).view(-1)
        targets = labels.long().view(-1)
        self.val_auroc.update(probabilities, targets)
        self.val_pr_auc.update(probabilities, targets)
        self.val_recall.update(probabilities, targets)
        self.val_precision.update(probabilities, targets)
        self.val_f1.update(probabilities, targets)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self) -> None:
        """Log validation metrics at epoch end."""
        self.log("val/roc_auc", self.val_auroc.compute(), prog_bar=True)
        self.log("val/pr_auc", self.val_pr_auc.compute(), prog_bar=True)
        self.log("val/recall", self.val_recall.compute(), prog_bar=True)
        self.log("val/precision", self.val_precision.compute())
        self.log("val/f1", self.val_f1.compute())
        self.val_auroc.reset()
        self.val_pr_auc.reset()
        self.val_recall.reset()
        self.val_precision.reset()
        self.val_f1.reset()

    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Run one test step.

        Args:
            batch: Pair of input volumes and labels.
            batch_idx: Batch index.

        Returns:
            Test loss tensor.
        """
        del batch_idx
        volumes, labels = batch
        logits = self(volumes)
        loss = self.loss_fn(logits, labels)
        probabilities = torch.sigmoid(logits).view(-1)
        targets = labels.long().view(-1)
        self.test_auroc.update(probabilities, targets)
        self.test_pr_auc.update(probabilities, targets)
        self.test_recall.update(probabilities, targets)
        self.test_precision.update(probabilities, targets)
        self.test_f1.update(probabilities, targets)
        self.log("test/loss", loss, on_epoch=True)
        return loss

    def on_test_epoch_end(self) -> None:
        """Log test metrics at epoch end."""
        self.log("test/roc_auc", self.test_auroc.compute())
        self.log("test/pr_auc", self.test_pr_auc.compute())
        self.log("test/recall", self.test_recall.compute())
        self.log("test/precision", self.test_precision.compute())
        self.log("test/f1", self.test_f1.compute())
        self.test_auroc.reset()
        self.test_pr_auc.reset()
        self.test_recall.reset()
        self.test_precision.reset()
        self.test_f1.reset()

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizer and learning-rate scheduler.

        Returns:
            Lightning optimizer configuration.
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=float(self.config.trainer.learning_rate),
            weight_decay=float(self.config.trainer.weight_decay),
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=2
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val/loss"},
        }
