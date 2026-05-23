"""Custom Lightning callbacks."""

from collections import defaultdict
from typing import Any

import pytorch_lightning as pl


class MetricsHistoryCallback(pl.Callback):
    """Collect epoch-level metrics emitted by Lightning."""

    def __init__(self) -> None:
        """Initialize empty metric history."""
        super().__init__()
        self.history: dict[str, list[float]] = defaultdict(list)

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Collect available metrics after a train epoch.

        Args:
            trainer: Active Lightning trainer.
            pl_module: Active Lightning module.
        """
        del pl_module
        self._collect_metrics(trainer.callback_metrics)

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Collect available metrics after a validation epoch.

        Args:
            trainer: Active Lightning trainer.
            pl_module: Active Lightning module.
        """
        del pl_module
        self._collect_metrics(trainer.callback_metrics)

    def _collect_metrics(self, metrics: dict[str, Any]) -> None:
        """Store scalar metrics from Lightning callback metrics.

        Args:
            metrics: Mapping of metric names to scalar-like values.
        """
        for name, value in metrics.items():
            if not _should_plot_metric(name):
                continue
            try:
                scalar = float(value.detach().cpu().item())
            except AttributeError:
                try:
                    scalar = float(value)
                except (TypeError, ValueError):
                    continue
            if not self.history[name] or self.history[name][-1] != scalar:
                self.history[name].append(scalar)


def _should_plot_metric(name: str) -> bool:
    """Return whether a metric should be stored for static plots.

    Args:
        name: Lightning metric name.

    Returns:
        True when the metric is one of the tracked training curves.
    """
    allowed_prefixes = (
        "train/loss",
        "val/loss",
        "val/roc_auc",
        "val/pr_auc",
        "val/recall",
        "val/f1",
    )
    return name.startswith(allowed_prefixes)
