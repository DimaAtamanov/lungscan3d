"""Decision-threshold optimization utilities."""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from lungscan3d.data.datamodule import LungScanDataModule
from lungscan3d.models import build_model
from lungscan3d.models.outputs import extract_positive_logits
from lungscan3d.training.lightning_module import LungScanLightningModule
from lungscan3d.utils.paths import ensure_dir

LOGGER = logging.getLogger(__name__)


ThresholdStrategy = Literal["recall_at_min_precision", "precision_at_min_recall", "f1"]
DatasetSplit = Literal["val", "test"]


@dataclass(frozen=True)
class ThresholdSearchResult:
    """Result of threshold optimization.

    Attributes
    ----------
        threshold: Selected decision threshold.
        strategy: Optimization strategy name.
        min_precision: Minimum precision constraint used by the strategy.
        min_recall: Minimum recall constraint used by the strategy.
        precision: Precision at the selected threshold.
        recall: Recall at the selected threshold.
        f1: F1-score at the selected threshold.
        roc_auc: ROC-AUC computed from probabilities.
        pr_auc: PR-AUC computed from probabilities.
        positive_rate: Fraction of positive predictions at the selected threshold.
        num_samples: Number of evaluated samples.

    """

    threshold: float
    strategy: str
    min_precision: float
    min_recall: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    positive_rate: float
    num_samples: int


def metrics_at_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Compute binary metrics at a fixed decision threshold.

    Args:
    ----
        labels: Binary ground-truth labels with shape ``(N,)``.
        probabilities: Positive-class probabilities with shape ``(N,)``.
        threshold: Decision threshold in the ``[0, 1]`` range.

    Returns:
    -------
        Dictionary with precision, recall, F1-score, and positive prediction rate.

    """
    predictions = (probabilities >= threshold).astype(np.int64)
    return {
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "positive_rate": float(predictions.mean()),
    }


def build_threshold_grid(probabilities: np.ndarray, num_thresholds: int = 1001) -> np.ndarray:
    """Build a stable threshold grid for optimization.

    The grid combines a regular ``[0, 1]`` grid with observed probability values. This makes the
    search deterministic and avoids missing a useful operating point on small validation sets.

    Args:
    ----
        probabilities: Positive-class probabilities with shape ``(N,)``.
        num_thresholds: Number of regular grid thresholds.

    Returns:
    -------
        Sorted unique threshold values.

    """
    regular_grid = np.linspace(0.0, 1.0, num_thresholds, dtype=np.float64)
    observed_values = np.asarray(probabilities, dtype=np.float64)
    return np.unique(np.concatenate([regular_grid, observed_values]))


def select_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    strategy: ThresholdStrategy = "recall_at_min_precision",
    min_precision: float = 0.4,
    min_recall: float = 0.8,
) -> ThresholdSearchResult:
    """Select a decision threshold for LUNA16 candidate classification.

    For this project the default strategy is ``recall_at_min_precision``.
    Among thresholds that keep precision above the configured lower bound,
    choose the threshold with the highest recall. This is aligned with the
    context of false-positive reduction.

    Args:
    ----
        labels: Binary ground-truth labels with shape ``(N,)``.
        probabilities: Positive-class probabilities with shape ``(N,)``.
        strategy: Threshold selection strategy.
        min_precision: Precision lower bound for ``recall_at_min_precision``.
        min_recall: Recall lower bound for ``precision_at_min_recall``.

    Returns:
    -------
        Threshold search result with metrics at the selected threshold.

    Raises:
    ------
        ValueError: If labels and probabilities are invalid or if the strategy is unknown.

    """
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if labels.shape != probabilities.shape:
        raise ValueError("labels and probabilities must have the same shape")
    if labels.size == 0:
        raise ValueError("threshold optimization requires at least one sample")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("labels must be binary: 0 or 1")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("probabilities must be in the [0, 1] range")

    LOGGER.info("Selecting threshold: strategy=%s, samples=%d", strategy, labels.size)
    rows: list[dict[str, float]] = []
    for threshold in build_threshold_grid(probabilities):
        row = metrics_at_threshold(labels, probabilities, float(threshold))
        row["threshold"] = float(threshold)
        rows.append(row)

    if strategy == "recall_at_min_precision":
        candidates = [row for row in rows if row["precision"] >= min_precision]
        if not candidates:
            candidates = rows
        best_row = max(
            candidates,
            key=lambda row: (
                row["recall"],
                row["f1"],
                row["precision"],
                -row["threshold"],
            ),
        )
    elif strategy == "precision_at_min_recall":
        candidates = [row for row in rows if row["recall"] >= min_recall]
        if not candidates:
            candidates = rows
        best_row = max(
            candidates,
            key=lambda row: (
                row["precision"],
                row["f1"],
                row["recall"],
                row["threshold"],
            ),
        )
    elif strategy == "f1":
        best_row = max(rows, key=lambda row: (row["f1"], row["recall"], row["precision"]))
    else:
        raise ValueError(f"Unknown threshold selection strategy: {strategy}")

    roc_auc = _safe_roc_auc(labels, probabilities)
    pr_auc = _safe_pr_auc(labels, probabilities)
    return ThresholdSearchResult(
        threshold=float(best_row["threshold"]),
        strategy=strategy,
        min_precision=float(min_precision),
        min_recall=float(min_recall),
        precision=float(best_row["precision"]),
        recall=float(best_row["recall"]),
        f1=float(best_row["f1"]),
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        positive_rate=float(best_row["positive_rate"]),
        num_samples=int(labels.size),
    )


def collect_probabilities(
    lightning_module: LungScanLightningModule,
    dataloader: torch.utils.data.DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect labels and predicted probabilities from a dataloader.

    Args:
    ----
        lightning_module: Trained Lightning module.
        dataloader: Validation or test dataloader.
        device: Device used for inference.

    Returns:
    -------
        Pair ``(labels, probabilities)`` as NumPy arrays.

    """
    lightning_module.eval()
    lightning_module.to(device)
    collected_labels: list[np.ndarray] = []
    collected_probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for volumes, labels in dataloader:
            volumes = volumes.to(device)
            logits = extract_positive_logits(lightning_module(volumes))
            probabilities = torch.sigmoid(logits).view(-1).detach().cpu().numpy()
            collected_probabilities.append(probabilities)
            collected_labels.append(labels.view(-1).detach().cpu().numpy())
    return np.concatenate(collected_labels), np.concatenate(collected_probabilities)


def optimize_threshold(
    config: Any,
    checkpoint: str | Path | None = None,
    split: DatasetSplit = "val",
    output: str | Path | None = None,
) -> ThresholdSearchResult:
    """Optimize and persist the decision threshold for a trained model.

    Args:
    ----
        config: Hydra configuration object.
        checkpoint: Path to a trained Lightning checkpoint. If omitted, ``infer.checkpoint_path`` is
            used.
        split: Dataset split used for threshold selection: ``val`` or ``test``. Validation is the
            recommended default.
        output: Optional output JSON path. If omitted, ``postprocess.threshold_artifact_path`` is
            used.

    Returns:
    -------
        Threshold search result.

    """
    LOGGER.info("Starting threshold optimization on split=%s", split)
    checkpoint_path = Path(checkpoint or config.infer.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    datamodule = LungScanDataModule(config)
    datamodule.prepare_data()
    datamodule.setup(stage="fit")
    dataloader = datamodule.val_dataloader() if split == "val" else datamodule.test_dataloader()

    model = build_model(config)
    lightning_module = LungScanLightningModule.load_from_checkpoint(
        checkpoint_path=str(checkpoint_path),
        model=model,
        config=config,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Collecting validation probabilities on device=%s", device)
    labels, probabilities = collect_probabilities(lightning_module, dataloader, device)

    result = select_threshold(
        labels=labels,
        probabilities=probabilities,
        strategy=str(config.postprocess.threshold_selection_metric),
        min_precision=float(config.postprocess.min_precision),
        min_recall=float(config.postprocess.min_recall),
    )
    output_path = Path(output or config.postprocess.threshold_artifact_path)
    save_threshold_result(result, output_path)
    LOGGER.info(
        "Threshold optimization finished: threshold=%.6f, recall=%.4f, precision=%.4f",
        result.threshold,
        result.recall,
        result.precision,
    )
    return result


def save_threshold_result(result: ThresholdSearchResult, output_path: str | Path) -> None:
    """Save threshold search result as JSON.

    Args:
    ----
        result: Threshold optimization result.
        output_path: Target JSON file path.

    """
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    output_path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    LOGGER.info("Threshold artifact saved: %s", output_path)


def load_threshold_from_artifact(path: str | Path, fallback: float) -> float:
    """Load an optimized threshold from JSON if the artifact exists.

    Args:
    ----
        path: Threshold artifact path.
        fallback: Fallback threshold from static config.

    Returns:
    -------
        Optimized threshold or fallback value.

    """
    artifact_path = Path(path)
    if not artifact_path.exists():
        LOGGER.info("Threshold artifact not found; using fallback threshold %.6f", fallback)
        return float(fallback)
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    threshold = float(payload.get("threshold", fallback))
    LOGGER.info("Loaded threshold %.6f from %s", threshold, artifact_path)
    return threshold


def _safe_roc_auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Compute ROC-AUC while handling single-class validation splits.

    Args:
    ----
        labels: Binary ground-truth labels.
        probabilities: Positive-class probabilities.

    Returns:
    -------
        ROC-AUC or ``nan`` when the metric is undefined.

    """
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, probabilities))


def _safe_pr_auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Compute PR-AUC while handling degenerate validation splits.

    Args:
    ----
        labels: Binary ground-truth labels.
        probabilities: Positive-class probabilities.

    Returns:
    -------
        PR-AUC or ``nan`` when the metric is undefined.

    """
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(average_precision_score(labels, probabilities))
