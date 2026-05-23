"""Utilities for hard negative mining."""

import logging
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger(__name__)


def select_hard_negative_indices(
    labels: np.ndarray,
    probabilities: np.ndarray,
    top_fraction: float,
    min_probability: float,
) -> np.ndarray:
    """Select high-confidence false positives as hard negatives.

    Args:
    ----
        labels: Binary labels with shape ``(N,)``.
        probabilities: Positive-class probabilities with shape ``(N,)``.
        top_fraction: Fraction of negative samples to keep among the hardest examples.
        min_probability: Minimum positive-class probability for a negative to be considered hard.

    Returns:
    -------
        Array of selected hard-negative indices.

    """
    if labels.shape != probabilities.shape:
        raise ValueError("labels and probabilities must have the same shape")
    if not 0.0 < top_fraction <= 1.0:
        raise ValueError("top_fraction must be in (0, 1]")
    negative_indices = np.flatnonzero(labels < 0.5)
    candidate_indices = negative_indices[probabilities[negative_indices] >= min_probability]
    if len(candidate_indices) == 0:
        LOGGER.info("No hard negatives found above probability %.4f", min_probability)
        return np.array([], dtype=np.int64)
    sorted_indices = candidate_indices[np.argsort(probabilities[candidate_indices])[::-1]]
    keep_count = max(1, int(round(len(sorted_indices) * top_fraction)))
    selected = sorted_indices[:keep_count].astype(np.int64)
    LOGGER.info(
        "Selected %d hard negatives from %d negative samples",
        len(selected),
        len(negative_indices),
    )
    return selected


def save_hard_negative_indices(indices: np.ndarray, output_path: str | Path) -> None:
    """Save hard-negative indices for the next training run.

    Args:
    ----
        indices: Hard-negative train indices.
        output_path: Destination ``.npy`` path.

    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, indices.astype(np.int64))
    LOGGER.info("Saved %d hard-negative indices to %s", len(indices), path)
