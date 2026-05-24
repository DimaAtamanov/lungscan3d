"""Tests for threshold optimization utilities."""

import json
from pathlib import Path

import numpy as np
from lungscan3d.inference.thresholds import (
    load_threshold_from_artifact,
    metrics_at_threshold,
    save_threshold_result,
    select_threshold,
)


def test_metrics_at_threshold_computes_expected_values() -> None:
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.4, 0.8, 0.9])

    metrics = metrics_at_threshold(labels, probabilities, threshold=0.5)

    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["positive_rate"] == 0.5


def test_select_threshold_recall_at_min_precision() -> None:
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.2, 0.6, 0.7, 0.9])

    result = select_threshold(
        labels=labels,
        probabilities=probabilities,
        strategy="recall_at_min_precision",
        min_precision=0.6,
    )

    assert 0.0 <= result.threshold <= 1.0
    assert result.precision >= 0.6
    assert result.recall == 1.0
    assert result.num_samples == 4


def test_threshold_artifact_roundtrip(tmp_path: Path) -> None:
    labels = np.array([0, 1])
    probabilities = np.array([0.1, 0.9])
    result = select_threshold(labels, probabilities)
    output_path = tmp_path / "threshold.json"

    save_threshold_result(result, output_path)
    loaded_threshold = load_threshold_from_artifact(output_path, fallback=0.5)

    assert output_path.exists()
    assert json.loads(output_path.read_text(encoding="utf-8"))["threshold"] == result.threshold
    assert loaded_threshold == result.threshold


def test_load_threshold_uses_fallback_for_missing_artifact(tmp_path: Path) -> None:
    assert load_threshold_from_artifact(tmp_path / "missing.json", fallback=0.35) == 0.35
