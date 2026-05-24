"""Utilities for saving training curves."""

from pathlib import Path

import matplotlib.pyplot as plt

from lungscan3d.utils.paths import ensure_dir


def save_training_plots(history: dict[str, list[float]], plots_dir: str | Path) -> None:
    """Save static training plots as PNG files.

    Args:
    ----
        history: Metric history collected during training.
        plots_dir: Directory where plot images are written.

    """
    output_dir = ensure_dir(plots_dir)
    plot_specs = {
        "loss.png": ["train/loss_epoch", "val/loss"],
        "roc_auc.png": ["val/roc_auc", "test/roc_auc"],
        "recall_pr_auc.png": ["val/recall", "val/pr_auc", "val/f1"],
    }
    for file_name, metric_names in plot_specs.items():
        _save_single_plot(
            history=history,
            metric_names=metric_names,
            output_path=output_dir / file_name,
        )


def _save_single_plot(
    history: dict[str, list[float]],
    metric_names: list[str],
    output_path: Path,
) -> None:
    """Save one metric plot.

    Args:
    ----
        history: Metric history collected during training.
        metric_names: Names to include in the plot.
        output_path: Target PNG file path.

    """
    figure = plt.figure()
    has_values = False
    for metric_name in metric_names:
        values = history.get(metric_name, [])
        if not values:
            continue
        has_values = True
        plt.plot(range(1, len(values) + 1), values, label=metric_name)
    if not has_values:
        plt.plot([1], [0.0], label="not_available")
    plt.title(output_path.stem.replace("_", " "))
    plt.xlabel("epoch")
    plt.ylabel("value")
    plt.legend()
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)
