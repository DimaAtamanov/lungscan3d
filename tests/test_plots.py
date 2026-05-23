from pathlib import Path

from lungscan3d.training.plots import save_training_plots


def test_save_training_plots_creates_required_files(tmp_path: Path):
    history = {
        "train/loss_epoch": [1.0, 0.8],
        "val/loss": [1.1, 0.9],
        "val/roc_auc": [0.6, 0.7],
        "val/recall": [0.5, 0.6],
        "val/pr_auc": [0.4, 0.5],
    }

    save_training_plots(history, tmp_path)

    assert (tmp_path / "loss.png").exists()
    assert (tmp_path / "roc_auc.png").exists()
    assert (tmp_path / "recall_pr_auc.png").exists()
