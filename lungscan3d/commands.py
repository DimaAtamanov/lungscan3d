"""Public command-line entry point."""

import json
import logging
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import fire
import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from lungscan3d.data.download import download_data, download_luna16_dataset
from lungscan3d.data.preprocessing import preprocess
from lungscan3d.inference.infer import infer as run_infer
from lungscan3d.inference.onnx_export import export_onnx
from lungscan3d.inference.thresholds import optimize_threshold as run_optimize_threshold
from lungscan3d.inference.trt_export import export_tensorrt
from lungscan3d.serving.triton_client import call_triton
from lungscan3d.training.hard_negative_mining import save_hard_negative_indices
from lungscan3d.training.hard_negative_mining import (
    select_hard_negative_indices as run_select_hard_negative_indices,
)
from lungscan3d.training.train import train as run_train
from lungscan3d.utils.dvc import dvc_add as run_dvc_add
from lungscan3d.utils.dvc import dvc_pull as run_dvc_pull
from lungscan3d.utils.dvc import dvc_push as run_dvc_push
from lungscan3d.utils.logging import setup_logging

LOGGER = logging.getLogger(__name__)


def _load_config(overrides: list[str] | None = None) -> DictConfig:
    """Load Hydra config through the compose API.

    Args:
    ----
        overrides: Hydra-style overrides such as ``["data=synthetic"]``.

    Returns:
    -------
        Composed Hydra configuration.

    """
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        return compose(config_name="config", overrides=overrides or [])


def _run_with_config(function: Callable[[Any], Any], overrides: tuple[str, ...]) -> Any:
    """Load config and call a config-only function.

    Args:
    ----
        function: Function that accepts Hydra config.
        overrides: Hydra overrides.

    Returns:
    -------
        Function result.

    """
    config = _load_config(list(overrides))
    LOGGER.info(
        "Resolved config for command: project=%s, data=%s",
        config.project_name,
        config.data.name,
    )
    return function(config)


class Commands:
    """CLI commands for LungScan3D."""

    def show_config(self, *overrides: str) -> None:
        """Print the resolved Hydra configuration.

        Args:
        ----
            *overrides: Hydra overrides.

        """
        config = _load_config(list(overrides))
        print(OmegaConf.to_yaml(config, resolve=True))

    def dvc_add(self, target: str) -> None:
        """Add a file or directory to DVC tracking.

        Args:
        ----
            target: File or directory to track with DVC.

        """
        if not run_dvc_add(target=target):
            raise RuntimeError("DVC add failed; see logs above")

    def dvc_pull(self, target: str | None = None, remote: str | None = None) -> None:
        """Pull DVC-tracked data or model artifacts.

        Args:
        ----
            target: Optional DVC target, for example ``data/processed/luna16``.
            remote: Optional DVC remote, for example ``data_storage``.

        """
        if not run_dvc_pull(target=target, remote=remote):
            raise RuntimeError("DVC pull failed; see logs above")

    def dvc_push(self, target: str | None = None, remote: str | None = None) -> None:
        """Push DVC-tracked data or model artifacts.

        Args:
        ----
            target: Optional DVC target.
            remote: Optional DVC remote, for example ``model_storage``.

        """
        if not run_dvc_push(target=target, remote=remote):
            raise RuntimeError("DVC push failed; see logs above")

    def download_data(self, *overrides: str) -> None:
        """Download or generate data according to Hydra config.

        Args:
        ----
            *overrides: Hydra overrides.

        """
        _run_with_config(download_data, overrides)

    def download_luna16(
        self,
        raw_dir: str = "data/raw/luna16",
        subsets: str | None = None,
        max_subsets: int | None = None,
        include_metadata: bool = True,
        extract: bool = True,
        keep_archives: bool = True,
        overwrite: bool = False,
    ) -> None:
        """Download selected LUNA16 archives from Zenodo and optionally extract them.

        Args:
        ----
            raw_dir: Destination directory.
            subsets: Comma-separated subset ids, for example ``"0,1"``.
            max_subsets: Number of first subsets to download when ``subsets`` is omitted.
            include_metadata: Download ``annotations.csv`` and ``candidates.csv``.
            extract: Unzip downloaded subset archives.
            keep_archives: Keep zip archives after extraction.
            overwrite: Redownload/re-extract existing files.

        """
        download_luna16_dataset(
            raw_dir=raw_dir,
            subsets=subsets,
            max_subsets=max_subsets,
            include_metadata=include_metadata,
            extract=extract,
            keep_archives=keep_archives,
            overwrite=overwrite,
        )

    def preprocess(self, *overrides: str) -> None:
        """Run preprocessing pipeline.

        Args:
        ----
            *overrides: Hydra overrides.

        """
        _run_with_config(preprocess, overrides)

    def train(self, *overrides: str) -> None:
        """Train a model.

        Args:
        ----
            *overrides: Hydra overrides.

        """
        _run_with_config(run_train, overrides)

    def infer(self, input: str, *overrides: str) -> None:
        """Run inference for a preprocessed NumPy patch.

        Args:
        ----
            input: Path to ``.npy`` patch.
            *overrides: Hydra overrides.

        """
        config = _load_config(list(overrides))
        run_infer(config, input=input)

    def optimize_threshold(
        self,
        *overrides: str,
        checkpoint: str | None = None,
        split: str = "val",
        output: str | None = None,
    ) -> None:
        """Select and save an operating threshold on validation predictions.

        Args:
        ----
            *overrides: Hydra overrides.
            checkpoint: Optional trained checkpoint path.
            split: Dataset split used for threshold search: ``val`` or ``test``.
            output: Optional JSON output path.

        """
        if split not in {"val", "test"}:
            raise ValueError("split must be either 'val' or 'test'")
        config = _load_config(list(overrides))
        result = run_optimize_threshold(
            config,
            checkpoint=checkpoint,
            split=split,  # type: ignore[arg-type]
            output=output,
        )
        print(json.dumps(asdict(result), indent=2))

    def select_hard_negatives(
        self,
        labels: str,
        probabilities: str,
        output: str = "artifacts/hard_negatives/train_hard_negatives.npy",
        top_fraction: float = 0.25,
        min_probability: float = 0.5,
    ) -> None:
        """Select hard negatives from saved labels and model probabilities.

        Args:
        ----
            labels: Path to ``.npy`` array with binary labels.
            probabilities: Path to ``.npy`` array with positive-class probabilities.
            output: Destination path for selected hard-negative indices.
            top_fraction: Fraction of the hardest negative examples to keep.
            min_probability: Minimum positive-class probability for a negative example.

        """
        label_array = np.load(labels)
        probability_array = np.load(probabilities)
        indices = run_select_hard_negative_indices(
            labels=label_array,
            probabilities=probability_array,
            top_fraction=top_fraction,
            min_probability=min_probability,
        )
        save_hard_negative_indices(indices, output)
        print(json.dumps({"output": output, "num_hard_negatives": int(len(indices))}, indent=2))

    def export_onnx(
        self,
        *overrides: str,
        checkpoint: str | None = None,
        output: str | None = None,
    ) -> None:
        """Export a model to ONNX.

        Args:
        ----
            *overrides: Hydra overrides.
            checkpoint: Optional checkpoint path.
            output: Optional ONNX output path.

        """
        config = _load_config(list(overrides))
        path = export_onnx(config, checkpoint=checkpoint, output=output)
        print(path)

    def export_tensorrt(
        self,
        *overrides: str,
        output: str = "artifacts/tensorrt/lungscan3d.engine",
    ) -> None:
        """Export a model to TensorRT.

        Args:
        ----
            *overrides: Hydra overrides.
            output: Target TensorRT engine path.

        """
        config = _load_config(list(overrides))
        path = export_tensorrt(config, output=output)
        print(path)

    def triton_client(self, input: str, *overrides: str) -> None:
        """Call Triton server with a NumPy patch.

        Args:
        ----
            input: Path to input patch.
            *overrides: Hydra overrides.
            url: Triton HTTP endpoint URL.

        """
        config = _load_config(list(overrides))
        call_triton(config, input=input)


def main() -> None:
    """Run Fire CLI."""
    setup_logging()
    fire.Fire(Commands)


if __name__ == "__main__":
    main()
