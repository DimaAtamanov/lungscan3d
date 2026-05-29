"""Public command-line entry point."""

import json
import logging
import subprocess
import sys
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from lungscan3d.data.download import download_data, download_luna16_dataset
from lungscan3d.data.preprocessing import preprocess
from lungscan3d.inference.infer import infer as run_infer
from lungscan3d.inference.onnx_export import export_onnx
from lungscan3d.inference.thresholds import optimize_threshold as run_optimize_threshold
from lungscan3d.inference.trt_export import export_tensorrt
from lungscan3d.serving.triton_client import call_triton
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
    """Load config and call a config-only function."""
    config = _load_config(list(overrides))
    LOGGER.info(
        "Resolved config for command: project=%s, data=%s",
        config.project_name,
        config.data.name,
    )
    return function(config)


def _require_config_value(value: Any, name: str) -> Any:
    """Return a required Hydra value or raise a CLI-friendly error."""
    if value in (None, ""):
        raise ValueError(f"Hydra parameter '{name}' is required for this command")
    return value


def _download_luna16_from_config(config: DictConfig) -> None:
    """Download LUNA16 using only Hydra config values."""
    download_luna16_dataset(
        raw_dir=Path(config.data.raw_dir),
        subsets=config.data.download_subsets,
        max_subsets=config.data.download_max_subsets,
        include_metadata=bool(config.data.download_metadata),
        extract=bool(config.data.extract_archives),
        keep_archives=bool(config.data.keep_archives),
        overwrite=bool(config.data.overwrite_downloads),
    )


def _dvc_add_from_config(config: DictConfig) -> None:
    """Run ``dvc add`` using ``dvc.target`` from Hydra config."""
    target = _require_config_value(config.dvc.target, "dvc.target")
    if not run_dvc_add(target=target):
        raise RuntimeError("DVC add failed; see logs above")


def _dvc_pull_from_config(config: DictConfig) -> None:
    """Run ``dvc pull`` using ``dvc.*`` from Hydra config."""
    if not run_dvc_pull(target=config.dvc.target, remote=config.dvc.remote):
        raise RuntimeError("DVC pull failed; see logs above")


def _dvc_push_from_config(config: DictConfig) -> None:
    """Run ``dvc push`` using ``dvc.*`` from Hydra config."""
    if not run_dvc_push(target=config.dvc.target, remote=config.dvc.remote):
        raise RuntimeError("DVC push failed; see logs above")


def _infer_from_config(config: DictConfig) -> None:
    """Run local inference using ``infer.input_path`` from Hydra config."""
    input_path = _require_config_value(config.infer.input_path, "infer.input_path")
    run_infer(config, input=str(input_path))


def _optimize_threshold_from_config(config: DictConfig) -> None:
    """Optimize postprocessing threshold using only Hydra config values."""
    if str(config.postprocess.split) not in {"val", "test"}:
        raise ValueError("postprocess.split must be either 'val' or 'test'")
    result = run_optimize_threshold(
        config,
        checkpoint=config.infer.checkpoint_path,
        split=str(config.postprocess.split),  # type: ignore[arg-type]
        output=config.postprocess.threshold_artifact_path,
    )
    print(json.dumps(asdict(result), indent=2))


def _export_onnx_from_config(config: DictConfig) -> None:
    """Export ONNX using ``infer.checkpoint_path`` and ``infer.onnx_path``."""
    path = export_onnx(
        config,
        checkpoint=config.infer.checkpoint_path,
        output=config.infer.onnx_path,
    )
    print(path)


def _export_tensorrt_from_config(config: DictConfig) -> None:
    """Export TensorRT engine using ``tensorrt.engine_path``."""
    path = export_tensorrt(config, output=config.tensorrt.engine_path)
    print(path)


def _triton_client_from_config(config: DictConfig) -> None:
    """Call Triton using ``triton.input_path`` from Hydra config."""
    input_path = _require_config_value(config.triton.input_path, "triton.input_path")
    call_triton(config, input=str(input_path))


def _validate_triton_repository(model_repository: Path, model_name: str) -> None:
    """Validate that a Triton repository has the expected model layout."""
    model_dir = model_repository / model_name
    config_path = model_dir / "config.pbtxt"
    version_dir = model_dir / "1"
    if not config_path.exists():
        raise FileNotFoundError(f"Triton config not found: {config_path}")
    if not version_dir.exists():
        raise FileNotFoundError(
            f"Triton model version directory not found: {version_dir}"
        )
    LOGGER.info("Triton repository layout looks valid: %s", model_dir)


def _run_self_test(config: DictConfig) -> None:
    """Run an end-to-end smoke workflow on synthetic data."""
    LOGGER.info("Starting LungScan3D self-test")
    base_overrides = [
        "data=synthetic",
        "data.num_samples=8",
        "model=dlwpt_baseline",
        "trainer.max_epochs=1",
        "trainer.fast_dev_run=true",
        "data.batch_size=2",
        "trainer.accelerator=cpu",
        "trainer.devices=1",
        "logging.mode=none",
        "paths.artifacts_dir=artifacts/self_test",
        "paths.checkpoints_dir=artifacts/self_test/checkpoints",
        "paths.plots_dir=artifacts/self_test/plots",
        "paths.processed_dir=data/processed",
        "infer.onnx_path=artifacts/self_test/onnx/lungscan3d.onnx",
        "infer.checkpoint_path=artifacts/self_test/checkpoints/best.ckpt",
        "tensorrt.engine_path=artifacts/self_test/tensorrt/lungscan3d.plan",
    ]
    download_data(_load_config(base_overrides))
    run_train(_load_config(base_overrides))
    export_onnx(_load_config(base_overrides))
    export_tensorrt(_load_config(base_overrides))
    _validate_triton_repository(
        Path(config.triton.model_repository), str(config.triton.model_name)
    )

    if bool(config.self_test.run_pytest):
        command = [sys.executable, "-m", "pytest", *list(config.self_test.pytest_args)]
        LOGGER.info("Running pytest: %s", " ".join(command))
        subprocess.run(command, check=True)

    LOGGER.info("Self-test finished successfully")


class Commands:
    """Fire CLI commands for LungScan3D.

    Every command accepts only Hydra overrides. Command-specific values such as
    checkpoint paths, output paths, DVC targets, LUNA16 subsets, inference inputs,
    TensorRT engine paths, and Triton inputs live in config files and can be
    overridden as ``key=value`` arguments.
    """

    def show_config(self, *overrides: str) -> None:
        """Print the resolved Hydra configuration."""
        config = _load_config(list(overrides))
        print(OmegaConf.to_yaml(config, resolve=True))

    def dvc_add(self, *overrides: str) -> None:
        """Add ``dvc.target`` to DVC tracking."""
        _run_with_config(_dvc_add_from_config, overrides)

    def dvc_pull(self, *overrides: str) -> None:
        """Pull DVC-tracked data or artifacts using ``dvc.*`` config."""
        _run_with_config(_dvc_pull_from_config, overrides)

    def dvc_push(self, *overrides: str) -> None:
        """Push DVC-tracked data or artifacts using ``dvc.*`` config."""
        _run_with_config(_dvc_push_from_config, overrides)

    def download_data(self, *overrides: str) -> None:
        """Download or generate data according to Hydra config."""
        _run_with_config(download_data, overrides)

    def download_luna16(self, *overrides: str) -> None:
        """Download selected LUNA16 archives using ``data.*`` config."""
        effective_overrides = _with_default_data_luna16(list(overrides))
        _run_with_config(_download_luna16_from_config, tuple(effective_overrides))

    def preprocess(self, *overrides: str) -> None:
        """Run preprocessing pipeline."""
        _run_with_config(preprocess, overrides)

    def train(self, *overrides: str) -> None:
        """Train a model."""
        _run_with_config(run_train, overrides)

    def infer(self, *overrides: str) -> None:
        """Run inference for ``infer.input_path``."""
        _run_with_config(_infer_from_config, overrides)

    def optimize_threshold(self, *overrides: str) -> None:
        """Select and save an operating threshold on validation/test predictions."""
        _run_with_config(_optimize_threshold_from_config, overrides)

    def export_onnx(self, *overrides: str) -> None:
        """Export a model to ONNX."""
        _run_with_config(_export_onnx_from_config, overrides)

    def export_tensorrt(self, *overrides: str) -> None:
        """Export a model to TensorRT through ``trtexec``."""
        _run_with_config(_export_tensorrt_from_config, overrides)

    def triton_client(self, *overrides: str) -> None:
        """Call Triton HTTP endpoint for ``triton.input_path``."""
        _run_with_config(_triton_client_from_config, overrides)

    def self_test(self, *overrides: str) -> None:
        """Run synthetic end-to-end package smoke test."""
        _run_with_config(_run_self_test, overrides)


def _with_default_data_luna16(overrides: list[str]) -> list[str]:
    """Select ``data=luna16`` for download-luna16 unless user chose another data group."""
    if not any(value.startswith("data=") for value in overrides):
        return ["data=luna16", *overrides]
    return overrides


def main() -> None:
    """Run Fire CLI."""
    setup_logging()
    import fire

    commands = Commands()
    fire.Fire(
        {
            "show-config": commands.show_config,
            "dvc-add": commands.dvc_add,
            "dvc-pull": commands.dvc_pull,
            "dvc-push": commands.dvc_push,
            "download-data": commands.download_data,
            "download-luna16": commands.download_luna16,
            "preprocess": commands.preprocess,
            "train": commands.train,
            "infer": commands.infer,
            "optimize-threshold": commands.optimize_threshold,
            "export-onnx": commands.export_onnx,
            "export-tensorrt": commands.export_tensorrt,
            "triton-client": commands.triton_client,
            "self-test": commands.self_test,
        }
    )


if __name__ == "__main__":
    main()
