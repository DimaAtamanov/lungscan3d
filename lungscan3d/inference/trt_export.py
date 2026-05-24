"""TensorRT export helper."""

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any

from lungscan3d.inference.onnx_export import export_onnx
from lungscan3d.utils.paths import ensure_dir

LOGGER = logging.getLogger(__name__)


def export_tensorrt(config: Any, output: str | None = None) -> Path:
    """Convert ONNX model to TensorRT engine through ``trtexec``."""
    LOGGER.info("Preparing TensorRT export")
    trtexec_path = str(getattr(config.tensorrt, "trtexec_path", "trtexec"))
    if shutil.which(trtexec_path) is None and not bool(getattr(config.tensorrt, "dry_run", False)):
        raise FileNotFoundError(
            "TensorRT CLI 'trtexec' was not found in PATH. Install NVIDIA TensorRT "
            "on the host or use the TensorRT container described in README.md."
        )

    onnx_path = Path(config.infer.onnx_path)
    if not onnx_path.exists():
        LOGGER.info("ONNX model is missing; exporting it first")
        onnx_path = export_onnx(config)

    engine_path = Path(output or config.tensorrt.engine_path)
    ensure_dir(engine_path.parent)
    command = _build_trtexec_command(config, trtexec_path, onnx_path, engine_path)
    LOGGER.info("Running TensorRT export: %s", " ".join(command))
    if bool(getattr(config.tensorrt, "dry_run", False)):
        LOGGER.info("TensorRT dry-run enabled; command was built but not executed")
        return engine_path

    subprocess.run(command, check=True)
    LOGGER.info("TensorRT engine saved: %s", engine_path)
    return engine_path


def _build_trtexec_command(
    config: Any,
    trtexec_path: str,
    onnx_path: Path,
    engine_path: Path,
) -> list[str]:
    """Build a deterministic trtexec command from Hydra config."""
    input_name = str(config.infer.input_name)
    channels = int(config.model.in_channels)
    depth, height, width = (int(value) for value in config.data.patch_size)
    min_batch = int(config.tensorrt.min_batch_size)
    opt_batch = int(config.tensorrt.opt_batch_size)
    max_batch = int(config.tensorrt.max_batch_size)
    shape_suffix = f"{channels}x{depth}x{height}x{width}"
    command = [
        trtexec_path,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--memPoolSize=workspace:{int(config.tensorrt.workspace_mb)}",
        f"--minShapes={input_name}:{min_batch}x{shape_suffix}",
        f"--optShapes={input_name}:{opt_batch}x{shape_suffix}",
        f"--maxShapes={input_name}:{max_batch}x{shape_suffix}",
    ]
    precision = str(config.tensorrt.precision).lower()
    if precision == "fp16":
        command.append("--fp16")
    elif precision == "int8":
        command.append("--int8")
    elif precision not in {"fp32", "float32"}:
        raise ValueError("tensorrt.precision must be one of: fp32, fp16, int8")
    command.extend(str(arg) for arg in getattr(config.tensorrt, "extra_args", []))
    return command
