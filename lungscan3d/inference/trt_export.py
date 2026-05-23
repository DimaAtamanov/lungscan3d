"""TensorRT export helper."""

import logging
import subprocess
from pathlib import Path
from typing import Any

from lungscan3d.inference.onnx_export import export_onnx
from lungscan3d.utils.paths import ensure_dir

LOGGER = logging.getLogger(__name__)


def export_tensorrt(config: Any, output: str = "artifacts/tensorrt/lungscan3d.engine") -> Path:
    """Convert ONNX model to TensorRT engine through ``trtexec``.

    Args:
    ----
        config: Hydra configuration object.
        output: Target TensorRT engine path.

    Returns:
    -------
        Path to TensorRT engine.

    """
    LOGGER.info("Preparing TensorRT export")
    onnx_path = Path(config.infer.onnx_path)
    if not onnx_path.exists():
        LOGGER.info("ONNX model is missing; exporting it first")
        onnx_path = export_onnx(config)
    engine_path = Path(output)
    ensure_dir(engine_path.parent)
    LOGGER.info("Running trtexec: onnx=%s, engine=%s", onnx_path, engine_path)
    subprocess.run(
        ["trtexec", f"--onnx={onnx_path}", f"--saveEngine={engine_path}", "--fp16"],
        check=True,
    )
    LOGGER.info("TensorRT engine saved: %s", engine_path)
    return engine_path
