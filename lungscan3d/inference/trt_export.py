"""TensorRT export helper."""

import logging
from pathlib import Path
from typing import Any

from lungscan3d.inference.onnx_export import export_onnx
from lungscan3d.utils.paths import ensure_dir

LOGGER = logging.getLogger(__name__)


def export_tensorrt(config: Any, output: str | None = None) -> Path:
    """Convert ONNX model to a TensorRT engine using TensorRT Python API."""
    LOGGER.info("Preparing TensorRT export")

    try:
        import tensorrt as trt
    except ImportError as error:
        raise ImportError(
            "TensorRT Python package is not installed. "
            "Install NVIDIA TensorRT in the current environment or run inside "
            "an NVIDIA TensorRT container."
        ) from error

    onnx_path = Path(config.infer.onnx_path)
    if not onnx_path.exists():
        LOGGER.info("ONNX model is missing; exporting it first")
        onnx_path = export_onnx(config)

    engine_path = Path(output or config.tensorrt.engine_path)
    ensure_dir(engine_path.parent)

    if bool(getattr(config.tensorrt, "dry_run", False)):
        LOGGER.info("TensorRT dry-run enabled; engine build skipped")
        return engine_path

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)

    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    LOGGER.info("Parsing ONNX model: %s", onnx_path)
    if not parser.parse_from_file(str(onnx_path)):
        errors = [str(parser.get_error(index)) for index in range(parser.num_errors)]
        raise RuntimeError("Failed to parse ONNX model:\n" + "\n".join(errors))

    builder_config = builder.create_builder_config()

    workspace_bytes = int(config.tensorrt.workspace_mb) * 1024 * 1024
    builder_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)

    _configure_precision(config, trt, builder, builder_config)
    _configure_optimization_profile(config, builder, builder_config, network)

    LOGGER.info("Building TensorRT engine: %s", engine_path)
    serialized_engine = builder.build_serialized_network(network, builder_config)

    if serialized_engine is None:
        raise RuntimeError("TensorRT engine build failed")

    engine_path.write_bytes(bytes(serialized_engine))
    LOGGER.info("TensorRT engine saved: %s", engine_path)

    return engine_path


def _configure_precision(
    config: Any,
    trt: Any,
    builder: Any,
    builder_config: Any,
) -> None:
    """Configure TensorRT precision mode."""
    precision = str(config.tensorrt.precision).lower()

    if precision in {"fp32", "float32"}:
        LOGGER.info("Using TensorRT FP32 precision")
        return

    if precision == "fp16":
        if not builder.platform_has_fast_fp16:
            LOGGER.warning("FP16 requested, but fast FP16 is not reported by this platform")
        builder_config.set_flag(trt.BuilderFlag.FP16)
        LOGGER.info("Using TensorRT FP16 precision")
        return

    if precision == "int8":
        raise ValueError(
            "INT8 TensorRT export requires a calibration pipeline. "
            "Use tensorrt.precision=fp16 or tensorrt.precision=fp32 for now."
        )

    raise ValueError("tensorrt.precision must be one of: fp32, fp16, int8")


def _configure_optimization_profile(
    config: Any,
    builder: Any,
    builder_config: Any,
    network: Any,
) -> None:
    """Configure dynamic batch profile for 3D input tensor."""
    input_tensor = network.get_input(0)
    if input_tensor is None:
        raise RuntimeError("TensorRT network has no inputs")

    input_name = str(input_tensor.name)

    configured_input_name = str(config.infer.input_name)
    if input_name != configured_input_name:
        LOGGER.warning(
            "Configured input name '%s' differs from ONNX input name '%s'. "
            "Using ONNX input name.",
            configured_input_name,
            input_name,
        )

    channels = int(config.model.in_channels)
    depth, height, width = (int(value) for value in config.data.patch_size)

    min_batch = int(config.tensorrt.min_batch_size)
    opt_batch = int(config.tensorrt.opt_batch_size)
    max_batch = int(config.tensorrt.max_batch_size)

    if not (1 <= min_batch <= opt_batch <= max_batch):
        raise ValueError(
            "TensorRT batch profile must satisfy: "
            "1 <= min_batch_size <= opt_batch_size <= max_batch_size"
        )

    min_shape = (min_batch, channels, depth, height, width)
    opt_shape = (opt_batch, channels, depth, height, width)
    max_shape = (max_batch, channels, depth, height, width)

    LOGGER.info(
        "TensorRT optimization profile for '%s': min=%s opt=%s max=%s",
        input_name,
        min_shape,
        opt_shape,
        max_shape,
    )

    profile = builder.create_optimization_profile()
    profile.set_shape(input_name, min_shape, opt_shape, max_shape)
    builder_config.add_optimization_profile(profile)
