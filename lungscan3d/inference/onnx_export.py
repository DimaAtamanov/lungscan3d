"""ONNX export utilities."""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lungscan3d.models import build_model
from lungscan3d.models.outputs import extract_positive_logits
from lungscan3d.utils.paths import ensure_dir

LOGGER = logging.getLogger(__name__)


class PositiveLogitWrapper(torch.nn.Module):
    """Wrap a model so ONNX export exposes one positive-class logit."""

    def __init__(self, model: torch.nn.Module) -> None:
        """Initialize wrapper.

        Args:
            model: Source classifier.
        """
        super().__init__()
        self.model = model

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Run wrapped model and return a single positive-class logit.

        Args:
            input_tensor: Input tensor with shape ``(B, C, D, H, W)``.

        Returns:
            Positive-class logits with shape ``(B, 1)``.
        """
        return extract_positive_logits(self.model(input_tensor))


def export_onnx(
    config: Any, checkpoint: str | None = None, output: str | None = None
) -> Path:
    """Export configured model to ONNX and run a lightweight validation.

    Args:
        config: Hydra configuration object.
        checkpoint: Optional checkpoint path override.
        output: Optional ONNX output path override.

    Returns:
        Path to exported ONNX file.
    """
    LOGGER.info("Exporting model to ONNX: model=%s", config.model.name)
    model = build_model(config)
    checkpoint_path = Path(checkpoint or config.infer.checkpoint_path)
    if checkpoint_path.exists():
        LOGGER.info("Loading checkpoint for ONNX export: %s", checkpoint_path)
        payload = torch.load(checkpoint_path, map_location="cpu")
        state_dict = payload.get("state_dict", payload)
        model_state_dict = {
            key.replace("model.", ""): value for key, value in state_dict.items()
        }
        model.load_state_dict(model_state_dict, strict=False)
    model.eval()
    export_model = PositiveLogitWrapper(model)
    export_model.eval()
    output_path = Path(output or config.infer.onnx_path)
    ensure_dir(output_path.parent)
    patch_size = [int(value) for value in config.data.patch_size]
    dummy_input = torch.zeros(
        1, int(config.model.in_channels), *patch_size, dtype=torch.float32
    )
    LOGGER.info("Writing ONNX model to %s", output_path)
    torch.onnx.export(
        export_model,
        dummy_input,
        output_path,
        input_names=[str(config.infer.input_name)],
        output_names=[str(config.infer.output_name)],
        dynamic_axes={
            str(config.infer.input_name): {0: "batch"},
            str(config.infer.output_name): {0: "batch"},
        },
        opset_version=int(config.infer.opset_version),
    )
    validate_onnx_export(
        output_path=output_path,
        input_name=str(config.infer.input_name),
        dummy_input=dummy_input,
    )
    LOGGER.info("ONNX export validated successfully: %s", output_path)
    return output_path


def validate_onnx_export(
    output_path: Path, input_name: str, dummy_input: torch.Tensor
) -> None:
    """Validate exported ONNX graph with checker and ONNX Runtime.

    Args:
        output_path: Path to exported ONNX file.
        input_name: ONNX input tensor name.
        dummy_input: Example tensor used for inference validation.
    """
    import onnx
    import onnxruntime as ort

    LOGGER.info("Validating ONNX graph: %s", output_path)
    model_proto = onnx.load(output_path)
    onnx.checker.check_model(model_proto)
    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    session.run(None, {input_name: dummy_input.cpu().numpy().astype(np.float32)})
