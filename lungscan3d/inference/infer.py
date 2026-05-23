"""Inference routines."""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lungscan3d.inference.postprocess import logits_to_prediction
from lungscan3d.inference.thresholds import load_threshold_from_artifact
from lungscan3d.models import build_model
from lungscan3d.models.outputs import extract_positive_logits

LOGGER = logging.getLogger(__name__)


def infer_patch(config: Any, input_path: str | Path) -> dict[str, float | int]:
    """Run PyTorch inference on a preprocessed NumPy patch.

    Args:
        config: Hydra configuration object.
        input_path: Path to ``.npy`` patch with shape ``(1, D, H, W)`` or ``(D, H, W)``.

    Returns:
        Prediction dictionary.
    """
    LOGGER.info("Loading input patch: %s", input_path)
    patch = np.load(input_path).astype(np.float32)
    if patch.ndim == 3:
        patch = patch[None, ...]
    if patch.ndim != 4:
        raise ValueError("Expected patch with shape (C, D, H, W) or (D, H, W)")
    input_tensor = torch.from_numpy(patch).unsqueeze(0)
    LOGGER.info("Building inference model: %s", config.model.name)
    model = build_model(config)
    model.eval()
    checkpoint_path = Path(config.infer.checkpoint_path)
    if checkpoint_path.exists():
        LOGGER.info("Loading checkpoint: %s", checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        model_state_dict = {
            key.replace("model.", ""): value for key, value in state_dict.items()
        }
        model.load_state_dict(model_state_dict, strict=False)
    with torch.no_grad():
        logits = extract_positive_logits(model(input_tensor))
    LOGGER.info("Model inference finished")
    threshold = float(config.postprocess.threshold)
    if bool(config.postprocess.get("use_threshold_artifact", False)):
        threshold = load_threshold_from_artifact(
            path=config.postprocess.threshold_artifact_path,
            fallback=threshold,
        )
    return logits_to_prediction(logits, threshold=threshold)


def infer(config: Any, input: str) -> None:
    """Run inference and print JSON result.

    Args:
        config: Hydra configuration object.
        input: Path to input NumPy patch.
    """
    result = infer_patch(config, input)
    LOGGER.info(
        "Inference result: label=%s probability=%.6f",
        result["label"],
        result["probability"],
    )
    print(json.dumps(result, indent=2))
