"""Postprocessing utilities."""

import torch


def logits_to_prediction(
    logits: torch.Tensor, threshold: float
) -> dict[str, float | int]:
    """Convert logits to probability and binary prediction.

    Args:
        logits: Tensor with a single binary logit.
        threshold: Decision threshold.

    Returns:
        Dictionary with probability, threshold, and label.
    """
    probability = float(torch.sigmoid(logits.detach().cpu()).view(-1)[0].item())
    label = int(probability >= threshold)
    return {"probability": probability, "threshold": float(threshold), "label": label}
