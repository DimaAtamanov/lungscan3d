"""Utilities for normalizing model outputs."""

import torch

ModelOutput = torch.Tensor | tuple[torch.Tensor, torch.Tensor]


def extract_class_logits(model_output: ModelOutput) -> torch.Tensor:
    """Extract raw class logits from a model output.

    Args:
    ----
        model_output: Either a tensor of logits or a tuple ``(logits, probabilities)``.

    Returns:
    -------
        Raw logits tensor.

    """
    if isinstance(model_output, tuple):
        return model_output[0]
    return model_output


def extract_positive_logits(model_output: ModelOutput) -> torch.Tensor:
    """Convert a model output to one positive-class binary logit.

    For a two-class classifier, the returned value is ``logit_1 - logit_0``. Applying
    sigmoid to this difference is equivalent to the class-1 softmax probability.

    Args:
    ----
        model_output: Model output tensor or ``(logits, probabilities)`` tuple.

    Returns:
    -------
        Positive-class logits with shape ``(B, 1)``.

    """
    logits = extract_class_logits(model_output)
    if logits.ndim == 2 and logits.size(1) == 2:
        return logits[:, 1:2] - logits[:, 0:1]
    if logits.ndim == 1:
        return logits.view(-1, 1)
    return logits
