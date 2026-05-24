import torch
from lungscan3d.training.losses import FocalLossWithLogits


def test_focal_loss_is_scalar_and_finite():
    loss_fn = FocalLossWithLogits(alpha=0.75, gamma=2.0)
    logits = torch.tensor([[0.0], [2.0], [-2.0]])
    targets = torch.tensor([[0.0], [1.0], [0.0]])

    loss = loss_fn(logits, targets)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
