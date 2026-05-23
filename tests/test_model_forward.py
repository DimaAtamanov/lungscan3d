import torch

from lungscan3d.models.baseline3d import LunaModel
from lungscan3d.models.outputs import extract_positive_logits
from lungscan3d.models.resnet3d import ResNet3DSE


def test_luna_model_forward_matches_dlwpt_shapes():
    model = LunaModel(in_channels=1, conv_channels=8)
    linear_output, probabilities = model(torch.zeros(2, 1, 32, 48, 48))

    assert tuple(linear_output.shape) == (2, 2)
    assert tuple(probabilities.shape) == (2, 2)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(2), atol=1e-6)


def test_extract_positive_logits_from_luna_output():
    model = LunaModel(in_channels=1, conv_channels=8)
    model_output = model(torch.zeros(2, 1, 32, 48, 48))
    positive_logits = extract_positive_logits(model_output)

    assert tuple(positive_logits.shape) == (2, 1)


def test_resnet3d_se_forward_shape():
    model = ResNet3DSE(
        in_channels=1,
        base_channels=8,
        blocks_per_stage=[1, 1],
        se_reduction=4,
        dropout=0.0,
    )
    logits = model(torch.zeros(2, 1, 32, 32, 32))

    assert tuple(logits.shape) == (2, 1)
