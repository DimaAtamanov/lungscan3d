"""Exact DLwP LUNA baseline 3D CNN model."""

import math

import torch
from torch import nn


class LunaBlock(nn.Module):
    """Convolutional block from the DLwP LUNA baseline.

    The block applies two 3D convolutions with ReLU activations and then downsamples the
    feature map with ``MaxPool3d(2, 2)``.
    """

    def __init__(self, in_channels: int, conv_channels: int) -> None:
        """Initialize a LUNA convolutional block.

        Args:
        ----
            in_channels: Number of input channels.
            conv_channels: Number of output channels for both convolutions in the block.

        """
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels,
            conv_channels,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(
            conv_channels,
            conv_channels,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(2, 2)

    def forward(self, input_batch: torch.Tensor) -> torch.Tensor:
        """Run the block forward pass.

        Args:
        ----
            input_batch: Input tensor with shape ``(B, C, D, H, W)``.

        Returns:
        -------
            Downsampled output tensor.

        """
        block_out = self.conv1(input_batch)
        block_out = self.relu1(block_out)
        block_out = self.conv2(block_out)
        block_out = self.relu2(block_out)
        return self.maxpool(block_out)


class LunaModel(nn.Module):
    """LUNA candidate classifier reproduced from Deep Learning with PyTorch.

    Expected input shape is ``(B, 1, 32, 48, 48)``. After four max-pooling operations the
    spatial shape becomes ``(2, 3, 3)`` and the last convolutional stage has 64 channels,
    therefore the flattened feature vector has size ``64 * 2 * 3 * 3 = 1152``.

    The forward method returns both raw class logits and softmax probabilities, matching
    the book-style implementation.
    """

    def __init__(self, in_channels: int = 1, conv_channels: int = 8) -> None:
        """Initialize the DLwP LUNA baseline model.

        Args:
        ----
            in_channels: Number of input channels. CT patches use one grayscale channel.
            conv_channels: Number of channels in the first convolutional block.

        """
        super().__init__()
        self.tail_batchnorm = nn.BatchNorm3d(1)
        self.block1 = LunaBlock(in_channels, conv_channels)
        self.block2 = LunaBlock(conv_channels, conv_channels * 2)
        self.block3 = LunaBlock(conv_channels * 2, conv_channels * 4)
        self.block4 = LunaBlock(conv_channels * 4, conv_channels * 8)
        self.head_linear = nn.Linear(1152, 2)
        self.head_activation = nn.Softmax(dim=1)
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize convolutional and linear layers as in the DLwP baseline."""
        for module in self.modules():
            if type(module) in {
                nn.Linear,
                nn.Conv3d,
                nn.Conv2d,
                nn.ConvTranspose2d,
                nn.ConvTranspose3d,
            }:
                nn.init.kaiming_normal_(
                    module.weight.data,
                    a=0,
                    mode="fan_out",
                    nonlinearity="relu",
                )
                if module.bias is not None:
                    _, fan_out = nn.init._calculate_fan_in_and_fan_out(module.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(module.bias, -bound, bound)

    def forward(self, input_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run model forward pass.

        Args:
        ----
            input_batch: Input tensor with shape ``(B, 1, 32, 48, 48)``.

        Returns:
        -------
            Pair ``(linear_output, probabilities)`` where ``linear_output`` has shape
            ``(B, 2)`` and ``probabilities`` is ``softmax(linear_output)``.

        """
        bn_output = self.tail_batchnorm(input_batch)
        block_out = self.block1(bn_output)
        block_out = self.block2(block_out)
        block_out = self.block3(block_out)
        block_out = self.block4(block_out)
        conv_flat = block_out.view(block_out.size(0), -1)
        linear_output = self.head_linear(conv_flat)
        return linear_output, self.head_activation(linear_output)
