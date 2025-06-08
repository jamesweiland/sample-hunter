import torch
import torch.nn as nn
from torch import Tensor
from torchsummary import summary
from typing import List, Tuple
from sample_hunter._util import (
    STRIDE,
    PADDING,
    POOL_KERNEL_SIZE,
    CONV_LAYER_DIMS,
    NUM_BRANCHES,
    DIVIDE_AND_ENCODE_HIDDEN_DIM,
    EMBEDDING_DIM,
    N_MELS,
    SAMPLE_RATE,
    WINDOW_SIZE,
    DEVICE,
)

INPUT_SHAPE = (1, N_MELS, WINDOW_SIZE)


class EncoderNet(nn.Module):
    def __init__(
        self,
        conv_layer_dims: List[Tuple[int, int]],
        stride: int,
        padding: int,
        pool_kernel_size: int,
        num_branches: int,
        divide_and_encode_hidden_dim: int,
        embedding_dim: int,
        input_shape: Tuple[int, int, int],
    ):
        super().__init__()

        # set up the conv blocks
        self.conv_blocks = nn.ModuleList()
        for i, dims in enumerate(conv_layer_dims):
            in_ch, out_ch = dims
            kernel_size = (1, 3) if i % 2 == 0 else (3, 1)
            block = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                ),
                nn.BatchNorm2d(out_ch),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=pool_kernel_size),
            )
            self.conv_blocks.append(block)

        # flatten the tensor before passing it to the divide-and-encode block
        self.flatten = nn.Flatten()
        h, w = self._calculate_conv_output_shape(input_shape)
        conv_out_dim = conv_layer_dims[-1][1] * h * w

        # set up the divide and encode block
        assert embedding_dim % num_branches == 0
        embedding_dim_per_branch = embedding_dim // num_branches
        self.divide_and_encode = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(conv_out_dim, divide_and_encode_hidden_dim),
                    nn.ELU(),
                    nn.BatchNorm1d(divide_and_encode_hidden_dim),
                    nn.Linear(divide_and_encode_hidden_dim, embedding_dim_per_branch),
                )
                for _ in range(num_branches)
            ]
        )
        self.num_branches = num_branches

    def forward(self, x: Tensor):
        for block in self.conv_blocks:
            x = block(x)

        x = self.flatten(x)

        splits = [fc(x) for fc in self.divide_and_encode]

        return torch.cat(splits, dim=1)

    def _calculate_conv_output_shape(self, input_shape: Tuple[int, int, int]):
        """Make a dummy forward pass to the conv stack to calculate the shape of the
        output tensor (the shape of the input to the divide and encode block)"""
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            for block in self.conv_blocks:
                dummy = block(dummy)
        _, _, h, w = dummy.shape
        return h, w


if __name__ == "__main__":
    print(INPUT_SHAPE)

    model = EncoderNet(
        conv_layer_dims=CONV_LAYER_DIMS,
        stride=STRIDE,
        padding=PADDING,
        pool_kernel_size=POOL_KERNEL_SIZE,
        num_branches=NUM_BRANCHES,
        divide_and_encode_hidden_dim=DIVIDE_AND_ENCODE_HIDDEN_DIM,
        embedding_dim=EMBEDDING_DIM,
        input_shape=INPUT_SHAPE,
    ).to(DEVICE)

    summary(model=model, input_size=INPUT_SHAPE, device=DEVICE)
