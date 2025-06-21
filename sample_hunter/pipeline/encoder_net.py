import torch
import torch.nn as nn
from torch import Tensor
from torchsummary import summary
from typing import List, Tuple

from sample_hunter.pipeline.song_pairs_dataset import SongPairsDataset
from sample_hunter._util import (
    DEFAULT_STRIDE,
    DEFAULT_PADDING,
    DEFAULT_POOL_KERNEL_SIZE,
    CONV_LAYER_DIMS,
    DEFAULT_NUM_BRANCHES,
    DEFAULT_DIVIDE_AND_ENCODE_HIDDEN_DIM,
    DEFAULT_EMBEDDING_DIM,
    DEVICE,
    AUDIO_DIR,
    ANNOTATIONS_PATH,
    DEFAULT_MEL_SPECTROGRAM,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_WINDOW_NUM_SAMPLES,
)


class EncoderNet(nn.Module):
    def __init__(
        self,
        input_shape: torch.Size,
        conv_layer_dims: List[Tuple[int, int]] = CONV_LAYER_DIMS,
        stride: int = DEFAULT_STRIDE,
        padding: int = DEFAULT_PADDING,
        pool_kernel_size: int = DEFAULT_POOL_KERNEL_SIZE,
        num_branches: int = DEFAULT_NUM_BRANCHES,
        divide_and_encode_hidden_dim: int = DEFAULT_DIVIDE_AND_ENCODE_HIDDEN_DIM,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
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
        out = torch.cat(splits, dim=1)

        return out

    def _calculate_conv_output_shape(self, input_shape: torch.Size):
        """Make a dummy forward pass to the conv stack to calculate the shape of the
        output tensor (the shape of the input to the divide and encode block)"""
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            for block in self.conv_blocks:
                dummy = block(dummy)
        _, _, h, w = dummy.shape
        return h, w


if __name__ == "__main__":

    dataset = SongPairsDataset(
        audio_dir=AUDIO_DIR,
        annotations_file=ANNOTATIONS_PATH,
        mel_spectrogram=DEFAULT_MEL_SPECTROGRAM,
        target_sample_rate=DEFAULT_SAMPLE_RATE,
        num_samples=DEFAULT_WINDOW_NUM_SAMPLES,
        device=DEVICE,
    )

    input_shape = dataset.shape()

    model = EncoderNet(
        conv_layer_dims=CONV_LAYER_DIMS,
        stride=DEFAULT_STRIDE,
        padding=DEFAULT_PADDING,
        pool_kernel_size=DEFAULT_POOL_KERNEL_SIZE,
        num_branches=DEFAULT_NUM_BRANCHES,
        divide_and_encode_hidden_dim=DEFAULT_DIVIDE_AND_ENCODE_HIDDEN_DIM,
        embedding_dim=DEFAULT_EMBEDDING_DIM,
        input_shape=input_shape,
    ).to(DEVICE)

    summary(model=model, input_size=input_shape, device=DEVICE)

    """
    
    ----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1          [-1, 16, 66, 173]              64
       BatchNorm2d-2          [-1, 16, 66, 173]              32
               ELU-3          [-1, 16, 66, 173]               0
         MaxPool2d-4           [-1, 16, 33, 86]               0
            Conv2d-5           [-1, 32, 33, 88]           1,568
       BatchNorm2d-6           [-1, 32, 33, 88]              64
               ELU-7           [-1, 32, 33, 88]               0
         MaxPool2d-8           [-1, 32, 16, 44]               0
            Conv2d-9           [-1, 64, 18, 44]           6,208
      BatchNorm2d-10           [-1, 64, 18, 44]             128
              ELU-11           [-1, 64, 18, 44]               0
        MaxPool2d-12            [-1, 64, 9, 22]               0
           Conv2d-13           [-1, 128, 9, 24]          24,704
      BatchNorm2d-14           [-1, 128, 9, 24]             256
              ELU-15           [-1, 128, 9, 24]               0
        MaxPool2d-16           [-1, 128, 4, 12]               0
           Conv2d-17           [-1, 256, 6, 12]          98,560
      BatchNorm2d-18           [-1, 256, 6, 12]             512
              ELU-19           [-1, 256, 6, 12]               0
        MaxPool2d-20            [-1, 256, 3, 6]               0
          Flatten-21                 [-1, 4608]               0
           Linear-22                  [-1, 192]         884,928
              ELU-23                  [-1, 192]               0
      BatchNorm1d-24                  [-1, 192]             384
           Linear-25                   [-1, 24]           4,632
           Linear-26                  [-1, 192]         884,928
              ELU-27                  [-1, 192]               0
      BatchNorm1d-28                  [-1, 192]             384
           Linear-29                   [-1, 24]           4,632
           Linear-30                  [-1, 192]         884,928
              ELU-31                  [-1, 192]               0
      BatchNorm1d-32                  [-1, 192]             384
           Linear-33                   [-1, 24]           4,632
           Linear-34                  [-1, 192]         884,928
              ELU-35                  [-1, 192]               0
      BatchNorm1d-36                  [-1, 192]             384
           Linear-37                   [-1, 24]           4,632
================================================================
Total params: 3,691,872
Trainable params: 3,691,872
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.04
Forward/backward pass size (MB): 9.27
Params size (MB): 14.08
Estimated Total Size (MB): 23.40
----------------------------------------------------------------
    """
