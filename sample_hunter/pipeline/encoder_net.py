import torch
import torch.nn as nn
from typing import List, Tuple

from sample_hunter._util import config, INPUT_SHAPE


class EncoderNet(nn.Module):
    def __init__(
        self,
        input_shape: torch.Size = INPUT_SHAPE,
        conv_layer_dims: List[Tuple[int, int]] = config.network.conv_layer_dims,
        minimum_dims: Tuple[int, int] = config.network.min_dims,
        stride: int = config.network.stride,
        padding: int = config.network.padding,
        pool_kernel_size: int = config.network.pool_kernel_size,
        num_branches: int = config.network.num_branches,
        divide_and_encode_hidden_dim: int = config.network.divide_and_encode_hidden_dim,
        embedding_dim: int = config.network.embedding_dim,
    ):
        super().__init__()

        # set up the conv blocks
        self.conv_blocks = nn.ModuleList()
        current_shape = input_shape
        for i, dims in enumerate(conv_layer_dims):
            in_ch, out_ch = dims
            kernel_size = (1, 3) if i % 2 == 0 else (3, 1)

            h_conv = self.conv_output_dim(
                current_shape[1], kernel_size[0], stride, padding
            )
            w_conv = self.conv_output_dim(
                current_shape[2], kernel_size[1], stride, padding
            )
            h_pool = self.pool_output_dim(h_conv, pool_kernel_size, pool_kernel_size)
            w_pool = self.pool_output_dim(w_conv, pool_kernel_size, pool_kernel_size)

            layers = [
                nn.Conv2d(
                    in_ch,
                    out_ch,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                ),
                nn.BatchNorm2d(out_ch),
                nn.ELU(),
            ]

            if h_pool > minimum_dims[0] and w_pool > minimum_dims[1]:
                layers.append(nn.MaxPool2d(kernel_size=pool_kernel_size))
                current_shape = (out_ch, h_pool, w_pool)
            else:
                current_shape = (out_ch, h_conv, w_conv)

            block = nn.Sequential(*layers)
            self.conv_blocks.append(block)

        # flatten the tensor before passing it to the divide-and-encode block
        self.flatten = nn.Flatten()
        _, h, w = current_shape
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

    def forward(self, x: torch.Tensor):
        for block in self.conv_blocks:
            x = block(x)

        x = self.flatten(x)

        splits = [fc(x) for fc in self.divide_and_encode]
        out = torch.cat(splits, dim=1)

        return out

    def conv_output_dim(self, dim, kernel_size, stride, padding):
        return (dim + 2 * padding - (kernel_size - 1) - 1) // stride + 1

    def pool_output_dim(self, dim, kernel_size, stride):
        return (dim - kernel_size) // stride + 1


if __name__ == "__main__":
    from sample_hunter._util import WINDOW_NUM_SAMPLES, DEVICE, INPUT_SHAPE
    from torchsummary import summary

    model = EncoderNet(
        input_shape=INPUT_SHAPE,
    ).to(DEVICE)

    summary(model=model, input_size=INPUT_SHAPE, device=DEVICE)

    """
    
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 66, 87]              64
       BatchNorm2d-2           [-1, 16, 66, 87]              32
               ELU-3           [-1, 16, 66, 87]               0
         MaxPool2d-4           [-1, 16, 33, 43]               0
            Conv2d-5           [-1, 32, 33, 45]           1,568
       BatchNorm2d-6           [-1, 32, 33, 45]              64
               ELU-7           [-1, 32, 33, 45]               0
         MaxPool2d-8           [-1, 32, 16, 22]               0
            Conv2d-9           [-1, 64, 18, 22]           6,208
      BatchNorm2d-10           [-1, 64, 18, 22]             128
              ELU-11           [-1, 64, 18, 22]               0
        MaxPool2d-12            [-1, 64, 9, 11]               0
           Conv2d-13           [-1, 128, 9, 13]          24,704
      BatchNorm2d-14           [-1, 128, 9, 13]             256
              ELU-15           [-1, 128, 9, 13]               0
           Conv2d-16          [-1, 256, 11, 13]          98,560
      BatchNorm2d-17          [-1, 256, 11, 13]             512
              ELU-18          [-1, 256, 11, 13]               0
           Conv2d-19          [-1, 384, 11, 15]         295,296
      BatchNorm2d-20          [-1, 384, 11, 15]             768
              ELU-21          [-1, 384, 11, 15]               0
           Conv2d-22          [-1, 512, 13, 15]         590,336
      BatchNorm2d-23          [-1, 512, 13, 15]           1,024
              ELU-24          [-1, 512, 13, 15]               0
        MaxPool2d-25            [-1, 512, 6, 7]               0
           Conv2d-26            [-1, 768, 6, 9]       1,180,416
      BatchNorm2d-27            [-1, 768, 6, 9]           1,536
              ELU-28            [-1, 768, 6, 9]               0
          Flatten-29                [-1, 41472]               0
           Linear-30                  [-1, 200]       8,294,600
              ELU-31                  [-1, 200]               0
      BatchNorm1d-32                  [-1, 200]             400
           Linear-33                   [-1, 32]           6,432
           Linear-34                  [-1, 200]       8,294,600
              ELU-35                  [-1, 200]               0
      BatchNorm1d-36                  [-1, 200]             400
           Linear-37                   [-1, 32]           6,432
           Linear-38                  [-1, 200]       8,294,600
              ELU-39                  [-1, 200]               0
      BatchNorm1d-40                  [-1, 200]             400
           Linear-41                   [-1, 32]           6,432
           Linear-42                  [-1, 200]       8,294,600
              ELU-43                  [-1, 200]               0
      BatchNorm1d-44                  [-1, 200]             400
           Linear-45                   [-1, 32]           6,432
================================================================
Total params: 35,407,200
Trainable params: 35,407,200
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.02
Forward/backward pass size (MB): 10.44
Params size (MB): 135.07
Estimated Total Size (MB): 145.53
----------------------------------------------------------------

    """
