import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_channels=1, embedding_dim=96, num_parallel_branches=4):
        super(Encoder, self).__init__()

        # Convolutional layer 1: 1 input, 32 dim output, 3x1 kernel
