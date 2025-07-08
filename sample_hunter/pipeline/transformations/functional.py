"""
Utility functions used for transforming the audio into a trainable form
"""

from typing import Tuple, Generator
from torch import Tensor
import torch

from sample_hunter._util import STEP_NUM_SAMPLES, WINDOW_NUM_SAMPLES


def num_windows(
    length: int,
    window_size: int = WINDOW_NUM_SAMPLES,
    step_size: int = STEP_NUM_SAMPLES,
) -> int:
    """Count the number of windows that a tensor with length would produce
    for the given window and step size."""
    if length < window_size:
        return 1
    # Number of full windows
    base_windows = 1 + (length - window_size) // step_size
    # Check for leftover samples after the last full window
    remaining_samples = length - ((base_windows - 1) * step_size + window_size)
    if remaining_samples > 0:
        return base_windows + 1
    else:
        return base_windows


def resize(signal: Tensor, desired_length: int) -> Tensor:
    """Set the length of the signal to the desired length"""
    if signal.shape[-1] < desired_length:
        num_missing_samples = desired_length - signal.shape[-1]
        return torch.nn.functional.pad(signal, (0, num_missing_samples))
    if signal.shape[-1] > desired_length:
        return signal[..., :desired_length]
    return signal


def slice_tensor(
    signal: Tensor, slice_length: int, dim: int = 0
) -> Generator[Tuple[Tuple[slice, ...], Tensor], None, None]:
    """Slice a signal into sub_batches of size `slice_length` along the given dimension."""
    dim = dim if dim >= 0 else signal.ndim + dim
    start = 0
    while start < signal.shape[dim]:
        end = min(start + slice_length, signal.shape[dim])
        slices = [slice(None)] * signal.ndim
        slices[dim] = slice(start, end)
        slices = tuple(slices)
        yield slices, signal[slices]
        start = end
