from typing import Dict, List, Tuple, Generator

from torch import Tensor
import torch


def collate_spectrograms(
    batch: List[Dict[str, Tensor]], col: str | List[str]
) -> Tensor | Tuple[Tensor, ...]:
    """
    Collate a batch of mappings of transformed tensors before passing to the dataloader.

    This function expects tensors with shape (batch_size, num_windows, num_channels, n_mels, time_frames)
    and returns a tensor with shape (new_batch_size, num_channels, n_mels, time_frames)
    """

    if isinstance(col, str):
        return torch.cat([example[col] for example in batch], dim=0)
    else:
        to_return = [
            torch.cat([example[name] for example in batch], dim=0) for name in col
        ]
        return tuple(to_return)


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
