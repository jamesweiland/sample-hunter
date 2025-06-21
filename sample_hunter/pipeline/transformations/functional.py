from typing import Dict, List, Tuple, Generator

from torch import Tensor
import torch


def collate_spectrograms(
    batch: List[Dict[str, Tensor]],
) -> Tensor | Tuple[Tensor, Tensor]:
    """
    Collate a batch of mappings of transformed tensors before passing to the dataloader.

    This function expects tensors with shape (batch_size, num_windows, num_channels, n_mels, time_frames)
    and returns a tensor with shape (new_batch_size, num_channels, n_mels, time_frames)
    """
    if batch[0].get("transform") is not None:
        return torch.cat([example["transform"] for example in batch], dim=0)
    return torch.cat([example["anchor"] for example in batch], dim=0), torch.cat(
        [example["positive"] for example in batch], dim=0
    )


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
