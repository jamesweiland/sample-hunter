from typing import Dict, List, Tuple, Generator

from torch import Tensor
import torch

from sample_hunter._util import STEP_NUM_SAMPLES, WINDOW_NUM_SAMPLES
from sample_hunter.cfg import config


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


def flatten_sub_batches(
    dataloader: torch.utils.data.DataLoader,
) -> Generator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], None, None]:
    """
    A generator to wrap around a torch dataloader with a collate function that
    returns a list of tensors. This yields the tensors in the list, one at a time.
    This expects the dataloader to yield a list of tuples
    """
    dataloader_iter = iter(dataloader)
    while True:
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            break  # End of dataloader
        except Exception as e:
            print("An error occurred while fetching a batch from the dataloader")
            print(str(e))
            continue

        for sub_batch in batch:
            yield sub_batch


def collate_spectrograms(
    batch: List[Dict[str, Tensor]],
    col: str | List[str],
    sub_batch_size: int = config.network.sub_batch_size,
) -> Tuple[torch.Tensor] | List[Tuple[torch.Tensor, ...]]:
    """
    Collate a batch of mappings of transformed tensors before passing to the dataloader.

    This function expects tensors with shape (batch_size, num_windows, num_channels, n_mels, time_frames)
    and returns a tensor with shape (new_batch_size, num_channels, n_mels, time_frames)
    """

    if isinstance(col, str):
        full_tensor = torch.cat([example[col] for example in batch], dim=0)
        perm = torch.randperm(full_tensor.shape[0])
        shuffled = full_tensor[perm]

        sub_batches = shuffled.split(sub_batch_size)
    else:
        full_tensors = [
            torch.cat([example[name] for example in batch], dim=0) for name in col
        ]
        perm = torch.randperm(full_tensors[0].shape[0])
        shuffled = [t[perm] for t in full_tensors]

        sub_batches = (t.split(sub_batch_size) for t in shuffled)

    return list(zip(*sub_batches))


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
