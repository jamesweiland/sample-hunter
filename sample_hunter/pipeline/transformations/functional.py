"""
Utility functions used for transforming the audio into a trainable form
"""

from typing import Tuple, Generator, List
import warnings
import torch
import torch
import torchaudio

from sample_hunter._util import DEVICE


def rms_normalize(signal: torch.Tensor, target_rms: float, eps: float = 1e-8):
    current_rms = torch.sqrt(torch.mean(signal.pow(2), dim=-1, keepdim=True))

    # Avoid division by zero for silent audio
    scale_factor = target_rms / (current_rms + eps)

    return signal * scale_factor


def num_windows(
    length: int,
    window_size: int,
    step_size: int,
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


def resize(signal: torch.Tensor, desired_length: int) -> torch.Tensor:
    """Set the length of the signal to the desired length"""
    if signal.shape[-1] < desired_length:
        num_missing_samples = desired_length - signal.shape[-1]
        return torch.nn.functional.pad(signal, (0, num_missing_samples))
    elif signal.shape[-1] > desired_length:
        return signal[..., :desired_length]
    return signal


def slice_tensor(
    signal: torch.Tensor, slice_length: int, dim: int = 0
) -> Generator[Tuple[Tuple[slice, ...], torch.Tensor], None, None]:
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


def mix_channels(signal: torch.Tensor) -> torch.Tensor:
    """Mix the channels of the signal down to mono"""
    if signal.ndim == 2:
        channel_dim = 0
    elif signal.ndim == 3:
        channel_dim = 1
    else:
        raise ValueError(f"Unsupported tensor shape: {signal.shape}")
    if signal.shape[channel_dim] != 1:
        return torch.mean(signal, dim=0, keepdim=True)
    return signal


def resample(
    signal: torch.Tensor, sample_rate: int, target_sample_rate: int
) -> torch.Tensor:
    if sample_rate != target_sample_rate:
        torch.set_num_threads(1)
        signal = torchaudio.functional.resample(signal, sample_rate, target_sample_rate)
        torch.set_num_threads(torch.multiprocessing.cpu_count())
    return signal


def dbfs(signal: torch.Tensor) -> torch.Tensor:
    """
    Convert an audio signal to dbfs scale, for determining the absolute power of the signal

    signal: an audio tensor that is normalized to (-1, 1) with shape (B?, 1, S) where
    B is optional batch size and S is num samples
    """
    if signal.ndim == 2:
        # reshape it to have a batch size of 1
        signal = signal.unsqueeze(0)

    ref = 1.0  # full scale for normalized audio
    eps = 1e-10  # to avoid log(0)
    window_rms = signal.square().mean(dim=(1, 2)).sqrt() + eps
    dbfs = 20 * torch.log10(window_rms / ref)
    return dbfs


def remove_low_volume(
    signal: torch.Tensor, step_num_samples: int, vol_threshold: int
) -> torch.Tensor:
    """
    Like `remove_low_volume_windows`, but takes a full signal instead of a batch of windows

    signal: a tensor of shape (1, S) representing a full audio song

    step_num_samples: the step to use for creating the windows that will be removed and then concatenated back together

    vol_threshold: the threshold to remove all windows with a dbfs less than it
    """
    windows = signal.unfold(
        1, min(step_num_samples, signal.shape[1]), step_num_samples
    )  # (1, W, step)

    windows = windows.squeeze(0).unsqueeze(1)  # (W, 1, step)

    windows = remove_low_volume_windows(windows, vol_threshold)

    # cat windows back together
    windows = windows.flatten()  # (S_new,)

    windows = windows.unsqueeze(0)  # (1, S_new)

    return windows


def remove_low_volume_windows(signal: torch.Tensor, vol_threshold: int) -> torch.Tensor:
    """
    signal is a batch of windows with size (W, 1, S), W is number of windows that represents the song

    note: if there are no windows above the threshold, this function will return the
    passed signal unmodified
    """
    with torch.no_grad():
        signal_dbfs = dbfs(signal)
        signals_below_threshold = signal_dbfs < vol_threshold
        if signals_below_threshold.all():
            warnings.warn(
                "No windows were found to be above the threshold. "
                "Returning the original signal unmodified."
            )
            return signal

        return signal[~signals_below_threshold]


def offset(signal: torch.Tensor, offset_num_samples: int) -> torch.Tensor:
    """
    offset a signal of shape (1, S) by offset_num_samples amount

    if offset_num_samples is negative, signal will be left shifted, otherwise it'll be right shifted
    """

    if offset_num_samples < 0:
        offset_signal = signal[:, abs(offset_num_samples) :]
    elif offset_num_samples > 0:
        offset_signal = torch.nn.functional.pad(signal, (offset_num_samples, 0))
    else:
        offset_signal = signal

    return offset_signal


def offset_with_span(
    signal: torch.Tensor, span_num_samples: int, step_num_samples: int
) -> List[torch.Tensor]:
    """
    Offset signal, which is a tensor with shape (1, S), where S is number of samples,
    from -span_num_samples to span_num_samples, stepped through by step_num_samples
    """

    results = []
    for offset_num_samples in range(
        -span_num_samples, span_num_samples + step_num_samples, step_num_samples
    ):
        offset_signal = offset(signal, offset_num_samples)

        results.append(offset_signal)
    return results


def create_windows(
    signal: torch.Tensor,
    window_num_samples: int,
    step_num_samples: int,
    target_length: int | None = None,
    device: str = DEVICE,
) -> torch.Tensor:
    if target_length is None:
        # Default behavior

        # handle signals shorter than window size
        if signal.size(1) < window_num_samples:
            num_missing_samples = window_num_samples - signal.size(1)
            signal = torch.nn.functional.pad(signal, (0, num_missing_samples))
            return signal.unsqueeze(0)

        windows = signal.unfold(
            dimension=1, size=window_num_samples, step=step_num_samples
        )

        # calculate remaining samples after last full window
        signal_length = signal.size(1)
        n_windows = windows.size(1)
        remaining_samples = signal_length - (
            (n_windows - 1) * step_num_samples + window_num_samples
        )

        # pad the last window if necessary
        if remaining_samples > 0:
            last_segment = signal[
                :,
                (n_windows - 1) * step_num_samples + window_num_samples :,
            ]
            last_segment = resize(last_segment, window_num_samples).unsqueeze(1)
            windows = torch.cat([windows, last_segment], dim=1)

        return windows.transpose(0, 1).to(device)

    # if n_windows is specified, do per-window padding to reach the target length

    proportion = signal.shape[1] / target_length
    n_windows = num_windows(target_length, window_num_samples, step_num_samples)
    windows = []
    start = 0
    for _ in range(n_windows - 1):
        # use floating point representation to calculate end
        end = start + window_num_samples * proportion
        window = signal[:, int(start) : int(end)]
        if window.shape[1] != window_num_samples:
            window = resize(window, window_num_samples)
        windows.append(window)
        start += step_num_samples * proportion

    # do the last window separately, because it has to get
    # all the remaining samples no matter what
    window = signal[:, int(start) :]
    if window.shape[1] != window_num_samples:
        window = resize(window, window_num_samples)
    windows.append(window)

    return torch.stack(windows, dim=0)
