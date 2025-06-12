"""
Custom generator functions to transform/pre-process data that is sitting
in the huggingface dataset.
"""

import torch
import torchaudio
from torch import Tensor
from typing import Generator
from datasets import Audio

from sample_hunter._util import DEVICE


def hf_audio_to_spectrogram(
    audio: Audio,
    target_sampling_rate: int,
    window_sample_size: int,
    window_step_size: int,
    mel_spectrogram: torchaudio.transforms.MelSpectrogram,
    device: str = DEVICE,
) -> Generator[Tensor, None, None]:
    """
    Perform the necessary pre-processing
    operations from a huggingface Audio object
    on the audio and transform it to several spectrograms
    """
    signal, sr = (
        torch.tensor(audio["array"], dtype=torch.float32, device=device),  # type: ignore
        audio["sampling_rate"],  # type: ignore
    )
    signal = signal.unsqueeze(0)  # the np array are one-dimensional

    # (we don't have to mix anything down because HF has down this for us)

    # resample to target sampling rate
    signal = resample(signal, sr, target_sampling_rate)
    # make windows with overlay
    signal = create_windows(signal, window_sample_size, window_step_size)

    for window in signal:
        yield mel_spectrogram(window)


def prepare_and_obfuscate_spectrograms() -> Generator[Tensor, None, None]:
    """
    Similar to prepare_spectrograms, but also create an obfuscated
    version of the audio snippet, the spectrogram of which is also yielded
    as the second element
    """
    pass


def resample(signal: Tensor, sr: int, target_sr: int, device: str = DEVICE) -> Tensor:
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr).to(device)
        return resampler(signal)
    return signal


def create_windows(signal: Tensor, size: int, step: int) -> Tensor:
    # handle signals shorter than window size
    if signal.size(1) < size:
        num_missing_samples = size - signal.size(1)
        signal = torch.nn.functional.pad(signal, (0, num_missing_samples))
        return signal.unsqueeze(0)

    windows = signal.unfold(dimension=1, size=size, step=step)

    # calculate remaining samples after last full window
    signal_length = signal.size(1)
    num_windows = windows.size(1)
    remaining_samples = signal_length - ((num_windows - 1) * step + size)

    # pad if necessary
    if remaining_samples > 0:
        last_segment = signal[:, (num_windows - 1) * step + size :]
        num_missing_samples = size - remaining_samples
        padded_segment = torch.nn.functional.pad(last_segment, (0, num_missing_samples))
        windows = torch.cat([windows, padded_segment], dim=1)

    return windows.transpose(0, 1)
