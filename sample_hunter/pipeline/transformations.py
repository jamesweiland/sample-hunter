"""
Custom generator functions to transform/pre-process data that is sitting
in the huggingface dataset.
"""

import torch
import torchaudio
from torch import Tensor
from typing import Dict, Generator, List
from datasets import Audio, load_dataset
from torch.utils.data import DataLoader

from sample_hunter._util import DEVICE, DEFAULT_MEL_SPECTROGRAM


def collate_spectrograms(batch: List[Dict[str, Tensor]]) -> Tensor:
    """
    Collate a batch of mappings of transformed tensors before passing to the dataloader.

    This function expects tensors with shape (batch_size, num_windows, num_channels, n_mels, time_frames)
    and returns a tensor with shape (new_batch_size, num_channels, n_mels, time_frames)
    """
    print("First batch element shape:")
    print(batch[0]["transformed"].shape)
    print("Second batch element shape:")
    print(batch[1]["transformed"].shape)

    return torch.cat([example["transformed"] for example in batch], dim=0)


def hf_audio_to_spectrogram(
    example,
    audio_field_name: str,
    target_sampling_rate: int,
    window_sample_size: int,
    window_step_size: int,
    mel_spectrogram: torchaudio.transforms.MelSpectrogram,
    device: str = DEVICE,
) -> Dict[str, Tensor]:
    """
    Perform the necessary pre-processing
    operations from a huggingface Audio object
    on the audio and transform it to several spectrograms.
    Returns the example with a new field `transformed` that contains the transformed tensor.
    The tensor that is returned has shape (num_windows, num_channels, n_mels, time_frames)
    """
    audio = example[audio_field_name]
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

    return {
        **example,
        "transformed": torch.stack([mel_spectrogram(window) for window in signal]),
    }


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
        padded_segment = torch.nn.functional.pad(
            last_segment, (0, num_missing_samples)
        ).unsqueeze(1)
        windows = torch.cat([windows, padded_segment], dim=1)

    return windows.transpose(0, 1)


if __name__ == "__main__":
    # test hf_audio_to_spectrogram and the collate fn
    ds = load_dataset("samplr/songs", streaming=True, split="train")
    ds = ds.map(
        hf_audio_to_spectrogram,
        fn_kwargs={
            "audio_field_name": "audio",
            "target_sampling_rate": 44_100,
            "window_sample_size": 44_100 * 2,
            "window_step_size": 44_100 * 1,
            "mel_spectrogram": DEFAULT_MEL_SPECTROGRAM,
        },
    )

    dataloader = DataLoader(ds, batch_size=2, collate_fn=collate_spectrograms)

    for batch in dataloader:
        print("Concatenated batch shape:")
        print(batch.size())
        exit()
