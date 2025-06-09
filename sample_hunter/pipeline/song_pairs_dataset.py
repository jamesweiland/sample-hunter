from typing import Tuple
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchaudio.transforms import MelSpectrogram
from pathlib import Path
import pandas as pd
import torchaudio

from sample_hunter._util import (
    read_into_df,
    SAMPLE_RATE,
    N_FFT,
    HOP_LENGTH,
    N_MELS,
    plot_spectrogram,
    WINDOW_SIZE,
    DEVICE,
)


class SongPairsDataset(Dataset):
    annotations: pd.DataFrame
    audio_dir: Path
    mel_spectrogram: MelSpectrogram
    target_sample_rate: int
    window_size: int
    step_size: int
    device: str

    def __init__(
        self,
        audio_dir: Path,
        annotations_file: Path,
        mel_spectrogram: MelSpectrogram,
        target_sample_rate: int,
        num_samples: int,
        device: str,
    ):
        """Initialize a SongPairs dataset. Each pair consists of an original audio
        file with it's obfuscated version. audio_dir is the
        path to the directory of audio files."""

        self.annotations = read_into_df(annotations_file)
        self.audio_dir = audio_dir
        self.mel_spectrogram = mel_spectrogram.to(device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        self.device = device

        if "song_id" not in self.annotations.columns:
            self.annotations["song_id"] = self.annotations["unob_full"].apply(
                lambda p: hash(Path(p).stem)
            )

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        # load both unobfuscated and obfuscated waveform
        unobfuscated_path = self._get_unobfuscated_path(idx)
        obfuscated_path = self._get_obfuscated_path(idx)

        unobfuscated_signal, unobfuscated_sr = torchaudio.load(
            unobfuscated_path, format="mp3", backend="ffmpeg"
        )
        obfuscated_signal, obfuscated_sr = torchaudio.load(
            obfuscated_path, format="mp3", backend="ffmpeg"
        )
        unobfuscated_signal = unobfuscated_signal.to(self.device)
        obfuscated_signal = obfuscated_signal.to(self.device)
        # the signals are tensors with dimensions: (num_channels, samples)

        # mix signals down to mono if they are more than one channel
        # (set num_channels = 1)
        unobfuscated_signal = self._mix_down(obfuscated_signal)
        obfuscated_signal = self._mix_down(unobfuscated_signal)

        # downsample to target sampling rate
        unobfuscated_signal = self._resample(unobfuscated_sr, unobfuscated_signal)
        obfuscated_signal = self._resample(obfuscated_sr, obfuscated_signal)

        # truncate/pad signals if necessary
        unobfuscated_signal = self._pad_or_truncate(unobfuscated_signal)
        obfuscated_signal = self._pad_or_truncate(obfuscated_signal)

        # apply the transformation (spectrogram) to each window
        unob_spec = self.mel_spectrogram(unobfuscated_signal)  # anchor
        ob_spec = self.mel_spectrogram(obfuscated_signal)  # positive

        # finally, get the song id associated with this snippet
        song_id = self.annotations.at[idx, "song_id"]

        return unob_spec, ob_spec, song_id

    def shape(self) -> torch.Size:
        """Returns the shape of the tensors in the dataset"""
        time_frames = int(
            (
                self.num_samples
                + self.mel_spectrogram.n_fft
                - self.mel_spectrogram.hop_length
            )
            / self.mel_spectrogram.hop_length
        )
        return torch.Size((1, self.mel_spectrogram.n_mels, time_frames))

    def _get_unobfuscated_path(self, idx):
        return self.annotations.at[idx, "unob_seg"]

    def _get_obfuscated_path(self, idx):
        return self.annotations.at[idx, "ob_seg"]

    def _mix_down(self, signal: Tensor) -> Tensor:
        """If the signal is more than one channel, mix down to mono"""
        if signal.shape[0] == 1:
            return signal

        return torch.mean(signal, dim=0, keepdim=True)

    def _resample(self, original_sampling_rate: int, signal: Tensor) -> Tensor:
        """If the signal has a different sampling rate than the target sample rate, downsample it"""
        if original_sampling_rate == self.target_sample_rate:
            return signal

        resampler = torchaudio.transforms.Resample(
            original_sampling_rate, self.target_sample_rate
        ).to(self.device)
        return resampler(signal)

    def _pad_or_truncate(self, signal: Tensor) -> Tensor:
        if signal.shape[1] > self.num_samples:
            # truncate
            return signal[:, : self.num_samples]
        if signal.shape[1] < self.num_samples:
            # pad
            num_missing_samples = self.num_samples - signal.shape[1]
            return torch.nn.functional.pad(signal, (0, num_missing_samples))
        return signal


if __name__ == "__main__":
    # test SongPairs

    mel_spectrogram = MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )

    dataset = SongPairsDataset(
        audio_dir=Path("/home/james/code/sample-hunter/_data/audio-dir/"),
        annotations_file=Path(
            "/home/james/code/sample-hunter/_data/new_annotations.csv"
        ),
        mel_spectrogram=mel_spectrogram,
        target_sample_rate=SAMPLE_RATE,
        num_samples=WINDOW_SIZE,
        device=DEVICE,
    )

    print(dataset.shape())

    print(len(dataset))
    X, Y, id = dataset[100]

    print(id)
    print(type(id))

    print(X.shape, Y.shape)
    plot_spectrogram(X, "unobfuscated")
    plot_spectrogram(Y, "obfuscated")
