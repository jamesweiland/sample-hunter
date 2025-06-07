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
)


class SongPairs(Dataset):
    annotations: pd.DataFrame
    audio_dir: Path
    mel_spectrogram: MelSpectrogram
    target_sample_rate: int

    def __init__(
        self,
        audio_dir: Path,
        annotations_file: Path,
        mel_spectrogram: MelSpectrogram,
        target_sample_rate: int,
    ):
        """Initialize a SongPairs dataset. Each pair consists of an original audio
        file with it's obfuscated version. audio_dir is the
        path to the directory of audio files."""

        self.annotations = read_into_df(annotations_file)
        self.audio_dir = audio_dir
        self.mel_spectrogram = mel_spectrogram
        self.target_sample_rate = target_sample_rate

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
        # the signals are tensors with dimensions: (num_channels, samples)

        # mix signals down to mono if they are more than one channel
        # (set num_channels = 1)
        unobfuscated_signal = self._mix_down(obfuscated_signal)
        obfuscated_signal = self._mix_down(unobfuscated_signal)

        # downsample to target sampling rate
        unobfuscated_signal = self._resample(unobfuscated_sr, unobfuscated_signal)
        obfuscated_signal = self._resample(obfuscated_sr, obfuscated_signal)

        # apply the transformation (spectrogram) to the signals
        unobfuscated_signal = self.mel_spectrogram(unobfuscated_signal)
        obfuscated_signal = self.mel_spectrogram(obfuscated_signal)

        return unobfuscated_signal, obfuscated_signal

    def _get_unobfuscated_path(self, idx):
        return self.annotations.at[idx, "unobfuscate"]

    def _get_obfuscated_path(self, idx):
        return self.annotations.at[idx, "obfuscate"]

    def _mix_down(self, signal: Tensor):
        """If the signal is more than one channel, mix down to mono"""
        if signal.shape[0] == 1:
            return signal

        return torch.mean(signal, dim=0, keepdim=True)

    def _resample(self, original_sampling_rate: int, signal: Tensor):
        """If the signal has a different sampling rate than the target sample rate, downsample it"""
        if original_sampling_rate == self.target_sample_rate:
            return signal

        resampler = torchaudio.transforms.Resample(
            original_sampling_rate, self.target_sample_rate
        )
        return resampler(signal)


if __name__ == "__main__":
    # test SongPairs

    mel_spectrogram = MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )

    dataset = SongPairs(
        audio_dir=Path("/home/james/code/sample-hunter/_data/audio-dir/"),
        annotations_file=Path("/home/james/code/sample-hunter/_data/annotations.csv"),
        mel_spectrogram=mel_spectrogram,
        target_sample_rate=SAMPLE_RATE,
    )

    print(len(dataset))
    x, y = dataset[0]

    plot_spectrogram(x, "unobfuscated")
    plot_spectrogram(y, "obfuscated")
