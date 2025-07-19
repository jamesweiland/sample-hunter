import torch
import torchaudio
from torchaudio.prototype.datasets import Musan
from pathlib import Path

from .functional import mix_channels, remove_low_volume_windows, create_windows
from sample_hunter._util import WINDOW_NUM_SAMPLES, config

# every song in the dataset is the same sample rate
_MUSAN_ORIGINAL_SAMPLE_RATE = 16_000


class MusanException(BaseException):
    pass


class MyMusan(torch.utils.data.Dataset):
    """A wrapper around torchaudio.prototype.datasets.Musan to do some extra preprocessing before yielding samples"""

    def __init__(self, root: Path, subset: str):
        self.musan = Musan(root, subset=subset)
        self.resample = torchaudio.transforms.Resample(
            _MUSAN_ORIGINAL_SAMPLE_RATE, config.preprocess.sample_rate
        )

    def __len__(self):
        return len(self.musan)

    def __getitem__(self, n):
        signal, sr, name = self.musan[n]

        if signal.ndim != 2:
            raise MusanException(
                f"Loaded MUSAN signal should be 2D (channels, samples), got shape {signal.shape}"
            )

        # regular preprocessing stuff
        signal = mix_channels(signal)
        signal = self.resample(signal)
        signal = create_windows(
            signal, step_num_samples=WINDOW_NUM_SAMPLES
        )  # we don't want these to be overlaying
        signal = remove_low_volume_windows(signal, config.preprocess.volume_threshold)

        return signal, name
