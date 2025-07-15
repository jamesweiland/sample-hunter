import torch
import torchaudio
from torchaudio.prototype.datasets import Musan
from pathlib import Path

from .functional import mix_channels, remove_low_volume_windows, create_windows
from sample_hunter._util import config

# every song in the dataset is the same sample rate
_MUSAN_ORIGINAL_SAMPLE_RATE = 16_000


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

        # regular preprocessing stuff
        signal = mix_channels(signal)
        signal = self.resample(signal)
        signal = create_windows(signal)
        signal = remove_low_volume_windows(signal, config.preprocess.volume_threshold)

        return signal
