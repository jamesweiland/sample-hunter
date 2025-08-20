import torch
import torchaudio
from torchaudio.prototype.datasets import Musan
from pathlib import Path

from .functional import (
    mix_channels,
    remove_low_volume,
    remove_low_volume_windows,
    create_windows,
    rms_normalize,
)
from sample_hunter.config import PreprocessConfig

# every song in the dataset is the same sample rate
_MUSAN_ORIGINAL_SAMPLE_RATE = 16_000


class MusanException(BaseException):
    pass


_LOCKS = None
_MANAGER = None


def set_global_locks(locks_dict, manager):
    global _LOCKS
    global _MANAGER
    _LOCKS = locks_dict
    _MANAGER = manager


class MyMusan(torch.utils.data.Dataset):
    """A wrapper around torchaudio.prototype.datasets.Musan to do some extra preprocessing before yielding samples"""

    def __init__(
        self,
        root: Path,
        subset: str,
        sample_rate: int | None = None,
        spec_num_sec: float | None = None,
        step_num_sec: float | None = None,
        volume_threshold: int | None = None,
        target_rms: float | None = None,
    ):
        if (
            not sample_rate
            or not spec_num_sec
            or not volume_threshold
            or not target_rms
        ):
            default_config = PreprocessConfig()
            self.sample_rate = sample_rate or default_config.sample_rate
            self.spec_num_sec = spec_num_sec or default_config.spec_num_sec
            self.spec_num_samples = int(self.spec_num_sec * self.sample_rate)
            self.step_num_sec = step_num_sec or default_config.step_num_sec
            self.step_num_samples = int(self.step_num_sec * self.sample_rate)
            self.volume_threshold = volume_threshold or default_config.volume_threshold
            self.target_rms = target_rms or default_config.target_rms
        else:
            self.sample_rate = sample_rate
            self.spec_num_sec = spec_num_sec
            self.spec_num_samples = int(self.sample_rate * self.spec_num_sec)
            self.volume_threshold = volume_threshold
            self.target_rms = target_rms

        self.musan = Musan(root, subset=subset)
        self.resample = torchaudio.transforms.Resample(
            _MUSAN_ORIGINAL_SAMPLE_RATE, self.sample_rate
        )

    def __len__(self):
        return len(self.musan)

    def __getitem__(self, n):

        if _LOCKS is None or _MANAGER is None:
            # fall back to single process reading
            signal, sr, name = self.musan[n]

        else:
            # ensure a lock exists for this name
            lock = _LOCKS.get(n)
            if lock is None:
                # create a new lock for this idx using the shared global manager
                lock = _MANAGER.Lock()

                # use setdefault in case a different process set this concurrently
                existing_lock = _LOCKS.setdefault(n, lock)
                if existing_lock != lock:
                    lock = existing_lock

            # now, we can safely read the file
            with lock:
                try:
                    signal, sr, name = self.musan[n]
                except Exception as e:
                    raise MusanException(f"Failed to access Musan item {n} data: {e}")

        if signal.ndim != 2:
            raise MusanException(
                f"Loaded MUSAN signal should be 2D (channels, samples), got shape {signal.shape}"
            )

        # regular preprocessing stuff
        signal = mix_channels(signal)
        signal = self.resample(signal)
        signal = create_windows(
            signal,
            window_num_samples=self.spec_num_samples,
            step_num_samples=self.spec_num_samples,
        )  # we don't want these to be overlaying

        signal = remove_low_volume_windows(signal, self.volume_threshold)

        return signal, name
