"""
Custom generator functions to transform/pre-process data that is sitting
in the huggingface dataset.
"""

import math
import torch
import torchaudio
from typing import Dict, Tuple, Union

from sample_hunter.pipeline.data_loading import load_tensor_from_bytes


from .obfuscator import Obfuscator
from .functional import (
    create_windows,
    mix_channels,
    remove_low_volume_windows,
    resize,
    num_windows,
)
from sample_hunter._util import (
    config,
    DEVICE,
    MEL_SPECTROGRAM,
    STEP_NUM_SAMPLES,
    WINDOW_NUM_SAMPLES,
)


class SpectrogramPreprocessor:
    """
    A callable object to transform full songs into trainable spectrograms.
    To be used as the callable when overloading `.map` for an IterableDataset.
    """

    def __init__(
        self,
        mel_spectrogram: torchaudio.transforms.MelSpectrogram = MEL_SPECTROGRAM,
        target_sample_rate: int = config.preprocess.sample_rate,
        volume_threshold: int = config.preprocess.volume_threshold,
        take_rate: float = config.preprocess.take_rate,
        window_num_samples: int = WINDOW_NUM_SAMPLES,
        step_num_samples: int = STEP_NUM_SAMPLES,
        obfuscator: Obfuscator = Obfuscator(),
        device: str = DEVICE,
    ):
        """
        Store the necessary settings for transforming the audio.

        Args:
            mel_spectrogram: the mel spectrogram to apply
            target_sample_rate: the sample rate to resample the audio to if necessary
            window_sample_size: the number of samples in each spectrogram window
            obfuscate: if true, return a tuple of two spectrograms when called: the first being the unobfuscated
            spectrogram, the second being the obfuscated one
            device: the device to use for torch
        """
        self.mel_spectrogram = mel_spectrogram
        self.target_sample_rate = target_sample_rate
        self.window_num_samples = window_num_samples
        self.step_num_samples = step_num_samples
        self.device = device
        self.obfuscator = obfuscator
        self.vol_threshold = volume_threshold
        self.take_rate = take_rate

    def __enter__(self):
        """Set up the resources for the downstream classes"""
        self.obfuscator.__enter__()
        return self

    def __exit__(self, *exc):
        """Clean up the downstream resources"""
        self.obfuscator.__exit__(*exc)

    def __call__(
        self,
        data: Union[Dict, torch.Tensor, bytes],
        train: bool = False,
        target_length: int | None = None,
        sample_rate: int | None = None,
    ):
        """
        Transform `data` into a mel spectrogram.

        If `obfuscate == True`, a tuple of tensors will be returned, the
        first being the obfuscated data, the second being the anchor.

        `sample_rate` must be provided if a `torch.Tensor` is passed.
        """
        with torch.no_grad():
            if isinstance(data, Dict):
                # it is a huggingface Audio object
                signal = torch.tensor(
                    data["array"], dtype=torch.float32, device=self.device
                ).to(self.device)
                if signal.ndim == 1:
                    signal = signal.unsqueeze(0)

            elif isinstance(data, bytes):
                # try to read the bytes as an mp3 file
                signal, sample_rate = load_tensor_from_bytes(data)
            elif isinstance(data, torch.Tensor):
                # if a tensor is passed, a sample_rate has to be passed too
                assert sample_rate is not None
                signal = data.to(self.device)
            else:
                raise RuntimeError("Unsupported data type")

            return self.transform(
                signal=signal,
                sample_rate=sample_rate,  # type: ignore
                train=train,
                target_length=target_length,
            )

    def transform(
        self,
        signal: torch.Tensor,
        sample_rate: int,
        train: bool = False,
        target_length: int | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """
        Perform the necessary pre-processing
        operations from a huggingface Audio object
        on the audio and transform it to several spectrograms.
        Returns the example with a new field `transformed` that contains the transformed tensor.

        Expects a tensor with shape (N, S), N=num_channels, S=num_samples
        The tensor that is returned has shape (num_windows, num_channels, n_mels, time_frames)
        """

        # mix to mono
        signal = mix_channels(signal)

        # resample to target sampling rate
        signal = self.resample(signal, sample_rate)

        if train:
            # make windows without overlay for training
            signal = create_windows(
                signal,
                target_length=target_length,
                window_num_samples=self.window_num_samples,
                step_num_samples=self.window_num_samples,
            )

            signal = remove_low_volume_windows(signal, vol_threshold=self.vol_threshold)

            if self.take_rate < 1.0:
                # take only a fraction of windows from the song
                k = math.ceil(signal.shape[0] * self.take_rate)
                idx = torch.randperm(signal.shape[0])[:k]
                signal = signal[idx]

            anchor = self.mel_spectrogram(signal)

            positive = self.mel_spectrogram(self.obfuscate_window(signal))
            return positive, anchor
        else:
            # make windows with overlay
            signal = create_windows(
                signal,
                target_length=target_length,
                window_num_samples=self.window_num_samples,
                step_num_samples=self.step_num_samples,
            )

            signal = remove_low_volume_windows(signal, vol_threshold=self.vol_threshold)

            anchor = self.mel_spectrogram(signal)

            # if not train, just return the regular transformation
            return anchor

    def obfuscate_window(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Obfuscate a segment of an audio file represented as a tensor, and return
        the obfuscation as a tensor.

        Args:
            window: a tensor of shape (num_channels, num_samples)
            sample_rate: the sample rate of the audio
        Returns:
            A Tensor, where the tensor is the obfuscated signal.
            The returned tensor has the same shape as the input tensor. If
            the obfuscation time-distorted the tensor, it will be truncated
            or padded to match the exact shape of the input
        """
        ob_sig = self.obfuscator(signal)
        return ob_sig

    def resample(self, signal: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if sample_rate != self.target_sample_rate:
            torch.set_num_threads(1)
            signal = torchaudio.functional.resample(
                signal, sample_rate, self.target_sample_rate
            )
            torch.set_num_threads(torch.multiprocessing.cpu_count())
        return signal
