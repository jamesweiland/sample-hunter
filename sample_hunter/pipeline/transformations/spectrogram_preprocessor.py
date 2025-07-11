"""
Custom generator functions to transform/pre-process data that is sitting
in the huggingface dataset.
"""

import torch
import torchaudio
from typing import Dict, Tuple, Union

from sample_hunter.pipeline.data_loading import load_tensor_from_bytes


from .obfuscator import Obfuscator
from .functional import resize, num_windows
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
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

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
        target_length: int | None = None,
        obfuscate: bool = False,
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
                obfuscate=obfuscate,
                target_length=target_length,
            )

    def transform(
        self,
        signal: torch.Tensor,
        sample_rate: int,
        obfuscate: bool,
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
        signal = self.mix_channels(signal)

        # resample to target sampling rate
        signal = self.resample(signal, sample_rate)
        # make windows with overlay
        signal = self.create_windows(signal, target_length=target_length)

        anchor = self.mel_spectrogram(signal)
        if obfuscate:
            # obfuscate each window

            positive = self.mel_spectrogram(self.obfuscate_window(signal))
            return positive, anchor
        # if not obfuscate, just return the regular transformation
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

    def mix_channels(self, signal: torch.Tensor) -> torch.Tensor:
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

    def resample(self, signal: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if sample_rate != self.target_sample_rate:
            torch.set_num_threads(1)
            signal = torchaudio.functional.resample(
                signal, sample_rate, self.target_sample_rate
            )
            torch.set_num_threads(torch.multiprocessing.cpu_count())
        return signal

    def remove_low_volume_windows(self, signal: torch.Tensor) -> torch.Tensor:
        signal_db = self.amplitude_to_db(signal)
        mean_db_per_example = signal_db.mean(dim=(1, 2))
        print(f"Max volume in signal: {torch.max(signal_db)}")
        print(f"Min volume in signal: {torch.min(signal_db)}")
        signals_above_threshold = mean_db_per_example > self.vol_threshold
        return signal[signals_above_threshold]

    def create_windows(
        self, signal: torch.Tensor, target_length: int | None = None
    ) -> torch.Tensor:

        if target_length is None:
            # Default behavior

            # handle signals shorter than window size
            if signal.size(1) < self.window_num_samples:
                num_missing_samples = self.window_num_samples - signal.size(1)
                signal = torch.nn.functional.pad(signal, (0, num_missing_samples))
                return signal.unsqueeze(0)

            windows = signal.unfold(
                dimension=1, size=self.window_num_samples, step=self.step_num_samples
            )

            # calculate remaining samples after last full window
            signal_length = signal.size(1)
            n_windows = windows.size(1)
            remaining_samples = signal_length - (
                (n_windows - 1) * self.step_num_samples + self.window_num_samples
            )

            # pad the last window if necessary
            if remaining_samples > 0:
                last_segment = signal[
                    :,
                    (n_windows - 1) * self.step_num_samples + self.window_num_samples :,
                ]
                last_segment = resize(last_segment, self.window_num_samples).unsqueeze(
                    1
                )
                windows = torch.cat([windows, last_segment], dim=1)

            return windows.transpose(0, 1).to(self.device)

        # if n_windows is specified, do per-window padding to reach the target length

        proportion = signal.shape[1] / target_length
        n_windows = num_windows(target_length)
        windows = []
        start = 0
        for _ in range(n_windows - 1):
            # use floating point representation to calculate end
            end = start + self.window_num_samples * proportion
            window = signal[:, int(start) : int(end)]
            if window.shape[1] != self.window_num_samples:
                window = resize(window, self.window_num_samples)
            windows.append(window)
            start += self.step_num_samples * proportion

        # do the last window separately, because it has to get
        # all the remaining samples no matter what
        window = signal[:, int(start) :]
        if window.shape[1] != self.window_num_samples:
            window = resize(window, self.window_num_samples)
        windows.append(window)

        return torch.stack(windows, dim=0)
