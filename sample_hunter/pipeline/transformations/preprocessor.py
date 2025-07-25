"""
Custom generator functions to transform/pre-process data that is sitting
in the huggingface dataset.
"""

import torch
from typing import Dict, Tuple, Union, List


from .obfuscator import Obfuscator
from .functional import (
    create_windows,
    mix_channels,
    offset,
    remove_low_volume_windows,
    resample,
)
from sample_hunter.config import PreprocessConfig
from sample_hunter.pipeline.data_loading import load_tensor_from_bytes
from sample_hunter._util import DEVICE


class Preprocessor:
    """
    A callable object to transform full songs into trainable spectrograms.
    To be used as the callable when overloading `.map` for an IterableDataset.
    """

    def __init__(
        self,
        config: PreprocessConfig | None = None,
        obfuscator: Obfuscator | None = None,
        device: str = DEVICE,
        **kwargs,
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
        self.config = config or PreprocessConfig()
        self.config = self.config.merge_kwargs(**kwargs)
        self.device = device
        self.obfuscator = obfuscator

    def __enter__(self):
        """Set up the resources for the downstream classes"""
        if self.obfuscator is not None:
            self.obfuscator.__enter__()
        self._entered_context = True
        return self

    def __exit__(self, *exc):
        """Clean up the downstream resources"""
        if self.obfuscator is not None:
            self.obfuscator.__exit__(*exc)

    def __call__(
        self,
        data: Union[Dict, torch.Tensor, bytes],
        train: bool = False,
        target_length: int | None = None,
        sample_rate: int | None = None,
        **kwargs,
    ):
        """
        Transform `data` into a mel spectrogram.

        If `obfuscate == True`, a tuple of tensors will be returned, the
        first being the obfuscated data, the second being the anchor.

        `sample_rate` must be provided if a `torch.Tensor` is passed.

        **kwargs can be used to change the parameters at call-time
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
                **kwargs,
            )

    def transform(
        self,
        signal: torch.Tensor,
        sample_rate: int,
        train: bool = False,
        target_length: int | None = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor] | List[torch.Tensor]:
        """
        Perform the necessary pre-processing
        operations from a huggingface Audio object
        on the audio and transform it to several spectrograms.
        Returns the example with a new field `transformed` that contains the transformed tensor.

        Expects a tensor with shape (N, S), N=num_channels, S=num_samples
        The tensor that is returned has shape (num_windows, num_channels, n_mels, time_frames)

        **kwargs can be used to change the parameters at call-time
        """
        config = self.config.merge_kwargs(**kwargs)
        config.mel_spectrogram = config.mel_spectrogram.to(DEVICE)

        # mix to mono
        signal = mix_channels(signal)

        # resample to target sampling rate
        signal = resample(signal, sample_rate, config.sample_rate)

        if train:
            # make windows without overlay for training
            signal = create_windows(
                signal,
                target_length=target_length,
                window_num_samples=config.spec_num_samples,
                step_num_samples=config.spec_num_samples,
            )

            signal = remove_low_volume_windows(
                signal, vol_threshold=config.volume_threshold
            )

            anchor = config.mel_spectrogram(signal)
            positive = config.mel_spectrogram(self.obfuscate_window(signal))

            return positive, anchor
        else:
            # make a list of audio signals that are offset from
            # -offset_span to offset_span, stepped through by offset_step
            offsets = offset(
                signal, config.offset_span_num_samples, config.offset_step_num_samples
            )

            # make windows with overlay
            offsets = [
                create_windows(
                    offset,
                    target_length=target_length,
                    window_num_samples=config.spec_num_samples,
                    step_num_samples=config.step_num_samples,
                )
                for offset in offsets
            ]

            offsets = [
                remove_low_volume_windows(offset, vol_threshold=config.volume_threshold)
                for offset in offsets
            ]

            anchors = [config.mel_spectrogram(offset) for offset in offsets]

            # if not train, just return the regular transformation
            return anchors

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
        # we can lazily intialize obfuscator if it wasn't passed during construction
        if self.obfuscator is None:
            self.obfuscator = Obfuscator()
            if self._entered_context:
                self.obfuscator.__enter__()
        ob_sig = self.obfuscator(signal)
        return ob_sig
