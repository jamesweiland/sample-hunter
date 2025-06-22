"""
Custom generator functions to transform/pre-process data that is sitting
in the huggingface dataset.
"""

import argparse
from fractions import Fraction
from functools import cached_property
import math
import random
import sys
import memray
import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram
from torch import Tensor
from typing import Dict, List, Tuple, Union
from pathlib import Path
from datasets import load_dataset, Audio
from torch.utils.data import DataLoader
from pydantic import BaseModel, field_validator

from sample_hunter._util import (
    DEFAULT_HOP_LENGTH,
    DEFAULT_N_FFT,
    DEVICE,
    DEFAULT_MEL_SPECTROGRAM,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_WINDOW_NUM_SAMPLES,
    DEFAULT_STEP_NUM_SAMPLES,
    PROCS,
    CACHE_DIR,
    HF_TOKEN,
    plot_spectrogram,
)
from sample_hunter.pipeline.transformations.functional import (
    collate_spectrograms,
    resize,
    slice_tensor,
)


class BatchedPitchPerturbation(torch.nn.Module):
    def __init__(self, sub_batch_size: int, rates_list: List[float], sample_rate: int):
        super().__init__()
        self.sub_batch_size = sub_batch_size
        n_steps_list = [-math.log2(rate) for rate in rates_list]
        # ensure that the n_steps are rational by converting them to a Fraction
        n_steps_list = [
            Fraction(n_steps).limit_denominator(1000) for n_steps in n_steps_list
        ]

        self.shifters = [
            torchaudio.transforms.PitchShift(
                sample_rate,
                n_steps=n_steps.numerator,
                bins_per_octave=n_steps.denominator,
            )
            for n_steps in n_steps_list
        ]

    def forward(self, batch: Tensor):
        ob = torch.empty_like(batch, device=batch.device, dtype=batch.dtype)
        for idx, sub_batch in slice_tensor(batch, self.sub_batch_size):
            i = random.randint(0, len(self.shifters) - 1)
            for j, shifter in enumerate(self.shifters):
                if i == j:
                    print(f"Picking shifter {i} with n_steps {shifter.n_steps}")
                    ob[idx] = shifter(sub_batch)
                    if getattr(shifter, "kernel", None) is not None:
                        print(
                            f"This shifter's kernel has size {sys.getsizeof(shifter.kernel) * 1e-9} GB"
                        )
        return ob


class BatchedTimeStretchPerturbation(torch.nn.Module):
    def __init__(
        self,
        factors: List[float],
        sub_batch_size: int,
        n_fft: int = DEFAULT_N_FFT,
        hop_length: int = DEFAULT_HOP_LENGTH,
        device: str = DEVICE,
    ):
        super().__init__()
        self.sub_batch_size = sub_batch_size
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.window = torch.hann_window(window_length=self.n_fft, device=device)
        self.stretchers = [
            torchaudio.transforms.TimeStretch(
                hop_length=self.hop_length,
                fixed_rate=factor,
                n_freq=self.hop_length + 1,
            )
            for factor in factors
        ]

    def forward(self, batch: Tensor) -> Tensor:
        ob = torch.empty_like(batch, device=batch.device, dtype=batch.dtype)
        for idx, sub_batch in slice_tensor(batch, self.sub_batch_size):
            print("Do we get here?")
            i = random.randint(0, len(self.stretchers) - 1)
            print(i)
            for j, stretcher in enumerate(self.stretchers):
                if i == j:
                    print("Stretching")
                    spec = torch.stft(
                        sub_batch.squeeze(1),
                        n_fft=self.n_fft,
                        hop_length=self.hop_length,
                        return_complex=True,
                        window=self.window,
                    )
                    spec = stretcher(spec)
                    ob[idx] = resize(
                        torch.istft(spec, n_fft=self.n_fft, hop_length=self.hop_length),
                        batch.shape[-1],
                    ).unsqueeze(1)

        return ob


class Obfuscator(BaseModel):
    """
    A callable object to obfuscate audio. Randomly distort an audio signal in a variety of ways.

    The following distortions are applied:

    - The audio is time-distorted by increasing or decreasing the tempo.
    - The pitch is increased or decreased.
    - Reverb is added.
    - A low or high pass filter is added.
    - White or pink noise is added.

    Distortions are made using the Sox library.

    Args:
        `tempo_range`: The range with which to time-distort the signal.
        `pitch_range`: The range with which to pitch-distort the signal.
        `lowpass_range`: The range with which to add a low-pass filter.
        `highpass_range`: The range with which to add a high-pass filter.
        `whitenoise_range`: The volume range of white noise to add.
        `lowpass_frac`: The proportion of signals to apply a lowpass filter to (otherwise,
        a highpass filter will be applied).
        `sample_rate`: the sample rate of the signal to distort
        `n_fft`: n_fft to use for the window for the FFT
        `hop_length`: the hop length for the FFT
    """

    time_stretch_factors: List[float] = [0.75, 1, 1.25, 1.5]
    pitch_factors: List[float] = [0.5, 0.75, 1, 1.2, 1.5]
    sub_batch_size: int = 25
    lowpass_range: Tuple[int, int] = (2000, 3000)
    highpass_range: Tuple[int, int] = (500, 1500)
    whitenoise_range: Tuple[float, float] = (0.00, 0.1)
    lowpass_frac: float = 0.5
    device: str = DEVICE
    sample_rate: int = DEFAULT_SAMPLE_RATE
    n_fft: int = DEFAULT_N_FFT
    hop_length: int = DEFAULT_HOP_LENGTH

    @cached_property
    def time_stretch_perturbation(self) -> BatchedTimeStretchPerturbation:
        return BatchedTimeStretchPerturbation(
            sub_batch_size=self.sub_batch_size,
            factors=self.time_stretch_factors,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        ).to(self.device)

    @cached_property
    def pitch_perturbation(self) -> BatchedPitchPerturbation:
        return BatchedPitchPerturbation(
            sub_batch_size=self.sub_batch_size,
            sample_rate=self.sample_rate,
            rates_list=self.pitch_factors,
        ).to(self.device)

    @field_validator(
        "lowpass_range",
        "highpass_range",
        "whitenoise_range",
    )
    def check_range(cls, v, field):
        if len(v) != 2:
            raise ValueError(f"{field.name} must be a tuple of length 2")
        if v[0] >= v[1]:
            raise ValueError(
                f"In {field.name}, the first value must be smaller than the second value"
            )
        return v

    @field_validator("lowpass_frac")
    def check_fraction(cls, v, field):
        if not (0 <= v <= 1):
            raise ValueError(f"{field.name} must be between 0 and 1")
        return v

    def __call__(self, batch: Tensor) -> Tensor:
        """
        Accepts a torch.tensor
        and returns the obfuscated tensor.
        """
        with torch.no_grad():
            batch = batch.contiguous() if not batch.is_contiguous() else batch
            batch = self.time_stretch_perturbation(batch)
            batch = self.pitch_perturbation(batch)
            batch = self.apply_filter(batch)
            batch = self.add_noise(batch)
            return batch

    def apply_filter(self, signal: Tensor) -> Tensor:
        """Apply either a high or low pass filter to `signal`."""

        # generate decisions per snippet in the batch
        lowpass_mask = (
            torch.rand(signal.shape[0], device=signal.device) < self.lowpass_frac
        )
        frequencies = torch.zeros(signal.shape[0], dtype=torch.int)

        low_idx = lowpass_mask.nonzero().squeeze(1)
        low_freqs = torch.randint(
            low=self.lowpass_range[0],
            high=self.lowpass_range[1] + 1,
            size=(low_idx.shape[0],),
            device=signal.device,
            dtype=torch.int,
        )
        frequencies[lowpass_mask] = low_freqs
        high_idx = (~lowpass_mask).nonzero().squeeze(1)
        high_freqs = torch.randint(
            low=self.highpass_range[0],
            high=self.highpass_range[1] + 1,
            size=(high_idx.shape[0],),
            device=signal.device,
            dtype=torch.int,
        )
        frequencies[~lowpass_mask] = high_freqs
        signal_low = self._lowpass_biquad(
            signal[lowpass_mask], self.sample_rate, frequencies[lowpass_mask]
        )
        signal_high = self._highpass_biquad(
            signal[~lowpass_mask], self.sample_rate, frequencies[~lowpass_mask]
        )
        ob = torch.empty_like(signal, dtype=signal.dtype, device=signal.device)
        ob[lowpass_mask] = signal_low
        ob[~lowpass_mask] = signal_high
        return ob

    def _lowpass_biquad(
        self, signal: Tensor, sample_rate: int, cutoff_freq: Tensor, Q: float = 0.707
    ) -> Tensor:
        Q = torch.as_tensor(Q, dtype=signal.dtype, device=signal.device)  # type: ignore
        w0 = 2 * math.pi * cutoff_freq / sample_rate
        alpha = torch.sin(w0) / 2 / Q

        b0 = (1 - torch.cos(w0)) / 2
        b1 = 1 - torch.cos(w0)
        b2 = b0
        a0 = 1 + alpha
        a1 = -2 * torch.cos(w0)
        a2 = 1 - alpha
        return self._biquad(signal, b0, b1, b2, a0, a1, a2)

    def _highpass_biquad(
        self, signal: Tensor, sample_rate: int, cutoff_freq: Tensor, Q: float = 0.707
    ) -> Tensor:
        Q = torch.as_tensor(Q, dtype=signal.dtype, device=signal.device)  # type: ignore
        w0 = 2 * math.pi * cutoff_freq / sample_rate
        alpha = torch.sin(w0) / 2.0 / Q

        b0 = (1 + torch.cos(w0)) / 2
        b1 = -1 - torch.cos(w0)
        b2 = b0
        a0 = 1 + alpha
        a1 = -2 * torch.cos(w0)
        a2 = 1 - alpha
        return self._biquad(signal, b0, b1, b2, a0, a1, a2)

    def _biquad(
        self,
        signal: Tensor,
        b0: Tensor,
        b1: Tensor,
        b2: Tensor,
        a0: Tensor,
        a1: Tensor,
        a2: Tensor,
    ) -> Tensor:
        a_coeffs = torch.stack([a0, a1, a2], dim=1)
        b_coeffs = torch.stack([b0, b1, b2], dim=1)

        batch = signal.reshape((1, signal.shape[0], signal.shape[-1]))
        output = torchaudio.functional.filtering.lfilter(
            batch, a_coeffs, b_coeffs, clamp=True, batching=True
        )
        return output.reshape(signal.shape)

    def add_noise(self, signal: Tensor) -> Tensor:
        """Add either white or pink noise to `signal`."""
        print(f"Signal: {signal.nbytes * 1e-9} GB")
        noise = torch.randn_like(signal, device=signal.device, dtype=signal.dtype)
        noise_levels = self.whitenoise_range[0] + (
            self.whitenoise_range[1] - self.whitenoise_range[0]
        ) * torch.rand(signal.shape[0], device=signal.device, dtype=torch.float16)
        return torchaudio.functional.add_noise(
            waveform=signal, noise=noise, snr=noise_levels.unsqueeze(1)
        )


class SpectrogramPreprocessor:
    """
    A callable object to transform full songs into trainable spectrograms.
    To be used as the callable when overloading `.map` for an IterableDataset.
    """

    def __init__(
        self,
        mel_spectrogram: MelSpectrogram = DEFAULT_MEL_SPECTROGRAM,
        target_sample_rate: int = DEFAULT_SAMPLE_RATE,
        window_num_samples: int = DEFAULT_WINDOW_NUM_SAMPLES,
        step_num_samples: int = DEFAULT_STEP_NUM_SAMPLES,
        obfuscator: Obfuscator = Obfuscator(),
        num_workers: int = PROCS,
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

    def __call__(
        self,
        data: Union[Dict, Tensor],
        obfuscate: bool = False,
        sample_rate: int | None = None,
    ):
        """
        Update `example` with fields for preprocessed data.
        """
        with torch.no_grad():

            if isinstance(data, Dict):
                # it is a huggingface Audio object
                signal = torch.tensor(
                    data["array"], dtype=torch.float32, device=self.device
                ).to(self.device)
                if signal.ndim == 1:
                    signal = signal.unsqueeze(0)
                return self.transform(
                    signal=signal,
                    sample_rate=data["sampling_rate"],
                    obfuscate=obfuscate,
                )
            else:
                # if a tensor is passed, a sample_rate has to be passed too
                assert sample_rate is not None
                data = data.to(self.device)
                return self.transform(
                    signal=data, sample_rate=sample_rate, obfuscate=obfuscate
                )

    def transform(
        self, signal: Tensor, sample_rate: int, obfuscate: bool
    ) -> Tuple[Tensor, Tensor] | Tensor:
        """
        Perform the necessary pre-processing
        operations from a huggingface Audio object
        on the audio and transform it to several spectrograms.
        Returns the example with a new field `transformed` that contains the transformed tensor.

        Expects a tensor with shape (N, S), N=num_channels, S=num_samples
        The tensor that is returned has shape (num_windows, num_channels, n_mels, time_frames)
        """
        print(f"Start with: {signal.nbytes * 1e-9} GB")

        # mix to mono
        signal = self.mix_channels(signal)

        # resample to target sampling rate
        signal = self.resample(signal, sample_rate)
        # make windows with overlay
        signal = self.create_windows(signal)

        anchor = self.mel_spectrogram(signal)
        if obfuscate:
            # obfuscate each window

            positive = self.mel_spectrogram(self.obfuscate_window(signal))
            return positive, anchor
        # if not obfuscate, just return the regular transformation
        return anchor

    def obfuscate_window(self, signal: Tensor) -> Tensor:
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

    def mix_channels(self, signal: Tensor) -> Tensor:
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

    def resample(self, signal: Tensor, sample_rate: int) -> Tensor:
        if sample_rate != self.target_sample_rate:
            print("We actually have to resample")
            torch.set_num_threads(1)
            signal = torchaudio.functional.resample(
                signal, sample_rate, self.target_sample_rate
            )
            torch.set_num_threads(torch.multiprocessing.cpu_count())
        return signal

    def create_windows(self, signal: Tensor) -> Tensor:
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
        num_windows = windows.size(1)
        remaining_samples = signal_length - (
            (num_windows - 1) * self.step_num_samples + self.window_num_samples
        )

        # pad the last window if necessary
        if remaining_samples > 0:
            last_segment = signal[
                :, (num_windows - 1) * self.step_num_samples + self.window_num_samples :
            ]
            last_segment = resize(last_segment, self.window_num_samples).unsqueeze(1)
            windows = torch.cat([windows, last_segment], dim=1)

        return windows.transpose(0, 1).to(self.device)


if __name__ == "__main__":
    # test the obfuscations
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", type=str, required=False)
    args = parser.parse_args()
    if args.token is None:
        args.token = HF_TOKEN

    # test hf_audio_to_spectrogram and the collate fn
    preprocessor = SpectrogramPreprocessor()
    # print("Downloading dataset")
    # ds = load_dataset("samplr/songs", streaming=True, split="train")
    # print("Download complete, applying map")
    # ds = ds.map(lambda ex: {**ex, "transform": preprocessor(ex["audio"])})
    # print("map complete, making dataloader")

    # dataloader = DataLoader(
    #     ds, batch_size=2, collate_fn=SpectrogramPreprocessor.collate_spectrograms
    # )
    # print("dataloader done, starting loop")
    # for batch in dataloader:

    #     print("Concatenated batch shape:")
    #     print(batch.size())

    # test the obfuscation stuff
    def map_fn(ex):
        positive, anchor = preprocessor(ex["audio"], obfuscate=True)
        return {**ex, "positive": positive, "anchor": anchor}

    with memray.Tracker("output.bin"):

        obf_ds = load_dataset(
            "samplr/songs",
            streaming=True,
            split="train",
            cache_dir=Path(CACHE_DIR / "songs").__str__(),
            token=args.token,
        ).cast_column("audio", Audio(decode=True))
        obf_ds = obf_ds.map(map_fn)

        dataloader = DataLoader(
            obf_ds,
            batch_size=2,
            collate_fn=collate_spectrograms,
        )
        i = 0
        for anchor, positive in dataloader:
            print("Concatenated batch shape:")
            print(anchor.size())
            print(positive.size())

            for i in range(2):
                plot_spectrogram(anchor[i], f"Anchor {i}")
                plot_spectrogram(positive[i], f"Positive {i}")

            break
            i += 1
            if i == 10:
                break
    print("done")
