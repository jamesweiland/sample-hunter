from functools import cached_property
import math
import torch
from pydantic import BaseModel, field_validator
from typing import List, Sequence, Tuple

import torchaudio

from .batched_pitch_perturbation import BatchedPitchPerturbation
from .batched_time_stretch_perturbation import BatchedTimeStretchPerturbation
from sample_hunter._util import config, DEVICE


class Obfuscator(BaseModel):
    """
    A callable object to obfuscate audio. Randomly distort an audio signal in a variety of ways.

    The following distortions are applied:

    - The audio is time-distorted by increasing or decreasing the tempo.
    - The pitch is increased or decreased.
    - A low or high pass filter is added.
    - White or pink noise is added.

    Distortions are made using the Sox library.

    Args:
        `time_stretch_factors`: The list of factors used to time-distort the signal via phase vocoding.
        `pitch_factors`: The range with which to pitch-distort the signal (via phase vocoding, then resampling).
        `lowpass_range`: The range with which to add a low-pass filter.
        `highpass_range`: The range with which to add a high-pass filter.
        `whitenoise_range`: The volume range of white noise to add.
        `lowpass_frac`: The proportion of signals to apply a lowpass filter to (otherwise,
        a highpass filter will be applied).
        `sample_rate`: the sample rate of the signal to distort
        `n_fft`: n_fft to use for the window for the FFT
        `hop_length`: the hop length for the FFT
        `device`: either 'cuda' or 'cpu'
    """

    time_stretch_factors: Sequence[float] = (
        config.preprocess.obfuscator.time_stretch_factors
    )
    pitch_factors: Sequence[float] = config.preprocess.obfuscator.pitch_factors
    lowpass_range: Tuple[int, int] = config.preprocess.obfuscator.lowpass_range
    highpass_range: Tuple[int, int] = config.preprocess.obfuscator.highpass_range
    whitenoise_range: Tuple[float, float] = (
        config.preprocess.obfuscator.whitenoise_range
    )
    lowpass_frac: float = config.preprocess.obfuscator.lowpass_frac
    device: str = DEVICE
    sample_rate: int = config.preprocess.sample_rate
    n_fft: int = config.preprocess.n_fft
    hop_length: int = config.preprocess.hop_length

    @cached_property
    def time_stretch_perturbation(self) -> BatchedTimeStretchPerturbation:
        return BatchedTimeStretchPerturbation(
            factors=self.time_stretch_factors,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            num_workers=len(self.time_stretch_factors),
            device=self.device,
        )

    @cached_property
    def pitch_perturbation(self) -> BatchedPitchPerturbation:
        return BatchedPitchPerturbation(
            sample_rate=self.sample_rate,
            factors=self.pitch_factors,
            device=self.device,
            num_workers=len(self.pitch_factors),
        )

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

    def __enter__(self):
        self.time_stretch_perturbation.__enter__()
        self.pitch_perturbation.__enter__()
        return self

    def __exit__(self, *exc):
        self.time_stretch_perturbation.__exit__(*exc)
        self.pitch_perturbation.__exit__(*exc)

    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
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

    def apply_filter(self, signal: torch.Tensor) -> torch.Tensor:
        """Apply either a high or low pass filter to `signal`."""
        # generate decisions per snippet in the batch
        lowpass_mask = (
            torch.rand(signal.shape[0], device=signal.device) < self.lowpass_frac
        ).to(signal.device)
        frequencies = torch.zeros(
            signal.shape[0], dtype=torch.int, device=signal.device
        )

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
        self,
        signal: torch.Tensor,
        sample_rate: int,
        cutoff_freq: torch.Tensor,
        Q: float = 0.707,
    ) -> torch.Tensor:
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
        self,
        signal: torch.Tensor,
        sample_rate: int,
        cutoff_freq: torch.Tensor,
        Q: float = 0.707,
    ) -> torch.Tensor:
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
        signal: torch.Tensor,
        b0: torch.Tensor,
        b1: torch.Tensor,
        b2: torch.Tensor,
        a0: torch.Tensor,
        a1: torch.Tensor,
        a2: torch.Tensor,
    ) -> torch.Tensor:
        a_coeffs = torch.stack([a0, a1, a2], dim=1)
        b_coeffs = torch.stack([b0, b1, b2], dim=1)

        batch = signal.reshape((1, signal.shape[0], signal.shape[-1]))
        output = torchaudio.functional.filtering.lfilter(
            batch, a_coeffs, b_coeffs, clamp=True, batching=True
        )
        return output.reshape(signal.shape)

    def add_noise(self, signal: torch.Tensor) -> torch.Tensor:
        """Add either white or pink noise to `signal`."""
        noise = torch.randn_like(signal, device=signal.device, dtype=signal.dtype)
        noise_levels = self.whitenoise_range[0] + (
            self.whitenoise_range[1] - self.whitenoise_range[0]
        ) * torch.rand(signal.shape[0], device=signal.device, dtype=torch.float16)
        return torchaudio.functional.add_noise(
            waveform=signal, noise=noise, snr=noise_levels.unsqueeze(1)
        )
