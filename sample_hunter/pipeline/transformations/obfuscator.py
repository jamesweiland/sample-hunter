from functools import cached_property
import math
import torch
from pydantic import BaseModel, field_validator
from typing import Sequence, Tuple

import torchaudio

from .tone_generator import ToneGenerator
from .batched_pitch_perturbation import BatchedPitchPerturbation
from .batched_time_stretch_perturbation import BatchedTimeStretchPerturbation
from sample_hunter._util import config, DEVICE, plot_spectrogram


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
    num_tones_to_add: int = config.preprocess.obfuscator.num_tones_to_add
    tone_gen_frequency_range: Tuple[int, int] = (
        config.preprocess.obfuscator.tone_gen_frequency_range
    )
    tone_gen_amplitude_range: Tuple[float, float] = (
        config.preprocess.obfuscator.tone_gen_amplitude_range
    )
    tone_gen_duration_range: Tuple[float, float] = (
        config.preprocess.obfuscator.tone_gen_duration_range
    )
    lowpass_frac: float = config.preprocess.obfuscator.lowpass_frac
    device: str = DEVICE
    sample_rate: int = config.preprocess.sample_rate
    n_fft: int = config.preprocess.n_fft
    hop_length: int = config.preprocess.hop_length

    # this is probably deprecated but i'm gonna keep it here just in case
    @cached_property
    def tone_gen(self) -> ToneGenerator:
        return ToneGenerator(self.sample_rate)

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
            # batch = self.generate_tone(batch)
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
        """Apply a random amount of noise to `signal`."""
        noise = torch.randn_like(signal, device=signal.device, dtype=signal.dtype)
        noise_levels = self.whitenoise_range[0] + (
            self.whitenoise_range[1] - self.whitenoise_range[0]
        ) * torch.rand(signal.shape[0], device=signal.device, dtype=torch.float16)
        return torchaudio.functional.add_noise(
            waveform=signal, noise=noise, snr=noise_levels.unsqueeze(1)
        )

    def generate_tone(
        self,
        signal: torch.Tensor,
    ):
        """
        Use `ToneGenerator` to generate a tone and add it to `signal`

        signal has shape (B, 1, S), S is number of samples and B is batch size

        This is likely deprecated as we have a model that was trained
        without it that's pretty good, but i'll keep it here just in case
        """
        tone_gens = [
            self.tone_gen.generate_sine_wave,
            self.tone_gen.generate_square_wave,
            self.tone_gen.generate_triangle_wave,
            self.tone_gen.generate_sawtooth_wave,
        ]
        num_waves = len(tone_gens)
        for _ in range(self.num_tones_to_add):
            perm = torch.randperm(signal.shape[0])
            # calculate the inverse permutation so we can restore the original order after
            inv_perm = torch.empty_like(perm)
            inv_perm[perm] = torch.arange(signal.shape[0])
            signal = signal[perm]
            segments = torch.chunk(signal, num_waves)
            for i, segment in enumerate(segments):
                frequencies = torch.randint(
                    low=self.tone_gen_frequency_range[0],
                    high=self.tone_gen_frequency_range[1],
                    size=(segment.shape[0],),
                )
                amplitudes = (
                    self.tone_gen_amplitude_range[1] - self.tone_gen_amplitude_range[0]
                ) * torch.rand(segment.shape[0]) + self.tone_gen_amplitude_range[0]
                starts = config.preprocess.spectrogram_width * torch.rand(
                    segment.shape[0]
                )
                ends = torch.clamp(
                    starts
                    + (
                        self.tone_gen_duration_range[1]
                        - self.tone_gen_duration_range[0]
                    )
                    * torch.rand(segment.shape[0])
                    + self.tone_gen_duration_range[0],
                    max=config.preprocess.spectrogram_width,
                )
                segment = tone_gens[i](
                    signal=segment,
                    frequencies=frequencies,
                    amplitudes=amplitudes,
                    starts=starts,
                    ends=ends,
                )
                signal[
                    sum(seg.shape[0] for seg in segments[:i]) : sum(
                        seg.shape[0] for seg in segments[: i + 1]
                    )
                ] = segment

            signal = signal[inv_perm]

        return signal


if __name__ == "__main__":
    pass
    # listen to obfuscated vs original audio
    from .spectrogram_preprocessor import SpectrogramPreprocessor
    from sample_hunter.pipeline.data_loading import load_webdataset
    from sample_hunter._util import HF_TOKEN, play_tensor_audio
    from sample_hunter.pipeline.data_loading import load_tensor_from_bytes

    with SpectrogramPreprocessor() as preprocessor:
        dataset = load_webdataset("samplr/songs", "train", HF_TOKEN)

        def map_fn(ex):
            # load the audio as tensor form so it can be passed to obfuscator
            audio, sr = load_tensor_from_bytes(ex["mp3"])
            audio = preprocessor.mix_channels(audio)
            audio = preprocessor.resample(audio, sr)
            anchor = preprocessor.create_windows(audio)
            positive = preprocessor.obfuscate_window(anchor)
            return {
                **ex,
                "anchor": anchor,
                "positive": positive,
                "anchor_spec": preprocessor.mel_spectrogram(anchor),
                "positive_spec": preprocessor.mel_spectrogram(positive),
            }

        dataset = dataset.map(map_fn)
        for ex in dataset:
            print(f"Song: {ex["json"]["title"]}\n")

            for i in range(ex["anchor"].shape[0]):
                play_tensor_audio(ex["anchor"][i], f"Playing anchor {i}...")
                play_tensor_audio(ex["positive"][i], f"Playing positive {i}...")
                plot_spectrogram(ex["anchor_spec"][i], f"anchor {i}")
                plot_spectrogram(ex["positive_spec"][i], f"positive {i}")

            print("--------------------------------------------------")
