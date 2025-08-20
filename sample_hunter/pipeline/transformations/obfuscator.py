import math
from pathlib import Path
import random
import torchaudio
import torch

from functools import cached_property

from .functional import create_windows, offset
from .my_musan import MusanException, MyMusan
from .batched_pitch_perturbation import BatchedPitchPerturbation
from .batched_time_stretch_perturbation import BatchedTimeStretchPerturbation
from sample_hunter._util import DEVICE
from sample_hunter.config import ObfuscatorConfig


class Obfuscator:
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

    def __init__(
        self, config: ObfuscatorConfig | None = None, device: str = DEVICE, **kwargs
    ):
        self.config = config or ObfuscatorConfig()
        self.config = self.config.merge_kwargs(**kwargs)
        self.device = device

    @cached_property
    def musan(self) -> MyMusan:
        return MyMusan(
            self.config.musan,
            subset="music",
            sample_rate=self.config.sample_rate,
            spec_num_sec=self.config.spec_num_sec,
            volume_threshold=self.config.volume_threshold,
        )

    @cached_property
    def time_stretch_perturbation(self) -> BatchedTimeStretchPerturbation:
        return BatchedTimeStretchPerturbation(
            factors=self.config.time_stretch_factors,
            num_workers=self.config.perturb_num_workers,
            device=self.device,
        )

    @cached_property
    def pitch_perturbation(self) -> BatchedPitchPerturbation:
        return BatchedPitchPerturbation(
            sample_rate=self.config.sample_rate,
            factors=self.config.pitch_factors,
            device=self.device,
            num_workers=self.config.perturb_num_workers,
        )

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

            # batch = self.make_offset_windows(batch)

            batch = self.time_stretch_perturbation(batch)
            batch = self.pitch_perturbation(batch)
            batch = self.apply_filter(batch)

            batch = self.overlay_musan(batch)

            return batch

    def make_offset_windows(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Randomly offset a signal with shape (1, S), and then create windows.

        Returns a tensor with shape (W, 1, spec_num_samples) where W is number of windows
        """
        offset_num_samples = random.randint(
            -self.config.offset_span_num_samples, self.config.offset_span_num_samples
        )

        offset_signal = offset(signal, offset_num_samples)

        windows = create_windows(
            offset_signal, self.config.spec_num_samples, self.config.step_num_samples
        )

        return windows

    def apply_filter(self, signal: torch.Tensor) -> torch.Tensor:
        """Apply either a high or low pass filter to `signal`."""
        # generate decisions per snippet in the batch
        lowpass_mask = (
            torch.rand(signal.shape[0], device=signal.device) < self.config.lowpass_frac
        ).to(signal.device)
        frequencies = torch.zeros(
            signal.shape[0], dtype=torch.int, device=signal.device
        )

        low_idx = lowpass_mask.nonzero().squeeze(1)
        low_freqs = torch.randint(
            low=self.config.lowpass_range[0],
            high=self.config.lowpass_range[1] + 1,
            size=(low_idx.shape[0],),
            device=signal.device,
            dtype=torch.int,
        )
        frequencies[lowpass_mask] = low_freqs
        high_idx = (~lowpass_mask).nonzero().squeeze(1)
        high_freqs = torch.randint(
            low=self.config.highpass_range[0],
            high=self.config.highpass_range[1] + 1,
            size=(high_idx.shape[0],),
            device=signal.device,
            dtype=torch.int,
        )
        frequencies[~lowpass_mask] = high_freqs
        signal_low = self._lowpass_biquad(
            signal[lowpass_mask], self.config.sample_rate, frequencies[lowpass_mask]
        )
        signal_high = self._highpass_biquad(
            signal[~lowpass_mask], self.config.sample_rate, frequencies[~lowpass_mask]
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

    def overlay_musan(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Overlay random samples from the MUSAN dataset onto `signal`

        `signal` is a tensor with shape (B, 1, S),
        B is batch size,
        S is number of samples

        each example in `signal` will have a different song from
        MUSAN overlaid
        """

        # build up a sample of songs from musan
        n = signal.shape[0]
        sample = torch.empty_like(signal)
        num_windows_added = 0
        while num_windows_added < n:
            try:
                idx = random.randint(0, len(self.musan) - 1)
                musan_signal, name = self.musan[idx]

                remaining = n - num_windows_added
                windows_to_add = min(musan_signal.shape[0], remaining)

                # add musan signal to the correct sample slice, checking for overflow
                sample[num_windows_added : num_windows_added + windows_to_add] = (
                    musan_signal[:windows_to_add]
                )
                num_windows_added += windows_to_add
            except MusanException:
                continue

        noise_levels = self.config.musan_noise_range[0] + (
            self.config.musan_noise_range[1] - self.config.musan_noise_range[0]
        ) * torch.rand(signal.shape[0], device=signal.device, dtype=torch.float32)

        res = torchaudio.functional.add_noise(
            waveform=signal, noise=sample, snr=noise_levels.unsqueeze(1)
        )
        return res


if __name__ == "__main__":
    # listen to obfuscated vs original audio
    from .preprocessor import Preprocessor
    import webdataset as wds
    from .functional import (
        mix_channels,
        resample,
        remove_low_volume_windows,
        create_windows,
        rms_normalize,
    )
    from sample_hunter._util import HF_TOKEN, play_tensor_audio, plot_spectrogram
    from sample_hunter.pipeline.data_loading import load_tensor_from_bytes

    with Preprocessor() as preprocessor:
        train_tars = Path("./_data/webdataset-shards/train").glob("*.tar")
        train_tars = [str(tar) for tar in train_tars]
        dataset = wds.WebDataset(train_tars).decode()

        def map_fn(ex):
            positive, anchor = preprocessor(ex["mp3"], train=True)
            signal, sr = load_tensor_from_bytes(ex["mp3"])
            signal = mix_channels(signal)
            signal = resample(signal, sr, 44_100)
            signal = create_windows(signal, int(44_100 * 1), int(44_100 * 0.5))
            signal = remove_low_volume_windows(signal, -50)
            # signal = remove_low_volume_windows(signal, -50)
            positive_audio = preprocessor.obfuscate(signal)
            anchor = preprocessor.config.mel_spectrogram(signal)
            positive = preprocessor.config.mel_spectrogram(positive_audio)

            return {
                **ex,
                "anchor_audio": signal,
                "positive_audio": positive_audio,
                "anchor": anchor,
                "positive": positive,
            }

        dataset = dataset.map(map_fn)
        for ex in dataset:
            print(f"Song: {ex["json"]["title"]}\n")

            for i in range(min(ex["anchor"].shape[0], 10)):
                print(ex["anchor"][i].shape)
                play_tensor_audio(ex["anchor_audio"][i], f"Playing anchor {i}...")
                play_tensor_audio(ex["positive_audio"][i], f"Playing positive {i}...")
                # plot_spectrogram(ex["anchor"][i], f"anchor {i}")
                # plot_spectrogram(ex["positive"][i], f"positive {i}")

            print("--------------------------------------------------")
