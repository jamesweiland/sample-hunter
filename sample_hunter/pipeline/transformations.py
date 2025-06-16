"""
Custom generator functions to transform/pre-process data that is sitting
in the huggingface dataset.
"""

from functools import cached_property
import math
import multiprocessing
import random
import tempfile
import time
import memray
import numpy as np
import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram
from torch import Tensor
from typing import Dict, List, Tuple, Union
from pathlib import Path
from datasets import load_dataset, Audio
from torch.utils.data import DataLoader
from pydantic import BaseModel, field_validator
from sox import Transformer, Combiner

from sample_hunter._util import (
    DEFAULT_HOP_LENGTH,
    DEFAULT_N_FFT,
    DEFAULT_N_MELS,
    DEVICE,
    DEFAULT_MEL_SPECTROGRAM,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_WINDOW_NUM_SAMPLES,
    DEFAULT_STEP_NUM_SAMPLES,
    PROCS,
    CACHE_DIR,
)


class TempfileContext:
    """
    A wrapper around tempfile.NamedTemporaryFile
    to create temporary files without having to worry about cleanup.
    """

    def __init__(self, files: List[str], suffix: List[str] | str = ".mp3"):
        """
        Store the necessary settings for the manager.

        Args:
            files: a string or list of strings representing the filenames,
            that can be accessed as attributes of the TempfileContext object.
            suffixes: a string or list of strings representing the suffixes of each file.
            If only a string is given for suffixes but a list is given for files, the same suffix
            will be used for each file.
        """
        self._files = files
        if isinstance(suffix, str):
            self._suffix = [suffix] * len(self._files)
        else:
            self._suffix = suffix

    def __enter__(self):
        self._paths = []
        for filename, suffix in zip(self._files, self._suffix):
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
                setattr(self, filename, Path(f.name))
                self._paths.append(Path(f.name))
                # now, exit this context so other processes can write to the file
        return self

    def __exit__(self, *exc):
        for f in self._paths:
            f.unlink()

    def append(self, file: str, suffix: str | None = None):
        """
        Add a new temp file. if suffix is not specified then it will be the same suffix as the last file element in _suffix
        """
        if suffix is not None:
            self._suffix.append(suffix)
        else:
            self._suffix.append(self._suffix[-1])
        with tempfile.NamedTemporaryFile(suffix=self._suffix[-1], delete=False) as f:
            setattr(self, file, Path(f.name))
            self._paths.append(Path(f.name))
        self._files.append(file)


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
        `reverb_range`: The range with which to add reverb to the signal.
        `lowpass_range`: The range with which to add a low-pass filter.
        `highpass_range`: The range with which to add a high-pass filter.
        `whitenoise_range`: The volume range of white noise to add.
        `pinknoise_range`: The volume range of pink noise to add.
        `lowpass_frac`: The proportion of signals to apply a lowpass filter to (otherwise,
        a highpass filter will be applied).
        `whitenoise_frac`: The proportion of signals to apply white noise to (otherwise,
        pink noise will be applied).
    """

    tempo_range: Tuple[float, float] = (0.6, 1.5)
    pitch_range: Tuple[int, int] = (-24, 24)
    reverb_range: Tuple[int, int] = (40, 100)
    lowpass_range: Tuple[int, int] = (1000, 3000)
    highpass_range: Tuple[int, int] = (500, 1500)
    whitenoise_range: Tuple[float, float] = (0.05, 0.25)
    pinknoise_range: Tuple[float, float] = (0.05, 0.25)
    lowpass_frac: float = 0.5
    whitenoise_frac: float = 0.5
    n_fft: int = DEFAULT_N_FFT
    hop_length: int = DEFAULT_HOP_LENGTH
    device: str = DEVICE
    # num_workers: int = 1

    # @field_validator("num_workers")
    # def check_num_workers(cls, v, field):
    #     if v <= multiprocessing.cpu_count():
    #         return v
    #     raise ValueError(
    #         f"num_workers is greater than number of available cores: {v} > {multiprocessing.cpu_count()}"
    #     )

    @cached_property
    def window(self):
        return torch.hann_window(window_length=self.n_fft, device=self.device)

    @field_validator(
        "tempo_range",
        "pitch_range",
        "reverb_range",
        "lowpass_range",
        "highpass_range",
        "whitenoise_range",
        "pinknoise_range",
    )
    def check_range(cls, v, field):
        if len(v) != 2:
            raise ValueError(f"{field.name} must be a tuple of length 2")
        if v[0] >= v[1]:
            raise ValueError(
                f"In {field.name}, the first value must be smaller than the second value"
            )
        return v

    @field_validator("lowpass_frac", "whitenoise_frac")
    def check_fraction(cls, v, field):
        if not (0 <= v <= 1):
            raise ValueError(f"{field.name} must be between 0 and 1")
        return v

    def __call__(
        self, batch: Tensor, sample_rate: int, return_complex_spec: bool = False
    ) -> Tensor:
        """
        Accepts a torch.tensor
        and returns the obfuscated tensor.
        """
        with torch.no_grad():
            print(f"Received shape: {batch.shape}")
            batch = batch.contiguous() if not batch.is_contiguous() else batch
            mods = {}
            batch, pitch = self.pitch_distort(batch, sample_rate)
            mods.update({"pitch": round(pitch, 2)})
            batch, filter = self.apply_filter(batch, sample_rate)
            mods.update(filter)
            batch, noise_mods = self.add_noise(batch)
            mods.update(noise_mods)
            batch, tempo = self.time_distort(batch)  # this makes a complex spec
            mods.update({"tempo": round(tempo, 2)})
            if return_complex_spec:
                print(f"Returning shape: {batch.shape}")
                return batch
            # convert back to audio before returning
            return torch.istft(
                batch, n_fft=self.n_fft, hop_length=self.hop_length, window=self.window
            ).unsqueeze(1)

    def time_distort(self, signal: Tensor) -> Tuple[Tensor, float]:
        """Apply a tempo distortion to `signal`."""
        # rate = random.uniform(*self.tempo_range)
        rate = 1.5

        spec = torch.stft(
            input=signal.squeeze(1),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True,
        )
        freq_bins = self.n_fft // 2 + 1
        phase_advance = torch.linspace(
            0, math.pi * self.hop_length, freq_bins, device=self.device
        ).unsqueeze(-1)

        return (
            torchaudio.functional.phase_vocoder(
                complex_specgrams=spec, rate=rate, phase_advance=phase_advance
            ),
            rate,
        )

    def pitch_distort(self, signal: Tensor, sample_rate: int) -> Tuple[Tensor, float]:
        """Apply a pitch distortion to `signal`."""
        # pitch = random.randint(*self.pitch_range)
        pitch = random.choice([-24, -18, -12, -6, 0, 6, 12, 18, 24])
        signal = torchaudio.functional.pitch_shift(signal, sample_rate, n_steps=pitch)
        return signal, pitch

    def apply_filter(
        self, signal: Tensor, sample_rate: int
    ) -> Tuple[Tensor, Dict[str, int]]:
        """Apply either a high or low pass filter to `signal`."""
        if random.random() < self.lowpass_frac:
            # apply low pass
            freq = random.randint(*self.lowpass_range)
            signal = torchaudio.functional.lowpass_biquad(signal, sample_rate, freq)
            mods = {"lowpass": freq}
        else:
            # apply high pass
            freq = random.randint(*self.highpass_range)
            signal = torchaudio.functional.highpass_biquad(signal, sample_rate, freq)
            mods = {"highpass": freq}
        return signal, mods

    def add_noise(self, signal: Tensor) -> Tuple[Tensor, Dict[str, float]]:
        """Add either white or pink noise to `signal`."""
        if random.random() < self.whitenoise_frac:
            noise_level = random.uniform(*self.whitenoise_range)
            noise = torch.randn_like(signal) * noise_level
            mods = {"whitenoise": round(noise_level, 2)}
        else:
            noise_level = random.uniform(*self.pinknoise_range)
            whitenoise = torch.randn_like(signal)
            # filter whitenoise to get pinknoise
            pinknoise = torch.cumsum(whitenoise, dim=-1)
            pinknoise = pinknoise / pinknoise.abs().max(dim=-1, keepdim=True).values
            noise = pinknoise * noise_level
            mods = {"pinknoise": round(noise_level, 2)}
        noisy_signal = signal + noise
        return noisy_signal, mods


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
        self.num_workers = num_workers
        self._resampler_cache = {}

    def __call__(
        self,
        data: Union[Dict, Tensor],
        obfuscate: bool = False,
        sample_rate: int | None = None,
    ):
        """
        Update `example` with fields for preprocessed data.
        """
        print(type(data))
        print(data)
        with torch.no_grad():

            if isinstance(data, Dict):
                # it is a huggingface Audio object
                signal = torch.tensor(
                    data["array"], dtype=torch.float16, device=self.device
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
        print("Entered transform")
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
        # window = signal[0]
        # first, split signal into chunks

        ob_sig = SpectrogramPreprocessor.resize(
            self.obfuscator(signal, self.target_sample_rate), signal.shape[-1]
        )

        # ob_sig = torch.cat(
        #     [
        #         SpectrogramPreprocessor.resize(
        #             self.obfuscator(chunk, self.target_sample_rate), signal.shape[-1]
        #         )
        #         for chunk in self._split_into_chunks(signal)
        #     ],
        #     dim=0,
        # )

        print(f"Obfuscation returns shape: {ob_sig.shape}")

        torchaudio.save(
            uri="unob.mp3",
            src=signal[0].to(torch.float32),
            format="mp3",
            backend="ffmpeg",
            sample_rate=self.target_sample_rate,
        )

        torchaudio.save(
            uri="ob.mp3",
            src=ob_sig[0].to(torch.float32),
            format="mp3",
            backend="ffmpeg",
            sample_rate=self.target_sample_rate,
        )

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
        torch.set_num_threads(1)
        print("resampling")
        if sample_rate != self.target_sample_rate:
            if sample_rate in self._resampler_cache.keys():
                signal = self._resampler_cache[sample_rate](sample_rate)
            else:
                resampler = torchaudio.transforms.Resample(
                    sample_rate, self.target_sample_rate
                ).to(self.device)
                signal = resampler(signal)
        print("resampling done")
        torch.set_num_threads(torch.multiprocessing.cpu_count())
        return signal

    @staticmethod
    def resize(signal: Tensor, desired_length: int) -> Tensor:
        """Set the length of the signal to the desired length"""
        if signal.shape[-1] < desired_length:
            num_missing_samples = desired_length - signal.shape[-1]
            return torch.nn.functional.pad(signal, (0, num_missing_samples))
        if signal.shape[-1] > desired_length:
            return signal[..., :desired_length]
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
            last_segment = self.resize(last_segment, self.window_num_samples).unsqueeze(
                1
            )
            windows = torch.cat([windows, last_segment], dim=1)

        return windows.transpose(0, 1).to(self.device)

    @staticmethod
    def collate_spectrograms(
        batch: List[Dict[str, Tensor]],
    ) -> Tensor | Tuple[Tensor, Tensor]:
        """
        Collate a batch of mappings of transformed tensors before passing to the dataloader.

        This function expects tensors with shape (batch_size, num_windows, num_channels, n_mels, time_frames)
        and returns a tensor with shape (new_batch_size, num_channels, n_mels, time_frames)
        """
        print("Entered collate")
        if batch[0].get("transform") is not None:
            return torch.cat([example["transform"] for example in batch], dim=0)
        return torch.cat([example["anchor"] for example in batch], dim=0), torch.cat(
            [example["positive"] for example in batch], dim=0
        )

    def _split_into_chunks(self, signal: Tensor) -> Tuple[Tensor, ...]:
        """
        Split a tensor into chunks, with a maximum size of self.max_chunk_bytes

        """
        elements_per_window = signal.shape[-1] * signal.shape[-2]
        bytes_per_window = elements_per_window * signal.element_size()
        num_windows_in_chunk = max(self.max_chunk_bytes // bytes_per_window, 1)

        return torch.split(signal, num_windows_in_chunk, dim=0)


if __name__ == "__main__":
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
        ).cast_column("audio", Audio(decode=True))
        obf_ds = obf_ds.map(map_fn)

        dataloader = DataLoader(
            obf_ds,
            batch_size=2,
            collate_fn=SpectrogramPreprocessor.collate_spectrograms,
        )
        i = 0
        for anchor, positive in dataloader:
            print("Concatenated batch shape:")
            print(anchor.size())
            print(positive.size())
            break
            i += 1
            if i == 10:
                break
    print("done")
