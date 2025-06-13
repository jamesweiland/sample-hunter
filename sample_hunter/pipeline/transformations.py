"""
Custom generator functions to transform/pre-process data that is sitting
in the huggingface dataset.
"""

import multiprocessing
import random
import tempfile
import time
import numpy as np
import torch
import torchaudio
from torchaudio.transforms import MelSpectrogram
from torch import Tensor
from typing import Dict, List, Tuple, Union
from pathlib import Path
from datasets import load_dataset
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
    pitch_range: Tuple[float, float] = (0.6, 1.5)
    reverb_range: Tuple[int, int] = (40, 100)
    lowpass_range: Tuple[int, int] = (1000, 3000)
    highpass_range: Tuple[int, int] = (500, 1500)
    whitenoise_range: Tuple[float, float] = (0.05, 0.35)
    pinknoise_range: Tuple[float, float] = (0.05, 0.35)
    lowpass_frac: float = 0.5
    whitenoise_frac: float = 0.5

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

    def __call__(self, signal: Tensor | Path, sample_rate: int | None = None) -> Tensor:
        """
        Accepts either a torch.tensor or a path to an audio file,
        and returns the obfuscated tensor.
        """
        tmp_files = ["ob", "noise", "out"]
        with TempfileContext(files=tmp_files, suffix=".mp3") as tmp:
            if isinstance(signal, Tensor):
                # sox needs audio paths
                assert sample_rate is not None
                tmp.append("unob")
                torchaudio.save(
                    uri=tmp.unob,
                    src=signal,
                    sample_rate=sample_rate,
                    backend="ffmpeg",
                    format="mp3",
                )
                signal = tmp.unob
            return self.obfuscate(signal, tmp)

    def obfuscate(self, signal: Path, tmp: TempfileContext) -> Tensor:
        """
        Do the heavy lifting for __call__
        """
        tfm, mods = self.get_transformer()
        noise_mods = self.make_noise_like(signal, outfile=tmp.noise)
        mods = mods.update(noise_mods)

        tfm.build_file(str(signal), str(tmp.ob))

        try:
            combiner = Combiner().set_input_format(file_type=["mp3", "mp3"])
            combiner.build([str(tmp.ob), str(tmp.noise)], tmp.out, "mix")  # type: ignore
            return torchaudio.load(uri=tmp.out, backend="ffmpeg", format="mp3")[0]
        except Exception as e:
            print(f"An error occurred")
            raise e

    def get_transformer(self) -> Tuple[Transformer, Dict[str, Union[int, float]]]:
        """
        Create a Sox Transformer with the instance's ranges.

        Returns a tuple containing the transformer and a dictionary describing the distortions applied.
        """

        tfm = Transformer().set_globals(verbosity=1)
        mods = {}

        tempo = random.uniform(self.tempo_range[0], self.tempo_range[1])
        if abs(tempo - 1.0) <= 0.1:
            tfm.stretch(tempo)
        else:
            tfm.tempo(tempo)
        mods["tempo"] = round(tempo, 2)

        pitch = random.uniform(self.pitch_range[0], self.pitch_range[1])
        tfm.pitch(pitch)
        mods["pitch"] = round(pitch, 2)

        reverb = random.randint(self.reverb_range[0], self.reverb_range[1])
        tfm.reverb(reverb)
        mods["reverb"] = reverb

        if random.random() < 0.5:
            lowpass = random.randint(self.lowpass_range[0], self.lowpass_range[1])
            tfm.lowpass(lowpass)
            mods["lowpass"] = lowpass
        else:
            highpass = random.randint(self.highpass_range[0], self.highpass_range[1])
            tfm.highpass(highpass)
            mods["highpass"] = highpass

        return tfm, mods

    def make_noise_like(
        self, signal: Path, outfile: Path
    ) -> Dict[str, Union[int, float]]:
        """
        Generate a file containing white/pink noise with the same shape as the
        audio at `signal`.

        Returns a dictionary describing the noise generated.
        """
        tfm = Transformer().set_globals(verbosity=1)
        mods = {}

        if random.random() < self.whitenoise_frac:
            # White noise
            noise_level = random.uniform(
                self.whitenoise_range[0], self.whitenoise_range[1]
            )
            noise_type = "whitenoise"
            mods["whitenoise"] = round(noise_level, 2)
        else:
            # Pink noise
            noise_level = random.uniform(
                self.pinknoise_range[0], self.pinknoise_range[1]
            )
            noise_type = "pinknoise"
            mods["pinknoise"] = round(noise_level, 2)

        # Build the Sox synth command
        extra_args = ["synth", noise_type, "vol", str(noise_level)]

        tfm.build_file(signal, outfile, extra_args=extra_args)
        return mods


class SpectrogramPreprocessor:
    """
    A callable object to transform full songs into trainable spectrograms.
    To be used as the callable when overloading `.map` for an IterableDataset.
    """

    def __init__(
        self,
        mel_spectrogram: MelSpectrogram = DEFAULT_MEL_SPECTROGRAM,
        target_sample_rate: int = DEFAULT_SAMPLE_RATE,
        window_sample_size: int = DEFAULT_WINDOW_NUM_SAMPLES,
        window_step_size: int = DEFAULT_STEP_NUM_SAMPLES,
        n_fft: int = DEFAULT_N_FFT,
        hop_length: int = DEFAULT_HOP_LENGTH,
        n_mels: int = DEFAULT_N_MELS,
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
        self.window_num_samples = window_sample_size
        self.step_num_samples = window_step_size
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.device = device
        self.obfuscator = obfuscator
        self.num_workers = num_workers

    def __call__(
        self,
        data: Union[Dict, Tensor],
        obfuscate: bool = False,
        sample_rate: int | None = None,
    ):
        """
        Update `example` with fields for preprocessed data.
        """

        if isinstance(data, Dict):
            # it is a huggingface Audio object
            signal = (
                torch.tensor(data["array"], dtype=torch.float32, device=self.device)
                .unsqueeze(0)
                .to(self.device)
            )  # hf audio arrays are 1d
            return self.transform(
                signal=signal, sample_rate=data["sampling_rate"], obfuscate=obfuscate
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
        The tensor that is returned has shape (num_windows, num_channels, n_mels, time_frames)
        """
        print("Entered transform")
        # (we don't have to mix anything down because HF has done this for us)

        # resample to target sampling rate
        signal = self.resample(signal, sample_rate)
        # make windows with overlay
        signal = self.create_windows(signal)

        anchor = self.mel_spectrogram(signal)
        if obfuscate:
            # obfuscate each window
            positive = torch.stack(
                [self.mel_spectrogram(self.obfuscator(window)) for window in signal]
            )
            return positive, anchor
        # if not obfuscate, just return the regular transformation
        return anchor

    def obfuscate_window(self, window: Tensor) -> Tensor:
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
        ob_sig = self.obfuscator(window, self.target_sample_rate)
        ob_sig = self.resize(ob_sig, window.shape[1])
        return ob_sig

    def resample(self, signal: Tensor, sample_rate: int) -> Tensor:
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(
                sample_rate, self.target_sample_rate
            ).to(self.device)
            return resampler(signal)
        return signal

    def resize(self, signal: Tensor, desired_length: int) -> Tensor:
        """Set the length of the signal to the desired length"""
        if signal.shape[1] < desired_length:
            num_missing_samples = desired_length - signal.shape[1]
            return torch.nn.functional.pad(signal, (0, num_missing_samples))
        if signal.shape[1] > desired_length:
            return signal[:, :desired_length]
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

    def _worker(self, window: np.ndarray, obfuscate: bool = False) -> np.ndarray:
        """The worker function for transforming windows into spectrograms."""
        mel = self._create_mel()
        if obfuscate:
            pass
        else:
            return mel(window)

    def _create_mel(self) -> MelSpectrogram:
        return MelSpectrogram(
            sample_rate=self.target_sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            hop_length=self.hop_length,
        ).to(self.device)

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


if __name__ == "__main__":
    # test hf_audio_to_spectrogram and the collate fn
    preprocessor = SpectrogramPreprocessor()
    print("Downloading dataset")
    ds = load_dataset("samplr/songs", streaming=True, split="train")
    print("Download complete, applying map")
    ds = ds.map(lambda ex: {**ex, "transform": preprocessor(ex["audio"])})
    print("map complete, making dataloader")

    dataloader = DataLoader(
        ds, batch_size=2, collate_fn=SpectrogramPreprocessor.collate_spectrograms
    )
    print("dataloader done, starting loop")
    for batch in dataloader:

        print("Concatenated batch shape:")
        print(batch.size())

    # # test the obfuscation stuff
    # obf_ds = load_dataset("samplr/songs", streaming=True, split="train")
    # obf_ds = obf_ds.map(
    #     lambda ex: (
    #         lambda signals: {**ex, "positive": signals[0], "anchor": signals[1]}
    #     )(preprocessor(ex["audio"], obfuscate=True))
    # )

    # dataloader = DataLoader(
    #     obf_ds, batch_size=2, collate_fn=SpectrogramPreprocessor.collate_spectrograms
    # )
    # for anchor, positive in dataloader:
    #     print("Concatenated batch shape:")
    #     print(anchor.size())
    #     print(positive.size())
    #     break
