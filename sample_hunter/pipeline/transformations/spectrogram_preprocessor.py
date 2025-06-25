"""
Custom generator functions to transform/pre-process data that is sitting
in the huggingface dataset.
"""

import io
import torch
import torchaudio
from typing import Dict, Tuple, Union
from torch.utils.data import DataLoader

from .obfuscator import Obfuscator
from .functional import collate_spectrograms, resize
from sample_hunter._util import (
    DEVICE,
    DEFAULT_MEL_SPECTROGRAM,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_WINDOW_NUM_SAMPLES,
    DEFAULT_STEP_NUM_SAMPLES,
    CACHE_DIR,
)


class SpectrogramPreprocessor:
    """
    A callable object to transform full songs into trainable spectrograms.
    To be used as the callable when overloading `.map` for an IterableDataset.
    """

    def __init__(
        self,
        mel_spectrogram: torchaudio.transforms.MelSpectrogram = DEFAULT_MEL_SPECTROGRAM,
        target_sample_rate: int = DEFAULT_SAMPLE_RATE,
        window_num_samples: int = DEFAULT_WINDOW_NUM_SAMPLES,
        step_num_samples: int = DEFAULT_STEP_NUM_SAMPLES,
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
                return self.transform(
                    signal=signal,
                    sample_rate=data["sampling_rate"],
                    obfuscate=obfuscate,
                )
            elif isinstance(data, bytes):
                # try to read the bytes as an mp3 file
                with io.BytesIO(data) as buffer:
                    signal, sample_rate = torchaudio.load(
                        buffer, format="mp3", backend="ffmpeg"
                    )
                    if signal.ndim == 1:
                        signal = signal.unsqueeze(0)
                    return self.transform(
                        signal=signal, sample_rate=sample_rate, obfuscate=obfuscate
                    )
            elif isinstance(data, torch.Tensor):
                # if a tensor is passed, a sample_rate has to be passed too
                assert sample_rate is not None
                data = data.to(self.device)
                return self.transform(
                    signal=data, sample_rate=sample_rate, obfuscate=obfuscate
                )
            else:
                raise RuntimeError("Unsupported data type")

    def transform(
        self, signal: torch.Tensor, sample_rate: int, obfuscate: bool
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
        signal = self.create_windows(signal)

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

    def create_windows(self, signal: torch.Tensor) -> torch.Tensor:
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
    import argparse
    import memray
    from sample_hunter._util import plot_spectrogram, HF_TOKEN
    from pathlib import Path
    from datasets import load_dataset, Audio

    parser = argparse.ArgumentParser()
    parser.add_argument("--token", type=str, required=False)
    args = parser.parse_args()
    if args.token is None:
        args.token = HF_TOKEN

    # test hf_audio_to_spectrogram and the collate fn
    with SpectrogramPreprocessor() as preprocessor:
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
