from pathlib import Path
import os
import sys
import multiprocessing
import threading
import torch
import torchaudio
from torchaudio.transforms import AmplitudeToDB
from torch import Tensor
import matplotlib.pyplot as plt
from typing import Any
from abc import ABC, abstractmethod
import sounddevice as sd

import pandas as pd

from sample_hunter.config import set_config_path, get_config

CONFIG_PATH: Path = Path("configs/7_15_2025.yaml")
set_config_path(Path(CONFIG_PATH))
config = get_config()

WINDOW_NUM_SAMPLES: int = int(
    config.preprocess.sample_rate * config.preprocess.spectrogram_width
)  # 2 seconds per window
STEP_NUM_SAMPLES: int = int(
    config.preprocess.sample_rate * config.preprocess.step_length
)  # 1 second overlay between windows
INPUT_SHAPE: torch.Size = torch.Size(
    (
        1,
        config.preprocess.n_mels,
        1 + WINDOW_NUM_SAMPLES // config.preprocess.hop_length,
    )
)


DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
MEL_SPECTROGRAM = torchaudio.transforms.MelSpectrogram(
    sample_rate=config.preprocess.sample_rate,
    n_fft=config.preprocess.n_fft,
    hop_length=config.preprocess.hop_length,
    n_mels=config.preprocess.n_mels,
).to(DEVICE)

PROCS: int = multiprocessing.cpu_count()
HF_TOKEN: str = os.environ.get("HF_TOKEN", "")

# these are maybe deprecated
DEFAULT_REQUEST_TIMEOUT: float = 15.0
DEFAULT_DOWNLOAD_TIME: float = 2700.0
DEFAULT_RETRIES: int = 5
DEFAULT_RETRY_DELAY: float = 5.0


def load_model(model_path: Path):
    from sample_hunter.pipeline.encoder_net import EncoderNet

    if DEVICE == "cuda":
        state_dict = torch.load(model_path, weights_only=False)
    else:
        state_dict = torch.load(
            model_path, weights_only=False, map_location=torch.device("cpu")
        )
    model = EncoderNet().to(DEVICE)
    model.load_state_dict(state_dict)
    return model


def plot_spectrogram(tensor: Tensor, title: str = "Spectrogram"):
    """Plot a torch Tensor as a spectrogram"""
    tensor = tensor.squeeze().cpu()
    tensor = AmplitudeToDB()(tensor)

    plt.figure(figsize=(12, 4))
    plt.imshow(tensor.numpy(), aspect="auto", origin="lower", cmap="viridis")
    plt.colorbar(format="%+2.0f dB")
    plt.title(title)
    plt.ylabel("Mel Frequency Bin")
    plt.xlabel("Time Frame")
    plt.show()


def play_tensor_audio(
    tensor: torch.Tensor,
    message: str | None = None,
    sample_rate=config.preprocess.sample_rate,
):
    """Halt script execution to play a tensor as audio"""
    # tensor: shape (1, T) or (T,)
    arr = tensor.cpu().numpy()
    if arr.ndim == 2:
        if arr.shape[0] == 2:  # stereo
            arr = arr.T
        elif arr.shape[0] == 1:  # mono
            arr = arr.squeeze(0)  # sd doesn't like the extra dimension

    if message:
        print(message)
    else:
        print("Playing audio...")
    sd.play(arr, samplerate=sample_rate)
    sd.wait()  # Wait until playback is finished


def save_to_json_or_csv(path: Path, df: pd.DataFrame) -> None:
    """Save a dataframe to json to csv"""
    if path.suffix == ".csv":
        return df.to_csv(path, index=True)
    elif path.suffix == ".json":
        return df.to_json(path, orient="index", indent=4)
    raise RuntimeError("Illegal path suffix for writing df to file")


def read_into_df(path: Path) -> pd.DataFrame:
    """Read a json or csv file into a pandas dataframe"""

    if path.suffix == ".csv":
        return pd.read_csv(path)
    elif path.suffix == ".json":
        return pd.read_json(path)
    raise RuntimeError("Illegal path suffix for reading into df")


class SigintHandler:
    """Handler to save a dataframe before exiting a long-running script that needs to be interrupted"""

    def __init__(self, df_name: str, path: Path):
        """Store information about the df.

        Args:
            df_name (str): the name of the object that the df is assigned to in the script
            path (Path): the path to save the df to
        """

        self.df_name = df_name
        self.path = path
        self.lock = multiprocessing.Lock()  # to make it safe for parallelization

    def __call__(self, sig, frame):
        """On sigint, save the df"""

        curr = frame
        df = None
        found = False
        while curr:
            for name, val in curr.f_locals.items():
                if name == self.df_name:
                    df = val
                    found = True
                    break
            if found:
                break
            curr = curr.f_back
        if df is not None and self.path is not None:
            print(f"Saving data to {self.path} before exiting...")
            if self.path == ".csv":
                with self.lock:
                    old_df = pd.read_csv(self.path)
                    df.to_csv(self.path)
            else:
                df.to_json(self.path, orient="index", indent=4)
            sys.exit(0)
        print("df not found and no data was saved")
        sys.exit(0)


class AtomicCounter(ABC):
    """Abstract atomic counter for multiprocessing/threading"""

    @property
    @abstractmethod
    def lock(self) -> Any:
        pass

    @property
    @abstractmethod
    def value(self) -> int:
        pass

    @value.setter
    @abstractmethod
    def value(self, v):
        pass

    def __init__(self, end: int | None = None):
        self.end = end

    def fetch_and_increment(self, inc: int = 1) -> int:
        with self.lock:
            tmp = self.value
            self.value = tmp + inc
            if self.end is not None and self.value > self.end:
                raise ValueError("Can't increment past max value")
            return tmp


class ThreadAtomicCounter(AtomicCounter):
    def __init__(self, start: int, end: int | None = None):
        super().__init__(end)
        self._lock = threading.Lock()
        self._value = start

    @property
    def lock(self):
        return self._lock

    @property
    def value(self) -> int:
        return self._value

    @value.setter
    def value(self, v):
        self._value = v


class ProcessAtomicCounter(AtomicCounter):
    def __init__(
        self,
        start: int,
        end: int | None = None,
    ):
        super().__init__(end)
        self.manager = multiprocessing.Manager()
        self._shared = self.manager.Value("i", start)
        self._lock = self.manager.Lock()

    @property
    def value(self) -> int:
        return self._shared.value

    @value.setter
    def value(self, v):
        self._shared.value = v

    @property
    def lock(self):
        return self._lock
