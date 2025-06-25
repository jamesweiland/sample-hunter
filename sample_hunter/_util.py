from pathlib import Path
import os
import sys
import multiprocessing
import threading
import torch
from torchaudio.transforms import AmplitudeToDB, MelSpectrogram
from torch import Tensor
import matplotlib.pyplot as plt
from typing import Any, List, Tuple
from abc import ABC, abstractmethod

import pandas as pd

# pipeline hyperparameters
DEFAULT_SAMPLE_RATE: int = 44_100  # 44.1 kHz
DEFAULT_N_FFT: int = 1024
DEFAULT_HOP_LENGTH: int = 512
DEFAULT_N_MELS: int = 64
STEP_LENGTH: float = 1.0  # the seconds of overlay with previous spectrograms
SPECTROGRAM_WIDTH: float = 2.0  # how many seconds each spectrogram represents
NUM_FOLDS: int = 5
SNIPPET_LENGTH: float = 8.0  # number of seconds of each input
DEFAULT_WINDOW_NUM_SAMPLES: int = int(
    DEFAULT_SAMPLE_RATE * SPECTROGRAM_WIDTH
)  # 2 seconds per window
DEFAULT_STEP_NUM_SAMPLES: int = int(
    DEFAULT_SAMPLE_RATE * STEP_LENGTH
)  # 1 second overlay between windows
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_MEL_SPECTROGRAM = MelSpectrogram(
    sample_rate=DEFAULT_SAMPLE_RATE,
    n_fft=DEFAULT_N_FFT,
    hop_length=DEFAULT_HOP_LENGTH,
    n_mels=DEFAULT_N_MELS,
).to(DEVICE)

# CNN hyperparameters
DEFAULT_STRIDE: int = 1
DEFAULT_PADDING: int = 1
DEFAULT_POOL_KERNEL_SIZE: int = 2
CONV_LAYER_DIMS: List[Tuple[int, int]] = [
    (1, 16),
    (16, 32),
    (32, 64),
    (64, 128),
    (128, 256),
]
DEFAULT_NUM_BRANCHES: int = 4
DEFAULT_DIVIDE_AND_ENCODE_HIDDEN_DIM: int = 192
DEFAULT_EMBEDDING_DIM: int = 96
# note: this batch size is the *number of songs sampled*, not the number of spectrograms the
# networks trains on at a time
DEFAULT_BATCH_SIZE: int = 10
DEFAULT_LEARNING_RATE: float = 0.005
DEFAULT_NUM_EPOCHS: int = 10
DEFAULT_ALPHA: float = 0.2
DEFAULT_TEST_SPLIT: float = 0.25


TOR_BROWSER_DIR: Path = Path("/home/james/tor-browser-linux-x86_64-14.5.3/tor-browser/")
TEMP_TOR_DATA_DIR: Path = Path("/home/james/code/sample-hunter/temp_tor_data/")
TOR_PASSWORD: str = os.environ["TOR_PASSWORD"]


PARENT_SITEMAP_URL: str = "https://www.hiphopisread.com/sitemap.xml"
SITEMAP_SAVE_PATH: Path = Path("_data/sitemaps/")
DATA_SAVE_DIR: Path = Path("_data/")
TEMP_DIR: Path = Path(DATA_SAVE_DIR / "tmp/")
ZIP_ARCHIVE_DIR: Path = Path("_data/archive/")

DEFAULT_REQUEST_TIMEOUT: float = 15.0
DEFAULT_DOWNLOAD_TIME: float = 2700.0
DEFAULT_RETRIES: int = 5
DEFAULT_RETRY_DELAY: float = 5.0
CACHE_DIR: Path = Path(DATA_SAVE_DIR / "cache")
THREADS: int = 1
PROCS: int = multiprocessing.cpu_count()

ANNOTATIONS_PATH: Path = Path(DATA_SAVE_DIR / "new_annotations.csv")
AUDIO_DIR: Path = Path(DATA_SAVE_DIR / "audio-dir/")
MODEL_SAVE_PATH: Path = Path(DATA_SAVE_DIR / "test.pth")
TRAIN_LOG_DIR: Path = Path("_data/logs")
HF_DATASET_REPO_ID: str = "samplr/songs"
HF_DATASET_URL: str = "https://huggingface.co/datasets/samplr/songs"
HF_TOKEN: str = os.environ.get("HF_TOKEN", "")


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
