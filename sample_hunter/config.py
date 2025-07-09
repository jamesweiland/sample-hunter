from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence, Tuple
import yaml

_CONFIG_OBJ = None
_CONFIG_PATH = None


@dataclass
class ObfuscatorConfig:
    time_stretch_factors: Sequence[float]
    pitch_factors: Sequence[float]
    lowpass_range: Tuple[int, int]
    highpass_range: Tuple[int, int]
    whitenoise_range: Tuple[float, float]
    lowpass_frac: float


@dataclass
class PreprocessConfig:
    sample_rate: int
    n_fft: int
    hop_length: int
    n_mels: int
    step_length: int
    spectrogram_width: int
    obfuscator: ObfuscatorConfig


@dataclass
class NetworkConfig:
    stride: int
    padding: int
    pool_kernel_size: int
    conv_layer_dims: List[Tuple[int, int]]
    min_dims: Tuple[int, int]
    num_branches: int
    divide_and_encode_hidden_dim: int
    embedding_dim: int
    source_batch_size: int
    sub_batch_size: int
    learning_rate: int
    num_epochs: int
    alpha: float
    test_split: float


@dataclass
class PathsConfig:
    log_dir: Path
    cache_dir: Path


@dataclass
class HuggingfaceConfig:
    repo_id: str
    url: str


@dataclass
class Config:
    preprocess: PreprocessConfig
    network: NetworkConfig
    paths: PathsConfig
    hf: HuggingfaceConfig


def set_config_path(path: Path):
    global _CONFIG_PATH
    _CONFIG_PATH = path


def get_config() -> Config:
    global _CONFIG_OBJ, _CONFIG_PATH
    if _CONFIG_PATH is None:
        raise RuntimeError(
            "Attempting to get config without calling set_config_path first"
        )
    if _CONFIG_OBJ is None:
        _CONFIG_OBJ = load_config(_CONFIG_PATH)
    return _CONFIG_OBJ


def load_config(config_path: Path) -> Config:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    obfuscator = ObfuscatorConfig(
        time_stretch_factors=tuple(
            cfg["preprocess"]["obfuscator"]["time_stretch_factors"]
        ),
        pitch_factors=tuple(cfg["preprocess"]["obfuscator"]["pitch_factors"]),
        lowpass_range=tuple(cfg["preprocess"]["obfuscator"]["lowpass_range"]),
        highpass_range=tuple(cfg["preprocess"]["obfuscator"]["highpass_range"]),
        whitenoise_range=tuple(cfg["preprocess"]["obfuscator"]["whitenoise_range"]),
        lowpass_frac=cfg["preprocess"]["obfuscator"]["lowpass_frac"],
    )
    preprocess = PreprocessConfig(
        sample_rate=cfg["preprocess"]["sample_rate"],
        n_fft=cfg["preprocess"]["n_fft"],
        hop_length=cfg["preprocess"]["hop_length"],
        n_mels=cfg["preprocess"]["n_mels"],
        step_length=cfg["preprocess"]["step_length"],
        spectrogram_width=cfg["preprocess"]["spectrogram_width"],
        obfuscator=obfuscator,
    )
    network = NetworkConfig(
        stride=cfg["network"]["stride"],
        padding=cfg["network"]["padding"],
        pool_kernel_size=cfg["network"]["pool_kernel_size"],
        conv_layer_dims=[tuple(pair) for pair in cfg["network"]["conv_layer_dims"]],
        min_dims=cfg["network"]["min_dims"],
        num_branches=cfg["network"]["num_branches"],
        divide_and_encode_hidden_dim=cfg["network"]["divide_and_encode_hidden_dim"],
        embedding_dim=cfg["network"]["embedding_dim"],
        source_batch_size=cfg["network"]["source_batch_size"],
        sub_batch_size=cfg["network"]["sub_batch_size"],
        learning_rate=cfg["network"]["learning_rate"],
        num_epochs=cfg["network"]["num_epochs"],
        alpha=cfg["network"]["alpha"],
        test_split=cfg["network"]["test_split"],
    )
    paths = PathsConfig(
        log_dir=Path(cfg["paths"]["log_dir"]),
        cache_dir=Path(cfg["paths"]["cache_dir"]),
    )
    hf = HuggingfaceConfig(
        repo_id=cfg["hf"]["repo_id"],
        url=cfg["hf"]["url"],
    )
    return Config(
        preprocess=preprocess,
        network=network,
        paths=paths,
        hf=hf,
    )
