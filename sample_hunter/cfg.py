from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence, Tuple
import yaml


with open("sample_hunter/config.yaml") as f:
    cfg = yaml.safe_load(f)


@dataclass
class ObfuscatorConfig:
    time_stretch_factors: Sequence[float] = field(
        default_factory=lambda: tuple(
            cfg["preprocess"]["obfuscator"]["time_stretch_factors"]
        )
    )
    pitch_factors: Sequence[float] = field(
        default_factory=lambda: tuple(cfg["preprocess"]["obfuscator"]["pitch_factors"])
    )

    lowpass_range: Tuple[int, int] = field(
        default_factory=lambda: tuple(cfg["preprocess"]["obfuscator"]["lowpass_range"])
    )
    highpass_range: Tuple[int, int] = field(
        default_factory=lambda: tuple(cfg["preprocess"]["obfuscator"]["highpass_range"])
    )
    whitenoise_range: Tuple[float, float] = field(
        default_factory=lambda: tuple(
            cfg["preprocess"]["obfuscator"]["whitenoise_range"]
        )
    )
    lowpass_frac: float = cfg["preprocess"]["obfuscator"]["lowpass_frac"]


@dataclass
class PreprocessConfig:
    sample_rate: int = cfg["preprocess"]["sample_rate"]
    n_fft: int = cfg["preprocess"]["n_fft"]
    hop_length: int = cfg["preprocess"]["hop_length"]
    n_mels: int = cfg["preprocess"]["n_mels"]
    step_length: int = cfg["preprocess"]["step_length"]
    spectrogram_width: int = cfg["preprocess"]["spectrogram_width"]
    obfuscator: ObfuscatorConfig = field(default_factory=ObfuscatorConfig)


@dataclass
class NetworkConfig:
    stride: int = cfg["network"]["stride"]
    padding: int = cfg["network"]["padding"]
    pool_kernel_size: int = cfg["network"]["pool_kernel_size"]
    conv_layer_dims: List[Tuple[int, int]] = field(
        default_factory=lambda: [
            tuple(pair) for pair in cfg["network"]["conv_layer_dims"]
        ]
    )
    num_branches: int = cfg["network"]["num_branches"]
    divide_and_encode_hidden_dim: int = cfg["network"]["divide_and_encode_hidden_dim"]
    embedding_dim: int = cfg["network"]["embedding_dim"]
    source_batch_size: int = cfg["network"]["source_batch_size"]
    sub_batch_size: int = cfg["network"]["sub_batch_size"]
    learning_rate: int = cfg["network"]["learning_rate"]
    num_epochs: int = cfg["network"]["num_epochs"]
    alpha: float = cfg["network"]["alpha"]
    test_split: float = cfg["network"]["test_split"]


@dataclass
class PathsConfig:
    log_dir: Path = Path(cfg["paths"]["log_dir"])
    cache_dir: Path = Path(cfg["paths"]["cache_dir"])


@dataclass
class HuggingfaceConfig:
    repo_id: str = cfg["hf"]["repo_id"]
    url: str = cfg["hf"]["url"]


@dataclass
class Config:
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    hf: HuggingfaceConfig = field(default_factory=HuggingfaceConfig)


config = Config()
