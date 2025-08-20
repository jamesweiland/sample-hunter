from typing_extensions import Literal
import torchaudio
from functools import cached_property
from dataclasses import InitVar, dataclass, fields, field, replace, asdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, TypeVar, Type, Union, cast
import yaml
from transformers.configuration_utils import PretrainedConfig
from abc import ABC


DEFAULT_N_FFT: int = 1024
DEFAULT_HOP_LENGTH: int = 512
DEFAULT_N_MELS: int = 64
DEFAULT_SPEC_NUM_SEC: float = 1.0
DEFAULT_STEP_NUM_SEC: float = 0.5
DEFAULT_SAMPLE_RATE: int = 44_100
DEFAULT_EMBEDDING_DIM: int = 128
DEFAULT_TRIPLET_LOSS_MARGIN: float = 0.2
DEFAULT_MINE_STRATEGY: Literal["semi", "hard"] = "hard"
DEFAULT_TOP_K: int = 20
DEFAULT_VOLUME_THRESHOLD: int = -60  # dbfs, remove anything below this

DEFAULT_DATASET_REPO: str = "samplr/specs"
DEFAULT_CACHE_DIR: Path = Path("/content/drive/MyDrive/sample-hunter/cache")

T = TypeVar("T", bound="YAMLConfig")


@dataclass
class YAMLConfig(ABC):
    @classmethod
    def from_yaml(cls: Type[T], yaml_: Union[Path, str]) -> T:
        """
        Load config from a YAML file and instantiate the config dataclass.
        """

        with open(yaml_, "r") as f:
            cfg = yaml.safe_load(f)

        init_kwargs = {}
        for field in fields(cls):
            field_name = field.name
            if field_name in cfg:
                init_kwargs[field_name] = cfg[field_name]
            else:
                init_kwargs[field_name] = field.default

        return cls(**init_kwargs)

    def merge_kwargs(self, **kwargs):
        """
        Given kwargs, return a new instance of `cls` with updated attributes in kwargs.
        If a field is not in kwargs but is in this instance, the original instance's attribute will be used.
        """
        if kwargs:
            # kwargs might have keys that aren't attributes of the instance, so need to filter kwargs
            valid_fields = {f.name for f in fields(self)}
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_fields}
            return replace(self, **filtered_kwargs)
        return self

    def to_dict(self) -> Dict[str, Any]:
        """
        Returns the attributes as a python dictionary
        """
        return asdict(self)


@dataclass
class EncoderNetConfig(PretrainedConfig, YAMLConfig):
    """Stores the necessary hyperparameters for instantiating the model architecture."""

    kernel_stride: int = 1
    kernel_padding: int = 1
    pool_kernel_size: int = 2
    conv_layer_dims: List[Tuple[int, int]] = field(
        default_factory=lambda: [
            (1, 16),
            (16, 32),
            (32, 64),
            (64, 128),
            (128, 256),
            (256, 384),
            (384, 512),
        ]
    )
    min_dims: Tuple[int, int] = (1, 1)
    num_branches: int = 4
    divide_and_encode_hidden_dim: int = 256
    embedding_dim: int = DEFAULT_EMBEDDING_DIM
    sample_rate: int = 44100
    spec_num_sec: float = DEFAULT_SPEC_NUM_SEC
    n_mels: int = DEFAULT_N_MELS
    hop_length: int = DEFAULT_HOP_LENGTH

    extra_kwargs: InitVar[dict | None] = None  # extra kwargs for PretrainedConfig

    def __post__init__(self, extra_kwargs):
        """Initialize with args for PretrainedConfig"""
        if extra_kwargs is not None:
            super().__init__(**extra_kwargs)
        else:
            super().__init__()


@dataclass
class ObfuscatorConfig(YAMLConfig):
    time_stretch_factors: Sequence[float] = field(
        default_factory=lambda: (0.75, 1.0, 1.25, 1.5)
    )
    pitch_factors: Sequence[float] = field(
        default_factory=lambda: (0.6, 0.8, 1.0, 1.2, 1.5)
    )
    lowpass_range: Tuple[int, int] = (6_000, 12_000)
    highpass_range: Tuple[int, int] = (20, 1000)
    musan_noise_range: Tuple[float, float] = (15, 30)
    offset_span: float = (
        0.0  # offset each sample randomly from -offset_span seconds to offset_span seconds
    )
    lowpass_frac: float = 0.5
    musan: Path = Path("/home/james/code/sample-hunter/_data/musan")
    sample_rate: int = DEFAULT_SAMPLE_RATE
    n_fft: int = DEFAULT_N_FFT
    hop_length: int = DEFAULT_HOP_LENGTH
    spec_num_sec: float = DEFAULT_SPEC_NUM_SEC
    step_num_sec: float = DEFAULT_STEP_NUM_SEC
    volume_threshold: int = DEFAULT_VOLUME_THRESHOLD
    perturb_num_workers: int = 1

    @property
    def offset_span_num_samples(self) -> int:
        return int(self.sample_rate * self.offset_span)

    @property
    def spec_num_samples(self) -> int:
        return int(self.sample_rate * self.spec_num_sec)

    @property
    def step_num_samples(self) -> int:
        return int(self.sample_rate * self.step_num_sec)


@dataclass
class TrainConfig(YAMLConfig):
    batch_size: int = 2000
    learning_rate: float = 0.005
    num_epochs: int = 10
    triplet_loss_margin: float = DEFAULT_TRIPLET_LOSS_MARGIN
    tensorboard_log_dir: Path = Path("/home/james/code/sample-hunter/_data/logs")
    tensorboard: str = "epoch"  # can be "none", "batch", or "epoch"
    cache_dir: Path | None = None
    mine_strategy: Literal["semi", "hard"] = DEFAULT_MINE_STRATEGY
    num_threads: int = 2


@dataclass
class PreprocessConfig(YAMLConfig):
    sample_rate: int = DEFAULT_SAMPLE_RATE
    n_fft: int = DEFAULT_N_FFT
    hop_length: int = DEFAULT_HOP_LENGTH
    n_mels: int = DEFAULT_N_MELS
    step_num_sec: float = DEFAULT_STEP_NUM_SEC
    spec_num_sec: float = DEFAULT_SPEC_NUM_SEC
    volume_threshold: int = DEFAULT_VOLUME_THRESHOLD  # dB

    @property
    def step_num_samples(self) -> int:
        return int(self.sample_rate * self.step_num_sec)

    @property
    def spec_num_samples(self) -> int:
        return int(self.sample_rate * self.spec_num_sec)

    @cached_property
    def mel_spectrogram(self) -> torchaudio.transforms.MelSpectrogram:
        return torchaudio.transforms.MelSpectrogram(
            n_mels=self.n_mels,
            sample_rate=self.sample_rate,
            hop_length=self.hop_length,
            n_fft=self.n_fft,
        )


@dataclass
class PostprocessConfig(YAMLConfig):
    alpha: float = 9.0
    top_k: int = DEFAULT_TOP_K
    sample_rate: int = DEFAULT_SAMPLE_RATE


@dataclass
class FunkyFinderPipelineConfig(YAMLConfig):
    preprocess: PreprocessConfig = field(default_factory=lambda: PreprocessConfig())
    postprocess: PostprocessConfig = field(default_factory=lambda: PostprocessConfig())

    extra_kwargs: InitVar[dict | None] = None

    def __post_init__(self, extra_kwargs):
        if extra_kwargs is not None:
            self.preprocess = self.preprocess.merge_kwargs(**extra_kwargs)
            self.postprocess = self.postprocess.merge_kwargs(**extra_kwargs)

    def to_dict(self) -> Dict[str, Any]:
        to_return = asdict(self.preprocess)
        to_return.update(asdict(self.postprocess))
        return to_return
