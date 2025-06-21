from datasets import IterableDataset, Dataset, load_dataset
from typing import Callable, Generator, Dict, Iterator, Any
from torch import Tensor

from sample_hunter.pipeline.transformations.transformations import (
    hf_audio_to_spectrogram,
)


class SongDataset:
    """
    A wrapper around any hf IterableDataset or Dataset (including
    WebDataset) with custom iteration logic for preprocessing
    audio files.
    """

    def __init__(
        self,
        dataset: IterableDataset,
        transformation: Callable[..., Generator[Tensor, None, None]],
        transformation_kwargs: Dict[str, Any],
        transformation_dataset_kwargs: Dict[str, str],
    ):
        """
        Store the dataset and transformation settings.

        Args:
            dataset The dataset to wrap around.

            transformation the transformation to apply to the dataset that yields tensors
            that can be given as input to the model.

            transformation_kwargs: a list of arguments for `transformation` that that map to the name of
            the field in the dataset (for example, if the transformation has an argument for `audio: audio`,
            there should be a field in the dataset called `audio` that can be passed as this argument)
        """
        self.dataset = dataset
        self.transformation = transformation
        self.transformation_kwargs = transformation_kwargs
        self.transformation_dataset_kwargs = transformation_dataset_kwargs

    def __iter__(self) -> Iterator:
        """Return the transformed data"""
        for row in self.dataset:
            print(row)
            kwargs = {
                param: row[key]
                for param, key in self.transformation_dataset_kwargs.items()
            }
            kwargs.update(
                {param: obj for param, obj in self.transformation_kwargs.items()}
            )
            yield from self.transformation(**kwargs)


if __name__ == "__main__":
    ds = load_dataset("samplr/songs", streaming=True, split="train")

    sds = SongDataset(
        dataset=ds,
        transformation=hf_audio_to_spectrogram,
        transformation_kwargs={"target_sampling_rate": 1},
        transformation_dataset_kwargs={"audio": "audio"},
    )

    for ex in sds:
        print("test")
