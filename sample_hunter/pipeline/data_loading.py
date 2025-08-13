"""
Utility functions for loading in the webdataset.
"""

from collections.abc import Buffer
import io
import math
from pathlib import Path
import random
import torch
from typing import Dict, List, Tuple, Generator
from huggingface_hub import HfApi
from tqdm import tqdm
import torchaudio
import webdataset as wds
import traceback
from tqdm import tqdm
import fnmatch

from sample_hunter._util import HF_TOKEN, DEVICE
from sample_hunter.config import DEFAULT_CACHE_DIR


def load_tensor_from_pth_bytes(
    initial_bytes: Buffer, device: str = DEVICE
) -> torch.Tensor:
    with io.BytesIO(initial_bytes) as buffer:
        tensor = torch.load(buffer, map_location=device)
    return tensor


def load_tensor_from_mp3_bytes(
    initial_bytes: Buffer, device: str = DEVICE
) -> Tuple[torch.Tensor, int]:
    with io.BytesIO(initial_bytes) as buffer:
        audio, sr = torchaudio.load(buffer, format="mp3", backend="ffmpeg")
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)

    audio = audio.to(device)
    return audio, sr


def flatten_sub_batches(
    dataloader: torch.utils.data.DataLoader,
) -> Generator[Tuple[torch.Tensor, ...], None, None]:
    """
    A generator to wrap around a torch dataloader with a collate function that
    returns a list of tensors. This yields the tensors in the list, one at a time.
    This expects the dataloader to yield a list of tuples
    """
    with tqdm(desc="Iterating through sub-batches...") as pbar:
        dataloader_iter = iter(dataloader)
        pbar.update()
        while True:
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                break  # End of dataloader
            except Exception as e:
                print("An error occurred while fetching a batch from the dataloader")
                traceback.print_exc()
                continue

            for sub_batch in batch:
                yield sub_batch


def collate_spectrograms(
    batch: torch.Tensor | Tuple[torch.Tensor, ...],
    sub_batch_size: int,
    shuffle: bool = True,
) -> Tuple[torch.Tensor, ...] | List[Tuple[torch.Tensor, ...]]:
    """
    Collate a batch of mappings of transformed tensors before passing to the dataloader.

    This function expects tensors with shape (source_batch_size, num_windows, num_channels, n_mels, time_frames)
    and returns a tensor with shape (sub_batch_size, num_channels, n_mels, time_frames)

    source_batch_size is the batch_size given to the dataloader
    """
    if isinstance(batch, torch.Tensor):
        if shuffle:
            perm = torch.randperm(batch.shape[0])
            shuffled = batch[perm]
            chunks = math.ceil(batch.shape[0] / sub_batch_size)
            sub_batches = torch.chunk(shuffled, chunks)
        else:
            chunks = math.ceil(batch.shape[0] / sub_batch_size)
            sub_batches = torch.chunk(batch, chunks)
        return tuple(sub_batches)

    elif isinstance(batch, tuple):

        if shuffle:
            perm = torch.randperm(batch[0].shape[0])
            shuffled = [t[perm] for t in batch]
            chunks = math.ceil(batch[0].shape[0] / sub_batch_size)
            sub_batches = [torch.chunk(t, chunks) for t in shuffled]
        else:
            chunks = math.ceil(batch[0].shape[0] / sub_batch_size)
            sub_batches = [torch.chunk(t, chunks) for t in batch]
        return list(zip(*sub_batches))

    raise ValueError(f"Unsupported type for batch: {type(batch)}")


def get_tar_files(repo_id: str, split: str, token: str) -> List[str]:
    api = HfApi()
    files = api.list_repo_files(repo_id, repo_type="dataset", token=token)
    tar_files = fnmatch.filter(files, f"{split}/*.tar")
    return tar_files


def build_pipes(repo_id: str, split: str, token: str = HF_TOKEN) -> List[str]:
    tar_files = get_tar_files(repo_id, split, token)

    urls = [
        f"https://huggingface.co/datasets/{repo_id}/resolve/main/{tar}"
        for tar in tar_files
    ]
    pipes = [f"pipe:curl -s -L {url} -H 'Authorization:Bearer {token}'" for url in urls]

    return pipes


def reshuffle_batches(dataset: wds.WebDataset, buffersize: int = 2000):
    """
    Collect samples from dataset and reshuffle across a larger buffer.

    Since webdataset only shuffles on the shard level and within the shards
    themselves, but doesn't allow inter-shard shuffling, this shuffles examples
    from different shards.
    """

    buffer = []
    for example in dataset:
        buffer.append(example)

        if len(buffer) >= buffersize:
            random.shuffle(buffer)
            yield from buffer[: len(buffer) // 2]  # yield half, keep half for mixing
            buffer = buffer[len(buffer) // 2 :]

    # yield remaining samples
    random.shuffle(buffer)
    yield from buffer


def load_webdataset(
    repo_id: str,
    split: str | List[str],
    token: str = HF_TOKEN,
    cache_dir: Path | None = None,
) -> wds.WebDataset | Dict[str, wds.WebDataset]:
    """load a webdataset of a split containing tarfiles like {split}-{i:0nd}.tar, where n is some
    arbitary 0 padding, for all i found in the split."""
    if isinstance(split, str):
        pipes = build_pipes(repo_id, split, token=token)
        return wds.WebDataset(
            pipes, shardshuffle=len(pipes), cache_dir=cache_dir
        ).decode()
    else:
        # there are multiple splits, and we'll return a dict of datasets
        datasets = {}
        for s in split:
            pipes = build_pipes(repo_id, s, token=token)
            dataset = wds.WebDataset(
                pipes, shardshuffle=len(pipes), cache_dir=cache_dir
            ).decode()
            datasets.update({s: dataset})
        return datasets
