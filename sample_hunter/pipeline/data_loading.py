"""
Utility functions for loading in the webdataset.
"""

from collections.abc import Buffer
import io
import math
from pathlib import Path
import torch
from typing import Dict, List, Tuple, Generator
from huggingface_hub import HfApi
from tqdm.notebook import tqdm
import torchaudio
import webdataset as wds
import re
import traceback
from tqdm.notebook import tqdm

from sample_hunter._util import HF_TOKEN, DEVICE
from sample_hunter.config import DEFAULT_CACHE_DIR


def load_tensor_from_bytes(
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
                pbar.update()
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
    tar_files = [
        file for file in files if file.startswith(f"{split}/") and file.endswith(".tar")
    ]
    return tar_files


def extract_numbers_and_padding(tar_files: List[str], split: str):
    """thx chat"""
    numbers = []
    padding = None
    for name in tar_files:
        match = re.search(rf"{split}-(\d+)\.tar$", name)
        if match:
            num_str = match.group(1)
            numbers.append(int(num_str))
            if padding is None:
                padding = len(num_str)
    return numbers, padding


def build_pipe(repo_id: str, split: str, token: str = HF_TOKEN) -> str:
    tar_files = get_tar_files(repo_id, split, token)
    numbers, padding = extract_numbers_and_padding(tar_files, split)
    assert padding is not None
    min_num_str = str(min(numbers)).zfill(padding)
    max_num_str = str(max(numbers)).zfill(padding)
    pattern = f"{split}-{{{min_num_str}..{max_num_str}}}.tar"

    url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{split}/{pattern}"
    pipe = f"pipe:curl -s -L {url} -H 'Authorization:Bearer {token}'"

    return pipe


def load_webdataset(
    repo_id: str,
    split: str | List[str],
    token: str = HF_TOKEN,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> wds.WebDataset | Dict[str, wds.WebDataset]:
    """load a webdataset of a split containing tarfiles like {split}-{i:0nd}.tar, where n is some
    arbitary 0 padding, for all i found in the split."""
    if isinstance(split, str):
        pipe = build_pipe(repo_id, split, token=token)
        return (
            wds.WebDataset(pipe, shardshuffle=100, cache_dir=cache_dir)
            .shuffle(200)
            .decode()
        )
    else:
        # there are multiple splits, and we'll return a dict of datasets
        datasets = {}
        for s in split:
            pipe = build_pipe(repo_id, s, token=token)
            dataset = (
                wds.WebDataset(pipe, shardshuffle=100, cache_dir=cache_dir)
                .shuffle(200)
                .decode()
            )
            datasets.update({s: dataset})
        return datasets
