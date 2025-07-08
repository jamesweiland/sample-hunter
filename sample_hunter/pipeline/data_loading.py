"""
Utility functions for loading in the webdataset.
"""

import torch
from typing import Dict, List, Tuple, Generator
from huggingface_hub import HfApi
import webdataset as wds
import re

from sample_hunter._util import config


def flatten_sub_batches(
    dataloader: torch.utils.data.DataLoader,
) -> Generator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], None, None]:
    """
    A generator to wrap around a torch dataloader with a collate function that
    returns a list of tensors. This yields the tensors in the list, one at a time.
    This expects the dataloader to yield a list of tuples
    """
    dataloader_iter = iter(dataloader)
    while True:
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            break  # End of dataloader
        except Exception as e:
            print("An error occurred while fetching a batch from the dataloader")
            print(str(e))
            continue

        for sub_batch in batch:
            yield sub_batch


def collate_spectrograms(
    batch: List[Dict[str, torch.Tensor]],
    col: str | List[str],
    sub_batch_size: int = config.network.sub_batch_size,
) -> Tuple[torch.Tensor] | List[Tuple[torch.Tensor, ...]]:
    """
    Collate a batch of mappings of transformed tensors before passing to the dataloader.

    This function expects tensors with shape (batch_size, num_windows, num_channels, n_mels, time_frames)
    and returns a tensor with shape (new_batch_size, num_channels, n_mels, time_frames)
    """

    if isinstance(col, str):
        full_tensor = torch.cat([example[col] for example in batch], dim=0)
        perm = torch.randperm(full_tensor.shape[0])
        shuffled = full_tensor[perm]

        sub_batches = shuffled.split(sub_batch_size)
    else:
        full_tensors = [
            torch.cat([example[name] for example in batch], dim=0) for name in col
        ]
        perm = torch.randperm(full_tensors[0].shape[0])
        shuffled = [t[perm] for t in full_tensors]

        sub_batches = (t.split(sub_batch_size) for t in shuffled)

    return list(zip(*sub_batches))


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


def load_webdataset(repo_id: str, split: str, token: str) -> wds.WebDataset:
    """load a webdataset of a split containing tarfiles like {split}-{i:0nd}.tar, where n is some
    arbitary 0 padding, for all i found in the split."""
    tar_files = get_tar_files(repo_id, split, token)
    numbers, padding = extract_numbers_and_padding(tar_files, split)
    assert padding is not None
    min_num_str = str(min(numbers)).zfill(padding)
    max_num_str = str(max(numbers)).zfill(padding)
    pattern = f"{split}-{{{min_num_str}..{max_num_str}}}.tar"

    url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{split}/{pattern}"
    pipe = f"pipe:curl -s -L {url} -H 'Authorization:Bearer {token}'"
    return wds.WebDataset(pipe, shardshuffle=True).shuffle(200).decode()
