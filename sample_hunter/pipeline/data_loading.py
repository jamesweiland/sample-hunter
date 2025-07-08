"""
Utility functions for loading in the webdataset.
"""

from typing import List
from huggingface_hub import HfApi
import webdataset as wds
import re


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
