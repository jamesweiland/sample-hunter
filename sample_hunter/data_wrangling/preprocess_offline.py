"""
Preprocess all songs in samplr/songs (or the local dataset) into pairs of spectrograms, where one is the anchor and
one is the positive, and upload to samplr/specs.

Config used for most recent preprocessing: preprocess_8_6_2025.yaml
"""

import argparse
import webdataset as wds

from pathlib import Path

from sample_hunter.pipeline.data_loading import load_webdataset
from sample_hunter.config import PreprocessConfig, ObfuscatorConfig
from sample_hunter._util import HF_TOKEN, DEVICE

DEFAULT_SHARD_SIZE: int = int(1e9)


def upload_split(
    split: wds.WebDataset,
    split_name: str,
    preprocess_config: PreprocessConfig,
    obfuscator_config: ObfuscatorConfig,
    repo_id: str,
    token: str,
    procs: int | None = None,
    threads: int | None = None,
    shard_size: int = DEFAULT_SHARD_SIZE,
    device: str = DEVICE,
):
    """
    Upload a dataset split to `repo_id` under the `split_name` directory.
    """

    if device == "cpu":
        # use multiprocessing to preprocess songs in parallel

        # a single thread on the main process will periodically check a queue for preprocessed songs and add
        # them to a tar shard, and if the shard's size > shard_size, upload the shard to HF and start a new one


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        help="path to the yaml file to use for configuration,"
        "should have preprocessor and obfuscator config params set.",
    )

    parser.add_argument(
        "--shardsize, -s",
        type=int,
        help="Number of bytes to make each tar shard in the resultant webdataset",
        default=DEFAULT_SHARD_SIZE,
    )

    parser.add_argument(
        "--token",
        type=str,
        help="Your HF token to download and push to/from repos",
        default=HF_TOKEN,
    )

    parser.add_argument(
        "source",
        type=str,
        help="Either a HF repo id or a path to a local folder,"
        "containing the webdataset shards of songs to preprocess into spectrograms.",
    )

    parser.add_argument(
        "target", type=str, help="The hf repo to save the preprocessed spectrograms to"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if Path(args.source).exists():
        args.source = Path(args.source)
        train_dir = args.source / "train"
        test_dir = args.source / "test"

        train_split = wds.WebDataset(train_dir)
        test_split = wds.WebDataset(test_dir)

    else:
        d = load_webdataset(args.source, ["train", "test"], args.token)
        train_split = d["train"]
        test_split = d["test"]

    upload_split(train_split, preprocess_config, obfuscator_config)
