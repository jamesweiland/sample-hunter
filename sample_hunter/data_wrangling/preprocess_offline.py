"""
Preprocess all songs in samplr/songs (or the local dataset) into pairs of spectrograms, where one is the anchor and
one is the positive, and upload to samplr/specs.

Config used for most recent preprocessing: preprocess_8_6_2025.yaml
"""

import argparse
import tarfile
import time
import threading
import queue
import webdataset as wds

from huggingface_hub import HfApi
from concurrent.futures import ProcessPoolExecutor, as_completed
from uuid import uuid4
from pathlib import Path

from sample_hunter.pipeline.data_loading import load_webdataset
from sample_hunter.config import PreprocessConfig, ObfuscatorConfig
from sample_hunter._util import HF_TOKEN, DEVICE

DEFAULT_SHARD_SIZE: int = int(1e9)


def _fetch_tar_and_upload(
    q: queue.Queue, target_repo: str, split: str, token: str, max_retries: int = 3
):
    api = HfApi(token=token)

    retry_count = 0
    while True:
        try:
            shard = q.get(timeout=60)

            if shard is None:
                break

            path_in_repo = f"{split}/{str(uuid4())}.tar"

            api.upload_file(
                path_or_fileobj=shard,
                repo_id=target_repo,
                repo_type="dataset",
                path_in_repo=path_in_repo,
            )

            retry_count = 0
        except queue.Empty:
            if retry_count >= max_retries:
                print("Max retries exceeded waiting for tarfile")
                raise

            print(
                "Fetcher timed out waiting for a shard. Waiting 1 minute then retrying..."
            )
            retry_count += 1
            time.sleep(60)

        except Exception:
            raise


def upload_split(
    split: wds.WebDataset,
    split_name: str,
    preprocess_config: PreprocessConfig,
    obfuscator_config: ObfuscatorConfig,
    target_repo: str,
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
        assert procs is not None
        # use multiprocessing to preprocess songs in parallel

        # a single thread on the main process will periodically check a queue for preprocessed songs and add
        # them to a tar shard, and if the shard's size > shard_size, upload the shard to HF and start a new one

        q = queue.Queue()
        tar = None
        archive_size = 0

        uploader = threading.Thread(
            target=_fetch_tar_and_upload, args=(q, target_repo, split_name, token)
        )
        uploader.start()

        with ProcessPoolExecutor(max_workers=procs) as executor:
            futures = []

            for future in as_completed(futures):
                result = future.result()

                result_size = (
                    len(result["anchor"])
                    + len(result["positive"])
                    + len(result["json"])
                )

                if tar is None or (archive_size + result_size) > shard_size:
                    if tar is not None:
                        q.put(tar)

                    

        # sentinel to end uploader
        q.put(None)
        q.join()


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
