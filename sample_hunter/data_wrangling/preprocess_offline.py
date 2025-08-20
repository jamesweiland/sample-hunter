"""
Preprocess all songs in samplr/songs (or the local dataset) into pairs of spectrograms, where one is the anchor and
one is the positive, and upload to samplr/specs.

Config used for most recent preprocessing: preprocess_8_6_2025.yaml
"""

import itertools
import traceback
from typing import Any, Dict
import torch
import argparse
import io
import json
import tarfile
import sys
import threading
import queue
import webdataset as wds
import concurrent.futures
import torch.multiprocessing as mp

from huggingface_hub import HfApi
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from tqdm import tqdm
from uuid import uuid4
from pathlib import Path

from sample_hunter.pipeline.data_loading import (
    load_tensor_from_mp3_bytes,
    load_webdataset,
)
from sample_hunter.config import PreprocessConfig, ObfuscatorConfig
from sample_hunter._util import HF_TOKEN, DEVICE
from sample_hunter.pipeline.transformations.my_musan import set_global_locks
from sample_hunter.pipeline.transformations.obfuscator import Obfuscator
from sample_hunter.pipeline.transformations.preprocessor import Preprocessor

DEFAULT_SHARD_SIZE: int = int(1e9)
DEFAULT_PROCS: int = 8
DEFAULT_THREADS: int = 10


def _fetch_tar_and_upload(q: queue.Queue, target_repo: str, split: str, token: str):
    api = HfApi(token=token)

    while True:
        try:
            shard = q.get(timeout=600.0)  # 10 minute timeout

            if shard is None:
                break

            path_in_repo = f"{split}/{str(uuid4())}.tar"

            api.upload_file(
                path_or_fileobj=shard,
                repo_id=target_repo,
                repo_type="dataset",
                path_in_repo=path_in_repo,
            )
        except queue.Empty:
            print("timed out waiting for a shard")
            raise

        except Exception:
            raise


def write_tensor_to_tar(
    tar: tarfile.TarFile, tensor: torch.Tensor, arcname: str
) -> int:
    """
    Add the tensor as a .pth file to the tarfile by serializing it in-memory.

    arcname is what the name of the file will be in the tarfile.

    Returns the size of the .pth file that was added to the tarfile
    """

    # serialize to .pth format in-memory
    buffer = io.BytesIO()
    torch.save(tensor, buffer)

    # get bytes of the serialized .pth
    buffer.seek(0)

    info = tarfile.TarInfo(arcname)
    info.size = buffer.getbuffer().nbytes

    tar.addfile(info, buffer)

    return info.size


def write_json_to_tar(tar: tarfile.TarFile, json_: bytes | Dict, arcname: str) -> int:
    """
    Add a json-serialized bytes in-memory.

    Returns the size of the .pth file that was added to the tarfile
    """

    if isinstance(json_, dict):
        json_ = json.dumps(json_).encode("utf-8")

    info = tarfile.TarInfo(arcname)
    info.size = len(json_)

    tar.addfile(info, io.BytesIO(json_))

    return info.size


def _init_worker(
    func,
    preprocess_config: PreprocessConfig,
    obfuscator_config: ObfuscatorConfig,
    device: str,
    num_threads_per_worker: int | None = None,
):
    """Initialize each worker with it's own preprocessor"""
    func.preprocessor = Preprocessor(
        preprocess_config, obfuscator=Obfuscator(obfuscator_config)
    )
    func.device = device

    if device == "cpu":
        assert num_threads_per_worker is not None
        torch.set_num_threads(num_threads_per_worker)


def add_future_result_to_tar(
    future: concurrent.futures.Future,
    tar: tarfile.TarFile | None,
    tar_buf: io.BytesIO | None,
    tar_result_queue: queue.Queue,
    archive_size: int,
    shard_size: int,
):
    try:
        result = future.result()
    except Exception:
        print("an exception occured trying to access the result of a future")
        traceback.print_exc()
        raise

    anchor, positive, metadata = result["anchor"], result["positive"], result["json"]
    anchor = anchor.to("cpu")
    positive = positive.to("cpu")

    result_size = anchor.nbytes + positive.nbytes + len(metadata)

    metadata = json.loads(metadata.decode("utf-8"))

    if tar is None or (archive_size + result_size) > shard_size:
        if tar is not None:
            assert tar_buf is not None

            tar.close()
            tar_buf.seek(0)
            tar_result_queue.put(tar_buf)
            archive_size = 0

        tar_buf = io.BytesIO()
        tar = tarfile.open(fileobj=tar_buf, mode="w")

    example_id = metadata["example_id"]
    anchor_name = f"{example_id}.anchor.tar"
    positive_name = f"{example_id}.positive.tar"
    json_name = f"{example_id}.json"

    # add tensors to tar
    archive_size += write_tensor_to_tar(tar, anchor, anchor_name)
    archive_size += write_tensor_to_tar(tar, positive, positive_name)
    archive_size += write_json_to_tar(tar, metadata, json_name)
    return tar, tar_buf, archive_size


def preprocess(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Preprocess the example into a preprocessed result with keys 'anchor', 'positive', and 'json'
    """
    try:
        with torch.no_grad():
            with preprocess.preprocessor as preprocessor:  # type: ignore
                audio, sr = load_tensor_from_mp3_bytes(example["mp3"], preprocess.device)  # type: ignore

                positive, anchor = preprocessor(audio, sample_rate=sr, train=True)

            # create metadata file and encode to bytes
            metadata = {
                "song_id": example["json"]["id"],
                "example_id": str(uuid4()),
            }
            metadata = json.dumps(metadata).encode("utf-8")

            return {"anchor": anchor, "positive": positive, "json": metadata}
    except Exception:
        print("an error occured trying to preprocess a song")
        traceback.print_exc(file=sys.stderr)
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

    # a single thread on the main process will periodically check a queue for preprocessed songs and add
    # them to a tar shard, and if the shard's size > shard_size, upload the shard to HF and start a new one

    tar_result_queue = queue.Queue()
    tar = None
    tar_buf = None
    archive_size = 0
    if procs is not None:
        n = max(2 * procs, 16)  # size of task chunks to hold in memory at once
    elif threads is not None:
        n = max(2 * threads, 16)
    else:
        n = 16

    uploader = threading.Thread(
        target=_fetch_tar_and_upload,
        args=(tar_result_queue, target_repo, split_name, token),
    )
    uploader.start()

    with tqdm(desc="Preprocessing songs", total=1800) as pbar:

        if device == "cpu":
            assert procs is not None
            # use multiprocessing to preprocess songs in parallel
            num_threads_per_process = mp.cpu_count() // procs

            # set up some mp stuff
            manager = mp.Manager()
            locks = manager.dict()
            set_global_locks(locks, manager)

            with ProcessPoolExecutor(
                max_workers=procs,
                initializer=_init_worker,
                initargs=(
                    preprocess,
                    preprocess_config,
                    obfuscator_config,
                    device,
                    num_threads_per_process,
                ),
            ) as executor:

                # schedule the first n futures
                futures = {
                    executor.submit(preprocess, task)
                    for task in itertools.islice(split, n)
                }

                while futures:
                    done, futures = concurrent.futures.wait(
                        futures, return_when=concurrent.futures.FIRST_COMPLETED
                    )

                    for fut in done:
                        tar, tar_buf, archive_size = add_future_result_to_tar(
                            fut,
                            tar,
                            tar_buf,
                            tar_result_queue,
                            archive_size,
                            shard_size,
                        )
                        pbar.update(1)

                    # schedule the next set of futures
                    for task in itertools.islice(split, len(done)):
                        futures.add(executor.submit(preprocess, task))

        elif device == "cuda":
            assert threads is not None

            with ThreadPoolExecutor(
                max_workers=threads,
                initializer=_init_worker,
                initargs=(preprocess, preprocess_config, obfuscator_config, device),
            ) as executor:

                futures = {
                    executor.submit(preprocess, task)
                    for task in itertools.islice(split, n)
                }

                while futures:
                    done, futures = concurrent.futures.wait(
                        futures, return_when=concurrent.futures.FIRST_COMPLETED
                    )

                    for fut in done:
                        tar, tar_buf, archive_size = add_future_result_to_tar(
                            fut,
                            tar,
                            tar_buf,
                            tar_result_queue,
                            archive_size,
                            shard_size,
                        )
                        pbar.update(1)

                    # schedule the next set of futures
                    for task in itertools.islice(split, len(done)):
                        futures.add(executor.submit(preprocess, task))

        else:
            raise NotImplementedError("only supports cpu and cuda")

        # sentinel to end uploader
        tar_result_queue.put(None)
        uploader.join()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        help="path to the yaml file to use for configuration,"
        "should have preprocessor and obfuscator config params set.",
    )

    parser.add_argument(
        "--shardsize",
        "-s",
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
        "--procs",
        type=int,
        help="number of processes to run if using multiprocessing on cpu",
        default=DEFAULT_PROCS,
    )

    parser.add_argument(
        "--threads",
        type=int,
        help="number of threads to use if using multithreading on cuda",
        default=DEFAULT_THREADS,
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
    mp.set_start_method("spawn")
    args = parse_args()

    if Path(args.source).exists():
        args.source = Path(args.source)
        train_dir = args.source / "train"
        test_dir = args.source / "test"

        train_tars = [str(tar) for tar in train_dir.glob("*.tar")]
        test_tars = [str(tar) for tar in test_dir.glob("*.tar")]

        train_split = (
            wds.WebDataset(train_tars, shardshuffle=len(train_tars))
            .shuffle(200)
            .decode()
        )
        test_split = (
            wds.WebDataset(test_tars, shardshuffle=len(test_tars)).shuffle(200).decode()
        )

    else:
        d = load_webdataset(args.source, ["train", "test"], args.token)
        train_split = d["train"]
        test_split = d["test"]

    if args.config:
        preprocess_config = PreprocessConfig.from_yaml(args.config)
        obfuscator_config = ObfuscatorConfig.from_yaml(args.config)
    else:
        preprocess_config = PreprocessConfig()
        obfuscator_config = ObfuscatorConfig()

    upload_split(
        split=train_split,
        split_name="train",
        preprocess_config=preprocess_config,
        obfuscator_config=obfuscator_config,
        target_repo=args.target,
        token=args.token,
        procs=args.procs,
        threads=args.threads,
        shard_size=args.shardsize,
    )
