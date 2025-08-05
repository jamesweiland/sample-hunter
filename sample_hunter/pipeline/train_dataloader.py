import torch
import time
import json
import traceback
import threading
import queue
import webdataset as wds
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.notebook import tqdm
from typing import List, Tuple

from .transformations.preprocessor import Preprocessor
from .transformations.obfuscator import Obfuscator
from .data_loading import collate_spectrograms, load_tensor_from_bytes
from sample_hunter._util import DEVICE
from sample_hunter.config import TrainConfig, PreprocessConfig, ObfuscatorConfig


class TrainDataloaderBuffer:
    def __init__(
        self,
        data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        buffersize: int = 10,
    ):
        print("initializing data buffer")
        self.cpu_queue = queue.Queue()
        self.gpu_queue = queue.Queue(maxsize=buffersize)
        self.buffersize = buffersize

        # push the first buffersize sub batches to the gpu queue, and
        # move the rest to cpu then put on cpu queue

        for i in range(min(buffersize, len(data))):
            sub_batch = data[i]
            self.gpu_queue.put(sub_batch)

        for i in tqdm(range(buffersize, len(data)), desc="moving data to cpu"):
            with torch.no_grad():
                sub_batch = data[i]
                sub_batch_cpu = tuple(t.cpu() for t in sub_batch)
                self.cpu_queue.put(sub_batch_cpu)

                # we need to make sure that memory on the gpu is free'd or otherwise
                # training will be bad
                del sub_batch
                torch.cuda.empty_cache()

        print("all data moved to cpu")

        self.gpu_thread = threading.Thread(target=self._gpu_prefetcher)

        self.stop = False

    def __enter__(self):
        self.gpu_thread.start()
        return self

    def __exit__(self, *exc):
        self.gpu_thread.join()

    def __iter__(self):
        while True:
            sub_batch = self.gpu_queue.get()
            if sub_batch is None:
                break
            yield sub_batch
            time.sleep(0.05)

    def _gpu_prefetcher(self):
        """fetch a sub batch from the cpu if necessary"""
        while self.cpu_queue.not_empty:
            if self.gpu_queue.qsize() < self.buffersize:
                sub_batch = self.cpu_queue.get()

                # move the sub batch to cuda and schedule it by enqueueing
                sub_batch = tuple(t.to("cuda") for t in sub_batch)
                self.gpu_queue.put(sub_batch)

            time.sleep(0.1)
        # when we get here, it means there are no batches left
        self.gpu_queue.put(None)


class TrainDataloader:
    def __init__(
        self,
        dataset: wds.WebDataset,
        config: TrainConfig | None = None,
        preprocess_config: PreprocessConfig | None = None,
        obfuscator_config: ObfuscatorConfig | None = None,
        device: str = DEVICE,
        **kwargs,
    ):
        config = config or TrainConfig()
        config = config.merge_kwargs(**kwargs)
        self.config = config
        self.dataset = dataset
        self.dataset = self.dataset.map(self._map_fn)
        self.device = device
        self._preprocess_config = preprocess_config
        self._obfuscator_config = obfuscator_config
        self._lock = threading.Lock()
        self._queue = queue.Queue()

    def __iter__(self):
        self.batch_num = 1
        dataset_iter = iter(self.dataset)

        while True:
            try:
                batch = self._preprocess_batch(dataset_iter)

                yield from self._collate(batch)

            except Exception:
                print("An error occured collating the batch")
                traceback.print_exc()

    def _preprocess_batch(self, dataset_iter):
        # set up the preprocessors
        preprocessors = [
            Preprocessor(
                self._preprocess_config,
                obfuscator=Obfuscator(self._obfuscator_config),
            ).__enter__()
            for _ in range(self.config.num_threads)
        ]
        preprocessed_examples = []
        try:
            with tqdm(
                total=self.config.source_batch_size,
                desc=f"Processing batch {self.batch_num}...",
                mininterval=0,
                miniters=1,
            ) as pbar:
                with ThreadPoolExecutor(
                    max_workers=self.config.num_threads
                ) as executor:
                    futures = [
                        executor.submit(
                            self._preprocess_example,
                            dataset_iter,
                            preprocessors[i % len(preprocessors)],
                        )
                        for i in range(self.config.source_batch_size)
                    ]

                    for future in as_completed(futures):
                        result = future.result()
                        if result is not None:
                            preprocessed_examples.append(result)
                        pbar.update()

                self.batch_num += 1
                return preprocessed_examples

        except StopIteration:
            # resource cleanup
            for preprocessor in preprocessors:
                preprocessor.__exit__()
                del preprocessor
            torch.cuda.empty_cache()
            return preprocessed_examples

    def _collate(self, batch):
        with torch.no_grad():
            # filter out failed results
            batch = [ex for ex in batch if ex is not None]

            # one-hot encode ids
            unique_ids = [ex["id"] for ex in batch]
            uuid_to_int = {u: i for i, u in enumerate(unique_ids)}
            unique_ids = torch.tensor(
                [uuid_to_int[u] for u in unique_ids],
                device=self.device,
                dtype=torch.int32,
            )
            windows_per_song = torch.tensor(
                [ex["positive"].shape[0] for ex in batch],
                device=self.device,
                dtype=torch.int32,
            )
            ids = torch.repeat_interleave(unique_ids, windows_per_song)
            ids = ids.to(torch.int16)

            positives = torch.cat([ex["positive"] for ex in batch])
            anchors = torch.cat([ex["anchor"] for ex in batch])

            sub_batches = collate_spectrograms(
                (anchors, positives, ids), self.config.sub_batch_size, shuffle=True
            )
            print(sub_batches[0][0].dtype)
            print(sub_batches[0][1].dtype)
            print(sub_batches[0][2].dtype)

        with TrainDataloaderBuffer(sub_batches) as buffer:  # type: ignore
            yield from buffer

    def _preprocess_example(self, dataset_iter, preprocessor: Preprocessor):
        with torch.no_grad():
            example = None
            try:
                with self._lock:
                    example = next(dataset_iter)

                positive, anchor = preprocessor(
                    example["audio_tensor"],
                    sample_rate=example["json"]["sample_rate"],
                    train=True,
                )
                print("preprocessing done")
                return {
                    "positive": positive,
                    "anchor": anchor,
                    "id": example["json"]["id"],
                }

            except StopIteration:
                raise

            except Exception:
                title = (
                    example["json"].get("title", "unknown") if example else "unknown"
                )
                print(f"An error occurred trying to preprocess {title}")
                traceback.print_exc()
                return None

    def _map_fn(self, ex):
        if isinstance(ex["json"], bytes):
            ex["json"] = json.loads(ex["json"].decode("utf-8"))

        ex["json"]["sample_rate"] = int(ex["json"]["sample_rate"])

        audio_tensor, sr = load_tensor_from_bytes(ex["mp3"])
        ex["audio_tensor"] = audio_tensor.to(self.device)
        return ex
