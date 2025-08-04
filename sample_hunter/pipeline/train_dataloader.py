import torch
import json
import traceback
import threading
import webdataset as wds
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.notebook import tqdm

from .transformations.preprocessor import Preprocessor
from .transformations.obfuscator import Obfuscator
from .data_loading import collate_spectrograms, load_tensor_from_bytes
from sample_hunter._util import DEVICE
from sample_hunter.config import TrainConfig, PreprocessConfig, ObfuscatorConfig


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

    def __iter__(self):
        dataset_iter = iter(self.dataset)
        batch_num = 1

        while True:
            try:
                # set up the preprocessors
                preprocessors = [
                    Preprocessor(
                        self._preprocess_config,
                        obfuscator=Obfuscator(self._obfuscator_config),
                    ).__enter__()
                    for _ in range(self.config.num_threads)
                ]

                preprocessed_examples = []
                with tqdm(
                    total=self.config.source_batch_size,
                    desc=f"Processing batch {batch_num}...",
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

                # clean up preprocessors
                [preprocessor.__exit__() for preprocessor in preprocessors]
                batch_num += 1

                yield from self._collate(preprocessed_examples)
            except StopIteration:
                # this means the dataset is totally exhausted
                break

            except Exception:
                print("An error occured collating the batch")
                traceback.print_exc()

    def _collate(self, batch):
        # filter out failed results
        batch = [ex for ex in batch if ex is not None]

        # one-hot encode ids
        unique_ids = [ex["id"] for ex in batch]
        uuid_to_int = {u: i for i, u in enumerate(unique_ids)}
        unique_ids = torch.tensor(
            [uuid_to_int[u] for u in unique_ids], device=self.device
        )
        windows_per_song = torch.tensor(
            [ex["positive"].shape[0] for ex in batch], device=self.device
        )
        ids = torch.repeat_interleave(unique_ids, windows_per_song)

        positives = torch.cat([ex["positive"] for ex in batch])
        anchors = torch.cat([ex["anchor"] for ex in batch])

        sub_batches = collate_spectrograms(
            (anchors, positives, ids), self.config.sub_batch_size, shuffle=True
        )
        for sub_batch in sub_batches:
            yield sub_batch

    def _preprocess_example(self, dataset_iter, preprocessor: Preprocessor):
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
            return {"positive": positive, "anchor": anchor, "id": example["json"]["id"]}

        except StopIteration:
            raise

        except Exception:
            title = example["json"].get("title", "unknown") if example else "unknown"
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
