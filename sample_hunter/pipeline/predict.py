"""
Make a prediction for song snippets
"""

import argparse
from pathlib import Path
import torch
import faiss
from tqdm import tqdm
import webdataset as wds
import pandas as pd
from typing import Tuple, Dict

from .data_loading import (
    collate_spectrograms,
    flatten_sub_batches,
    load_tensor_from_bytes,
    load_webdataset,
)
from .transformations.spectrogram_preprocessor import SpectrogramPreprocessor
from .encoder_net import EncoderNet
from sample_hunter._util import config, HF_TOKEN, DEVICE, load_model


def build_index(
    model: EncoderNet,
    dataset: wds.WebDataset | Dict[str, wds.WebDataset],
    embedding_dim: int = config.network.embedding_dim,
    source_batch_size: int = config.network.source_batch_size,
    sub_batch_size: int = config.network.sub_batch_size,
    device: str = DEVICE,
) -> Tuple[faiss.IndexFlatL2, pd.DataFrame]:
    """
    Given a dataset of audio, build a faiss Index
    from it to later query for nearest neighbor searching.
    """
    with torch.no_grad():
        model.eval()

        index = faiss.IndexFlatL2(embedding_dim)
        metadata = pd.DataFrame(columns=["song_id", "snippet_id"])

        with SpectrogramPreprocessor() as preprocessor:

            def map_fn(ex):
                audio, sr = load_tensor_from_bytes(ex["mp3"])
                specs = preprocessor(audio, sample_rate=sr, obfuscate=False)
                return {**ex, "specs": specs}

            def collate_fn(batch):
                keys = torch.tensor([int(song["json"]["id"]) for song in batch])
                windows_per_song = [song["specs"].shape[0] for song in batch]
                keys = torch.repeat_interleave(keys, torch.tensor(windows_per_song))
                key_splits = keys.split(sub_batch_size)

                real_spec_shape = 0
                for spec in batch:
                    real_spec_shape += spec["specs"].shape[0]

                specs = collate_spectrograms(batch, col="specs")

                assert len(specs) == len(key_splits)

                return [(spec, k) for spec, k in zip(specs, key_splits)]

            if isinstance(dataset, dict):
                dataloaders = [
                    torch.utils.data.DataLoader(
                        d.map(map_fn),
                        batch_size=source_batch_size,
                        collate_fn=collate_fn,
                    )
                    for _, d in dataset.items()
                ]
            else:
                dataloaders = [
                    torch.utils.data.DataLoader(
                        dataset.map(map_fn), batch_size=source_batch_size, collate_fn=collate_fn  # type: ignore
                    )
                ]

            # track the vector ids in index with vector_id
            vector_id = 0
            batch = 1
            for dataloader in dataloaders:
                # iterate through each dataloader and add embeddings to index
                for specs, keys in tqdm(
                    flatten_sub_batches(dataloader), desc="Building index..."
                ):
                    print(f"Batch {batch}")
                    specs, keys = specs.to(device), keys.to(device)
                    embeddings = model(specs)

                    # add the embeddings to the index
                    index.add(embeddings.numpy().astype("float32"))  # type: ignore

                    # store ids into the metadata df
                    snippet_ids = list(
                        range(vector_id, vector_id + embeddings.shape[0])
                    )
                    song_ids = keys.tolist()
                    metadata = pd.concat(
                        [
                            metadata,
                            pd.DataFrame(
                                {"song_id": song_ids, "snippet_id": snippet_ids}
                            ),
                        ],
                        ignore_index=True,
                    )

                    vector_id += embeddings.shape[0]
                    batch += 1

        return index, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model",
        type=Path,
        help="The path to the model to use to generate embeddings from",
    )

    parser.add_argument(
        "--index", type=Path, help="Where to save the faiss index", required=True
    )

    parser.add_argument(
        "--metadata",
        type=Path,
        help="Where to save the metadata csv file",
        required=True,
    )

    parser.add_argument(
        "--repo-id",
        type=str,
        help="The name of the HF dataset to use to generate embeddings",
        default=config.hf.repo_id,
    )

    parser.add_argument("--token", type=str, help="Your HF token", default=HF_TOKEN)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    model = load_model(args.model)
    dataset = load_webdataset(args.repo_id, ["train", "test"], args.token)

    index, metadata = build_index(model, dataset)

    print("Index finished building")
    print(f"Index size: {index.ntotal}")
    print(f"metadata rows: {len(metadata)}")

    faiss.write_index(index, str(args.index))
    metadata.to_csv(args.metadata)
