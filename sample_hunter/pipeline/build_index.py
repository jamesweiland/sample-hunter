"""
Build a FAISS index from a trained embeddings model, to later use for prediction.
"""

import argparse
from pathlib import Path
import torch
import faiss
from tqdm import tqdm
import webdataset as wds
import pandas as pd
from typing import Tuple, Dict, List, cast

from .data_loading import (
    collate_spectrograms,
    flatten_sub_batches,
    load_tensor_from_mp3_bytes,
    load_webdataset,
)
from .transformations.preprocessor import Preprocessor
from .encoder_net import EncoderNet
from sample_hunter._util import HF_TOKEN, DEVICE, load_model
from sample_hunter.config import DEFAULT_EMBEDDING_DIM, DEFAULT_DATASET_REPO


def build_index(
    model: EncoderNet,
    dataset: wds.WebDataset | Dict[str, wds.WebDataset],
    song_titles: List[str],
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    batch_size: int = 1,
    device: str = DEVICE,
) -> Tuple[faiss.IndexFlatL2, pd.DataFrame]:
    """
    Given a dataset of audio, build a faiss Index
    from it to later query for nearest neighbor searching.
    """
    with torch.no_grad():
        model.eval()

        index = faiss.IndexFlatL2(embedding_dim)
        metadata = pd.DataFrame(columns=["song_id", "snippet_id", "song_title"])

        def map_fn(ex):
            print("in map fn")
            if ex["json"]["title"] in song_titles:
                preprocessor = Preprocessor(obfuscator=None)
                print(f"good ex {ex["json"]["title"]}")
                audio, sr = load_tensor_from_mp3_bytes(ex["a.mp3"])
                specs = preprocessor(audio, sample_rate=sr, train=False)
                return {**ex, "specs": specs}
            print(f"bad ex {ex["json"]["title"]}")
            return None

        def collate_fn(batch):
            batch = [ex for ex in batch if ex is not None]

            if len(batch) > 0:

                keys = torch.tensor(
                    [int(song["json"]["ground_song_id"]) for song in batch]
                )
                windows_per_song = torch.tensor(
                    [song["specs"].shape[0] for song in batch]
                )

                keys = torch.repeat_interleave(keys, windows_per_song)
                specs = torch.cat([song["specs"] for song in batch], dim=0)

                # we have to do titles separately as there's no way to convert
                # strings to tensors
                titles = []
                for song in batch:
                    titles.extend([song["json"]["title"]] * song["specs"].shape[0])

                return specs, keys, titles
            return None, None, None

        # set up the dataloaders
        if isinstance(dataset, dict):
            dataloaders = [
                torch.utils.data.DataLoader(
                    d.map(map_fn),
                    batch_size=batch_size,
                    collate_fn=collate_fn,
                )
                for _, d in dataset.items()
            ]
        else:
            dataloaders = [
                torch.utils.data.DataLoader(
                    dataset.map(map_fn), batch_size=batch_size, collate_fn=collate_fn  # type: ignore
                )
            ]

        # track the vector ids in index with vector_id
        vector_id = 0
        batch = 1
        for dataloader in dataloaders:
            # iterate through each dataloader and add embeddings to index
            for specs, keys, titles in tqdm(dataloader, desc="Building index..."):

                if specs is None:
                    print("none in current batch, continuing")
                    continue

                print(f"Batch {batch}")

                specs, keys = specs.to(device), keys.to(device)
                embeddings = model(specs)

                # add the embeddings to the index
                index.add(embeddings.numpy().astype("float32"))  # type: ignore

                # store ids into the metadata df
                snippet_ids = list(range(vector_id, vector_id + embeddings.shape[0]))
                song_ids = keys.tolist()
                metadata = pd.concat(
                    [
                        metadata,
                        pd.DataFrame(
                            {
                                "song_id": song_ids,
                                "snippet_id": snippet_ids,
                                "song_title": titles,
                            }
                        ),
                    ],
                    ignore_index=True,
                )

                vector_id += embeddings.shape[0]
                batch += 1

        # chat says this works, we'll see
        metadata["snippet_order"] = metadata.groupby("song_id").cumcount()

        return index, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model",
        type=Path,
        help="The path to the model to use to generate embeddings from",
    )

    parser.add_argument("index", type=Path, help="Where to save the faiss index")

    parser.add_argument(
        "metadata",
        type=Path,
        help="Where to save the metadata csv file",
    )

    parser.add_argument(
        "--repo-id",
        type=str,
        help="The name of the HF dataset to use to generate embeddings",
        default=DEFAULT_DATASET_REPO,
    )

    parser.add_argument("--token", type=str, help="Your HF token", default=HF_TOKEN)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    model = load_model(args.model)
    dataset = load_webdataset(args.repo_id, ["train", "test"], args.token)
    validation_dataset = cast(
        wds.WebDataset, load_webdataset(args.repo_id, "validation", args.token)
    )

    titles = []
    for ex in validation_dataset:
        titles.append(ex["json"]["title"])

    index, metadata = build_index(
        model,
        dataset,
        titles,
    )

    print("Index finished building")
    print(f"Index size: {index.ntotal}")
    print(f"metadata rows: {len(metadata)}")

    faiss.write_index(index, str(args.index))
    metadata.to_csv(args.metadata)
