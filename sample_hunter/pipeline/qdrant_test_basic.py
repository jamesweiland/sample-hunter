"""
A basic test to see how vector searches return the correct song as a neighbor
"""

import torch
import argparse

import webdataset as wds
from qdrant_client import QdrantClient
from qdrant_client.models import QueryRequest
from typing import Dict, Any, cast
from pathlib import Path

from .transformations.preprocessor import Preprocessor
from .data_loading import load_tensor_from_bytes
from .encoder_net import EncoderNet
from .make_qdrant import QDRANT_PORT
from sample_hunter.config import (
    EncoderNetConfig,
    PreprocessConfig,
    DEFAULT_TOP_K,
    DEFAULT_DATASET_REPO,
)
from sample_hunter._util import HF_TOKEN


def prepare_example_for_query(
    example: Dict[str, Any],
    preprocess_config: PreprocessConfig | None = None,
    **preprocess_kwargs,
) -> torch.Tensor:

    # load song as tensor
    song_bytes = example.get("mp3") or example.get("positive.mp3")
    assert song_bytes is not None
    song, sr = load_tensor_from_bytes(song_bytes)

    # preprocess songs into specs
    config = preprocess_config or PreprocessConfig()
    config = config.merge_kwargs(**preprocess_kwargs)
    preprocessor = Preprocessor(config)
    specs = cast(torch.Tensor, preprocessor(song, train=False, sample_rate=sr))

    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("model", type=Path, help="the path to the embeddings model")
    parser.add_argument(
        "--repo-id",
        type=str,
        help="repo id of the dataset to use",
        default=DEFAULT_DATASET_REPO,
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to a config to instantiate the model",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        help="number of nearest neighbors to return for each search",
        default=DEFAULT_TOP_K,
    )

    parser.add_argument(
        "--dev",
        action="store_true",
        help="option to only make an index out of the songs in the validation split",
    )

    parser.add_argument("--token", type=str, help="your hf token", default=HF_TOKEN)

    return parser.parse_args()


if __name__ == "__main__":
    import json
    from tqdm import tqdm

    args = parse_args()

    if args.config:
        config = EncoderNetConfig.from_yaml(args.config)
    else:
        config = EncoderNetConfig()

    model = EncoderNet.from_pretrained(args.model, config=config)

    client = QdrantClient(QDRANT_PORT)

    if args.dev:
        # load validation dataset saved locally
        dataset = wds.WebDataset(
            "./_data/validation-shards/validation/validation-0001.tar"
        )

        num_right = 0
        total = 0
        for ex in tqdm(dataset, desc="Testing"):
            ex["json"] = json.loads(ex["json"].decode("utf-8"))
            specs = prepare_example_for_query(ex)

            embeddings = model(specs)

            requests = [
                QueryRequest(query=embedding, limit=args.top_k, with_payload=True)
                for embedding in embeddings
            ]

            responses = client.query_batch_points(
                collection_name="dev", requests=requests
            )

            found = False
            for response in responses:
                # print(result)
                points = response.points
                for point in points:
                    if point.payload["ground_id"] == ex["json"]["ground_id"]:
                        num_right += 1
                        found = True
                        break
                if found:
                    break

                total += 1

        print(f"Accuracy: {num_right / total:.2%}")

    else:
        raise NotImplementedError("not implemented yet")
