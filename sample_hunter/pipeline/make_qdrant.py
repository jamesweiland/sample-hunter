import torch
import argparse

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from uuid import uuid4
from pathlib import Path

from .encoder_net import EncoderNet
from sample_hunter._util import HF_TOKEN
from sample_hunter.config import (
    EncoderNetConfig,
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_DATASET_REPO,
)

QDRANT_PORT: str = "http://localhost:6333"


def insert_song(
    qdrant: QdrantClient, model: EncoderNet, song: torch.Tensor, **song_metadata
):
    embeddings = model(song)

    # prepare payloads
    payloads = [None] * embeddings.shape[0]
    for i in range(embeddings.shape[0]):
        payloads[i] = song_metadata.copy()
        payloads[i].update({"order": i})

    assert None not in payloads

    # create points
    points = [
        PointStruct(id=str(uuid4()), vector=embeddings[i], payload=payloads[i])
        for i in range(embeddings.shape[0])
    ]

    operation_info = client.upsert(
        collection_name="test_collection", wait=True, points=points
    )
    print(operation_info)
    exit(0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("model", type=Path, help="the path to the embeddings model")
    parser.add_argument("index", type=Path, help="Where to save the index")
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
        "--dev",
        action="store_true",
        help="option to only make an index out of the songs in the validation split",
    )

    parser.add_argument("--token", type=str, help="your hf token", default=HF_TOKEN)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.config:
        config = EncoderNetConfig.from_yaml(args.config)
    else:
        config = EncoderNetConfig()

    model = EncoderNet.from_pretrained(args.model, config=config)

    # first, start the client
    client = QdrantClient(
        ":memory:"
    )  # just testing for now, so keep everything in memory and changes don't persist

    # create the qdrant collection
    client.create_collection(
        collection_name="dev",
        vectors_config=VectorParams(
            size=config.embedding_dim, distance=Distance.EUCLID
        ),
    )
