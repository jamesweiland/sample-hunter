import torch
import argparse
import json
import webdataset as wds

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    UpdateStatus,
)
from uuid import uuid4
from typing import Tuple, Dict, Any, cast
from pathlib import Path
from tqdm import tqdm

from .transformations.preprocessor import Preprocessor
from .data_loading import load_tensor_from_bytes
from .encoder_net import EncoderNet
from sample_hunter._util import HF_TOKEN
from sample_hunter.config import (
    EncoderNetConfig,
    PreprocessConfig,
    DEFAULT_DATASET_REPO,
)

QDRANT_PORT: str = "http://localhost:6333"

"""
to run the service:

docker run -p 6333:6333 -p 6334:6334 \
    -v "$(pwd)/qdrant_storage:/qdrant/storage:z" \
    qdrant/qdrant
"""


def prepare_example_for_insertion(
    example: Dict[str, Any],
    preprocess_config: PreprocessConfig | None = None,
    **preprocess_kwargs,
) -> Tuple[torch.Tensor, Dict[str, Any]]:

    # set up preprocessor
    config = preprocess_config or PreprocessConfig()
    config = config.merge_kwargs(**preprocess_kwargs)
    preprocessor = Preprocessor(config)

    # preprocess song
    song_bytes = example.get("mp3") or example.get("ground.mp3")
    assert song_bytes is not None

    song, sr = load_tensor_from_bytes(song_bytes)
    specs = cast(torch.Tensor, preprocessor(song, train=False, sample_rate=sr))

    # extract song metadata
    metadata = example["json"]

    return specs, metadata


def insert_song(
    client: QdrantClient, model: EncoderNet, specs: torch.Tensor, **song_metadata
) -> bool:
    with torch.no_grad():
        model.eval()
        embeddings = model(specs)

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

        operation_info = client.upsert(collection_name="dev", wait=True, points=points)

        return operation_info.status == UpdateStatus.COMPLETED


def make_qdrant(
    dataset: wds.WebDataset,
    client: QdrantClient,
    model: EncoderNet,
    collection_name: str = "dev",
    preprocess_config: PreprocessConfig | None = None,
    **preprocess_kwargs,
):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=model.config.embedding_dim, distance=Distance.EUCLID
        ),
    )

    for ex in tqdm(dataset, desc="building db"):
        if isinstance(ex["json"], bytes):
            ex["json"] = json.loads(ex["json"].decode("utf-8"))

            specs, metadata = prepare_example_for_insertion(
                ex, preprocess_config, **preprocess_kwargs
            )

            if not insert_song(client, model, specs, **metadata):
                raise RuntimeError(f"{metadata["ground_id"]} failed")


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

    # load local dataset
    # tars = glob.glob("./_data/webdataset-shards/*/*.tar")
    # dataset = wds.WebDataset(tars)

    # first, start the client
    client = QdrantClient(QDRANT_PORT)

    if args.dev:
        # load local validation dataset
        dataset = wds.WebDataset(
            "./_data/validation-shards/validation/validation-0001.tar"
        )

        make_qdrant(dataset, client, model)

        print("done!")
        exit(0)

        # # once it's fully built, let's play around
        # test_ground_id = "54f6a4ba-2a62-4788-8a7c-07cee640fabf"
        # test_validation_id = "28f07cf4-0c1b-4545-803b-afd6fa33a772"

        # # this should only return three examples
        # test_embedding = torch.rand((128,))
        # search_result = client.query_points(
        #     collection_name="dev",
        #     query=test_embedding.numpy(),
        #     query_filter=Filter(
        #         must=[
        #             FieldCondition(
        #                 key="validation_id", match=MatchValue(value=test_validation_id)
        #             )
        #         ]
        #     ),
        #     with_payload=True,
        #     with_vectors=True,
        # ).points

        # print(search_result)
        # print(len(search_result))

    else:
        raise NotImplementedError("not implemented yet")
