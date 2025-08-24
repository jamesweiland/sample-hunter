import json
import argparse
import uuid
from pathlib import Path
from typing import Tuple, cast
import torch
import torch.nn as nn
import webdataset as wds

from .transformations.functional import resample
from .evaluate import evaluate_batch
from .transformations.preprocessor import Preprocessor
from .data_loading import load_tensor_from_mp3_bytes, load_webdataset
from sample_hunter.config import (
    DEFAULT_TRIPLET_LOSS_MARGIN,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_DATASET_REPO,
    EncoderNetConfig,
)
from sample_hunter._util import (
    DEVICE,
    HF_TOKEN,
    load_model,
)


def validate(
    model: nn.Module,
    dataset: wds.WebDataset,
    alpha: float = DEFAULT_TRIPLET_LOSS_MARGIN,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    device: str = DEVICE,
    debug: bool = False,
) -> Tuple[float, float, int]:
    """
    Validate the model using a validation set. This is very similar to `sample_hunter.pipeline.evaluate.evaluate`, with the key difference
    being that no batching occurs in this function: all examples are validated at once.

    Returns the average triplet accuracy of the model on the dataset and the batch size of the validation dataset
    as a (float, int) tuple.
    """

    with torch.no_grad():
        with Preprocessor() as preprocessor:

            def map_fn(ex):
                anchor, anchor_sr = load_tensor_from_mp3_bytes(ex["ground.mp3"])
                positive, positive_sr = load_tensor_from_mp3_bytes(ex["positive.mp3"])

                if isinstance(ex["json"], bytes):
                    ex["json"] = json.loads(ex["json"].decode("utf-8"))

                # we resample them out here so we can mess around with the lengths
                anchor = resample(anchor, anchor_sr, sample_rate)
                positive = resample(positive, positive_sr, sample_rate)

                max_length = max(anchor.shape[1], positive.shape[1])

                anchor = preprocessor(
                    anchor,
                    sample_rate=sample_rate,
                    target_length=max_length,
                    train=False,
                )
                positive = preprocessor(
                    positive,
                    sample_rate=sample_rate,
                    target_length=max_length,
                    train=False,
                )

                return {**ex, "positive_tensor": positive, "anchor_tensor": anchor}

            def collate_fn(batch):
                anchors = batch["anchor_tensor"]
                positives = batch["positive_tensor"]

                assert anchors.shape[0] == positives.shape[0]

                keys = [batch["json"]["ground_id"]] * anchors.shape[0]

                anchor_audio, anchor_sr = load_tensor_from_mp3_bytes(
                    batch["positive.mp3"]
                )
                positive_audio, positive_sr = load_tensor_from_mp3_bytes(
                    batch["ground.mp3"]
                )

                return anchors, positives, keys, anchor_audio, positive_audio

            dataset = dataset.map(map_fn)

            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=None, collate_fn=collate_fn
            )

            all_anchors = []
            all_positives = []
            all_keys = []
            audio = {}
            for anchors, positives, keys, anchor_audio, positive_audio in dataloader:
                all_anchors.append(anchors.to(device))
                all_positives.append(positives.to(device))
                all_keys.extend(keys)
                key = keys[0]
                audio[key] = (anchor_audio, positive_audio)

            # encode uuid keys to ints
            all_keys = [uuid.UUID(key).int for key in all_keys]
            # have to map uuids to smaller ints, since torch only supports up to int64
            uuid_to_int64 = {uuid_: i for i, uuid_ in enumerate(all_keys)}
            all_keys = [uuid_to_int64[uuid_] for uuid_ in all_keys]

            # properly collate anchors and positives
            all_anchors = torch.cat(all_anchors, dim=0)
            all_positives = torch.cat(all_positives, dim=0)
            all_keys = torch.tensor(all_keys, dtype=torch.int64)

            assert (
                all_anchors.shape == all_positives.shape
                and all_keys.shape[0] == all_anchors.shape[0]
            )
            batch_size = all_anchors.shape[0]

            loss, accuracy, topk_accuracy = evaluate_batch(
                model=model,
                positive=all_positives,
                song_ids=all_keys,
                anchor=all_anchors,
                mine_strategy="hard",
                margin=alpha,
                device=device,
                debug=debug,
            )

            if debug:
                # for i in range(res.shape[0]):  # type: ignore
                #     if res[i] == False:  # type: ignore
                #         print(f"Example {i} failed")
                #         key = all_keys[i].item()
                #         print(f"Part of song {key}")
                #         play_tensor_audio(audio[key][0], message="Playing anchor...")
                #         play_tensor_audio(audio[key][1], message="Playing positive...")

                accuracy = cast(torch.Tensor, accuracy)
                keys_with_all_failures = []
                unique_keys = torch.unique(all_keys)
                for k in unique_keys:
                    mask = k == all_keys
                    if torch.all(~accuracy[mask]):
                        keys_with_all_failures.append(k)

                print(
                    f"Number of keys that completely failed: {len(keys_with_all_failures)}"
                )
                print(
                    f"Percentage of keys that completely failed: {len(keys_with_all_failures) / len(unique_keys):.2%}"
                )

                accuracy = accuracy.float().mean().item()  # type: ignore

            return accuracy, topk_accuracy, batch_size  # type: ignore


def parse_args() -> argparse.Namespace:
    """Set up the command line parser"""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model",
        type=Path,
        help="The path to the model to validate",
    )

    parser.add_argument("--config", type=Path, help="The path to the config.yaml file")

    parser.add_argument(
        "--repo-id",
        type=str,
        help="The HF repo id to use",
        default=DEFAULT_DATASET_REPO,
    )

    parser.add_argument(
        "--token",
        type=str,
        help="Your HF token",
        default=HF_TOKEN,
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Option to play all the samples that were incorrect",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.config:
        config = EncoderNetConfig.from_yaml(args.config)
    else:
        config = EncoderNetConfig()

    model = load_model(args.model, config)

    dataset = wds.WebDataset("./_data/validation-shards/validation/validation-0001.tar")

    accuracy, topk_accuracy, batch_size = validate(model, dataset, debug=args.debug)
    print(f"Average validation accuracy: {accuracy:.2%}")
    print(f"Average validation top k accuracy: {topk_accuracy:.2%}")
    print(f"Batch size: {batch_size}")
