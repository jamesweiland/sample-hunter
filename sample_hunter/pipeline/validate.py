import argparse
import io
from pathlib import Path
from typing import Tuple
import torch
import torch.nn as nn
import torchaudio
import webdataset as wds

from .encoder_net import EncoderNet
from .evaluate import evaluate_batch
from .transformations.spectrogram_preprocessor import SpectrogramPreprocessor
from .data_loading import load_webdataset
from sample_hunter.cfg import config
from sample_hunter._util import DEVICE, HF_TOKEN


def validate(
    model: nn.Module,
    dataset: wds.WebDataset,
    alpha: float = config.network.alpha,
    device: str = DEVICE,
) -> Tuple[float, int]:
    """
    Validate the model using a validation set. This is very similar to `sample_hunter.pipeline.evaluate.evaluate`, with the key difference
    being that no batching occurs in this function: all examples are validated at once.

    Returns the average triplet accuracy of the model on the dataset and the batch size of the validation dataset
    as a (float, int) tuple.
    """

    with torch.no_grad():
        with SpectrogramPreprocessor() as preprocessor:

            def map_fn(ex):
                anchor = preprocessor(ex["a.mp3"], obfuscate=False)
                positive = preprocessor(ex["b.mp3"], obfuscate=False)
                with io.BytesIO(ex["a.mp3"]) as buffer:
                    anchor, anchor_sr = torchaudio.load(
                        buffer, format="mp3", backend="ffmpeg"
                    )
                    if anchor.ndim == 1:
                        anchor = anchor.unsqueeze(0)

                with io.BytesIO(ex["b.mp3"]) as buffer:
                    positive, positive_sr = torchaudio.load(
                        buffer, format="mp3", backend="ffmpeg"
                    )
                    if positive.ndim == 1:
                        positive = positive.unsqueeze(0)

                # we resample them out here so we can mess around with the lengths
                anchor = preprocessor.resample(anchor, anchor_sr)
                positive = preprocessor.resample(positive, positive_sr)

                max_length = max(anchor.shape[1], positive.shape[1])

                anchor = preprocessor(
                    anchor,
                    sample_rate=config.preprocess.sample_rate,
                    target_length=max_length,
                    obfuscate=False,
                )
                positive = preprocessor(
                    positive,
                    sample_rate=config.preprocess.sample_rate,
                    target_length=max_length,
                    obfuscate=False,
                )

                return {**ex, "positive_tensor": positive, "anchor_tensor": anchor}

            def collate_fn(batch):
                anchors = batch["anchor_tensor"]
                positives = batch["positive_tensor"]

                keys = torch.tensor(int(batch["__key__"]))
                windows_per_song = anchors.shape[0]
                keys = torch.repeat_interleave(keys, torch.tensor(windows_per_song))
                return anchors, positives, keys

            dataset = dataset.map(map_fn)

            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=None, collate_fn=collate_fn
            )

            all_anchors = []
            all_positives = []
            all_keys = []
            for anchors, positives, keys in dataloader:
                all_anchors.append(anchors.to(device))
                all_positives.append(positives.to(device))
                all_keys.append(keys.to(device))

            all_anchors = torch.cat(all_anchors, dim=0)
            all_positives = torch.cat(all_positives, dim=0)
            all_keys = torch.cat(all_keys, dim=0)

            assert (
                all_anchors.shape == all_positives.shape
                and all_keys.shape[0] == all_anchors.shape[0]
            )
            batch_size = all_anchors.shape[0]

            accuracy = evaluate_batch(
                model=model,
                positive=all_positives,
                song_ids=all_keys,
                anchor=all_anchors,
                alpha=alpha,
                device=device,
            )
            return accuracy, batch_size


def parse_args() -> argparse.Namespace:
    """Set up the command line parser"""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model",
        type=Path,
        help="The path to the model to validate",
    )

    parser.add_argument(
        "--repo-id",
        type=str,
        help="The HF repo id to use",
        default=config.hf.repo_id,
    )

    parser.add_argument(
        "--token",
        type=str,
        help="Your HF token",
        default=HF_TOKEN,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if DEVICE == "cuda":
        state_dict = torch.load(args.model, weights_only=False)
    else:
        state_dict = torch.load(
            args.model, weights_only=False, map_location=torch.device("cpu")
        )
    model = EncoderNet().to(DEVICE)
    model.load_state_dict(state_dict)

    dataset = load_webdataset(args.repo_id, "validation", args.token)

    accuracy, batch_size = validate(model, dataset)
    print(f"Average validation accuracy: {accuracy}")
    print(f"Batch size: {batch_size}")
