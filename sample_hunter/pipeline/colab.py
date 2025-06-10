"""
Running colab code cells in the uv environment doesn't work,
so we have to set up python scripts that the notebook can clone
and then run in a code cell, like:

`uv run python3 -m sample_hunter.pipeline.colab`
"""

import argparse
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
import torch.nn as nn
import torch

from sample_hunter.pipeline.encoder_net import EncoderNet
from sample_hunter.pipeline.song_pairs_dataset import SongPairsDataset
from sample_hunter._util import BATCH_SIZE, DEVICE, LEARNING_RATE
from sample_hunter.pipeline.train import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--audio-dir", type=Path, help="The path to the audio files", required=True
    )

    parser.add_argument(
        "--annotations",
        type=Path,
        help="The path to the annotations csv",
        required=True,
    )

    parser.add_argument(
        "--out", type=Path, help="The path to save the trained model", required=True
    )

    parser.add_argument(
        "--test-frac",
        type=float,
        help="The percentage of data to use as the test split",
        required=True,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    annotations = pd.read_csv(args.annotations)

    # right now, we're only focused on fold 1
    annotations = annotations[annotations["fold"] == 1]
    # need to update the paths to connect to drive
    drive_path = Path("/content/MyDrive/sample-hunter/fold-1")
    annotations["anchor"] = annotations["anchor"].apply(
        lambda p: Path(drive_path / Path(p).name).__str__()
    )
    annotations["positive"] = annotations["positive"].apply(
        lambda p: Path(drive_path / Path(p).name).__str__()
    )

    song_ids = pd.Series(annotations["song_id"].unique())
    test_song_ids = song_ids.sample(frac=args.test_frac)
    train_song_ids = song_ids.drop(index=test_song_ids.index)

    test_annotations = annotations[annotations["song_id"].isin(test_song_ids)]
    train_annotations = annotations[annotations["song_id"].isin(train_song_ids)]

    train_dataset = SongPairsDataset(
        audio_dir=args.audio_dir,
        annotations_file=train_annotations,
    )

    test_dataset = SongPairsDataset(
        audio_dir=args.audio_dir, annotations_file=test_annotations
    )

    assert train_dataset.shape() == test_dataset.shape()

    input_shape = train_dataset.shape()
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = EncoderNet(input_shape=input_shape).to(DEVICE)
    adam = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    triplet_loss = nn.TripletMarginLoss()

    train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizer=adam,
        loss_fn=triplet_loss,
    )

    torch.save(model.state_dict(), args.out)
    print(f"Model saved to {args.out}")
