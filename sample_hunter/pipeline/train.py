import torch
import torch.nn as nn
from torchaudio.transforms import MelSpectrogram
from torch.utils.data import DataLoader
from torch import Tensor
from typing import Callable, List
import argparse
from pathlib import Path

from sample_hunter.pipeline.song_pairs_dataset import SongPairsDataset
from sample_hunter.pipeline.encoder_net import EncoderNet
from sample_hunter._util import (
    ANNOTATIONS_PATH,
    AUDIO_DIR,
    MODEL_SAVE_PATH,
    CONV_LAYER_DIMS,
    DIVIDE_AND_ENCODE_HIDDEN_DIM,
    EMBEDDING_DIM,
    NUM_BRANCHES,
    PADDING,
    POOL_KERNEL_SIZE,
    STRIDE,
    WINDOW_SIZE,
    SAMPLE_RATE,
    LEARNING_RATE,
    NUM_EPOCHS,
    ALPHA,
    BATCH_SIZE,
    DEFAULT_MEL_SPECTROGRAM,
    DEVICE,
)


def mine_negative_triplet(
    anchor_embeddings: Tensor,
    positive_embeddings: Tensor,
    song_ids: Tensor,
    alpha: float,
) -> Tensor:
    """
    Semi-hard online triplet mining implementation.
    Given an embedding and the batch it came from,
    find the best semi-hard negative example

    anchor_embeddings: (batch_size, embedding_dim) tensor that serves as the positive example

    positive_embeddings: (batch_size, embedding_dim) tensor that serves as the anchor

    song_ids: list of strings that have the song ids correspond to
    the batch. batch_ids[i] should be the song id of batch[i]

    alpha: margin for mining
    """
    batch_size = anchor_embeddings.shape[0]

    # pos_dists has shape (batch_size) and neg_dists has shape (batch_size, batch_size)
    pos_dists = torch.linalg.vector_norm(
        anchor_embeddings - positive_embeddings, ord=2, dim=1
    )
    # i don't really understand why this is only anchor embeddings,
    # but chat insisted
    neg_dists = torch.cdist(anchor_embeddings, anchor_embeddings, p=2)

    # create masks. within_margin_mask ensures that the tensor satisfies the 'semi-hard' criterion
    # different_song_mask checks that the song_id is different
    same_song_mask = song_ids.unsqueeze(dim=1) == song_ids.unsqueeze(dim=0)
    margin_mask = (neg_dists > pos_dists.unsqueeze(dim=1)) & (
        neg_dists < pos_dists.unsqueeze(dim=1) + alpha
    )
    valid_neg_mask = same_song_mask & margin_mask

    negatives = torch.zeros(anchor_embeddings.shape)
    for i in range(batch_size):
        valid_negs = valid_neg_mask[i].nonzero().squeeze(dim=1)
        if valid_negs.any():
            # find the hardest valid neg
            hardest = valid_negs[torch.argmin(neg_dists[i][valid_negs])]
            negatives[i] = anchor_embeddings[hardest]
        else:
            # if there's no semi-hard, just take the hardest from a different song
            different_song_negs = (~same_song_mask[i]).nonzero().squeeze(dim=1)
            if different_song_negs.any():
                hardest = different_song_negs[
                    torch.argmin(neg_dists[i, different_song_negs])
                ]
                negatives[i] = anchor_embeddings[hardest]
            else:
                # edge case: all samples in the batch are from the same song
                # this should never happen. if this happens then the batch size
                # is too small
                print(
                    "WARNING: all tensors in batch are from the same song. INCREASE THE BATCH SIZE!!!!"
                )
                hardest = neg_dists[i][torch.argmin(neg_dists[i])]
                negatives[i] = anchor_embeddings[hardest]

    return negatives


def train_single_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: Callable[..., Tensor],
    optimizer: torch.optim.Optimizer,
    device: str,
    alpha: float,
):
    loss = Tensor()
    for anchor_batch, positive_batch, song_ids in dataloader:
        anchor_batch, positive_batch, song_ids = (
            anchor_batch.to(device),
            positive_batch.to(device),
            song_ids.to(device),
        )

        # predict embeddings
        anchor_embeddings = model(anchor_batch)
        positive_embeddings = model(positive_batch)

        # mine the negative embedding
        negative_embeddings = mine_negative_triplet(
            anchor_embeddings=anchor_embeddings,
            positive_embeddings=positive_embeddings,
            song_ids=song_ids,
            alpha=alpha,
        )
        # calculate loss
        loss = loss_fn(anchor_embeddings, positive_embeddings, negative_embeddings)

        # backpropagate loss and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Loss: {loss.item()}")


def train(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: Callable,
    optimizer: torch.optim.Optimizer,
    device: str,
    num_epochs: int,
    alpha: float,
):
    for i in range(num_epochs):
        print(f"Epoch {i + 1}")
        train_single_epoch(
            model=model,
            dataloader=dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            alpha=alpha,
        )
        print("--------------------------------------------")
    print("Finished training")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--annotations",
        type=Path,
        help="The path to the file with annotated pairs",
        default=ANNOTATIONS_PATH,
    )

    parser.add_argument(
        "--audio-dir",
        type=Path,
        help="The path to the directory of audio files",
        default=AUDIO_DIR,
    )

    parser.add_argument(
        "--out",
        type=Path,
        help="The path to save the trained model to",
        default=MODEL_SAVE_PATH,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    spd = SongPairsDataset(
        audio_dir=args.audio_dir,
        annotations_file=args.annotations,
        mel_spectrogram=DEFAULT_MEL_SPECTROGRAM,
        target_sample_rate=SAMPLE_RATE,
        num_samples=WINDOW_SIZE,
        device=DEVICE,
    )
    input_shape = spd.shape()
    dataloader = DataLoader(spd, batch_size=BATCH_SIZE)

    model = EncoderNet(
        conv_layer_dims=CONV_LAYER_DIMS,
        stride=STRIDE,
        padding=PADDING,
        pool_kernel_size=POOL_KERNEL_SIZE,
        num_branches=NUM_BRANCHES,
        divide_and_encode_hidden_dim=DIVIDE_AND_ENCODE_HIDDEN_DIM,
        embedding_dim=EMBEDDING_DIM,
        input_shape=input_shape,
    ).to(DEVICE)

    adam = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    triplet_loss = nn.TripletMarginLoss()

    train(
        model=model,
        dataloader=dataloader,
        optimizer=adam,
        loss_fn=triplet_loss,
        device=DEVICE,
        num_epochs=NUM_EPOCHS,
        alpha=ALPHA,
    )

    torch.save(model.state_dict(), args.out)
    print(f"Model saved to {args.out}")
