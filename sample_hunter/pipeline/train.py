import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from typing import Callable, Tuple
import argparse
from pathlib import Path
from datasets import load_dataset, IterableDataset

from sample_hunter.pipeline.encoder_net import EncoderNet
from sample_hunter.pipeline.triplet_loss import triplet_accuracy, mine_negative_triplet
from sample_hunter.pipeline.evaluate import evaluate
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
    DEFAULT_TEST_SPLIT,
    TRAIN_LOG_DIR,
    PROCS,
    HF_DATASET,
    HF_TOKEN,
)


def train_single_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: Callable[..., Tensor],
    optimizer: torch.optim.Optimizer,
    device: str,
    alpha: float,
) -> Tuple[float, float]:
    """
    Train `model` for a single epoch. Returns a (loss, accuracy) tuple
    """
    model.train()
    epoch_total_loss = 0
    num_batches = 0
    epoch_total_accuracy = 0
    for batch in dataloader:
        anchor_batch = batch["anchor"].to(device)
        positive_batch = batch["positive"].to(device)
        song_ids = batch["song_id"].to(device)

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

        epoch_total_loss += loss.item()
        epoch_total_accuracy += triplet_accuracy(
            anchor=anchor_embeddings,
            positive=positive_embeddings,
            negative=negative_embeddings,
            alpha=alpha,
        )
        num_batches += 1

    epoch_average_loss = epoch_total_loss / num_batches
    print(f"Average loss of epoch: {epoch_average_loss}")
    epoch_average_accuracy = epoch_total_accuracy / num_batches
    print(f"Epoch accuracy: {epoch_total_accuracy}")
    return epoch_average_loss, epoch_average_accuracy


def train(
    model: nn.Module,
    train_dataloader: DataLoader,
    loss_fn: Callable,
    optimizer: torch.optim.Optimizer,
    test_dataloader: DataLoader | None = None,
    device: str = DEVICE,
    num_epochs: int = NUM_EPOCHS,
    alpha: float = ALPHA,
    log_dir: Path = TRAIN_LOG_DIR,
):
    writer = SummaryWriter(log_dir=log_dir)
    for i in range(num_epochs):
        print(f"Epoch {i + 1}")
        loss, accuracy = train_single_epoch(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            alpha=alpha,
        )
        writer.add_scalar("Training loss", loss, i)
        writer.add_scalar("Training accuracy", accuracy, i)

        if test_dataloader is not None:
            # evaluate accuracy as we go on the test set
            accuracy = evaluate(
                model=model, dataloader=test_dataloader, alpha=alpha, device=device
            )
            writer.add_scalar("Testing accuracy", accuracy, i)

        print("--------------------------------------------")
    writer.close()
    print("Finished training")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--out",
        type=Path,
        help="The path to save the trained model to",
        default=MODEL_SAVE_PATH,
    )

    parser.add_argument(
        "--repo-id", type=str, help="The path to the HF dataset", default=HF_DATASET
    )

    parser.add_argument(
        "--token", type=str, help="Your huggingface token", default=HF_TOKEN
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    train_dataset = load_dataset(
        path=args.repo_id, split="train_1", streaming=True, token=args.token
    ).with_format("torch")

    test_dataset = load_dataset(
        path=args.repo_id, split="test_1", streaming=True, token=args.token
    ).with_format("torch")

    assert isinstance(train_dataset, IterableDataset) and isinstance(
        test_dataset, IterableDataset
    )
    assert train_dataset.features["anchor"] == train_dataset.features["positive"]

    input_shape = train_dataset.features["anchor"].shape
    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, num_workers=PROCS  # type: ignore
    )
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=PROCS)  # type: ignore

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
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizer=adam,
        loss_fn=triplet_loss,
        device=DEVICE,
        num_epochs=NUM_EPOCHS,
        alpha=ALPHA,
    )

    torch.save(model.state_dict(), args.out)
    print(f"Model saved to {args.out}")
