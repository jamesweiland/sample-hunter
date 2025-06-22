import math
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from typing import Callable, Tuple, Union
import argparse
from pathlib import Path
from datasets import load_dataset, IterableDataset, Audio
from tqdm import tqdm

from sample_hunter.pipeline.encoder_net import EncoderNet
from sample_hunter.pipeline.transformations.functional import collate_spectrograms
from sample_hunter.pipeline.transformations.transformations import (
    SpectrogramPreprocessor,
)
from sample_hunter.pipeline.triplet_loss import triplet_accuracy, mine_negative_triplet
from sample_hunter.pipeline.evaluate import evaluate
from sample_hunter._util import (
    CACHE_DIR,
    DEFAULT_HOP_LENGTH,
    DEFAULT_MEL_SPECTROGRAM,
    DEFAULT_N_FFT,
    DEFAULT_WINDOW_NUM_SAMPLES,
    MODEL_SAVE_PATH,
    CONV_LAYER_DIMS,
    DEFAULT_DIVIDE_AND_ENCODE_HIDDEN_DIM,
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_NUM_BRANCHES,
    DEFAULT_PADDING,
    DEFAULT_POOL_KERNEL_SIZE,
    DEFAULT_STRIDE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_NUM_EPOCHS,
    DEFAULT_ALPHA,
    DEFAULT_BATCH_SIZE,
    DEVICE,
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
    writer: SummaryWriter | None = None,
) -> Tuple[float, float]:
    """
    Train `model` for a single epoch. Returns a (loss, accuracy) tuple
    """
    model.train()
    epoch_total_loss = 0
    num_batches = 0
    epoch_total_accuracy = 0
    for batch in tqdm(dataloader, desc="Training epoch..."):
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

        accuracy = triplet_accuracy(
            anchor=anchor_embeddings,
            positive=positive_embeddings,
            negative=negative_embeddings,
            alpha=alpha,
        )

        if writer:
            writer.add_scalar("Training loss", loss.item(), num_batches)
            writer.add_scalar("Training accuracy", accuracy, num_batches)

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
    tensorboard: str = "none",
    device: str = DEVICE,
    num_epochs: int = DEFAULT_NUM_EPOCHS,
    alpha: float = DEFAULT_ALPHA,
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
        if tensorboard == "epoch":
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

    parser.add_argument(
        "--procs",
        type=int,
        help="The number of processes to use for multiprocessing",
        default=PROCS,
    )

    tensorboard = parser.add_argument_group("tensorboard")
    tensorboard.add_argument(
        "--log-dir",
        type=Path,
        help="The path to the log directory for tensorboard",
        default=TRAIN_LOG_DIR,
    )
    tensorboard.add_argument(
        "--tensorboard",
        help="Specify whether to write tensorboard data per batch, epoch, or not at all",
        type=str,
        choices=["none", "batch", "epoch"],
    )

    hyperparams = parser.add_argument_group("Network hyperparameters")
    hyperparams.add_argument(
        "--num-epochs",
        type=int,
        help="The number of epochs to train the model for",
        default=DEFAULT_NUM_EPOCHS,
    )
    hyperparams.add_argument(
        "--stride",
        help="The stride of the convolutional kernel",
        type=int,
        default=DEFAULT_STRIDE,
    )
    hyperparams.add_argument(
        "--padding",
        help="The padding for the convolutions",
        type=int,
        default=DEFAULT_PADDING,
    )
    hyperparams.add_argument(
        "--pool-kernel-size",
        help="The kernel size to use for pooling",
        type=int,
        default=DEFAULT_POOL_KERNEL_SIZE,
    )
    hyperparams.add_argument(
        "--num-branches",
        help="The number of parallel branches to use in the divide and encode layer",
        type=int,
        default=DEFAULT_NUM_BRANCHES,
    )
    hyperparams.add_argument(
        "--hidden-dim",
        help="The hidden dimension of the divide and encode layer",
        default=DEFAULT_DIVIDE_AND_ENCODE_HIDDEN_DIM,
        type=int,
    )
    hyperparams.add_argument(
        "--embedding-dim",
        help="The dimension of the output embeddings",
        type=int,
        default=DEFAULT_EMBEDDING_DIM,
    )
    hyperparams.add_argument(
        "--alpha",
        help="The margin to use for triplet loss and mining",
        type=float,
        default=DEFAULT_ALPHA,
    )
    hyperparams.add_argument(
        "--learning-rate",
        help="The learning rate of the optimizer",
        type=float,
        default=DEFAULT_LEARNING_RATE,
    )
    hyperparams.add_argument(
        "--batch-size",
        help="The batch size for the dataloader",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )

    dev = parser.add_argument_group("dev")
    dev.add_argument("--num", help="train only a sample of the dataset", type=int)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    preprocessor = SpectrogramPreprocessor()

    def map_fn(ex):
        positive, anchor = preprocessor(ex["audio"], obfuscate=True)
        return {**ex, "positive": positive, "anchor": anchor}

    dataset = load_dataset(
        "samplr/songs",
        streaming=True,
        split="train",
        cache_dir=Path(CACHE_DIR / "songs").__str__(),
        token=args.token,
    ).cast_column("audio", Audio(decode=True))
    dataset = dataset.map(map_fn)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collate_spectrograms,
    )

    if args.num:
        dataset = train_dataset.shuffle()
        dataset = train_dataset.take(args.num)

    n_f = 1 + math.floor(DEFAULT_WINDOW_NUM_SAMPLES / DEFAULT_HOP_LENGTH)
    input_shape = torch.Size((args.n_mels, n_f))

    model = EncoderNet(
        conv_layer_dims=CONV_LAYER_DIMS,
        stride=args.stride,
        padding=args.padding,
        pool_kernel_size=args.pool_kernel_size,
        num_branches=args.num_branches,
        divide_and_encode_hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        input_shape=input_shape,
    ).to(DEVICE)

    adam = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    triplet_loss = nn.TripletMarginLoss()

    train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizer=adam,
        loss_fn=triplet_loss,
        log_dir=args.log_dir,
        tensorboard=args.tensorboard,
        device=DEVICE,
        num_epochs=args.num_epochs,
        alpha=args.alpha,
    )

    torch.save(model.state_dict(), args.out)
    print(f"Model saved to {args.out}")
