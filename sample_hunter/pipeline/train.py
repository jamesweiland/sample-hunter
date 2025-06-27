from huggingface_hub import HfApi
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from typing import Callable, List, Tuple
import argparse
from pathlib import Path
from tqdm import tqdm
import webdataset as wds


from .encoder_net import EncoderNet
from .transformations.functional import collate_spectrograms, flatten_sub_batches
from .transformations.spectrogram_preprocessor import SpectrogramPreprocessor
from .triplet_loss import triplet_accuracy, mine_negative_triplet
from .evaluate import evaluate
from sample_hunter._util import (
    DEVICE,
    HF_TOKEN,
)
from sample_hunter.cfg import config


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
    for anchor, positive, keys in tqdm(
        flatten_sub_batches(dataloader), desc="Training epoch..."
    ):
        anchor_batch = anchor.to(device)
        positive_batch = positive.to(device)
        keys = keys.to(device)

        # predict embeddings
        anchor_embeddings = model(anchor_batch)
        positive_embeddings = model(positive_batch)

        # mine the negative embedding
        negative_embeddings = mine_negative_triplet(
            anchor_embeddings=anchor_embeddings,
            positive_embeddings=positive_embeddings,
            song_ids=keys,
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
    num_epochs: range,
    alpha: float,
    tensorboard: str = "none",
    log_dir: Path | None = None,
    test_dataloader: DataLoader | None = None,
    save_per_epoch: Path | None = None,
    device: str = DEVICE,
):
    """
    `model`: The neural network to train.

    `train_dataloader`: A `torch.util.data.DataLoader` made from a train dataset. This dataloader must
    yield a (anchor, positive, key) triplet.

    `loss_fn`: The loss function to use for learning. Must be a triplet loss function or a derivation of it.

    `optimizer`: A `torch.optim.Optimizer` instance.

    `num_epochs`: A range object that represents the number of epochs to train.

    `alpha`: The alpha margin to use for triplet mining per-batch.

    `tensorboard`: Must be one of 'none', 'batch', or 'epoch'. If 'none', tensorboard will not be used.
    If `batch`, tensorboard metrics will be logged per-batch. If `epoch`, they will be logged per epoch.

    `log_dir`: If `tensorboard` is not `none`, then the path to the log_dir must be specified as well.

    `test_dataloader`: If given, the model will also be evaluated per epoch on a test dataset.

    `save_per_epoch`: If given, save the model to the provided path's directory at the end of each epoch,
    with '-<epoch_num>' appended to the end of the given path's stem.

    `device`: The device to use.
    """
    if tensorboard != "none":
        if tensorboard != "batch" and tensorboard != "epoch":
            raise ValueError(
                "An illegal option was given for tensorboard.\n"
                'tensorboard must be one of: ["none", "batch", "epoch"]\n'
                f"tensorboard was given as: {tensorboard}"
            )
        if log_dir == None:
            raise ValueError(
                "log_dir must be specified in order to use tensorboard.\n"
                f"tensorboard was given as: {tensorboard}"
            )

        writer = SummaryWriter(log_dir=log_dir)

    for i in num_epochs:
        print(f"Epoch {i}")
        loss, accuracy = train_single_epoch(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            alpha=alpha,
        )
        if tensorboard == "epoch":
            writer.add_scalar("Training loss", loss, i)  # type: ignore
            writer.add_scalar("Training accuracy", accuracy, i)  # type: ignore

        if test_dataloader is not None:
            # evaluate accuracy as we go on the test set
            accuracy = evaluate(
                model=model, dataloader=test_dataloader, alpha=alpha, device=device
            )
            if tensorboard != "none":
                writer.add_scalar("Testing accuracy", accuracy, i)  # type: ignore

        if save_per_epoch is not None and i != num_epochs.stop - 1:
            save_path = (
                save_per_epoch.parent
                / f"{save_per_epoch.stem}-{i}{save_per_epoch.suffix}"
            )
            torch.save(model.state_dict(), save_path)

        print("--------------------------------------------")
    if tensorboard != "none":
        writer.close()  # type: ignore
    print("Finished training")


def get_tar_files(repo_id: str, split: str, token: str) -> List[str]:
    api = HfApi()

    files = api.list_repo_files(repo_id, repo_type="dataset", token=token)
    tar_files = [
        file for file in files if file.startswith(f"{split}/") and file.endswith(".tar")
    ]
    return tar_files


def load_webdataset(repo_id: str, split: str, token: str) -> wds.WebDataset:
    tar_files = get_tar_files(repo_id, split, token)
    # get all the numbers of the files
    numbers = [int(name.split("-")[-1].split(".tar")[0]) for name in tar_files]
    max_num = max(numbers)

    max_num_str = f"{max_num:06d}"
    pattern = f"{split}-{{000001..{max_num_str}}}.tar"

    url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{split}/{pattern}"
    pipe = f"pipe:curl -s -L {url} -H 'Authorization:Bearer {token}'"
    return wds.WebDataset(pipe, shardshuffle=True).shuffle(200).decode()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--token", type=str, help="Your huggingface token", default=HF_TOKEN
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        help="The number of processes to use for dataloading",
        default=1,
    )

    parser.add_argument(
        "--from",
        dest="from_",
        type=Path,
        help="Option to load in a model to continue training instead of starting a new model",
    )

    tensorboard = parser.add_argument_group("tensorboard")
    tensorboard.add_argument(
        "--log-dir",
        type=Path,
        help="The path to the log directory for tensorboard",
        default=config.paths.log_dir,
    )
    tensorboard.add_argument(
        "--tensorboard",
        help="Specify whether to write tensorboard data per batch, epoch, or not at all",
        type=str,
        choices=["none", "batch", "epoch"],
        default="none",
    )

    dev = parser.add_argument_group("dev")
    dev.add_argument("--num", help="train only a sample of the dataset", type=int)
    dev.add_argument(
        "--save-per-epoch",
        help="Option to save the model upon training each epoch",
        action="store_true",
    )
    dev.add_argument(
        "--continue",
        dest="continue_",
        help="Option to continue from the last trained epoch to the config's num_epochs, if the saved model has a filename ending in -<last_trained_epoch>",
        action="store_true",
    )

    parser.add_argument("out", type=Path, help="The path to save the trained model to")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    with SpectrogramPreprocessor() as preprocessor:

        def map_fn(ex):
            positive, anchor = preprocessor(ex["mp3"], obfuscate=True)
            return {**ex, "positive": positive, "anchor": anchor}

        def collate_fn(songs):
            keys = torch.tensor([int(song["__key__"]) for song in songs])
            windows_per_song = [song["anchor"].shape[0] for song in songs]
            keys = torch.repeat_interleave(keys, torch.tensor(windows_per_song))
            key_splits = keys.split(config.network.sub_batch_size)

            specs = collate_spectrograms(songs, col=["anchor", "positive"])

            assert len(specs) == len(key_splits)

            return [
                (anchor, positive, k)
                for (anchor, positive), k in zip(specs, key_splits)
            ]

        train_dataset = load_webdataset(config.hf.repo_id, "train", args.token).map(
            map_fn
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.network.source_batch_size,
            collate_fn=collate_fn,
        )

        test_dataset = load_webdataset(config.hf.repo_id, "test", args.token).map(
            map_fn
        )

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=config.network.source_batch_size,
            collate_fn=collate_fn,
        )

        if args.num:
            train_dataset = train_dataset.shuffle()
            train_dataset = train_dataset.take(args.num)

        if args.continue_:
            if not args.from_:
                raise ValueError("--continue was given with no model to load in")
            if not args.from_.exists():
                raise ValueError(f"--from not found: {args.from_}")
            if (
                str(args.from_.stem).count("-") != 1
                and not str(args.from_.stem).split("-")[-1].isdigit()
            ):
                raise ValueError(
                    f"--from has a stem that does not follow the correct formatting: {args.from_.stem}\n"
                    "The stem must follow the format: <stem_name>-<epoch>"
                )

            epochs_already_trained = int(str(args.from_).split("-")[-1])
            if epochs_already_trained >= config.network.num_epochs:
                raise ValueError(
                    "The model has already been trained for more epochs than specified in the config for num_epochs\n"
                    f"Epochs already trained: {epochs_already_trained}\n"
                    f"Config num_epochs: {config.network.num_epochs}"
                )

            model = torch.load(args.from_, weights_only=False).to(DEVICE)
            num_epochs = range(
                epochs_already_trained + 1, config.network.num_epochs + 1
            )
        elif args.from_:
            if not args.from_.exists():
                raise ValueError(f"--from not found: {args.from_}")

            model = torch.load(args.from_, weights_only=False).to(DEVICE)
            num_epochs = range(1, config.network.num_epochs + 1)
        else:
            # make a new model
            model = EncoderNet().to(DEVICE)
            num_epochs = range(1, config.network.num_epochs + 1)

        save_per_epoch = args.out if args.save_per_epoch else None

        model = EncoderNet().to(DEVICE)

        adam = torch.optim.Adam(model.parameters(), lr=config.network.learning_rate)
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
            num_epochs=num_epochs,
            alpha=config.network.alpha,
            save_per_epoch=save_per_epoch,
        )

        torch.save(model.state_dict(), args.out)
        print(f"Model saved to {args.out}")
