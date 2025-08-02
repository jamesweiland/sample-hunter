import json
import traceback
import uuid
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from typing import Callable, Tuple, cast
import argparse
from pathlib import Path
from tqdm.notebook import tqdm
import webdataset as wds

from .data_loading import load_webdataset
from .encoder_net import EncoderNet
from .data_loading import collate_spectrograms, flatten_sub_batches
from .transformations.obfuscator import Obfuscator
from .transformations.preprocessor import Preprocessor
from .triplet_loss import triplet_accuracy, mine_negative
from .evaluate import evaluate
from sample_hunter.config import (
    PreprocessConfig,
    TrainConfig,
    ObfuscatorConfig,
    EncoderNetConfig,
    DEFAULT_DATASET_REPO,
)
from sample_hunter._util import (
    DEVICE,
    HF_TOKEN,
    load_model,
)


def train_single_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: Callable[..., Tensor],
    mine_strategy: str,
    optimizer: torch.optim.Optimizer,
    device: str,
    triplet_loss_margin: float,
    writer: SummaryWriter | None = None,
) -> Tuple[float, float]:
    """
    Train `model` for a single epoch. Returns a (loss, accuracy) tuple
    """
    model.train()
    epoch_total_loss = 0
    num_batches = 0
    epoch_total_accuracy = 0
    for anchor, positive, keys in flatten_sub_batches(dataloader):
        anchor_batch = anchor.to(device)
        positive_batch = positive.to(device)
        keys = keys.to(device)

        # predict embeddings
        anchor_embeddings = model(anchor_batch)
        positive_embeddings = model(positive_batch)

        # mine the negative embedding
        negative_embeddings = mine_negative(
            anchor_embeddings=anchor_embeddings,
            positive_embeddings=positive_embeddings,
            song_ids=keys,
            mine_strategy=mine_strategy,
            margin=triplet_loss_margin,
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
            margin=triplet_loss_margin,
        )

        if writer:
            writer.add_scalar("Training loss", loss.item(), num_batches)
            writer.add_scalar("Training accuracy", accuracy, num_batches)

        epoch_total_loss += loss.item()
        epoch_total_accuracy += accuracy
        num_batches += 1

    epoch_average_loss = epoch_total_loss / num_batches
    print(f"Average loss of epoch: {epoch_average_loss}")
    epoch_average_accuracy = epoch_total_accuracy / num_batches
    print(f"Epoch accuracy: {epoch_average_accuracy}")
    return epoch_average_loss, epoch_average_accuracy  # type: ignore


def train(
    model: nn.Module,
    train_dataloader: DataLoader,
    loss_fn: Callable,
    mine_strategy: str,
    optimizer: torch.optim.Optimizer,
    num_epochs: range,
    triplet_loss_margin: float,
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
    writer = None
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
            mine_strategy=mine_strategy,
            optimizer=optimizer,
            device=device,
            triplet_loss_margin=triplet_loss_margin,
            writer=writer if tensorboard == "batch" else None,
        )
        if tensorboard == "epoch":
            writer.add_scalar("Training loss", loss, i)  # type: ignore
            writer.add_scalar("Training accuracy", accuracy, i)  # type: ignore

        if save_per_epoch is not None and i != num_epochs.stop - 1:
            save_path = (
                save_per_epoch.parent
                / f"{save_per_epoch.stem}-{i}{save_per_epoch.suffix}"
            )
            torch.save(model.state_dict(), save_path)

        if test_dataloader is not None:
            # evaluate accuracy as we go on the test set
            accuracy = evaluate(
                model=model,
                dataloader=test_dataloader,
                alpha=triplet_loss_margin,
                device=device,
            )
            if tensorboard != "none":
                writer.add_scalar("Testing accuracy", accuracy, i)  # type: ignore

        print("--------------------------------------------")
    if tensorboard != "none":
        writer.close()  # type: ignore
    print("Finished training")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num-workers",
        type=int,
        help="The number of processes to use for dataloading",
    )

    parser.add_argument(
        "--config",
        type=Path,
        help="The path to a configuration file to instantiate the EncoderNetConfig and TrainConfig",
    )

    parser.add_argument(
        "--from",
        dest="from_",
        type=Path,
        help="Option to load in a model to continue training instead of starting a new model",
    )

    hf = parser.add_argument_group("hf")
    hf.add_argument(
        "--token", type=str, help="Your huggingface token", default=HF_TOKEN
    )
    hf.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET_REPO,
        help="The dataset to use for training and testing. Either a HF repo id or a local path",
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

    if args.config:
        preprocess_config = PreprocessConfig.from_yaml(args.config)
        train_config = TrainConfig.from_yaml(args.config)
        obfuscator_config = ObfuscatorConfig.from_yaml(args.config)
        encoder_net_config = EncoderNetConfig.from_yaml(args.config)
    else:
        preprocess_config = PreprocessConfig()
        train_config = TrainConfig()
        obfuscator_config = ObfuscatorConfig()
        encoder_net_config = EncoderNetConfig()

    batch_num = 1
    with tqdm(
        desc=f"Preprocessing batch {batch_num}",
        total=train_config.source_batch_size,
    ) as pbar:

        def map_fn(ex):
            with Preprocessor(
                preprocess_config, obfuscator=Obfuscator(obfuscator_config)
            ) as preprocessor:
                try:
                    if isinstance(ex["json"], bytes):
                        ex["json"] = json.loads(ex["json"].decode("utf-8"))
                    positive, anchor = preprocessor(ex["mp3"], train=True)
                    pbar.update()
                    return {**ex, "positive": positive, "anchor": anchor}
                except Exception as e:
                    print(f"An error occurred trying to process {ex["json"]["title"]}")
                    print(str(e))

                    traceback.print_exc()
                    return ex

        def collate_fn(songs):
            pbar.reset()
            global batch_num
            batch_num += 1

            # filter out songs where preprocessing failed
            songs = [song for song in songs if "anchor" in song and "positive" in song]

            keys = [song["json"]["id"] for song in songs]
            keys = [uuid.UUID(key).int for key in keys]
            # one-hot encode keys as int64
            uuid_to_key = {u: i for i, u in enumerate(keys)}
            keys = torch.tensor([uuid_to_key[u] for u in keys])

            windows_per_song = torch.tensor([song["anchor"].shape[0] for song in songs])
            keys = torch.repeat_interleave(keys, windows_per_song)

            anchors = torch.cat([song["anchor"] for song in songs], dim=0)
            positives = torch.cat([song["positive"] for song in songs], dim=0)
            sub_batches = collate_spectrograms(
                (anchors, positives, keys), train_config.sub_batch_size
            )

            return [(anchor, positive, k) for anchor, positive, k in sub_batches]

        # load the datasets, try to get them local first and if not, load from hf
        if Path(args.dataset).exists():
            # load locally
            dataset_dir = Path(args.dataset)

            train_tars = (dataset_dir / "train").glob("*.tar")
            test_tars = (dataset_dir / "test").glob("*.tar")

            # have to convert the paths to str
            train_tars = [str(tar) for tar in train_tars]
            test_tars = [str(tar) for tar in test_tars]

            train_dataset = wds.WebDataset(train_tars)
            test_dataset = wds.WebDataset(test_tars)

        else:
            # load from hf
            train_dataset = cast(
                wds.WebDataset,
                load_webdataset(
                    args.dataset,
                    "train",
                    token=args.token,
                    cache_dir=train_config.cache_dir,
                ),
            )
            test_dataset = cast(
                wds.WebDataset,
                load_webdataset(
                    args.dataset,
                    "test",
                    token=args.token,
                    cache_dir=train_config.cache_dir,
                ),
            )

        train_dataset = train_dataset.map(map_fn)
        test_dataset = test_dataset.map(map_fn)

        if args.num_workers and args.num_workers > 0:
            import torch.multiprocessing as mp

            ctx = mp.get_context("spawn")
        else:
            ctx = None

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=train_config.source_batch_size,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            multiprocessing_context=ctx,
        )

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=train_config.source_batch_size,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            multiprocessing_context=ctx,
        )

        if args.num:
            train_dataset = train_dataset.slice(args.num)

        if args.continue_:
            if not args.from_:
                raise ValueError("--continue was given with no model to load in")
            if not args.from_.exists():
                raise ValueError(f"--from not found: {args.from_}")
            if not str(args.from_.stem).split("-")[-1].isdigit():
                raise ValueError(
                    f"--from has a stem that does not follow the correct formatting: {args.from_.stem}\n"
                    "The stem must follow the format: <stem_name>-<epoch>"
                )

            epochs_already_trained = int(str(args.from_.stem).split("-")[-1])
            if epochs_already_trained >= train_config.num_epochs:
                raise ValueError(
                    "The model has already been trained for more epochs than specified in the config for num_epochs\n"
                    f"Epochs already trained: {epochs_already_trained}\n"
                    f"Config num_epochs: {train_config.num_epochs}"
                )

            model = load_model(args.from_, encoder_net_config)
            num_epochs = range(epochs_already_trained + 1, train_config.num_epochs + 1)
        elif args.from_:
            if not args.from_.exists():
                raise ValueError(f"--from not found: {args.from_}")

            model = load_model(args.from_, encoder_net_config)
            num_epochs = range(1, train_config.num_epochs + 1)
        else:
            # make a new model
            model = EncoderNet(encoder_net_config).to(DEVICE)  # type: ignore
            num_epochs = range(1, train_config.num_epochs + 1)

        save_per_epoch = args.out if args.save_per_epoch else None

        adam = torch.optim.Adam(model.parameters(), lr=train_config.learning_rate)
        triplet_loss = nn.TripletMarginLoss(margin=train_config.triplet_loss_margin)

        train(
            model=model,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            optimizer=adam,
            loss_fn=triplet_loss,
            mine_strategy=train_config.mine_strategy,
            log_dir=train_config.tensorboard_log_dir,
            tensorboard=train_config.tensorboard,
            device=DEVICE,
            num_epochs=num_epochs,
            triplet_loss_margin=train_config.triplet_loss_margin,
            save_per_epoch=save_per_epoch,
        )

        torch.save(model.state_dict(), args.out)
        print(f"Model saved to {args.out}")
