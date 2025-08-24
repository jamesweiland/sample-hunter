import gc
import uuid
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from typing import Any, Callable, Dict, List, Tuple, cast
import argparse
from pathlib import Path
import webdataset as wds
from tqdm import tqdm

from sample_hunter.pipeline.transformations.obfuscator import Obfuscator
from sample_hunter.pipeline.transformations.preprocessor import Preprocessor

from .data_loading import load_tensor_from_mp3_bytes, load_webdataset, flatten
from .encoder_net import EncoderNet
from .triplet_loss import topk_triplet_accuracy, triplet_accuracy, mine_negative
from .evaluate import evaluate
from sample_hunter.config import (
    PreprocessConfig,
    TrainConfig,
    ObfuscatorConfig,
    EncoderNetConfig,
    DEFAULT_DATASET_REPO,
    DEFAULT_TOP_K,
)
from sample_hunter._util import (
    DEVICE,
    HF_TOKEN,
    load_model,
)


def train_single_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: Callable[..., Tensor],
    batch_size: int,
    mine_strategy: str,
    optimizer: torch.optim.Optimizer,
    device: str,
    triplet_loss_margin: float,
    writer: SummaryWriter | None = None,
) -> Tuple[float, float, float]:
    """
    Train `model` for a single epoch. Returns a (loss, accuracy) tuple
    """
    model.train()

    num_batches = 0
    epoch_total_accuracy = 0.0
    epoch_total_topk_accuracy = 0.0
    epoch_total_loss = 0.0
    for batch in dataloader:
        sub_batches = flatten(batch, batch_size)
        for anchor, positive, keys in tqdm(sub_batches, desc="Training..."):
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

            topk_accuracy = topk_triplet_accuracy(
                anchor_embeddings, positive_embeddings
            )

            if writer:
                writer.add_scalar("Training loss", loss.item(), num_batches)
                writer.add_scalar("Training accuracy", accuracy, num_batches)
                writer.add_scalar("Training top K accuracy", topk_accuracy, num_batches)

            epoch_total_loss += loss.item()
            epoch_total_accuracy += accuracy
            epoch_total_topk_accuracy += topk_accuracy
            num_batches += 1

        del batch
        del sub_batches
        gc.collect()

    epoch_average_loss = epoch_total_loss / num_batches
    print(f"Average loss of epoch: {epoch_average_loss}")
    epoch_average_accuracy = epoch_total_accuracy / num_batches
    print(f"Epoch accuracy: {epoch_average_accuracy:.2%}")
    epoch_average_topk_accuracy = epoch_total_topk_accuracy / num_batches
    print(f"Epoch top K accuracy: {epoch_average_topk_accuracy:.2%}")
    return epoch_average_loss, epoch_average_accuracy, epoch_average_topk_accuracy  # type: ignore


def train(
    model: nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    loss_fn: Callable,
    mine_strategy: str,
    sub_batch_size: int,
    optimizer: torch.optim.Optimizer,
    num_epochs: range,
    triplet_loss_margin: float,
    k: int = DEFAULT_TOP_K,
    tensorboard: str = "none",
    log_dir: Path | None = None,
    test_dataloader: torch.utils.data.DataLoader | None = None,
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

    model.train()

    for i in num_epochs:
        print(f"Epoch {i}")
        loss, accuracy, topk_accuracy = train_single_epoch(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            batch_size=sub_batch_size,
            mine_strategy=mine_strategy,
            optimizer=optimizer,
            device=device,
            triplet_loss_margin=triplet_loss_margin,
            writer=writer if tensorboard == "batch" else None,
        )
        if tensorboard == "epoch":
            writer.add_scalar("Training loss", loss, i)  # type: ignore
            writer.add_scalar("Training accuracy", accuracy, i)  # type: ignore
            writer.add_scalar("Training top K accuracy", topk_accuracy, i)  # type: ignore

        if save_per_epoch is not None and i != num_epochs.stop - 1:
            save_path = (
                save_per_epoch.parent
                / f"{save_per_epoch.stem}-{i}{save_per_epoch.suffix}"
            )
            torch.save(model.state_dict(), save_path)

        if test_dataloader is not None:
            # evaluate accuracy as we go on the test set
            (
                test_loss,
                test_triplet_accuracy,
                test_topk_triplet_accuracy,
                test_song_accuracy,
                test_topk_song_accuracy,
            ) = evaluate(
                model=model,
                dataloader=test_dataloader,
                k=k,
                sub_batch_size=sub_batch_size,
                margin=triplet_loss_margin,
                device=device,
            )
            if tensorboard != "none":
                writer.add_scalar("Testing triplet accuracy", test_triplet_accuracy, i)  # type: ignore
                writer.add_scalar("Testing loss", test_loss, i)  # type: ignore
                writer.add_scalar("Testing top K triplet accuracy", test_topk_triplet_accuracy, i)  # type: ignore
                writer.add_scalar("Testing song accuracy", test_song_accuracy, i)  # type: ignore
                writer.add_scalar("Testing top K song accuracy", test_topk_song_accuracy, i)  # type: ignore

        print("--------------------------------------------")
    if tensorboard != "none":
        writer.close()  # type: ignore
    print("Finished training")


def train_collate_fn(
    batch: List[Dict[str, Any]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.no_grad():

        num_examples_per_song = torch.tensor(
            [ex["anchor"].shape[1] for ex in batch], device="cpu"
        )

        anchors = torch.cat([ex["anchor"].view(ex["anchor"].shape[1:]) for ex in batch])
        index = torch.randperm(anchors.shape[0])
        anchors = anchors[index]
        gc.collect()
        positives = torch.cat(
            [ex["positive"].view(ex["positive"].shape[1:]) for ex in batch]
        )
        positives = positives[index]
        gc.collect()
        uuids = [uuid.UUID(ex["json"]["id"][0]) for ex in batch]
        uuid_to_int = {u: i for i, u in enumerate(uuids)}
        uuids = torch.tensor([uuid_to_int[u] for u in uuids], device="cpu")
        uuids = torch.repeat_interleave(uuids, num_examples_per_song)
        uuids = uuids[index]
        del batch
        gc.collect()

    return anchors, positives, uuids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

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
        train_config = TrainConfig.from_yaml(args.config)
        encoder_net_config = EncoderNetConfig.from_yaml(args.config)
        preprocess_config = PreprocessConfig.from_yaml(args.config)
        obfuscator_config = ObfuscatorConfig.from_yaml(args.config)
    else:
        train_config = TrainConfig()
        encoder_net_config = EncoderNetConfig()
        preprocess_config = PreprocessConfig()
        obfuscator_config = ObfuscatorConfig()

    # load the datasets, try to get them local first and if not, load from hf
    if Path(args.dataset).exists():
        # load locally
        dataset_dir = Path(args.dataset)

        train_tars = (dataset_dir / "train").glob("*.tar")
        test_tars = (dataset_dir / "test").glob("*.tar")

        # have to convert the paths to str
        train_tars = [str(tar) for tar in train_tars]
        test_tars = [str(tar) for tar in test_tars]

        train_dataset = wds.WebDataset(train_tars).shuffle(200).decode()
        test_dataset = wds.WebDataset(test_tars).shuffle(200).decode()

    else:
        # load from hf
        train_dataset = (
            cast(
                wds.WebDataset,
                load_webdataset(
                    args.dataset,
                    "train",
                    token=args.token,
                    cache_dir=train_config.cache_dir,
                ),
            )
            .shuffle(200)
            .decode()
        )
        test_dataset = (
            cast(
                wds.WebDataset,
                load_webdataset(
                    args.dataset,
                    "test",
                    token=args.token,
                    cache_dir=train_config.cache_dir,
                ),
            )
            .shuffle(200)
            .decode()
        )

    with Preprocessor(
        config=preprocess_config, obfuscator=Obfuscator(config=obfuscator_config)
    ) as preprocessor:
        with tqdm(
            desc="preprocessing examples...", total=train_config.source_batch_size
        ) as pbar:

            def train_map_fn(example):
                with torch.no_grad():
                    audio, sr = load_tensor_from_mp3_bytes(example["mp3"], DEVICE)
                    anchor, positive = preprocessor(audio, sample_rate=sr, train=True)
                    example["anchor"] = torch.quantize_per_tensor(
                        anchor.to("cpu"),
                        train_config.quantize_scale,
                        train_config.zero_point,
                        torch.qint8,
                    )
                    example["positive"] = torch.quantize_per_tensor(
                        positive.to("cpu"),
                        train_config.quantize_scale,
                        train_config.zero_point,
                        torch.qint8,
                    )

                    pbar.update(1)

                    if (pbar.n - 1) == pbar.total:
                        pbar.reset()

                return example

            train_dataset = train_dataset.map(train_map_fn)
            test_dataset = test_dataset.map(train_map_fn)

            if args.num:
                train_dataset = train_dataset.slice(args.num)

            train_loader = wds.WebLoader(
                train_dataset,
            ).batched(train_config.source_batch_size, collation_fn=train_collate_fn)

            test_loader = wds.WebLoader(
                test_dataset,
            ).batched(train_config.source_batch_size, collation_fn=train_collate_fn)

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
                num_epochs = range(
                    epochs_already_trained + 1, train_config.num_epochs + 1
                )
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
                train_dataloader=train_loader,
                test_dataloader=test_loader,
                optimizer=adam,
                loss_fn=triplet_loss,
                sub_batch_size=train_config.sub_batch_size,
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
