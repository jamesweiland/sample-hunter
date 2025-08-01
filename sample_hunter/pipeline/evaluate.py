import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from typing import Literal

from sample_hunter.config import DEFAULT_TRIPLET_LOSS_MARGIN, DEFAULT_MINE_STRATEGY

from .data_loading import flatten_sub_batches
from .triplet_loss import triplet_accuracy, mine_negative
from sample_hunter._util import DEVICE


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    alpha: float = DEFAULT_TRIPLET_LOSS_MARGIN,
    device: str = DEVICE,
) -> float:
    """
    Evaluate on a test dataset and return the average
    accuracy for the dataset
    """
    with torch.no_grad():

        sum_accuracy = 0.0
        num_batches = 0
        for anchor, positive, keys in flatten_sub_batches(dataloader):
            anchor = anchor.to(device)
            positive = positive.to(device)
            keys = keys.to(device)

            batch_accuracy = evaluate_batch(
                model=model,
                positive=positive,
                anchor=anchor,
                song_ids=keys,
                alpha=alpha,
                device=device,
            )
            sum_accuracy += batch_accuracy
            num_batches += 1

        avg_accuracy = sum_accuracy / num_batches
        print(f"Average test accuracy: {avg_accuracy}")
        return avg_accuracy  # type: ignore


def evaluate_batch(
    model: nn.Module,
    positive: torch.Tensor,
    song_ids: torch.Tensor,
    anchor: torch.Tensor,
    mine_strategy: Literal["semi", "hard"] = DEFAULT_MINE_STRATEGY,
    alpha: float = DEFAULT_TRIPLET_LOSS_MARGIN,
    device: str = DEVICE,
    debug: bool = False,
) -> float | torch.Tensor:
    """
    Evaluate a single batch of tensors, and return the average triplet accuracy of the batch
    """
    assert mine_strategy in ["semi", "hard"]
    with torch.no_grad():

        model.eval()
        positive, anchor = positive.to(device), anchor.to(device)

        positive_embeddings = model(positive)
        anchor_embeddings = model(anchor)

        negative_embeddings = mine_negative(
            song_ids,
            positive_embeddings,
            anchor_embeddings,
            mine_strategy=mine_strategy,
            alpha=alpha,
        )

        res = triplet_accuracy(
            anchor=anchor_embeddings,
            positive=positive_embeddings,
            negative=negative_embeddings,
            alpha=alpha,
            debug=debug,
        )
        if debug:
            print(f"Result shape: {res.shape}")  # type: ignore
        return res
