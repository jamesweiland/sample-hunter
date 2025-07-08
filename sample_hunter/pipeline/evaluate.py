import torch.nn as nn
import torch
from torch.utils.data import DataLoader

from .data_loading import flatten_sub_batches
from .triplet_loss import triplet_accuracy, mine_negative_triplet
from sample_hunter._util import DEVICE
from sample_hunter.cfg import config


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    alpha: float = config.network.alpha,
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
        return avg_accuracy


def evaluate_batch(
    model: nn.Module,
    positive: torch.Tensor,
    song_ids: torch.Tensor,
    anchor: torch.Tensor,
    alpha: float,
    device: str,
) -> float:
    """
    Evaluate a single batch of tensors, and return the average triplet accuracy of the batch
    """

    model.eval()
    positive, anchor = positive.to(device), anchor.to(device)

    positive_embeddings = model(positive)
    anchor_embeddings = model(anchor)

    negative_embeddings = mine_negative_triplet(
        anchor_embeddings=anchor_embeddings,
        positive_embeddings=positive_embeddings,
        song_ids=song_ids,
        alpha=alpha,
    )

    return triplet_accuracy(
        anchor=anchor_embeddings,
        positive=positive_embeddings,
        negative=negative_embeddings,
        alpha=alpha,
    )
