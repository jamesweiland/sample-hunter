import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from sample_hunter.pipeline.triplet_loss import triplet_accuracy, mine_negative_triplet
from sample_hunter._util import DEVICE, ALPHA, ALPHA


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    alpha: float = ALPHA,
    device: str = DEVICE,
) -> float:
    """
    Evaluate on a test dataset and return the average
    accuracy for the dataset
    """

    sum_accuracy = 0.0
    num_batches = 0
    for anchor, positive, song_ids in dataloader:
        batch_accuracy = evaluate_batch(
            model=model,
            positive=positive,
            anchor=anchor,
            song_ids=song_ids,
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
    positive: Tensor,
    song_ids: Tensor,
    anchor: Tensor,
    alpha: float,
    device: str,
) -> float:
    """
    Train a single batch of tensors, and return the average triplet accuracy of the batch
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
