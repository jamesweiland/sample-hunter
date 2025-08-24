import torch.nn as nn
import torch
from typing import Callable, Literal, Tuple

from sample_hunter.config import DEFAULT_TRIPLET_LOSS_MARGIN, DEFAULT_MINE_STRATEGY

from .triplet_loss import (
    triplet_accuracy,
    mine_negative,
    topk_triplet_accuracy,
    song_accuracy,
    topk_song_accuracy,
)
from sample_hunter._util import DEVICE


def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    top_k: int,
    margin: float = DEFAULT_TRIPLET_LOSS_MARGIN,
    device: str = DEVICE,
) -> Tuple[float, float, float, float, float]:
    """
    Evaluate on a test dataset and return the average
    accuracy for the dataset
    """
    with torch.no_grad():

        sum_triplet_accuracy = 0.0
        sum_song_accuracy = 0.0
        sum_loss = 0.0
        sum_topk_triplet_accuracy = 0.0
        sum_topk_song_accuracy = 0.0
        num_batches = 0
        for anchor, positive, keys in dataloader:
            anchor = anchor.to(device)
            positive = positive.to(device)
            keys = keys.to(device)

            (
                batch_loss,
                batch_triplet_accuracy,
                batch_topk_triplet_accuracy,
                batch_song_accuracy,
                batch_topk_song_accuracy,
            ) = evaluate_batch(
                model=model,
                positive=positive,
                anchor=anchor,
                song_ids=keys,
                top_k=top_k,
                margin=margin,
                device=device,
            )
            sum_triplet_accuracy += batch_triplet_accuracy
            sum_loss += batch_loss
            sum_topk_triplet_accuracy += batch_topk_triplet_accuracy
            sum_song_accuracy += batch_song_accuracy
            sum_topk_song_accuracy += batch_topk_song_accuracy
            num_batches += 1

        avg_triplet_accuracy = sum_triplet_accuracy / num_batches
        avg_loss = sum_loss / num_batches
        avg_topk_triplet_accuracy = sum_topk_triplet_accuracy / num_batches
        avg_song_accuracy = sum_song_accuracy / num_batches
        avg_topk_song_accuracy = sum_song_accuracy / num_batches
        print(f"Average test loss: {avg_loss}")
        print(f"Average test triplet accuracy: {avg_triplet_accuracy:.2%}")
        print(f"Average top K triplet accuracy: {avg_topk_triplet_accuracy:.2%}")
        print(f"Average test song accuracy: {avg_song_accuracy:.2%}")
        print(f"Average top K song accuracy: {avg_topk_song_accuracy:.2%}")

        return avg_loss, avg_triplet_accuracy, avg_topk_triplet_accuracy, avg_song_accuracy, avg_topk_song_accuracy  # type: ignore


def evaluate_batch(
    model: nn.Module,
    positive: torch.Tensor,
    song_ids: torch.Tensor,
    anchor: torch.Tensor,
    top_k: int,
    mine_strategy: Literal["semi", "hard"] = DEFAULT_MINE_STRATEGY,
    margin: float = DEFAULT_TRIPLET_LOSS_MARGIN,
    loss_fn: Callable = torch.nn.TripletMarginLoss(margin=DEFAULT_TRIPLET_LOSS_MARGIN),
    device: str = DEVICE,
    debug: bool = False,
) -> Tuple[float, float | torch.Tensor, float, float, float]:
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
            margin=margin,
        )

        triplet_accuracy_ = triplet_accuracy(
            anchor=anchor_embeddings,
            positive=positive_embeddings,
            negative=negative_embeddings,
            margin=margin,
            debug=debug,
        )

        topk_triplet_accuracy_ = topk_triplet_accuracy(
            anchor_embeddings, positive_embeddings, top_k
        )

        song_accuracy_ = song_accuracy(anchor_embeddings, positive_embeddings, song_ids)

        topk_song_accuracy_ = topk_song_accuracy(
            anchor_embeddings, positive_embeddings, song_ids, top_k
        )

        loss = loss_fn(
            anchor_embeddings, positive_embeddings, negative_embeddings
        ).item()
        if debug:
            print(f"Result shape: {accuracy.shape}")  # type: ignore
        return (
            loss,
            triplet_accuracy_,
            topk_triplet_accuracy_,
            song_accuracy_,
            topk_song_accuracy_,
        )
