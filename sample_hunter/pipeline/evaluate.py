import torch.nn as nn
import torch
from typing import Callable, Literal, Tuple

from sample_hunter.config import DEFAULT_TRIPLET_LOSS_MARGIN, DEFAULT_MINE_STRATEGY

from .data_loading import flatten
from .triplet_loss import triplet_accuracy, mine_negative, topk_triplet_accuracy
from sample_hunter._util import DEVICE


def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    sub_batch_size: int,
    mine_strategy: Literal["semi", "hard"] = DEFAULT_MINE_STRATEGY,
    margin: float = DEFAULT_TRIPLET_LOSS_MARGIN,
    device: str = DEVICE,
) -> Tuple[float, float, float]:
    """
    Evaluate on a test dataset and return the average
    accuracy for the dataset
    """
    with torch.no_grad():

        sum_accuracy = 0.0
        sum_loss = 0.0
        sum_topk_accuracy = 0.0
        num_batches = 0
        for batch in dataloader:
            sub_batches = flatten(batch, sub_batch_size)
            for anchor, positive, keys in sub_batches:
                anchor = anchor.to(device)
                positive = positive.to(device)
                keys = keys.to(device)

                batch_loss, batch_accuracy, batch_topk_accuracy = evaluate_batch(
                    model=model,
                    positive=positive,
                    anchor=anchor,
                    mine_strategy=mine_strategy,
                    song_ids=keys,
                    margin=margin,
                    device=device,
                )
                sum_accuracy += batch_accuracy
                sum_loss += batch_loss
                sum_topk_accuracy += batch_topk_accuracy
                num_batches += 1

        avg_accuracy = sum_accuracy / num_batches
        avg_loss = sum_loss / num_batches
        avg_topk_accuracy = sum_topk_accuracy / num_batches
        print(f"Average test loss: {avg_loss}")
        print(f"Average test accuracy: {avg_accuracy:.2%}")
        print(f"Average top K accuracy: {avg_topk_accuracy:.2%}")
        return avg_loss, avg_accuracy, avg_topk_accuracy  # type: ignore


def evaluate_batch(
    model: nn.Module,
    positive: torch.Tensor,
    song_ids: torch.Tensor,
    anchor: torch.Tensor,
    mine_strategy: Literal["semi", "hard"] = DEFAULT_MINE_STRATEGY,
    margin: float = DEFAULT_TRIPLET_LOSS_MARGIN,
    loss_fn: Callable = torch.nn.TripletMarginLoss(margin=DEFAULT_TRIPLET_LOSS_MARGIN),
    device: str = DEVICE,
    debug: bool = False,
) -> Tuple[float, float | torch.Tensor, float]:
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

        accuracy = triplet_accuracy(
            anchor=anchor_embeddings,
            positive=positive_embeddings,
            negative=negative_embeddings,
            margin=margin,
            debug=debug,
        )

        topk_accuracy = topk_triplet_accuracy(anchor_embeddings, positive_embeddings)

        loss = loss_fn(
            anchor_embeddings, positive_embeddings, negative_embeddings
        ).item()
        if debug:
            print(f"Result shape: {accuracy.shape}")  # type: ignore
        return loss, accuracy, topk_accuracy
