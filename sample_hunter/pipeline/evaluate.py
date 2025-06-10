import torch.nn as nn
from torch import Tensor

from sample_hunter.pipeline.train import triplet_accuracy, mine_negative_triplet
from sample_hunter._util import DEVICE


def evaluate(
    model: nn.Module,
    positive: Tensor,
    song_ids: Tensor,
    anchor: Tensor,
    alpha: float,
    device: str,
) -> float:
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
