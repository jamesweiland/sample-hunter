"""
Triplet loss functions to be used in training and evaluation
"""

from typing import Tuple
import warnings
import torch

from sample_hunter.config import (
    DEFAULT_MINE_STRATEGY,
    DEFAULT_TRIPLET_LOSS_MARGIN,
    DEFAULT_TOP_K,
)


def triplet_accuracy(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    margin: float,
    debug: bool = False,
) -> float | torch.Tensor:
    """
    Calculates the accuracy of the embeddings by returning the ratio of positive embeddings
    closer to the anchor than negative ones. The positive embedding must be at least `alpha` closer
    to the anchor than the negative embedding

    anchor, positive, and negative are all tensors of shape (B, E)
    """
    pos_dists = torch.linalg.vector_norm(anchor - positive, ord=2, dim=1)
    neg_dists = torch.linalg.vector_norm(anchor - negative, ord=2, dim=1)

    correct = (pos_dists + margin) < neg_dists  # a boolean mask
    if debug:
        return correct
    else:
        return correct.float().mean().item()


def topk_triplet_accuracy(
    anchor: torch.Tensor, positive: torch.Tensor, top_k: int = DEFAULT_TOP_K
) -> float:
    """
    Calculates the accuracy of the embeddings by returning the ratio of positive embeddings that are in the
    'top k' nearest neighbors of the anchor embeddings.
    """

    dists = torch.cdist(anchor, positive, p=2)  # (B, B)

    try:
        nearest_neighbors = torch.topk(
            dists, top_k, dim=1, largest=False
        ).indices  # (B, k)
    except RuntimeError:
        # selected index k out of range
        nearest_neighbors = torch.topk(
            dists, dists.shape[1], dim=1, largest=False
        ).indices

    labels = torch.arange(
        0, anchor.shape[0], 1, device=anchor.device
    )  # these are the correct indices
    labels = labels.unsqueeze(1)  # (B, 1)

    correct = (labels == nearest_neighbors).any(dim=1)  # (B,)

    return correct.float().mean().item()


def song_accuracy(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    song_ids: torch.Tensor,
) -> float:
    dists = torch.cdist(anchor, positive, p=2)  # (B, B)
    nearest_neighbors = torch.min(dists, dim=1).indices  # (B,)

    predicted_song_ids = song_ids[nearest_neighbors]  # (B,)
    matches = song_ids == predicted_song_ids  # (B,)

    return matches.float().mean().item()


def topk_song_accuracy(
    anchor: torch.Tensor, positive: torch.Tensor, song_ids: torch.Tensor, k: int
) -> float:
    """Calculates the ratio of embeddings in positive_embeddings whose nearest neighbor has the same song id."""
    dists = torch.cdist(anchor, positive, p=2)  # (B, B)

    try:
        nearest_neighbors = torch.topk(dists, k, dim=1, largest=False).indices  # (B, k)
    except RuntimeError:
        # selected index k out of range
        nearest_neighbors = torch.topk(
            dists, dists.shape[1], dim=1, largest=False
        ).indices
    nearest_song_ids = song_ids[nearest_neighbors]

    # Expand song_ids to compare with each of the k nearest neighbors
    target_song_ids = song_ids.unsqueeze(1).expand(-1, k)  # (B, k)

    matches = (nearest_song_ids == target_song_ids).any(dim=1)  # (B,)
    accuracy = matches.float().mean().item()

    return accuracy


def make_triplets(
    song_ids: torch.Tensor,
    positive_embeddings: torch.Tensor,
    anchor_embeddings: torch.Tensor,
    filter: bool,
    mine_strategy: str = DEFAULT_MINE_STRATEGY,
    margin: float = DEFAULT_TRIPLET_LOSS_MARGIN,
):
    if mine_strategy == "semi":
        negative_embeddings = mine_semi_hard_negative(
            anchor_embeddings=anchor_embeddings,
            positive_embeddings=positive_embeddings,
            song_ids=song_ids,
            alpha=margin,
        )
    else:
        negative_embeddings = mine_hardest_negative(
            anchor_embeddings=anchor_embeddings,
            positive_embeddings=positive_embeddings,
            song_ids=song_ids,
        )

    if filter:
        # filter out any easy triplets
        return filter_triplets(
            anchor_embeddings, positive_embeddings, negative_embeddings, margin
        )
    else:
        return anchor_embeddings, positive_embeddings, negative_embeddings


def filter_triplets(
    anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, margin: float
) -> None | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Filter out possible triplets so we only have those that violate the triplet constraint of
    having the positive at least margin closer to the anchor than the negative.
    """

    pos_dists = torch.linalg.vector_norm(anchor - positive, ord=2, dim=1)
    neg_dists = torch.linalg.vector_norm(anchor - negative, ord=2, dim=1)

    violation_mask = (pos_dists + margin) >= neg_dists

    if violation_mask.sum() == 0:
        warnings.warn("No trainable triplets found in batch")
        return None

    anchor = anchor[violation_mask]
    positive = positive[violation_mask]
    negative = negative[violation_mask]

    return anchor, positive, negative


def mine_hardest_negative(
    anchor_embeddings: torch.Tensor,
    positive_embeddings: torch.Tensor,
    song_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Mine the hardest embedding from a batch of embeddings.

    Similar to mine_semi_hard_negative, except it considers all embeddings to mine from, not just those within a certain margin
    """
    with torch.no_grad():
        B = anchor_embeddings.shape[0]

        embeddings = torch.cat([anchor_embeddings, positive_embeddings], dim=0)
        doubled_song_ids = torch.cat([song_ids, song_ids], dim=0)

        dists = torch.cdist(embeddings, embeddings, p=2)

        valid_neg_mask = doubled_song_ids.unsqueeze(1) != doubled_song_ids.unsqueeze(0)

        negatives = _mine_negatives(
            embeddings=embeddings,
            dists=dists,
            valid_neg_mask=valid_neg_mask,
        )

        # we only want the negatives per anchor, so we have to cut down the extra negatives
        negatives = negatives[:B]

        return negatives


def mine_semi_hard_negative(
    anchor_embeddings: torch.Tensor,
    positive_embeddings: torch.Tensor,
    song_ids: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """
    Semi-hard online triplet mining implementation.
    Given an embedding and the batch it came from,
    find the best semi-hard negative example

    anchor_embeddings: (batch_size, embedding_dim) tensor that serves as the anchor example

    positive_embeddings: (batch_size, embedding_dim) tensor that serves as the positive

    song_ids: tensor of int ids corresponding to the batch. song_ids[i] should be the song id of anchor_embeddings[i]

    alpha: margin for mining
    """
    with torch.no_grad():

        # pos_dists has shape (batch_size) and neg_dists has shape (batch_size, 2*batch_size)
        pos_dists = torch.linalg.vector_norm(
            anchor_embeddings - positive_embeddings, ord=2, dim=1
        )

        # all embeddings should be considered as potential hard negatives
        embeddings = torch.cat([anchor_embeddings, positive_embeddings], dim=0)
        doubled_song_ids = torch.cat(
            [song_ids, song_ids], dim=0
        )  # we have to do this to match the catted embeddings

        # find dists between anchors and all other embeddings
        neg_dists = torch.cdist(anchor_embeddings, embeddings, p=2)  # (B, 2B)

        # create masks. within_margin_mask ensures that the tensor satisfies the 'semi-hard' criterion
        # different_song_mask checks that the song_id is different
        same_song_mask = song_ids.unsqueeze(dim=1) == doubled_song_ids.unsqueeze(
            dim=0
        )  # (B, 2B)
        margin_mask = (neg_dists > pos_dists.unsqueeze(dim=1)) & (
            neg_dists < (pos_dists.unsqueeze(dim=1) + alpha)
        )  # (B, 2B)
        valid_neg_mask = (~same_song_mask) & margin_mask  # (B, 2B)

        # check if any rows are invalid
        # if they are, drop the semi hard criterion
        no_valid_neg = ~valid_neg_mask.any(dim=1)
        valid_neg_mask[no_valid_neg] = ~same_song_mask[no_valid_neg]

        if ~valid_neg_mask.any(dim=1).all():
            # if we get here, it means that both the semi-hard criterion and
            # different song criterion failed to produce any valid negatives
            warnings.warn(
                "All tensors in batch are from the same song. INCREASE THE BATCH SIZE!!!!"
            )
            no_valid_neg = ~valid_neg_mask.any(dim=1)
            valid_neg_mask[no_valid_neg] = True

        # negatives will be (2B, E)
        negatives = _mine_negatives(
            embeddings=embeddings,  # (2B, E)
            dists=neg_dists,  # (B, 2B)
            valid_neg_mask=valid_neg_mask,  # (B, 2B)
        )

        # we only want the negatives per anchor, so have to cut out the second half
        negatives = negatives[: anchor_embeddings.shape[0]]  # (B, E)

        return negatives


def _mine_negatives(
    embeddings: torch.Tensor,
    dists: torch.Tensor,
    valid_neg_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Given a valid neg mask and a matrix of neg dists, find the hardest negative

    embeddings: a tensor of shape (B, E) B is batch size, E is embedding dim

    neg_dists: tensor of shape (B, B)

    valid_neg_mask: tensor of shape (B, B)
    """

    # to mask out invalid negatives, set their value to a large value
    large_value = torch.finfo(dists.dtype).max
    masked_dists = dists.clone()
    masked_dists[~valid_neg_mask] = large_value

    # next, find the indices of the hardest negatives
    hardest_neg_idxs = torch.argmin(masked_dists, dim=1)  # (B,)

    # gather the negative embeddings
    negatives = embeddings[hardest_neg_idxs]  # (B,)

    return negatives
