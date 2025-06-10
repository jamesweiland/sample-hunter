"""
Triplet loss functions to be used in training and evaluation
"""

import torch
from torch import Tensor


def triplet_accuracy(
    anchor: Tensor, positive: Tensor, negative: Tensor, alpha: float
) -> float:
    """Calculates the accuracy of the model by returning the ratio of positive embeddings
    closer to the anchor than negative ones. The positive embedding must be at least `alpha` closer
    to the anchor than the negative embedding"""
    pos_dists = torch.linalg.vector_norm(anchor - positive, ord=2, dim=1)
    neg_dists = torch.linalg.vector_norm(anchor - negative, ord=2, dim=1)

    correct = pos_dists < neg_dists + alpha  # a boolean mask
    return correct.float().mean().item()


def mine_negative_triplet(
    anchor_embeddings: Tensor,
    positive_embeddings: Tensor,
    song_ids: Tensor,
    alpha: float,
) -> Tensor:
    """
    Semi-hard online triplet mining implementation.
    Given an embedding and the batch it came from,
    find the best semi-hard negative example

    anchor_embeddings: (batch_size, embedding_dim) tensor that serves as the positive example

    positive_embeddings: (batch_size, embedding_dim) tensor that serves as the anchor

    song_ids: list of strings that have the song ids correspond to
    the batch. batch_ids[i] should be the song id of batch[i]

    alpha: margin for mining
    """
    batch_size = anchor_embeddings.shape[0]

    # pos_dists has shape (batch_size) and neg_dists has shape (batch_size, batch_size)
    pos_dists = torch.linalg.vector_norm(
        anchor_embeddings - positive_embeddings, ord=2, dim=1
    )
    # i don't really understand why this is only anchor embeddings,
    # but chat insisted
    neg_dists = torch.cdist(anchor_embeddings, anchor_embeddings, p=2)

    # create masks. within_margin_mask ensures that the tensor satisfies the 'semi-hard' criterion
    # different_song_mask checks that the song_id is different
    same_song_mask = song_ids.unsqueeze(dim=1) == song_ids.unsqueeze(dim=0)
    margin_mask = (neg_dists > pos_dists.unsqueeze(dim=1)) & (
        neg_dists < pos_dists.unsqueeze(dim=1) + alpha
    )
    valid_neg_mask = same_song_mask & margin_mask

    negatives = torch.zeros(anchor_embeddings.shape)
    for i in range(batch_size):
        valid_negs = valid_neg_mask[i].nonzero().squeeze(dim=1)
        if valid_negs.any():
            # find the hardest valid neg
            hardest = valid_negs[torch.argmin(neg_dists[i][valid_negs])]
            negatives[i] = anchor_embeddings[hardest]
        else:
            # if there's no semi-hard, just take the hardest from a different song
            different_song_negs = (~same_song_mask[i]).nonzero().squeeze(dim=1)
            if different_song_negs.any():
                hardest = different_song_negs[
                    torch.argmin(neg_dists[i, different_song_negs])
                ]
                negatives[i] = anchor_embeddings[hardest]
            else:
                # edge case: all samples in the batch are from the same song
                # this should never happen. if this happens then the batch size
                # is too small
                print(
                    "WARNING: all tensors in batch are from the same song. INCREASE THE BATCH SIZE!!!!"
                )
                hardest = neg_dists[i][torch.argmin(neg_dists[i])]
                negatives[i] = anchor_embeddings[hardest]

    return negatives
