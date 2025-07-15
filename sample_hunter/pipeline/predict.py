"""Use a built FAISS index to make predictions about song snippets."""

import argparse
from typing import Tuple
import faiss
import torch
import pandas as pd
from pathlib import Path

from .data_loading import load_tensor_from_bytes, load_webdataset
from .transformations.spectrogram_preprocessor import SpectrogramPreprocessor
from .encoder_net import EncoderNet
from sample_hunter._util import config, HF_TOKEN, load_model, play_tensor_audio


def validate_vector_search(
    spectrogram: torch.Tensor,
    index: faiss.IndexFlatL2,
    model: EncoderNet,
    metadata: pd.DataFrame,
    gt: int,
    k: int = 5,
) -> bool:
    """Returns True if gt is in the vector search for the spectrogram, False otherwise"""
    with torch.no_grad():
        model.eval()
        embeddings = model(spectrogram)

        D, I = index.search(embeddings, k)
        for neighbors in I:
            for neighbor in neighbors:
                song_id = metadata[metadata["snippet_id"] == neighbor][
                    "song_id"
                ].tolist()[0]
                if song_id == gt:
                    return True
        return False


def predict(
    spectrogram: torch.Tensor,
    index: faiss.IndexFlatL2,
    model: EncoderNet,
    metadata: pd.DataFrame,
    gt: int | None = None,
) -> Tuple[int, float]:
    """
    Predict embeddings of a spectrogram, then query the index to find
    the nearest neighbors of the spectrogram and use that to make a
    prediction about where the spectrogram was sampled from.

    spectrogram: a tensor of shape (B?, 1, M, T), where B is (optional) batch
    size, M is number of mel bins, and T is number of time frames.
    If B is given, `predict` will find the nearest neighbor for every
    spectrogram in the batch and return the song id with the most near neighbors
    """

    with torch.no_grad():
        model.eval()
        embeddings = model(spectrogram)
        D, I = index.search(embeddings, 1)

        I = I.squeeze(1)
        D = D.squeeze(1)

        # create a list of candidate songs from the index
        candidates = {}
        for neighbor in I:
            song_id = metadata[metadata["snippet_id"] == neighbor]["song_id"]
            assert len(song_id) == 1
            song_id = int(song_id.iloc[0])
            if song_id not in candidates.keys():
                # first, gather all embedding ids related to the song and retrieve them from index
                song_snippets = metadata[metadata["song_id"] == song_id].sort_values(
                    "snippet_order"
                )
                snippet_ids = torch.tensor(
                    song_snippets["snippet_id"].tolist(), device=embeddings.device
                )
                candidate = torch.tensor(
                    index.reconstruct_batch(snippet_ids),
                    dtype=embeddings.dtype,
                    device=embeddings.device,
                )

                candidates.update(
                    {
                        song_id: evaluate_candidate(
                            query_sequence=embeddings,
                            candidate_song=candidate,
                            candidate_song_id=song_id,
                            index=index,
                            metadata=metadata,
                        ),
                    }
                )

        best_candidate = max(candidates, key=candidates.get)
        best_score = candidates[best_candidate]
        if gt:
            if gt in candidates.keys():
                print("The correct song WAS a candidate")
                print(f"The correct song had a score of {candidates[gt]}")
            else:
                print("The correct song WAS NOT a candidate")
        return best_candidate, best_score

        # return the match with the least distance
        # distances = {}
        # for neighbor, dist in zip(I, D):
        #     song_id = metadata[metadata["snippet_id"] == neighbor]["song_id"]
        #     assert len(song_id) == 1
        #     song_id = int(song_id.iloc[0])
        #     if song_id not in distances:
        #         distances[song_id] = {"total_dist": 0.0, "count": 0}

        #     distances[song_id]["total_dist"] += dist
        #     distances[song_id]["count"] += 1

        # avg_distances = {k: v["total_dist"] / v["count"] for k, v in distances.items()}
        # best_match = min(avg_distances, key=avg_distances.get)

        # count up all the matches and return the one with the most matches
        # matches = {}
        # for neighbor in I:
        #     song_id = metadata[metadata["snippet_id"] == neighbor]["song_id"]
        #     assert len(song_id) == 1
        #     song_id = song_id.iloc[0]
        #     if matches.get(song_id) is None:
        #         matches[int(song_id)] = 1
        #     else:
        #         matches[int(song_id)] += 1

        # best_match = max(matches, key=matches.get)
        # return best_match


def evaluate_candidate(
    query_sequence: torch.Tensor,
    candidate_song: torch.Tensor,
    candidate_song_id: int,
    index: faiss.Index,
    metadata: pd.DataFrame,
) -> float:
    """
    Slide the query sequence over the candidate song and find the sequence in the
    candidate song with the highest similarity, and return that similarity score.

    query_sequence: a tensor of shape (B, E), E is embedding dim

    candidate_song: a tensor of shape (B, E)

    candidate_song_id: an int that is the candidate song's song_id in metadata.
    """
    if candidate_song.shape[0] < query_sequence.shape[0]:
        # pad candidate_song to query_sequence's length
        pad_size = query_sequence.shape[0] - candidate_song.shape[0]
        candidate_song = torch.nn.functional.pad(candidate_song, (0, 0, 0, pad_size))

    start = 0
    best_similarity = None
    while start <= (candidate_song.shape[0] - query_sequence.shape[0]):
        end = start + query_sequence.shape[0]
        candidate_sequence = candidate_song[start:end, ...]
        similarity = (
            sequence_similarity(
                x=query_sequence,
                y=candidate_sequence,
                y_song_id=candidate_song_id,
                index=index,
                metadata=metadata,
            )
            .mean()
            .item()
        )
        if best_similarity is None or similarity > best_similarity:
            best_similarity = float(similarity)
        start += 1
    assert best_similarity is not None

    return best_similarity


def sequence_similarity(
    x: torch.Tensor,
    y: torch.Tensor,
    y_song_id: int,
    index: faiss.Index,
    metadata: pd.DataFrame,
    alpha: float = 9.0,
) -> torch.Tensor:
    """
    Pairwise sequence similarity metric developed in https://openaccess.thecvf.com/content_cvpr_2013/papers/Qin_Query_Adaptive_Similarity_2013_CVPR_paper.pdf

    `x`: a query sequence

    `y`: a candidate sequence retrieved from `index`
    """
    # we take the mean of the pairwise sequence similarity to determine the total similarity
    # of the sequence
    return torch.exp(
        -alpha * normalized_distance(x, y, y_song_id, index, metadata) ** 4
    )


def normalized_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    y_song_id: int,
    index: faiss.Index,
    metadata: pd.DataFrame,
    sample_size: int = 200,
) -> torch.Tensor:
    """
    Compute the pairwise normalized distance between a query sequence of
    embeddings, x, and a sequence retrieved from the index, y. x and y
    must have identical shape. The normalized distance between x_i and y_i
    is defined as:

    `normalized_distance(x_i, y_i) = distance(x_i, y_i) / N_d(x_i)`, where

    N_d(x_i) is the expected distance from x_i to a set of non-matching (random) embeddings
    in the same index.
    """

    if x.shape != y.shape:
        raise ValueError(
            "x and y do not have the same shape."
            f"x shape: {x.shape}"
            f"y shape: {y.shape}"
        )

    # calculate pairwise distances between all elements in x and y
    d = torch.linalg.vector_norm(x - y, ord=2, dim=1)

    # find all embeddings in the index with a different song id
    non_matching_indices = torch.tensor(
        metadata[metadata["song_id"] != y_song_id]["snippet_id"].tolist(),
        device=x.device,
    )

    # take a random sample from the non_matching_indices,
    # and then retrieve these vectors from index
    sample_size = min(sample_size, len(non_matching_indices))
    sample_indices = non_matching_indices[
        torch.randperm(len(non_matching_indices))[:sample_size]
    ]
    sample = torch.tensor(index.reconstruct_batch(sample_indices), device=x.device)

    pairwise_dists = torch.cdist(x, sample, p=2)  # (N, sample_size)

    # For each x[i], compute mean distance to all sampled non-matching embeddings
    N_d = pairwise_dists.mean(dim=1)  # (N,)

    return d / N_d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model", type=Path, help="The model to use to generate prediction embeddings"
    )

    parser.add_argument(
        "index", type=Path, help="The path to the index to query for predictions"
    )

    parser.add_argument(
        "metadata",
        type=Path,
        help="The path to the metadata csv file with mappings between the vector ids in the index"
        "and song ids",
    )

    parser.add_argument(
        "--repo-id",
        type=str,
        help="The HF repo id that has data to make predictions on",
        default=config.hf.repo_id,
    )

    parser.add_argument("--token", type=str, help="Your HF token", default=HF_TOKEN)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    model = load_model(args.model)
    index = faiss.read_index(str(args.index))
    metadata = pd.read_csv(args.metadata)

    with SpectrogramPreprocessor() as preprocessor:

        def map_fn(ex):
            audio, sr = load_tensor_from_bytes(ex["b.mp3"])
            spec = preprocessor(audio, obfuscate=False, sample_rate=sr)
            return {**ex, "specs": spec}

        dataset = load_webdataset(args.repo_id, "validation", token=args.token)
        dataset = dataset.map(map_fn)

        num_right = 0
        num_total = 0
        for ex in dataset:
            print(f"iter {num_total}")
            anchor_id = ex["json"]["ground_id"]
            print(
                f"gt: {metadata[metadata["song_id"] == anchor_id]["song_title"].tolist()[0]}"
            )
            print(f"gt id: {anchor_id}")

            anchor, anchor_sr = load_tensor_from_bytes(ex["a.mp3"])
            positive, positive_sr = load_tensor_from_bytes(ex["b.mp3"])
            # play_tensor_audio(anchor, "Playing anchor...", sample_rate=anchor_sr)
            # play_tensor_audio(positive, "Playing positive...", sample_rate=positive_sr)

            res = validate_vector_search(ex["specs"], index, model, metadata, anchor_id)

            if res:
                num_right += 1
            num_total += 1

            # prediction, score = predict(
            #     ex["specs"], index, model, metadata, gt=anchor_id
            # )
            # print(f"predict id: {prediction}")
            # print(
            #     f"prediction: {metadata[metadata["song_id"] == prediction]["song_title"].tolist()[0]}"
            # )
            # print(f"score: {score}")

            # print("-----------------------------------------------------------")

            # if prediction == ex["json"]["ground_id"]:
            #     num_right += 1

            # num_total += 1

        print(f"Total accuracy: {(num_right / num_total):.2%}")
