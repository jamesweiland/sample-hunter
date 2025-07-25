"""Use a built FAISS index to make predictions about song snippets."""

import argparse
from typing import Tuple, List, cast
import faiss
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import webdataset as wds

from .data_loading import load_tensor_from_bytes, load_webdataset
from .transformations.preprocessor import Preprocessor
from .encoder_net import EncoderNet
from sample_hunter._util import HF_TOKEN, load_model
from sample_hunter.config import PostprocessConfig, DEFAULT_TOP_K, DEFAULT_REPO_ID


def validate_vector_search(
    spectrogram: torch.Tensor,
    index: faiss.IndexFlatL2,
    model: EncoderNet,
    metadata: pd.DataFrame,
    gt: int,
    top_k: int = DEFAULT_TOP_K,
) -> bool:
    """Returns True if gt is in the vector search for the spectrogram, False otherwise"""
    with torch.no_grad():
        model.eval()

        embeddings = model(spectrogram)
        D, I = index.search(embeddings, top_k)  # type: ignore
        for neighbors in I:
            for neighbor in neighbors:
                song_id = metadata[metadata["snippet_id"] == neighbor][
                    "song_id"
                ].tolist()[0]
                if song_id == gt:
                    return True
        return False


def infer_with_offset(
    embeddings: List[torch.Tensor],
    index: faiss.Index,
    metadata: pd.DataFrame,
    gt: int | None = None,
    config: PostprocessConfig | None = None,
    **kwargs,
) -> Tuple[int, float]:
    """
    Given a list of embeddings that represent an audio sample offset a certain number of times,
    find the best match across all offsets
    """

    candidates = {}
    for embedding in tqdm(embeddings):
        candidate, score = infer(
            embeddings=embedding,
            index=index,
            metadata=metadata,
            config=config,
            **kwargs,
        )
        if (
            candidate in candidates and candidates[candidate] < score
        ) or candidate not in candidates:
            candidates[candidate] = score

    best_candidate = max(candidates, key=candidates.get)  # type: ignore
    best_score = candidates[best_candidate]

    if gt:
        if gt in candidates.keys():
            print("The correct song WAS a candidate")
            print(f"The correct song had a score of {candidates[gt]}")
        else:
            print("The correct song WAS NOT a candidate")

    return best_candidate, best_score


def infer(
    embeddings: torch.Tensor,
    index: faiss.Index,
    metadata: pd.DataFrame,
    config: PostprocessConfig | None = None,
    **kwargs,
) -> Tuple[int, float]:
    """
    Predict embeddings of a spectrogram, then query the index to find
    the nearest neighbors of the spectrogram and use that to make a
    prediction about where the spectrogram was sampled from.

    spectrogram: a tensor of shape (B?, E), where B is (optional) batch
    size, E is embedding dimension
    If B is given, `predict` will find the nearest neighbor for every
    spectrogram in the batch and return the song id with the most near neighbors
    """

    config = config or PostprocessConfig()
    config = config.merge_kwargs(**kwargs)

    with torch.no_grad():
        D, I = index.search(embeddings, config.top_k)  # type: ignore

        # create a list of candidate songs from the index
        candidates = {}
        for query_idx in range(I.shape[0]):
            for neighbor_idx in range(I.shape[1]):
                neighbor = I[query_idx][neighbor_idx]

                song_id = metadata[metadata["snippet_id"] == neighbor]["song_id"]
                assert len(song_id) == 1
                song_id = int(song_id.iloc[0])
                if song_id not in candidates.keys():
                    # first, gather all embedding ids related to the song and retrieve them from index
                    song_snippets = metadata[
                        metadata["song_id"] == song_id
                    ].sort_values("snippet_order")
                    snippet_ids = torch.tensor(
                        song_snippets["snippet_id"].tolist(), device=embeddings.device
                    )
                    candidate = torch.tensor(
                        index.reconstruct_batch(snippet_ids),  # type: ignore
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
                                config=config,
                            ),
                        }
                    )

        best_candidate = max(candidates, key=candidates.get)  # type: ignore
        best_score = candidates[best_candidate]

        return best_candidate, best_score


def evaluate_candidate(
    query_sequence: torch.Tensor,
    candidate_song: torch.Tensor,
    candidate_song_id: int,
    index: faiss.Index,
    metadata: pd.DataFrame,
    config: PostprocessConfig | None = None,
) -> float:
    """
    Slide the query sequence over the candidate song and find the sequence in the
    candidate song with the highest similarity, and return that similarity score.

    query_sequence: a tensor of shape (B, E), E is embedding dim

    candidate_song: a tensor of shape (B, E)

    candidate_song_id: an int that is the candidate song's song_id in metadata.
    """
    config = config or PostprocessConfig()

    if candidate_song.shape[0] < query_sequence.shape[0]:
        # pad candidate_song to query_sequence's length
        pad_size = query_sequence.shape[0] - candidate_song.shape[0]
        candidate_song = torch.nn.functional.pad(candidate_song, (0, 0, 0, pad_size))

    start = 0
    best_similarity = None
    while start <= (candidate_song.shape[0] - query_sequence.shape[0]):
        end = start + query_sequence.shape[0]
        candidate_sequence = candidate_song[start:end, ...]

        # for each offset, calculate the sequence similarity with offset and store it in
        # similarities, then store that in some best_similarity
        similarities = [
            sequence_similarity(
                x=query_sequence,
                y=candidate_sequence,
                y_song_id=candidate_song_id,
                index=index,
                metadata=metadata,
                alpha=config.alpha,
            )
        ]
        if not best_similarity or max(similarities) > best_similarity:
            best_similarity = float(max(similarities))
        start += 1
    assert best_similarity

    return best_similarity


def sequence_similarity_with_offset(
    x: torch.Tensor,
    y: torch.Tensor,
    y_song_id: int,
    index: faiss.Index,
    metadata: pd.DataFrame,
    offset_num_samples: int,
    alpha: float,
) -> float:
    """
    return the similarity between x and y, where y is offset from x between -span and span with size of step

    offset is the number of samples to offset y from x
    """
    if offset_num_samples > 0:
        x = x[:, offset_num_samples:]
        y = y[:, : len(x)]
    elif offset_num_samples < 0:
        offset_num_samples = -offset_num_samples
        y = y[:, offset_num_samples:]
        x = x[:, : len(y)]

    similarity = sequence_similarity(
        x=x,
        y=y,
        y_song_id=y_song_id,
        index=index,
        metadata=metadata,
        alpha=alpha,
    )

    return similarity


def sequence_similarity(
    x: torch.Tensor,
    y: torch.Tensor,
    y_song_id: int,
    index: faiss.Index,
    metadata: pd.DataFrame,
    alpha: float,
) -> float:
    """
    Pairwise sequence similarity metric developed in https://openaccess.thecvf.com/content_cvpr_2013/papers/Qin_Query_Adaptive_Similarity_2013_CVPR_paper.pdf

    `x`: a query sequence

    `y`: a candidate sequence retrieved from `index`
    """
    # we take the mean of the pairwise sequence similarity to determine the total similarity
    # of the sequence
    return (
        torch.exp(-alpha * normalized_distance(x, y, y_song_id, index, metadata) ** 4)
        .mean()
        .item()
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
    sample = torch.tensor(index.reconstruct_batch(sample_indices), device=x.device)  # type: ignore

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
        default=DEFAULT_REPO_ID,
    )

    parser.add_argument("--token", type=str, help="Your HF token", default=HF_TOKEN)

    return parser.parse_args()


if __name__ == "__main__":
    pass
    # args = parse_args()

    # model = load_model(args.model)
    # index = faiss.read_index(str(args.index))
    # metadata = pd.read_csv(args.metadata)

    # with Preprocessor() as preprocessor:

    #     def map_fn(ex):
    #         audio, sr = load_tensor_from_bytes(ex["b.mp3"])

    #         span_num_samples = int(default_config.post.span * sr)
    #         step_num_samples = int(default_config.post.step * sr)

    #         offsets = []
    #         for offset_num_samples in range(
    #             -span_num_samples, span_num_samples + step_num_samples, step_num_samples
    #         ):
    #             if offset_num_samples < 0:
    #                 offset = audio[:, abs(offset_num_samples) :]
    #             elif offset_num_samples > 0:
    #                 offset = torch.nn.functional.pad(audio, (offset_num_samples, 0))
    #             else:
    #                 offset = audio
    #             offsets.append(offset)

    #         specs = [
    #             preprocessor(
    #                 offset,
    #                 sample_rate=sr,
    #                 train=False,
    #             )
    #             for offset in offsets
    #         ]
    #         return {**ex, "specs": specs}

    #     dataset = load_webdataset(args.repo_id, "validation", token=args.token)
    #     dataset = cast(wds.WebDataset, dataset.map(map_fn))

    #     num_right = 0
    #     num_total = 0
    #     for ex in dataset:
    #         print(f"iter {num_total}")
    #         anchor_id = ex["json"]["ground_id"]
    #         print(
    #             f"gt: {metadata[metadata["song_id"] == anchor_id]["song_title"].tolist()[0]}"
    #         )
    #         print(f"gt id: {anchor_id}")

    #         anchor, anchor_sr = load_tensor_from_bytes(ex["a.mp3"])
    #         positive, positive_sr = load_tensor_from_bytes(ex["b.mp3"])
    #         # play_tensor_audio(anchor, "Playing anchor...", sample_rate=anchor_sr)
    #         # play_tensor_audio(positive, "Playing positive...", sample_rate=positive_sr)

    #         BOLD = "\033[1m"
    #         RESET = "\033[0m"
    #         RED = "\033[91m"
    #         GREEN = "\033[92m"

    #         success = False
    #         for spec in ex["specs"]:
    #             if validate_vector_search(spec, index, model, metadata, anchor_id):
    #                 num_right += 1
    #                 success = True
    #                 print(f"{BOLD}{GREEN}SUCCESS: {ex["json"]["title"]}{RESET}")
    #                 break

    #         if not success:
    #             print(f"{BOLD}{RED}FAIL: {ex["json"]["title"]} failed{RESET}")

    #         # play_tensor_audio(anchor, "Playing anchor...", sample_rate=anchor_sr)
    #         # play_tensor_audio(positive, "Playing positive...", sample_rate=positive_sr)
    #         num_total += 1

    #         # prediction, score = predict(
    #         #     ex["specs"], index, model, metadata, gt=anchor_id
    #         # )
    #         # print(f"predict id: {prediction}")
    #         # print(
    #         #     f"prediction: {metadata[metadata["song_id"] == prediction]["song_title"].tolist()[0]}"
    #         # )
    #         # print(f"score: {score}")

    #         # print("-----------------------------------------------------------")

    #         # if prediction == ex["json"]["ground_id"]:
    #         #     num_right += 1

    #         # num_total += 1

    #     print(f"Total accuracy: {(num_right / num_total):.2%}")
