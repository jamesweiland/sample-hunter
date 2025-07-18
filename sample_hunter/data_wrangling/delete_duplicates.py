import argparse
import numpy as np
import json
from pathlib import Path
from itertools import combinations
from time import perf_counter
from tqdm import tqdm
import multiprocessing as mp

MIN_OVERLAP = 20
THRESHOLD = 0.7
SPAN = 20
STEP = 1


def similarity(a: np.ndarray, b: np.ndarray):
    """Calculate the bitwise similarity between fingerprint a and fingerprint b"""
    # Truncate to min length
    minlen = min(len(a), len(b))
    a = a[:minlen]
    b = b[:minlen]

    assert a.itemsize == b.itemsize and a.size == b.size

    xor_result = np.bitwise_xor(a, b)
    different_bits = np.unpackbits(xor_result.view(np.uint8)).sum()

    total_bits = a.itemsize * 8 * a.size
    return 1 - different_bits / total_bits


def similarity_with_offset(a: np.ndarray, b: np.ndarray, offset: int, min_overlap: int):
    """Return bitwise similarity with fingerprint b offset from fingerprint a"""
    if offset > 0:
        a = a[offset:]
        b = b[: len(a)]
    elif offset < 0:
        offset = -offset
        b = b[offset:]
        a = a[: len(b)]
    if min(len(a), len(b)) < MIN_OVERLAP:
        # we'll throw an error for now
        raise RuntimeError(f"Overlap is less than MIN_OVERLAP: {min(len(a), len(b))}")
    return similarity(a, b)


def compare(a: np.ndarray, b: np.ndarray, span: int, step: int, min_overlap):
    if span > min(len(a), len(b)):
        raise ValueError(
            f"span is greater than the fingerprints: {min(len(a), len(b))}"
        )

    sims_ab = []
    for offset in np.arange(-span, span + 1, step, dtype=int):
        sims_ab.append(similarity_with_offset(a, b, offset, min_overlap))

    return max(sims_ab)


def parse_args():
    """set up the command line parser"""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--threshold",
        type=float,
        default=THRESHOLD,
        help="The minimum threshold to identify two songs as a match",
    )

    parser.add_argument(
        "--span", type=int, default=SPAN, help="The span to use for offsetting"
    )

    parser.add_argument(
        "--step", type=int, default=STEP, help="The step to use for offsetting"
    )

    parser.add_argument(
        "--overlap",
        type=int,
        default=MIN_OVERLAP,
        help="The minimum overlap when offsetting two songs",
    )

    return parser.parse_args()


def worker_func(pair):
    a, b, span, step, overlap, threshold, lock = pair
    a_path = Path(a["filename"])
    b_path = Path(b["filename"])
    # they could've already been handled
    if not a_path.exists() or not b_path.exists():
        return

    a_fp = np.array(a["fingerprint"], dtype=np.uint32)
    b_fp = np.array(b["fingerprint"], dtype=np.uint32)
    corr = compare(a_fp, b_fp, span, step, overlap)

    if corr > threshold:
        match = f"{a_path.stem} and {b_path.stem} match with a similarity of {corr}"
        print(match)

        with lock:
            if b_path.exists():
                b_path.unlink()

        return match


if __name__ == "__main__":
    args = parse_args()
    with open("./_data/fingerprints.json") as f:
        fingerprints = json.load(f)

    n_iter = len(fingerprints) * (len(fingerprints) - 1) // 2

    manager = mp.Manager()
    lock = manager.Lock()
    jobs = [
        (a, b, args.span, args.step, args.overlap, args.threshold, lock)
        for a, b in combinations(fingerprints, 2)
    ]

    with mp.Pool(mp.cpu_count()) as pool:
        results = tqdm(pool.imap_unordered(worker_func, jobs), total=n_iter)

        with open("./_data/matches.txt", "w") as out:
            for result in results:
                if result is not None:
                    print(result, file=out)

    # with open("./_data/matches.txt", 'w') as out:
    #     for a, b in tqdm(combinations(fingerprints, 2), total=n_iter):

    #         a_fingerprint = np.array(a["fingerprint"], dtype=np.uint32)
    #         b_fingerprint = np.array(b["fingerprint"], dtype=np.uint32)

    #         corr = compare(a_fingerprint, b_fingerprint, args.span, args.step, args.overlap)
    #         if corr > args.threshold:
    #             a_path = Path(a["filename"])
    #             b_path = Path(b["filename"])

    #             match = f"{a_path.stem} and {b_path.stem} match with a similarity of {corr}"

    #             print(match)
    #             print(match, file=out)
