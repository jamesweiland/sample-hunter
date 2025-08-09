from fractions import Fraction
import math
from typing import List, Sequence
import torch
import torchaudio
import torch.multiprocessing as mp

from sample_hunter._util import DEVICE, PROCS

__all__ = ["BatchedPitchPerturbation"]


def _init_worker(
    function,
    factors: List[Fraction],
    sample_rate: int,
    threads_per_worker: int,
    device: str = DEVICE,
):
    function.shifters = [
        torchaudio.transforms.PitchShift(
            sample_rate=sample_rate,
            n_steps=factor.numerator,
            bins_per_octave=factor.denominator,
        ).to(device)
        for factor in factors
    ]
    torch.set_num_threads(threads_per_worker)


def _worker_func(i, mask, sub_batch):
    res = _worker_func.shifters[i](sub_batch).detach()  # type: ignore
    return res, mask


class BatchedPitchPerturbation:
    def __init__(
        self, factors: Sequence[float], sample_rate: int, num_workers: int, device: str
    ):
        factors = [-math.log2(rate) for rate in factors]
        # ensure that the n_steps are rational by converting them to a Fraction
        self.factors = [
            Fraction(n_steps).limit_denominator(1000) for n_steps in factors
        ]

        if device == "cuda" or device == "cpu":
            self.device = device
        else:
            raise ValueError(f"Unsupported device: {device}")
        # set up cuda streams
        self.num_workers = min(len(self.factors), num_workers)
        self.sample_rate = sample_rate

    def __enter__(self):
        """Initialize the proper resources"""
        self._pool = None
        self._streams = None
        if self.device == "cpu" and self.num_workers > 1:
            threads_per_worker = PROCS // self.num_workers
            if self.num_workers > 1:
                ctx = mp.get_context("spawn")
                self._pool = ctx.Pool(
                    processes=self.num_workers,
                    initializer=_init_worker,
                    initargs=[
                        _worker_func,
                        self.factors,
                        self.sample_rate,
                        threads_per_worker,
                    ],
                )
        else:
            self.shifters = [
                torchaudio.transforms.PitchShift(
                    sample_rate=self.sample_rate,
                    n_steps=factor.numerator,
                    bins_per_octave=factor.denominator,
                ).to(self.device)
                for factor in self.factors
            ]

            if self.device == "cuda":
                self._streams = [
                    torch.cuda.Stream(device=self.device)
                    for _ in range(self.num_workers)
                ]

        return self

    def __exit__(self, *exc):
        """Always make sure to clean up at end"""
        if self.device == "cpu" and self._pool:
            # clean up the mp pool
            self._pool.close()
            self._pool.join()
            self._pool = None
        elif self._streams:
            for stream in self._streams:
                stream.synchronize()
            self._streams = None

    def __call__(self, batch: torch.Tensor):
        """Perturb the batch by applying a randomly selected torchaudio.transforms.PitchShift to each example."""
        ob = torch.empty_like(batch, device=batch.device, dtype=batch.dtype)

        indices = torch.randint(
            0, len(self.factors), (batch.size(0),), device=self.device
        )
        shifter_tasks = [
            (i, (indices == i).detach(), batch[indices == i].detach())
            for i in range(len(self.factors))
        ]

        if self.device == "cuda":
            if self._streams is None:
                raise RuntimeError(
                    "BatchedPitchPerturbation must be used within context manager for CUDA"
                )
            for i, mask, sub_batch in shifter_tasks:
                # skip empty sub_batches, which can happen because of randomness
                if sub_batch.numel() == 0:
                    continue

                with torch.cuda.Stream(self._streams[i]):
                    ob[mask] = self.shifters[i](sub_batch)
            for i, _, _ in shifter_tasks:
                self._streams[i].synchronize()

        elif self.device == "cpu":
            if self._pool:
                results = self._pool.starmap(_worker_func, shifter_tasks)
                for result, mask in results:
                    ob[mask] = result
            else:
                for i, mask, sub_batch in shifter_tasks:
                    # skip empty sub_batches, which can happen because of randomness
                    if sub_batch.numel() == 0:
                        continue

                    ob[mask] = self.shifters[i](sub_batch)

        return ob
