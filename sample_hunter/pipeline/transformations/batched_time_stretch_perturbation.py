from typing import List, Sequence
import torch
import torch.multiprocessing as mp
import torchaudio

from .functional import resize
from sample_hunter._util import DEVICE, PROCS
from sample_hunter.config import DEFAULT_N_FFT, DEFAULT_HOP_LENGTH

__all__ = ["BatchedTimeStretchPerturbation"]


def _init_worker(
    function,
    factors: List[float],
    n_fft: int,
    hop_length: int,
    window: torch.Tensor,
    threads_per_worker: int,
    device: str = DEVICE,
):
    function.stretchers = [
        torchaudio.transforms.TimeStretch(
            hop_length=hop_length, fixed_rate=factor, n_freq=hop_length + 1
        ).to(device)
        for factor in factors
    ]
    function.window = window
    function.n_fft = n_fft
    function.hop_length = hop_length
    torch.set_num_threads(threads_per_worker)


def _worker_func(i: int, mask: torch.Tensor, spec: torch.Tensor, ori_size: int):
    return (
        resize(
            torch.istft(
                _worker_func.stretchers[i](spec),  # type: ignore
                n_fft=_worker_func.n_fft,  # type: ignore
                hop_length=_worker_func.hop_length,  # type: ignore
                window=_worker_func.window,  # type: ignore
            ),
            ori_size,
        ).unsqueeze(1),
        mask,
    )


class BatchedTimeStretchPerturbation:
    def __init__(
        self,
        factors: Sequence[float],
        n_fft: int = DEFAULT_N_FFT,
        hop_length: int = DEFAULT_HOP_LENGTH,
        num_workers: int = PROCS,
        device: str = DEVICE,
    ):
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.window = torch.hann_window(window_length=self.n_fft, device=device)
        self.factors = factors

        self.num_workers = min(num_workers, len(factors))

        if device == "cpu" or device == "cuda":
            self.device = device
        else:
            raise ValueError(f"Unsupported device: {device}")

    def __enter__(self):
        """Set up the proper resources"""
        self._pool = None
        self._streams = None
        if self.device == "cpu" and self.num_workers > 1:
            threads_per_worker = PROCS // self.num_workers
            ctx = mp.get_context("spawn")
            self._pool = ctx.Pool(
                processes=self.num_workers,
                initializer=_init_worker,
                initargs=[
                    _worker_func,
                    self.factors,
                    self.n_fft,
                    self.hop_length,
                    self.window,
                    threads_per_worker,
                ],
            )
        else:
            self.stretchers = [
                torchaudio.transforms.TimeStretch(
                    hop_length=self.hop_length,
                    fixed_rate=factor,
                    n_freq=self.hop_length + 1,
                ).to(self.device)
                for factor in self.factors
            ]

            if self.device == "cuda" and self.num_workers > 1:
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

    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Perturb the batch by applying a randomly selected torchaudio.transforms.TimeStretch
        to each example.
        """
        ob = torch.empty_like(batch, dtype=batch.dtype, device=batch.device)
        spec = torch.stft(
            batch.squeeze(1),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True,
        )
        indices = torch.randint(0, len(self.factors), (spec.size(0),))
        stretcher_tasks = [
            (i, (indices == i).detach(), spec[indices == i].detach(), batch.shape[-1])
            for i in range(len(self.factors))
        ]

        if self.device == "cuda" and self._streams:
            if self._streams is not None:
                for i, mask, sub_batch, ori_size in stretcher_tasks:
                    # skip empty sub batches, which can happen due to randomness
                    if sub_batch.numel() == 0:
                        continue

                    with torch.cuda.Stream(self._streams[i]):
                        ob[mask] = resize(
                            torch.istft(
                                self.stretchers[i](sub_batch),
                                n_fft=self.n_fft,
                                hop_length=self.hop_length,
                                window=self.window,
                            ),
                            ori_size,
                        ).unsqueeze(1)
                for i, _, _, _ in stretcher_tasks:
                    self._streams[i].synchronize()

        elif self.device == "cpu" and self._pool:
            if self._pool:
                results = self._pool.starmap(_worker_func, stretcher_tasks)
                for result, mask in results:
                    ob[mask] = result
        else:
            for i, mask, sub_batch, ori_size in stretcher_tasks:
                # skip empty sub batches, which can happen due to randomness
                if sub_batch.numel() == 0:
                    continue

                ob[mask] = resize(
                    torch.istft(
                        self.stretchers[i](sub_batch),
                        n_fft=self.n_fft,
                        hop_length=self.hop_length,
                        window=self.window,
                    ),
                    ori_size,
                ).unsqueeze(1)

        return ob
