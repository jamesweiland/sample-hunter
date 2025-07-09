"""
A class to generate tone waveforms, like a synth.

Can generate sine, square, triangle, or sawtooth waves.

This is likely deprecated but i'm gonna keep it here just in case
"""

import torch
import math


class ToneGenerator:

    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate

    def generate_sine_wave(
        self,
        signal: torch.Tensor,
        frequencies: torch.Tensor,
        amplitudes: torch.Tensor,
        starts: torch.Tensor,
        ends: torch.Tensor,
    ) -> torch.Tensor:
        """
        Add a unique sine wave to each example in the batch.
        frequencies, amplitudes, starts, ends: (batch_size,)
        signal: (batch_size, 1, time)
        """
        new_signal = signal.clone()
        batch_size = signal.shape[0]
        for b in range(batch_size):
            start_idx = int(starts[b].item() * self.sample_rate)
            end_idx = int(ends[b].item() * self.sample_rate)
            num_samples = end_idx - start_idx
            if num_samples <= 0:
                continue
            t = torch.arange(num_samples, device=signal.device) / self.sample_rate
            sine_wave = amplitudes[b] * torch.sin(2 * math.pi * frequencies[b] * t)
            new_signal[b, 0, start_idx:end_idx] += sine_wave
        return new_signal

    def generate_square_wave(
        self,
        signal: torch.Tensor,
        frequencies: torch.Tensor,
        amplitudes: torch.Tensor,
        starts: torch.Tensor,
        ends: torch.Tensor,
    ) -> torch.Tensor:
        """
        Add a unique square wave to each example in the batch.
        """
        new_signal = signal.clone()
        batch_size = signal.shape[0]
        for b in range(batch_size):
            start_idx = int(starts[b].item() * self.sample_rate)
            end_idx = int(ends[b].item() * self.sample_rate)
            num_samples = end_idx - start_idx
            if num_samples <= 0:
                continue
            t = torch.arange(num_samples, device=signal.device) / self.sample_rate
            square_wave = amplitudes[b] * torch.sign(
                torch.sin(2 * math.pi * frequencies[b] * t)
            )
            new_signal[b, 0, start_idx:end_idx] += square_wave
        return new_signal

    def generate_triangle_wave(
        self,
        signal: torch.Tensor,
        frequencies: torch.Tensor,
        amplitudes: torch.Tensor,
        starts: torch.Tensor,
        ends: torch.Tensor,
    ) -> torch.Tensor:
        """
        Add a unique triangle wave to each example in the batch.
        """
        new_signal = signal.clone()
        batch_size = signal.shape[0]
        for b in range(batch_size):
            start_idx = int(starts[b].item() * self.sample_rate)
            end_idx = int(ends[b].item() * self.sample_rate)
            num_samples = end_idx - start_idx
            if num_samples <= 0:
                continue
            t = torch.arange(num_samples, device=signal.device) / self.sample_rate
            triangle_wave = amplitudes[b] * (
                2
                * torch.abs(
                    2 * (frequencies[b] * t - torch.floor(frequencies[b] * t + 0.5))
                )
                - 1
            )
            new_signal[b, 0, start_idx:end_idx] += triangle_wave
        return new_signal

    def generate_sawtooth_wave(
        self,
        signal: torch.Tensor,
        frequencies: torch.Tensor,
        amplitudes: torch.Tensor,
        starts: torch.Tensor,
        ends: torch.Tensor,
    ) -> torch.Tensor:
        """
        Add a unique sawtooth wave to each example in the batch.
        """
        new_signal = signal.clone()
        batch_size = signal.shape[0]
        for b in range(batch_size):
            start_idx = int(starts[b].item() * self.sample_rate)
            end_idx = int(ends[b].item() * self.sample_rate)
            num_samples = end_idx - start_idx
            if num_samples <= 0:
                continue
            t = torch.arange(num_samples, device=signal.device) / self.sample_rate
            sawtooth_wave = amplitudes[b] * (
                2 * (frequencies[b] * t - torch.floor(frequencies[b] * t + 0.5))
            )
            new_signal[b, 0, start_idx:end_idx] += sawtooth_wave
        return new_signal
