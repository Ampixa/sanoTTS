#!/usr/bin/env python3
"""Shared frame-rate spectral decoder blocks for the FSD model family."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


HOP_LENGTH = 256

__all__ = [
    "HOP_LENGTH",
    "FactorizedSpectralHead",
    "FsdConvNeXtBlock",
    "initialize_factorized_spectral_head",
    "logmag_phase_synthesize",
]


class FsdConvNeXtBlock(nn.Module):
    """ConvNeXt1d block used by the FSD-family spectral decoders.

    The original decoder FSD path uses a 4x pointwise expansion and FiLM from
    the decoder input. Racer A reuses the same implementation with a smaller
    expansion and no FiLM so its default budget remains below the U600 cap.
    """

    def __init__(
        self,
        *,
        channels: int,
        in_channels: int | None = None,
        film_rank: int = 0,
        kernel_size: int = 7,
        expansion: int = 4,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")
        if expansion <= 0:
            raise ValueError(f"expansion must be positive, got {expansion}")
        if film_rank < 0:
            raise ValueError(f"film_rank must be non-negative, got {film_rank}")
        if film_rank > 0 and (in_channels is None or in_channels <= 0):
            raise ValueError(f"in_channels must be positive when film_rank > 0, got {in_channels}")

        self.channels = int(channels)
        self.film_rank = int(film_rank)
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=kernel_size // 2,
            groups=channels,
        )
        self.norm = nn.GroupNorm(1, channels)
        hidden = int(channels) * int(expansion)
        self.pointwise0 = nn.Conv1d(channels, hidden, 1)
        self.pointwise1 = nn.Conv1d(hidden, channels, 1)
        self.scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

        if self.film_rank > 0:
            self.film_reduce = nn.Conv1d(int(in_channels), self.film_rank, 1)
            self.film_out = nn.Conv1d(self.film_rank, channels * 2, 1)
            nn.init.zeros_(self.film_out.weight)
            if self.film_out.bias is not None:
                nn.init.zeros_(self.film_out.bias)
        else:
            self.film_reduce = None
            self.film_out = None

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor | None = None) -> torch.Tensor:
        residual = self.depthwise(x)
        residual = self.norm(residual)
        if self.film_rank > 0:
            if conditioning is None:
                raise RuntimeError("FsdConvNeXtBlock with FiLM requires conditioning")
            if self.film_reduce is None or self.film_out is None:
                raise RuntimeError("FsdConvNeXtBlock FiLM modules were not initialized")
            film = self.film_out(self.film_reduce(conditioning))
            scale, shift = film.chunk(2, dim=1)
            residual = residual * (1.0 + scale) + shift
        residual = self.pointwise0(residual)
        residual = F.gelu(residual)
        residual = self.pointwise1(residual)
        return x + self.scale * residual


class FactorizedSpectralHead(nn.Module):
    def __init__(self, *, in_channels: int, rank: int, n_fft: int) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        if n_fft <= 0 or n_fft % 2 != 0:
            raise ValueError(f"n_fft must be a positive even integer, got {n_fft}")
        self.in_channels = int(in_channels)
        self.rank = int(rank)
        self.n_fft = int(n_fft)
        self.bins = self.n_fft // 2 + 1
        self.in_proj = nn.Conv1d(self.in_channels, self.rank, 1)
        self.out_proj = nn.Conv1d(self.rank, self.bins * 3, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = F.gelu(self.in_proj(x))
        params = self.out_proj(hidden)
        log_magnitude = params[:, : self.bins, :]
        phase_logits = params[:, self.bins :, :]
        return log_magnitude, phase_logits, params


def logmag_phase_synthesize(
    log_magnitude: torch.Tensor,
    phase_logits: torch.Tensor,
    *,
    latent_frames: int,
    n_fft: int,
    window: torch.Tensor,
    hop_length: int = HOP_LENGTH,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if log_magnitude.ndim != 3:
        raise RuntimeError(f"expected log magnitude [batch, bins, frames], got {log_magnitude.shape}")
    if phase_logits.ndim != 3:
        raise RuntimeError(f"expected phase logits [batch, 2*bins, frames], got {phase_logits.shape}")
    if n_fft <= 0 or n_fft % 2 != 0:
        raise ValueError(f"n_fft must be a positive even integer, got {n_fft}")
    bins = n_fft // 2 + 1
    if log_magnitude.shape[1] != bins:
        raise RuntimeError(f"log magnitude bins {log_magnitude.shape[1]} != expected {bins}")
    if phase_logits.shape[1] != bins * 2:
        raise RuntimeError(f"phase channels {phase_logits.shape[1]} != expected {bins * 2}")
    if log_magnitude.shape[2] != latent_frames or phase_logits.shape[2] != latent_frames:
        raise RuntimeError(
            f"log-mag/phase frame mismatch: log={log_magnitude.shape[2]}, "
            f"phase={phase_logits.shape[2]}, latent={latent_frames}"
        )
    phase = phase_logits.reshape(phase_logits.shape[0], 2, bins, phase_logits.shape[2])
    phase = F.normalize(phase, dim=1, eps=1e-6)
    log_magnitude = log_magnitude.clamp(min=-12.0, max=8.0)
    magnitude = torch.exp(log_magnitude).clamp_min(1e-7)
    real = magnitude * phase[:, 0]
    imag = magnitude * phase[:, 1]
    complex_spec = torch.complex(real, imag)
    audio = torch.istft(
        complex_spec,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window.to(device=log_magnitude.device, dtype=log_magnitude.dtype),
        center=True,
        length=int(latent_frames) * int(hop_length),
    ).unsqueeze(1)
    return audio, {
        "fsd_log_magnitude": log_magnitude,
        "ap_amp_logits": log_magnitude,
        "ap_log_amplitude": torch.log1p(magnitude),
        "ap_phase_unit": phase,
        "ap_real": real,
        "ap_imag": imag,
    }


def initialize_factorized_spectral_head(
    head: FactorizedSpectralHead | nn.Module,
    *,
    mode: str,
    scale: float,
    amp_bias: float,
    phase_real_bias: float,
) -> dict[str, Any] | None:
    if mode == "default":
        return None
    if mode not in {"zero", "small"}:
        raise ValueError(f"unsupported spectral-head init mode: {mode}")
    if scale <= 0.0:
        raise ValueError(f"scale must be positive, got {scale}")
    out_proj = getattr(head, "out_proj", None)
    bins = int(getattr(head, "bins", 0))
    if not isinstance(out_proj, nn.Conv1d) or bins <= 0:
        raise RuntimeError("factorized spectral head init expected out_proj Conv1d and positive bins")
    with torch.no_grad():
        if mode == "zero":
            nn.init.zeros_(out_proj.weight)
        else:
            nn.init.normal_(out_proj.weight, mean=0.0, std=float(scale))
        if out_proj.bias is None:
            raise RuntimeError("factorized spectral head out_proj has no bias")
        out_proj.bias[:bins].fill_(float(amp_bias))
        out_proj.bias[bins : bins * 2].fill_(float(phase_real_bias))
        out_proj.bias[bins * 2 :].zero_()
    return {
        "mode": mode,
        "scale": float(scale),
        "amp_bias": float(amp_bias),
        "phase_real_bias": float(phase_real_bias),
        "head_out_weight_std": float(out_proj.weight.detach().float().std(unbiased=False).cpu()),
        "head_out_bias_mean": float(out_proj.bias.detach().float().mean().cpu()),
    }
