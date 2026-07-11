#!/usr/bin/env python3
"""Train a tiny Root A latent student on a Piper-native pack.

This is intentionally not a full final TTS model. It isolates one question:
can a small duration-conditioned student learn Piper's post-flow generator
input well enough to render speech through the verified decoder cut?
"""

from __future__ import annotations

import argparse
import hashlib
import html
import importlib.util
import json
import math
import os
import pathlib
import random
import sys
import wave
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
from torch import nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACK_DIR = (
    ROOT
    / "artifacts"
    / "sub10m-search"
    / "root-a-piper-vits"
    / "chitwan-medium-a1-32row-piper-native-pack-20260625"
)
DEFAULT_DECODER = (
    ROOT
    / "artifacts"
    / "sub10m-search"
    / "root-a-piper-vits"
    / "chitwan-medium-a2-decoder-cut-smoke-20260625"
    / "chitwan-decoder-from-generator-input.onnx"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "artifacts"
    / "sub10m-search"
    / "root-a-piper-vits"
    / "chitwan-medium-a3-latent-student-overfit-20260625"
)
SOURCE_FILTER_HOP_LENGTH = 256


@dataclass(frozen=True)
class ChunkSample:
    row_id: str
    row_index: int
    text: str
    chunk_index: int
    phoneme_ids: np.ndarray
    durations: np.ndarray
    target: np.ndarray
    tensor_path: Path
    audio_samples: int
    target_npz_path: Path | None = None
    decoder_target_audio: np.ndarray | None = None


@dataclass(frozen=True)
class ExpandedFeatures:
    ids: torch.Tensor
    durations: torch.Tensor
    expanded_ids: torch.Tensor
    frame_pos: torch.Tensor
    token_pos: torch.Tensor
    duration_pos: torch.Tensor


@dataclass(frozen=True)
class LatentChannelStats:
    mean: torch.Tensor
    std: torch.Tensor


@dataclass(frozen=True)
class SourceFilterAwareTarget:
    normalized_source: torch.Tensor
    target_audio: torch.Tensor
    frame_count: int


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
            nn.SiLU(),
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
        )
        self.scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.scale * self.net(x)


class LatentDiscriminator(nn.Module):
    """PatchGAN-style 1D critic over the time axis of a [T, C] latent.

    Training-only. It pushes the student's predicted latent toward the teacher
    latent's *distribution* (defeating L2 regression-to-the-mean, i.e. the
    over-smoothing that flattens prosody and depresses SCOREQ) instead of its
    per-frame conditional mean. It is never saved and never runs at inference:
    the acoustic student's output stays a single deterministic [T, C] tensor,
    so the decoder / joint / render paths are entirely untouched. Convolving
    over time yields many overlapping patch scores per utterance, so the
    adversarial signal is dense even at batch size 1.
    """

    def __init__(self, channels: int, hidden: int, layers: int, kernel_size: int = 5) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError(f"latent discriminator layers must be >= 1, got {layers}")
        padding = kernel_size // 2
        widths = [channels] + [hidden] * layers
        net: list[nn.Module] = []
        for i in range(layers):
            net.append(nn.Conv1d(widths[i], widths[i + 1], kernel_size, padding=padding))
            net.append(nn.LeakyReLU(0.2))
        net.append(nn.Conv1d(widths[layers], 1, kernel_size, padding=padding))
        self.net = nn.Sequential(*net)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        # [T, C] -> [1, C, T] -> conv stack -> [1, 1, T'] -> [T'] patch scores
        x = latent.transpose(0, 1).unsqueeze(0)
        return self.net(x).squeeze(0).squeeze(0)


class LatentStudent(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        hidden: int,
        depth: int,
        kernel_size: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden)
        self.input_proj = nn.Conv1d(hidden + 3, hidden, 1)
        self.blocks = nn.ModuleList([ResidualConvBlock(hidden, kernel_size) for _ in range(depth)])
        self.output = nn.Conv1d(hidden, out_channels, 1)

    def forward(
        self,
        expanded_ids: torch.Tensor,
        frame_pos: torch.Tensor,
        token_pos: torch.Tensor,
        duration_pos: torch.Tensor,
    ) -> torch.Tensor:
        if expanded_ids.ndim != 1:
            raise RuntimeError(f"expected 1D expanded_ids, got {expanded_ids.shape}")
        x = self.embedding(expanded_ids).transpose(0, 1).unsqueeze(0)
        features = torch.stack([frame_pos, token_pos, duration_pos], dim=0).unsqueeze(0)
        x = self.input_proj(torch.cat([x, features], dim=1))
        for block in self.blocks:
            x = block(x)
        return self.output(x).squeeze(0).transpose(0, 1)


class ContextualLatentStudent(nn.Module):
    """Token-context acoustic student with Piper/VITS duration expansion.

    The old frame-only student receives repeated phoneme IDs, so every frame of
    a token starts with the same local state. This variant first encodes the
    token sequence and duration hints, then repeats the contextual token states
    into frames before the frame-level stack.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        hidden: int,
        depth: int,
        token_depth: int,
        kernel_size: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        if token_depth < 1:
            raise ValueError(f"token_depth must be >= 1, got {token_depth}")
        self.embedding = nn.Embedding(vocab_size, hidden)
        self.token_input_proj = nn.Conv1d(hidden + 2, hidden, 1)
        self.token_blocks = nn.ModuleList([ResidualConvBlock(hidden, kernel_size) for _ in range(token_depth)])
        self.frame_input_proj = nn.Conv1d(hidden + 3, hidden, 1)
        self.frame_blocks = nn.ModuleList([ResidualConvBlock(hidden, kernel_size) for _ in range(depth)])
        self.output = nn.Conv1d(hidden, out_channels, 1)

    def forward(self, features: ExpandedFeatures) -> torch.Tensor:
        ids = features.ids
        durations = features.durations
        if ids.ndim != 1:
            raise RuntimeError(f"expected 1D ids, got {ids.shape}")
        if durations.shape != ids.shape:
            raise RuntimeError(f"duration/id shape mismatch: {durations.shape} vs {ids.shape}")
        token_frames = int(features.expanded_ids.numel())
        if token_frames <= 0:
            raise RuntimeError("empty expanded token sequence")

        token_count = int(ids.numel())
        token_pos = torch.linspace(0.0, 1.0, token_count, device=ids.device)
        durations_f = durations.to(dtype=torch.float32)
        max_duration = torch.clamp(torch.max(durations_f), min=1.0)
        duration_hint = torch.log1p(durations_f) / torch.log1p(max_duration)

        token_x = self.embedding(ids).transpose(0, 1).unsqueeze(0)
        token_features = torch.stack([token_pos, duration_hint], dim=0).unsqueeze(0)
        token_x = self.token_input_proj(torch.cat([token_x, token_features], dim=1))
        for block in self.token_blocks:
            token_x = block(token_x)

        contextual_tokens = token_x.squeeze(0).transpose(0, 1)
        expanded_context = torch.repeat_interleave(contextual_tokens, durations, dim=0)
        if int(expanded_context.shape[0]) != token_frames:
            raise RuntimeError(
                f"context expansion length {expanded_context.shape[0]} != frame length {token_frames}"
            )
        x = expanded_context.transpose(0, 1).unsqueeze(0)
        frame_features = torch.stack(
            [features.frame_pos, features.token_pos, features.duration_pos],
            dim=0,
        ).unsqueeze(0)
        x = self.frame_input_proj(torch.cat([x, frame_features], dim=1))
        for block in self.frame_blocks:
            x = block(x)
        return self.output(x).squeeze(0).transpose(0, 1)


class SeparableContextualLatentStudent(nn.Module):
    """Token-context student using depthwise-separable residual frame blocks."""

    def __init__(
        self,
        *,
        vocab_size: int,
        hidden: int,
        depth: int,
        token_depth: int,
        kernel_size: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        if token_depth < 1:
            raise ValueError(f"token_depth must be >= 1, got {token_depth}")
        self.embedding = nn.Embedding(vocab_size, hidden)
        self.token_input_proj = nn.Conv1d(hidden + 2, hidden, 1)
        self.token_blocks = nn.ModuleList([SeparableResidualConvBlock(hidden, kernel_size) for _ in range(token_depth)])
        self.frame_input_proj = nn.Conv1d(hidden + 3, hidden, 1)
        self.frame_blocks = nn.ModuleList([SeparableResidualConvBlock(hidden, kernel_size) for _ in range(depth)])
        self.output = nn.Conv1d(hidden, out_channels, 1)

    def forward(self, features: ExpandedFeatures) -> torch.Tensor:
        ids = features.ids
        durations = features.durations
        if ids.ndim != 1:
            raise RuntimeError(f"expected 1D ids, got {ids.shape}")
        if durations.shape != ids.shape:
            raise RuntimeError(f"duration/id shape mismatch: {durations.shape} vs {ids.shape}")
        token_frames = int(features.expanded_ids.numel())
        if token_frames <= 0:
            raise RuntimeError("empty expanded token sequence")

        token_count = int(ids.numel())
        token_pos = torch.linspace(0.0, 1.0, token_count, device=ids.device)
        durations_f = durations.to(dtype=torch.float32)
        max_duration = torch.clamp(torch.max(durations_f), min=1.0)
        duration_hint = torch.log1p(durations_f) / torch.log1p(max_duration)

        token_x = self.embedding(ids).transpose(0, 1).unsqueeze(0)
        token_features = torch.stack([token_pos, duration_hint], dim=0).unsqueeze(0)
        token_x = self.token_input_proj(torch.cat([token_x, token_features], dim=1))
        for block in self.token_blocks:
            token_x = block(token_x)

        contextual_tokens = token_x.squeeze(0).transpose(0, 1)
        expanded_context = torch.repeat_interleave(contextual_tokens, durations, dim=0)
        if int(expanded_context.shape[0]) != token_frames:
            raise RuntimeError(
                f"context expansion length {expanded_context.shape[0]} != frame length {token_frames}"
            )
        x = expanded_context.transpose(0, 1).unsqueeze(0)
        frame_features = torch.stack(
            [features.frame_pos, features.token_pos, features.duration_pos],
            dim=0,
        ).unsqueeze(0)
        x = self.frame_input_proj(torch.cat([x, frame_features], dim=1))
        for block in self.frame_blocks:
            x = block(x)
        return self.output(x).squeeze(0).transpose(0, 1)


class SeparableResidualConvBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be positive and odd, got {kernel_size}")
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, groups=channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, 1),
        )
        self.scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.scale * self.net(x)


class FactorizedOutputHead(nn.Module):
    def __init__(
        self,
        *,
        hidden: int,
        out_channels: int,
        head_depth: int,
        kernel_size: int,
    ) -> None:
        super().__init__()
        if out_channels <= 0:
            raise ValueError(f"out_channels must be positive, got {out_channels}")
        if head_depth < 0:
            raise ValueError(f"head_depth must be non-negative, got {head_depth}")
        self.blocks = nn.ModuleList(
            [SeparableResidualConvBlock(hidden, kernel_size) for _ in range(int(head_depth))]
        )
        self.output = nn.Conv1d(hidden, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.output(x)


class FactorizedContextualLatentStudent(nn.Module):
    """Token-context student with separate envelope, gain, and source heads."""

    def __init__(
        self,
        *,
        vocab_size: int,
        hidden: int,
        depth: int,
        token_depth: int,
        kernel_size: int,
        out_channels: int,
        envelope_channels: int,
        gain_channels: int,
        head_depth: int,
        head_kernel_size: int,
    ) -> None:
        super().__init__()
        if token_depth < 1:
            raise ValueError(f"token_depth must be >= 1, got {token_depth}")
        source_channels = int(out_channels) - int(envelope_channels) - int(gain_channels)
        if envelope_channels <= 0 or gain_channels <= 0 or source_channels <= 0:
            raise ValueError(
                "factorized channels must be positive and sum below out_channels, "
                f"got envelope={envelope_channels} gain={gain_channels} out={out_channels}"
            )
        self.envelope_channels = int(envelope_channels)
        self.gain_channels = int(gain_channels)
        self.source_channels = int(source_channels)
        self.embedding = nn.Embedding(vocab_size, hidden)
        self.token_input_proj = nn.Conv1d(hidden + 2, hidden, 1)
        self.token_blocks = nn.ModuleList([ResidualConvBlock(hidden, kernel_size) for _ in range(token_depth)])
        self.frame_input_proj = nn.Conv1d(hidden + 3, hidden, 1)
        self.frame_blocks = nn.ModuleList([ResidualConvBlock(hidden, kernel_size) for _ in range(depth)])
        self.envelope_head = FactorizedOutputHead(
            hidden=hidden,
            out_channels=int(envelope_channels),
            head_depth=int(head_depth),
            kernel_size=int(head_kernel_size),
        )
        self.gain_head = FactorizedOutputHead(
            hidden=hidden,
            out_channels=int(gain_channels),
            head_depth=int(head_depth),
            kernel_size=int(head_kernel_size),
        )
        self.source_head = FactorizedOutputHead(
            hidden=hidden,
            out_channels=int(source_channels),
            head_depth=int(head_depth),
            kernel_size=int(head_kernel_size),
        )

    def forward(self, features: ExpandedFeatures) -> torch.Tensor:
        ids = features.ids
        durations = features.durations
        if ids.ndim != 1:
            raise RuntimeError(f"expected 1D ids, got {ids.shape}")
        if durations.shape != ids.shape:
            raise RuntimeError(f"duration/id shape mismatch: {durations.shape} vs {ids.shape}")
        token_frames = int(features.expanded_ids.numel())
        if token_frames <= 0:
            raise RuntimeError("empty expanded token sequence")

        token_count = int(ids.numel())
        token_pos = torch.linspace(0.0, 1.0, token_count, device=ids.device)
        durations_f = durations.to(dtype=torch.float32)
        max_duration = torch.clamp(torch.max(durations_f), min=1.0)
        duration_hint = torch.log1p(durations_f) / torch.log1p(max_duration)

        token_x = self.embedding(ids).transpose(0, 1).unsqueeze(0)
        token_features = torch.stack([token_pos, duration_hint], dim=0).unsqueeze(0)
        token_x = self.token_input_proj(torch.cat([token_x, token_features], dim=1))
        for block in self.token_blocks:
            token_x = block(token_x)

        contextual_tokens = token_x.squeeze(0).transpose(0, 1)
        expanded_context = torch.repeat_interleave(contextual_tokens, durations, dim=0)
        if int(expanded_context.shape[0]) != token_frames:
            raise RuntimeError(
                f"context expansion length {expanded_context.shape[0]} != frame length {token_frames}"
            )
        x = expanded_context.transpose(0, 1).unsqueeze(0)
        frame_features = torch.stack(
            [features.frame_pos, features.token_pos, features.duration_pos],
            dim=0,
        ).unsqueeze(0)
        x = self.frame_input_proj(torch.cat([x, frame_features], dim=1))
        for block in self.frame_blocks:
            x = block(x)
        outputs = [
            self.envelope_head(x),
            self.gain_head(x),
            self.source_head(x),
        ]
        return torch.cat(outputs, dim=1).squeeze(0).transpose(0, 1)


class LatentOutputAdapter(nn.Module):
    """Small decoder-facing latent calibration layer.

    The adapter is initialized near identity so it can be added to an existing
    acoustic checkpoint without immediately destroying the learned latent map.
    """

    def __init__(
        self,
        *,
        channels: int,
        mode: str,
        kernel_size: int,
        rank: int,
        start_channel: int = 0,
        end_channel: int | None = None,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        self.channels = int(channels)
        self.mode = str(mode)
        self.kernel_size = int(kernel_size)
        self.rank = int(rank)
        self.start_channel = int(start_channel)
        self.end_channel = int(channels if end_channel is None else end_channel)
        if self.start_channel < 0 or self.end_channel > self.channels or self.start_channel >= self.end_channel:
            raise ValueError(
                "invalid adapter channel slice: "
                f"start={self.start_channel} end={self.end_channel} channels={self.channels}"
            )
        self.adapter_channels = int(self.end_channel - self.start_channel)
        self.scale = nn.Parameter(torch.ones(self.adapter_channels, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(self.adapter_channels, dtype=torch.float32))
        if self.mode == "affine":
            self.depthwise = None
            self.lowrank_down = None
            self.lowrank_up = None
        elif self.mode in {"depthwise", "depthwise_lowrank"}:
            if self.kernel_size < 1 or self.kernel_size % 2 == 0:
                raise ValueError(f"adapter kernel size must be odd and positive, got {self.kernel_size}")
            self.depthwise = nn.Conv1d(
                self.adapter_channels,
                self.adapter_channels,
                self.kernel_size,
                padding=self.kernel_size // 2,
                groups=self.adapter_channels,
                bias=False,
            )
            nn.init.zeros_(self.depthwise.weight)
            self.depthwise.weight.data[:, 0, self.kernel_size // 2] = 1.0
            self.lowrank_down = None
            self.lowrank_up = None
        elif self.mode == "lowrank":
            self.depthwise = None
            self.lowrank_down = None
            self.lowrank_up = None
        else:
            raise ValueError(f"unsupported output adapter mode: {self.mode}")

        if self.mode in {"lowrank", "depthwise_lowrank"}:
            if self.rank <= 0:
                raise ValueError(f"adapter rank must be positive, got {self.rank}")
            self.lowrank_down = nn.Conv1d(self.adapter_channels, self.rank, 1)
            self.lowrank_up = nn.Conv1d(self.rank, self.adapter_channels, 1)
            nn.init.normal_(self.lowrank_down.weight, mean=0.0, std=0.01)
            nn.init.zeros_(self.lowrank_down.bias)
            nn.init.zeros_(self.lowrank_up.weight)
            nn.init.zeros_(self.lowrank_up.bias)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 2 or int(latent.shape[1]) != self.channels:
            raise RuntimeError(f"expected latent [frames,{self.channels}], got {latent.shape}")
        source = latent[:, self.start_channel : self.end_channel]
        x = source.transpose(0, 1).unsqueeze(0)
        if self.depthwise is not None:
            x = self.depthwise(x)
        if self.lowrank_down is not None or self.lowrank_up is not None:
            if self.lowrank_down is None or self.lowrank_up is None:
                raise RuntimeError("incomplete low-rank adapter")
            x = x + self.lowrank_up(torch.tanh(self.lowrank_down(x)))
        adapted = x.squeeze(0).transpose(0, 1)
        adapted = adapted * self.scale.reshape(1, -1) + self.bias.reshape(1, -1)
        if self.start_channel == 0 and self.end_channel == self.channels:
            return adapted
        output = latent.clone()
        output[:, self.start_channel : self.end_channel] = adapted
        return output


class CalibratedLatentStudent(nn.Module):
    def __init__(self, base: nn.Module, adapter: LatentOutputAdapter) -> None:
        super().__init__()
        self.base = base
        self.adapter = adapter

    def forward(self, features: ExpandedFeatures) -> torch.Tensor:
        return self.adapter(predict_latent_tensor(self.base, features))


class SourceBranchCalibratedLatentStudent(nn.Module):
    """Preserve a base model and replace only a contiguous source channel slice."""

    def __init__(
        self,
        base: nn.Module,
        source_branch: nn.Module,
        *,
        start_channel: int,
        end_channel: int,
    ) -> None:
        super().__init__()
        self.base = base
        self.source_branch = source_branch
        self.start_channel = int(start_channel)
        self.end_channel = int(end_channel)
        if self.start_channel < 0 or self.start_channel >= self.end_channel:
            raise ValueError(f"invalid source slice: {self.start_channel}:{self.end_channel}")

    def forward(self, features: ExpandedFeatures) -> torch.Tensor:
        base_latent = predict_latent_tensor(self.base, features)
        if base_latent.ndim != 2 or int(base_latent.shape[1]) < self.end_channel:
            raise RuntimeError(
                f"base latent shape {base_latent.shape} incompatible with source slice "
                f"{self.start_channel}:{self.end_channel}"
            )
        source = predict_latent_tensor(self.source_branch, features)
        expected_channels = self.end_channel - self.start_channel
        if source.ndim != 2 or source.shape != (base_latent.shape[0], expected_channels):
            raise RuntimeError(
                f"source branch shape {source.shape} != expected "
                f"({base_latent.shape[0]}, {expected_channels})"
            )
        output = base_latent.clone()
        output[:, self.start_channel : self.end_channel] = source
        return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack-dir", type=Path, default=DEFAULT_PACK_DIR)
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=None,
        help=(
            "Optional target directory produced by a target builder such as "
            "build_roota_mel_targets.py. When omitted, the trainer uses "
            "generator_input from the Piper tensor NPZs."
        ),
    )
    parser.add_argument(
        "--target-key",
        type=str,
        default="log_mel",
        help="Array key to read from --target-dir target NPZs.",
    )
    parser.add_argument(
        "--target-duration-key",
        type=str,
        default=None,
        help=(
            "Optional duration array key to read from --target-dir target NPZs. "
            "Use this when the target tensor has been retimed to a different "
            "duration grid than the source Piper w_ceil."
        ),
    )
    parser.add_argument(
        "--eval-pack-dir",
        type=Path,
        default=None,
        help="Optional held-out Piper-native pack used only for latent evaluation and dashboard rendering.",
    )
    parser.add_argument(
        "--eval-target-dir",
        type=Path,
        default=None,
        help="Optional target directory for --eval-pack-dir. Required when --target-dir and --eval-pack-dir are both set.",
    )
    parser.add_argument("--decoder", type=Path, default=DEFAULT_DECODER)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--load-checkpoint", type=Path, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--mse-weight", type=float, default=0.25)
    parser.add_argument(
        "--norm-l1-weight",
        type=float,
        default=0.0,
        help="Extra L1 loss after train-set per-channel latent normalization.",
    )
    parser.add_argument(
        "--delta-l1-weight",
        type=float,
        default=0.0,
        help="Extra temporal-delta L1 loss on adjacent latent frames.",
    )
    parser.add_argument(
        "--channel-stat-weight",
        type=float,
        default=0.0,
        help="Extra per-sample channel mean/std alignment loss.",
    )
    parser.add_argument(
        "--channel-priority-report",
        type=Path,
        default=None,
        help=(
            "Optional channel-error report from analyze_roota_latent_channel_errors.py. "
            "When set, top_channels severity values define a per-channel priority loss."
        ),
    )
    parser.add_argument(
        "--channel-priority-weight",
        type=float,
        default=0.0,
        help="Extra normalized L1 loss weighted by --channel-priority-report severity.",
    )
    parser.add_argument(
        "--channel-priority-delta-weight",
        type=float,
        default=0.0,
        help="Extra temporal-delta L1 loss weighted by --channel-priority-report severity.",
    )
    parser.add_argument(
        "--channel-priority-scale",
        type=float,
        default=4.0,
        help="Maximum additive multiplier for the highest-severity channel.",
    )
    parser.add_argument(
        "--decoder-aware-checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional decoder-student.pt checkpoint used as a frozen differentiable "
            "decoder during acoustic training."
        ),
    )
    parser.add_argument(
        "--decoder-aware-l1-weight",
        type=float,
        default=0.0,
        help="Extra waveform L1 loss after passing predicted latents through the frozen decoder student.",
    )
    parser.add_argument(
        "--decoder-aware-stft-weight",
        type=float,
        default=0.0,
        help="Extra multi-resolution STFT loss after passing predicted latents through the frozen decoder student.",
    )
    parser.add_argument(
        "--decoder-aware-feature-weight",
        type=float,
        default=0.0,
        help=(
            "Extra L1 loss on selected frozen decoder-student internal features, "
            "comparing predicted-latent features to teacher-latent features on the same crop."
        ),
    )
    parser.add_argument(
        "--decoder-aware-feature-keys",
        type=str,
        default="stage0_mix,stage1_mix,stage2_mix,pre_tanh",
        help=(
            "Comma-separated decoder feature keys for --decoder-aware-feature-weight. "
            "Common keys: pre,up0_raw,stage0_mix,up1_raw,stage1_mix,up2_raw,stage2_mix,pre_tanh,audio_pre_filter,audio."
        ),
    )
    parser.add_argument(
        "--decoder-aware-crop-frames",
        type=int,
        default=64,
        help="Latent frames used per decoder-aware waveform crop.",
    )
    parser.add_argument(
        "--decoder-aware-target-cache-dir",
        type=Path,
        default=None,
        help="Optional directory for cached teacher-decoder audio targets keyed by tensor and decoder file metadata.",
    )
    parser.add_argument(
        "--source-filter-aware-spectral-weight",
        type=float,
        default=0.0,
        help=(
            "Extra differentiable source-filter spectral composition loss for B7/B9 "
            "phase-template targets. Uses fixed target source residual, predicted "
            "gain, and predicted reflection-logit envelope."
        ),
    )
    parser.add_argument(
        "--source-filter-aware-rms-weight",
        type=float,
        default=0.0,
        help="Extra RMS loss on source-filter spectral magnitudes.",
    )
    parser.add_argument(
        "--source-filter-aware-waveform-weight",
        type=float,
        default=0.0,
        help="Extra frame-local differentiable LPC waveform L1 loss using torchaudio lfilter.",
    )
    parser.add_argument(
        "--source-filter-aware-crop-frames",
        type=int,
        default=64,
        help="Frames used per source-filter-aware spectral crop.",
    )
    parser.add_argument(
        "--source-filter-aware-n-fft",
        type=int,
        default=512,
        help="FFT size for source-filter-aware per-frame spectra.",
    )
    parser.add_argument(
        "--source-filter-aware-target-audio-key",
        choices=("oracle_reconstruction", "teacher_audio"),
        default="oracle_reconstruction",
        help="Audio array in each target NPZ used as the source-filter spectral target.",
    )
    parser.add_argument(
        "--architecture",
        choices=("frame_conv", "token_context", "separable_token_context", "factorized_token_context"),
        default="frame_conv",
    )
    parser.add_argument(
        "--output-adapter",
        choices=("none", "affine", "depthwise", "lowrank", "depthwise_lowrank"),
        default="none",
        help="Optional small output adapter added around a loaded acoustic checkpoint.",
    )
    parser.add_argument(
        "--output-adapter-scope",
        choices=("all", "factorized_envelope", "factorized_gain", "factorized_env_gain", "factorized_source"),
        default="all",
        help="Restrict an added output adapter to all channels or one factorized channel slice.",
    )
    parser.add_argument("--output-adapter-kernel-size", type=int, default=5)
    parser.add_argument("--output-adapter-rank", type=int, default=16)
    parser.add_argument(
        "--output-adapter-lr-multiplier",
        type=float,
        default=1.0,
        help="Learning-rate multiplier for adapter params when training a calibrated checkpoint.",
    )
    parser.add_argument(
        "--freeze-loaded-base",
        action="store_true",
        help=(
            "When adapting --load-checkpoint, freeze the loaded base acoustic model. "
            "For source_branch_calibrated checkpoints, this leaves the existing branch trainable."
        ),
    )
    parser.add_argument("--hidden", type=int, default=192)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--token-depth", type=int, default=3)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument(
        "--latent-adv-weight",
        type=float,
        default=0.0,
        help="Weight of the latent adversarial (de-smoothing) loss. 0 disables it entirely (default), "
        "leaving training identical to before.",
    )
    parser.add_argument(
        "--latent-adv-start-step",
        type=int,
        default=1,
        help="Step at which the latent discriminator and adversarial term activate.",
    )
    parser.add_argument("--latent-adv-lr", type=float, default=2e-4, help="Latent discriminator learning rate.")
    parser.add_argument("--latent-disc-hidden", type=int, default=64, help="Latent discriminator hidden channels.")
    parser.add_argument("--latent-disc-layers", type=int, default=3, help="Latent discriminator conv layers.")
    parser.add_argument("--factorized-envelope-channels", type=int, default=16)
    parser.add_argument("--factorized-gain-channels", type=int, default=4)
    parser.add_argument("--factorized-head-depth", type=int, default=1)
    parser.add_argument("--factorized-head-kernel-size", type=int, default=5)
    parser.add_argument(
        "--source-branch-hidden",
        type=int,
        default=0,
        help="When >0, add a compact contextual source branch to a loaded factorized checkpoint.",
    )
    parser.add_argument(
        "--source-branch-architecture",
        choices=("token_context", "separable_token_context"),
        default="token_context",
    )
    parser.add_argument(
        "--source-branch-scope",
        choices=("factorized_envelope", "factorized_gain", "factorized_env_gain", "factorized_source"),
        default="factorized_source",
        help="Factorized channel slice replaced by --source-branch-hidden.",
    )
    parser.add_argument("--source-branch-depth", type=int, default=2)
    parser.add_argument("--source-branch-token-depth", type=int, default=1)
    parser.add_argument("--source-branch-kernel-size", type=int, default=5)
    parser.add_argument("--envelope-norm-weight", type=float, default=0.0)
    parser.add_argument("--gain-norm-weight", type=float, default=0.0)
    parser.add_argument("--source-norm-weight", type=float, default=0.0)
    parser.add_argument("--source-delta-weight", type=float, default=0.0)
    parser.add_argument("--source-smooth-weight", type=float, default=0.0)
    parser.add_argument(
        "--freeze-factorized-trunk",
        action="store_true",
        help="Freeze the shared token/frame trunk of a factorized_token_context model.",
    )
    parser.add_argument(
        "--freeze-factorized-source-head",
        action="store_true",
        help="Freeze the source head of a factorized_token_context model.",
    )
    parser.add_argument(
        "--freeze-factorized-envelope-head",
        action="store_true",
        help="Freeze the envelope head of a factorized_token_context model.",
    )
    parser.add_argument(
        "--freeze-factorized-gain-head",
        action="store_true",
        help="Freeze the gain head of a factorized_token_context model.",
    )
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--render-rows", type=int, default=16)
    parser.add_argument(
        "--skip-render",
        action="store_true",
        help=(
            "Skip Piper-decoder audio rendering. Use this for targets that do not "
            "have a matching decoder cut, such as log_mel."
        ),
    )
    parser.add_argument("--sentence-silence", type=float, default=0.12)
    parser.add_argument(
        "--sample-weight-mode",
        choices=("uniform", "frames"),
        default="uniform",
        help=(
            "Training sample distribution. 'uniform' samples chunks equally. "
            "'frames' samples proportional to latent frame count so short suffix chunks "
            "do not dominate template-heavy packs."
        ),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    return parser.parse_args()


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def require_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{label} not found: {path}")


def read_json(path: Path) -> Any:
    require_file(path, "JSON")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def load_torch_checkpoint_windows_safe(path: Path, label: str) -> dict[str, Any]:
    require_file(path, label)
    original_windows_path = pathlib.WindowsPath
    original_posix_path = pathlib.PosixPath
    if os.name == "nt":
        pathlib.PosixPath = pathlib.WindowsPath
    else:
        pathlib.WindowsPath = pathlib.PosixPath
    try:
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location="cpu")
    finally:
        pathlib.WindowsPath = original_windows_path
        pathlib.PosixPath = original_posix_path
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"{label} must be a dict checkpoint: {path}")
    return checkpoint


def load_target_index(target_dir: Path) -> dict[tuple[str, int], Path]:
    manifest_path = target_dir / "manifest.jsonl"
    require_file(manifest_path, "target manifest")
    index: dict[tuple[str, int], Path] = {}
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{manifest_path}:{line_no}: invalid JSON") from exc
            row_id = str(row.get("row_id") or "")
            if not row_id:
                raise RuntimeError(f"{manifest_path}:{line_no}: missing row_id")
            chunk_index = int(row.get("chunk_index") or 0)
            target_path = Path(str(row.get("target_npz") or ""))
            require_file(target_path, "target NPZ")
            key = (row_id, chunk_index)
            if key in index:
                raise RuntimeError(f"{manifest_path}:{line_no}: duplicate target key {key}")
            index[key] = target_path
    if not index:
        raise RuntimeError(f"{manifest_path} contains no target rows")
    return index


def normalize_frame_target(
    value: np.ndarray,
    *,
    tensor_path: Path,
    target_key: str,
    duration_frames: int,
) -> np.ndarray:
    target = np.asarray(value, dtype=np.float32)
    if target.ndim == 3:
        if int(target.shape[0]) != 1:
            raise RuntimeError(f"{tensor_path}:{target_key}: expected [1, C, T], got {target.shape}")
        target = target.squeeze(0).transpose(1, 0)
    elif target.ndim == 2:
        if int(target.shape[0]) != int(duration_frames) and int(target.shape[1]) == int(duration_frames):
            target = target.transpose(1, 0)
    else:
        raise RuntimeError(f"{tensor_path}:{target_key}: invalid target shape {target.shape}")
    if target.ndim != 2 or target.shape[1] <= 0:
        raise RuntimeError(f"{tensor_path}:{target_key}: invalid target shape {target.shape}")
    if int(duration_frames) != int(target.shape[0]):
        raise RuntimeError(f"{tensor_path}:{target_key}: duration sum {duration_frames} != target frames {target.shape[0]}")
    if not np.isfinite(target).all():
        raise RuntimeError(f"{tensor_path}:{target_key}: non-finite target")
    return np.ascontiguousarray(target, dtype=np.float32)


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    audio_f = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio_f.size <= 0:
        raise RuntimeError(f"refusing to write empty WAV: {path}")
    pcm = (np.clip(audio_f, -1.0, 1.0) * 32767.0).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())


def pick_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
        return torch.device("cuda")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but torch.backends.mps.is_available() is false")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_samples(
    pack_dir: Path,
    *,
    target_index: dict[tuple[str, int], Path] | None = None,
    target_key: str = "generator_input",
    target_duration_key: str | None = None,
) -> tuple[list[dict[str, Any]], list[ChunkSample], int, int]:
    require_dir(pack_dir, "pack directory")
    rows = read_json(pack_dir / "rows.json")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"{pack_dir / 'rows.json'} must contain a non-empty list")

    samples: list[ChunkSample] = []
    max_id = 0
    out_channels = 0
    for row in rows:
        row_id = str(row.get("row_id") or "")
        text = str(row.get("text") or "")
        row_index = int(row.get("index") or len(samples) + 1)
        chunks = row.get("chunks")
        if not row_id or not isinstance(chunks, list) or not chunks:
            raise RuntimeError(f"invalid row metadata: {row!r}")
        for chunk in chunks:
            tensor_path = Path(str(chunk.get("tensor_npz") or ""))
            require_file(tensor_path, "tensor NPZ")
            target_npz_path: Path | None = None
            with np.load(tensor_path) as tensors:
                missing = {"phoneme_ids", "w_ceil", "generator_input"} - set(tensors.files)
                if missing:
                    raise RuntimeError(f"{tensor_path}: missing tensors {sorted(missing)}")
                phoneme_ids = np.asarray(tensors["phoneme_ids"], dtype=np.int64).reshape(-1)
                durations = np.rint(np.asarray(tensors["w_ceil"], dtype=np.float32).reshape(-1)).astype(np.int64)
                if target_index is None:
                    target = np.asarray(tensors["generator_input"], dtype=np.float32)
                    target_source_path = tensor_path
                    target_source_key = "generator_input"
                else:
                    key = (row_id, int(chunk.get("chunk_index") or 0))
                    target_path = target_index.get(key)
                    if target_path is None:
                        raise RuntimeError(f"missing target for {key} in {pack_dir}")
                    target_npz_path = target_path
                    with np.load(target_path) as target_npz:
                        if target_key not in target_npz.files:
                            raise RuntimeError(
                                f"{target_path}: missing target key {target_key!r}; "
                                f"available={target_npz.files}"
                            )
                        target = np.asarray(target_npz[target_key], dtype=np.float32)
                        target_source_path = target_path
                        target_source_key = target_key
                        if target_duration_key is not None:
                            if target_duration_key not in target_npz.files:
                                raise RuntimeError(
                                    f"{target_path}: missing target duration key {target_duration_key!r}; "
                                    f"available={target_npz.files}"
                                )
                            durations = np.rint(
                                np.asarray(target_npz[target_duration_key], dtype=np.float32).reshape(-1)
                            ).astype(np.int64)
            if phoneme_ids.size <= 0:
                raise RuntimeError(f"{tensor_path}: empty phoneme_ids")
            if durations.shape[0] != phoneme_ids.shape[0]:
                raise RuntimeError(f"{tensor_path}: duration/id length mismatch")
            if np.any(durations < 0):
                raise RuntimeError(f"{tensor_path}: negative duration")
            target = normalize_frame_target(
                target,
                tensor_path=target_source_path,
                target_key=target_source_key,
                duration_frames=int(durations.sum()),
            )
            max_id = max(max_id, int(phoneme_ids.max()))
            out_channels = int(target.shape[1])
            samples.append(
                ChunkSample(
                    row_id=row_id,
                    row_index=row_index,
                    text=text,
                    chunk_index=int(chunk.get("chunk_index") or 0),
                    phoneme_ids=phoneme_ids,
                    durations=durations,
                    target=target,
                    tensor_path=tensor_path,
                    audio_samples=int(chunk.get("audio_samples") or 0),
                    target_npz_path=target_npz_path,
                )
            )
    if not samples:
        raise RuntimeError("pack produced no chunk samples")
    return rows, samples, max_id + 1, out_channels


def expand_features(sample: ChunkSample, device: torch.device) -> ExpandedFeatures:
    ids = torch.as_tensor(sample.phoneme_ids, dtype=torch.long, device=device)
    durations = torch.as_tensor(sample.durations, dtype=torch.long, device=device)
    expanded_ids = torch.repeat_interleave(ids, durations)
    if int(expanded_ids.numel()) != int(sample.target.shape[0]):
        raise RuntimeError(
            f"{sample.row_id} chunk {sample.chunk_index}: expanded length "
            f"{expanded_ids.numel()} != target length {sample.target.shape[0]}"
        )
    frames = int(expanded_ids.numel())
    frame_pos = torch.linspace(0.0, 1.0, frames, device=device)

    token_positions = []
    duration_positions = []
    token_count = max(int(ids.numel()) - 1, 1)
    for token_index, duration in enumerate(sample.durations.tolist()):
        duration_int = int(duration)
        if duration_int <= 0:
            continue
        token_positions.extend([float(token_index) / float(token_count)] * duration_int)
        if duration_int == 1:
            duration_positions.append(0.0)
        else:
            duration_positions.extend([float(i) / float(duration_int - 1) for i in range(duration_int)])
    token_pos = torch.as_tensor(token_positions, dtype=torch.float32, device=device)
    duration_pos = torch.as_tensor(duration_positions, dtype=torch.float32, device=device)
    if token_pos.numel() != expanded_ids.numel() or duration_pos.numel() != expanded_ids.numel():
        raise RuntimeError(f"{sample.row_id} chunk {sample.chunk_index}: feature expansion mismatch")
    return ExpandedFeatures(
        ids=ids,
        durations=durations,
        expanded_ids=expanded_ids,
        frame_pos=frame_pos,
        token_pos=token_pos,
        duration_pos=duration_pos,
    )


def target_tensor(sample: ChunkSample, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(sample.target, dtype=torch.float32, device=device)


def sample_weights_for(samples: list[ChunkSample], mode: str) -> list[float] | None:
    if mode == "uniform":
        return None
    if mode != "frames":
        raise ValueError(f"unsupported sample weight mode: {mode!r}")
    weights: list[float] = []
    for sample in samples:
        frames = int(sample.target.shape[0])
        if frames <= 0:
            raise RuntimeError(f"{sample.tensor_path}: cannot frame-weight an empty target")
        weights.append(float(frames))
    if not weights or sum(weights) <= 0.0:
        raise RuntimeError("sample weighting produced no positive weights")
    return weights


def summarize_sample_weights(samples: list[ChunkSample], weights: list[float] | None) -> dict[str, Any]:
    frames = np.asarray([int(sample.target.shape[0]) for sample in samples], dtype=np.float64)
    if frames.size <= 0:
        raise RuntimeError("cannot summarize empty sample list")
    summary: dict[str, Any] = {
        "mode": "uniform" if weights is None else "frames",
        "chunks": int(len(samples)),
        "frame_min": int(np.min(frames)),
        "frame_median": float(np.median(frames)),
        "frame_mean": float(np.mean(frames)),
        "frame_max": int(np.max(frames)),
    }
    chunk_counts: dict[str, int] = {}
    probability_by_chunk_index: dict[str, float] = {}
    total_probability = float(len(samples)) if weights is None else float(sum(weights))
    if total_probability <= 0.0:
        raise RuntimeError("non-positive sample probability mass")
    for sample, weight in zip(samples, weights or [1.0] * len(samples), strict=True):
        key = str(int(sample.chunk_index))
        chunk_counts[key] = chunk_counts.get(key, 0) + 1
        probability_by_chunk_index[key] = probability_by_chunk_index.get(key, 0.0) + float(weight) / total_probability
    summary["chunk_counts"] = dict(sorted(chunk_counts.items(), key=lambda item: int(item[0])))
    summary["probability_by_chunk_index"] = dict(
        sorted(probability_by_chunk_index.items(), key=lambda item: int(item[0]))
    )
    return summary


def count_parameters(model: nn.Module) -> int:
    return int(sum(param.numel() for param in model.parameters()))


def latent_parameter_table(model: nn.Module) -> dict[str, Any]:
    total = count_parameters(model)
    output_modules: list[dict[str, Any]] = []
    output_parameters = 0
    for name, module in model.named_modules():
        if name and name.split(".")[-1] == "output":
            params = count_parameters(module)
            output_modules.append({"module": name, "parameters": int(params)})
            output_parameters += params
    if output_parameters == 0:
        output_parameter_names = [
            name for name, _parameter in model.named_parameters() if ".output." in f".{name}."
        ]
        output_parameters = int(
            sum(parameter.numel() for name, parameter in model.named_parameters() if name in output_parameter_names)
        )
        output_modules = [{"module": "output", "parameters": int(output_parameters)}] if output_parameters else []
    trunk_parameters = int(total - output_parameters)
    return {
        "trunk_parameters": trunk_parameters,
        "output_head_parameters": int(output_parameters),
        "total_parameters": int(total),
        "output_modules": output_modules,
    }


def trainable_optimizer_groups(model: nn.Module, lr: float, adapter_lr_multiplier: float) -> list[dict[str, Any]]:
    if not math.isfinite(lr) or lr <= 0.0:
        raise ValueError(f"learning rate must be finite and positive, got {lr!r}")
    if not math.isfinite(adapter_lr_multiplier) or adapter_lr_multiplier <= 0.0:
        raise ValueError(
            f"--output-adapter-lr-multiplier must be finite and positive, got {adapter_lr_multiplier!r}"
        )
    if isinstance(model, CalibratedLatentStudent):
        base_params = [param for param in model.base.parameters() if param.requires_grad]
        adapter_params = [param for param in model.adapter.parameters() if param.requires_grad]
        groups: list[dict[str, Any]] = []
        if base_params:
            groups.append({"params": base_params, "lr": float(lr), "name": "base"})
        if adapter_params:
            groups.append(
                {
                    "params": adapter_params,
                    "lr": float(lr) * float(adapter_lr_multiplier),
                    "name": "adapter",
                }
            )
        return groups
    params = [param for param in model.parameters() if param.requires_grad]
    return [{"params": params, "lr": float(lr), "name": "model"}] if params else []


def apply_factorized_freezing(
    model: nn.Module,
    *,
    freeze_trunk: bool,
    freeze_envelope_head: bool,
    freeze_gain_head: bool,
    freeze_source_head: bool,
) -> dict[str, Any]:
    if not freeze_trunk and not freeze_envelope_head and not freeze_gain_head and not freeze_source_head:
        return {"trunk": False, "envelope_head": False, "gain_head": False, "source_head": False}
    if not isinstance(model, FactorizedContextualLatentStudent):
        raise RuntimeError("factorized freezing requires a factorized_token_context model")
    if freeze_trunk:
        trunk_modules = (
            model.embedding,
            model.token_input_proj,
            model.token_blocks,
            model.frame_input_proj,
            model.frame_blocks,
        )
        for module in trunk_modules:
            for parameter in module.parameters():
                parameter.requires_grad_(False)
    if freeze_envelope_head:
        for parameter in model.envelope_head.parameters():
            parameter.requires_grad_(False)
    if freeze_gain_head:
        for parameter in model.gain_head.parameters():
            parameter.requires_grad_(False)
    if freeze_source_head:
        for parameter in model.source_head.parameters():
            parameter.requires_grad_(False)
    return {
        "trunk": bool(freeze_trunk),
        "envelope_head": bool(freeze_envelope_head),
        "gain_head": bool(freeze_gain_head),
        "source_head": bool(freeze_source_head),
    }


def compute_latent_channel_stats(samples: list[ChunkSample], device: torch.device) -> LatentChannelStats:
    if not samples:
        raise RuntimeError("cannot compute latent stats from empty samples")
    channels = int(samples[0].target.shape[1])
    total_frames = 0
    channel_sum = np.zeros(channels, dtype=np.float64)
    channel_sumsq = np.zeros(channels, dtype=np.float64)
    for sample in samples:
        target = np.asarray(sample.target, dtype=np.float64)
        if target.ndim != 2 or int(target.shape[1]) != channels:
            raise RuntimeError(
                f"{sample.tensor_path}: expected target shape [frames,{channels}], got {target.shape}"
            )
        total_frames += int(target.shape[0])
        channel_sum += np.sum(target, axis=0)
        channel_sumsq += np.sum(np.square(target), axis=0)
    if total_frames <= 0:
        raise RuntimeError("latent stats saw zero frames")
    mean = channel_sum / float(total_frames)
    variance = channel_sumsq / float(total_frames) - np.square(mean)
    std = np.sqrt(np.maximum(variance, 1e-6))
    return LatentChannelStats(
        mean=torch.as_tensor(mean, dtype=torch.float32, device=device).unsqueeze(0),
        std=torch.as_tensor(std, dtype=torch.float32, device=device).unsqueeze(0),
    )


def latent_loss_components(
    prediction: torch.Tensor,
    target: torch.Tensor,
    stats: LatentChannelStats,
    channel_priority: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    if prediction.shape != target.shape:
        raise RuntimeError(f"prediction shape {prediction.shape} != target shape {target.shape}")
    diff = prediction - target
    norm_diff = torch.abs((prediction - stats.mean) / stats.std - (target - stats.mean) / stats.std)
    components = {
        "l1": torch.mean(torch.abs(diff)),
        "mse": torch.mean(torch.square(diff)),
        "norm_l1": torch.mean(norm_diff),
        "channel_mean_l1": torch.mean(torch.abs((prediction.mean(dim=0) - target.mean(dim=0)) / stats.std.squeeze(0))),
        "channel_std_l1": torch.mean(
            torch.abs(
                (
                    prediction.std(dim=0, unbiased=False)
                    - target.std(dim=0, unbiased=False)
                )
                / stats.std.squeeze(0)
            )
        ),
    }
    if int(prediction.shape[0]) > 1:
        components["delta_l1"] = torch.mean(
            torch.abs((prediction[1:] - prediction[:-1]) - (target[1:] - target[:-1]))
        )
    else:
        components["delta_l1"] = torch.zeros((), dtype=prediction.dtype, device=prediction.device)
    if channel_priority is not None:
        if channel_priority.ndim != 1 or int(channel_priority.shape[0]) != int(prediction.shape[1]):
            raise RuntimeError(
                f"channel priority shape {tuple(channel_priority.shape)} incompatible with prediction {tuple(prediction.shape)}"
            )
        priority = channel_priority.to(dtype=prediction.dtype, device=prediction.device).view(1, -1)
        priority_mean = torch.clamp(priority.mean(), min=1e-6)
        components["priority_norm_l1"] = torch.mean(norm_diff * priority) / priority_mean
        if int(prediction.shape[0]) > 1:
            delta_diff = torch.abs(
                ((prediction[1:] - prediction[:-1]) - (target[1:] - target[:-1])) / stats.std
            )
            components["priority_delta_l1"] = torch.mean(delta_diff * priority) / priority_mean
        else:
            components["priority_delta_l1"] = torch.zeros((), dtype=prediction.dtype, device=prediction.device)
    else:
        components["priority_norm_l1"] = torch.zeros((), dtype=prediction.dtype, device=prediction.device)
        components["priority_delta_l1"] = torch.zeros((), dtype=prediction.dtype, device=prediction.device)
    components["channel_stat"] = components["channel_mean_l1"] + components["channel_std_l1"]
    return components


def weighted_latent_loss(
    components: dict[str, torch.Tensor],
    *,
    mse_weight: float,
    norm_l1_weight: float,
    delta_l1_weight: float,
    channel_stat_weight: float,
    channel_priority_weight: float,
    channel_priority_delta_weight: float,
) -> torch.Tensor:
    return (
        components["l1"]
        + float(mse_weight) * components["mse"]
        + float(norm_l1_weight) * components["norm_l1"]
        + float(delta_l1_weight) * components["delta_l1"]
        + float(channel_stat_weight) * components["channel_stat"]
        + float(channel_priority_weight) * components["priority_norm_l1"]
        + float(channel_priority_delta_weight) * components["priority_delta_l1"]
    )


def factorized_base_config(config: dict[str, Any]) -> dict[str, Any] | None:
    architecture = str(config.get("architecture") or "")
    if architecture == "factorized_token_context":
        return config
    if architecture in {"calibrated", "source_branch_calibrated"}:
        base_config = config.get("base_config")
        if not isinstance(base_config, dict):
            raise RuntimeError(f"{architecture} config missing base_config: {config!r}")
        return factorized_base_config(base_config)
    return None


def factorized_channel_slices(config: dict[str, Any], channels: int) -> tuple[slice, slice, slice] | None:
    factorized_config = factorized_base_config(config)
    if factorized_config is None:
        return None
    envelope_channels = int(factorized_config.get("envelope_channels") or 0)
    gain_channels = int(factorized_config.get("gain_channels") or 0)
    if envelope_channels <= 0 or gain_channels <= 0:
        raise RuntimeError(f"invalid factorized channel config: {factorized_config!r}")
    source_start = envelope_channels + gain_channels
    if source_start >= int(channels):
        raise RuntimeError(
            f"factorized source start {source_start} outside channel count {channels}: {factorized_config!r}"
        )
    return (
        slice(0, envelope_channels),
        slice(envelope_channels, source_start),
        slice(source_start, int(channels)),
    )


def factorized_loss_components(
    prediction: torch.Tensor,
    target: torch.Tensor,
    stats: LatentChannelStats,
    config: dict[str, Any],
) -> dict[str, torch.Tensor]:
    if prediction.shape != target.shape:
        raise RuntimeError(f"prediction shape {prediction.shape} != target shape {target.shape}")
    slices = factorized_channel_slices(config, int(prediction.shape[1]))
    zero = prediction.new_tensor(0.0)
    if slices is None:
        return {
            "envelope_norm_l1": zero,
            "gain_norm_l1": zero,
            "source_norm_l1": zero,
            "source_delta_l1": zero,
            "source_smooth_l1": zero,
        }
    envelope_slice, gain_slice, source_slice = slices
    pred_norm = (prediction - stats.mean) / stats.std
    target_norm = (target - stats.mean) / stats.std
    source_prediction = pred_norm[:, source_slice]
    source_target = target_norm[:, source_slice]
    components = {
        "envelope_norm_l1": torch.mean(torch.abs(pred_norm[:, envelope_slice] - target_norm[:, envelope_slice])),
        "gain_norm_l1": torch.mean(torch.abs(pred_norm[:, gain_slice] - target_norm[:, gain_slice])),
        "source_norm_l1": torch.mean(torch.abs(source_prediction - source_target)),
    }
    if int(prediction.shape[0]) > 1:
        components["source_delta_l1"] = torch.mean(
            torch.abs(
                (source_prediction[1:] - source_prediction[:-1])
                - (source_target[1:] - source_target[:-1])
            )
        )
    else:
        components["source_delta_l1"] = zero
    if int(prediction.shape[0]) > 2:
        second_difference = source_prediction[2:] - 2.0 * source_prediction[1:-1] + source_prediction[:-2]
        components["source_smooth_l1"] = torch.mean(torch.abs(second_difference))
    else:
        components["source_smooth_l1"] = zero
    return components


def weighted_factorized_loss(
    components: dict[str, torch.Tensor],
    *,
    envelope_norm_weight: float,
    gain_norm_weight: float,
    source_norm_weight: float,
    source_delta_weight: float,
    source_smooth_weight: float,
) -> torch.Tensor:
    return (
        float(envelope_norm_weight) * components["envelope_norm_l1"]
        + float(gain_norm_weight) * components["gain_norm_l1"]
        + float(source_norm_weight) * components["source_norm_l1"]
        + float(source_delta_weight) * components["source_delta_l1"]
        + float(source_smooth_weight) * components["source_smooth_l1"]
    )


def load_channel_priority(
    report_path: Path | None,
    *,
    channels: int,
    scale: float,
    device: torch.device,
) -> tuple[torch.Tensor | None, dict[str, Any] | None]:
    if report_path is None:
        return None, None
    if channels <= 0:
        raise ValueError(f"channels must be positive, got {channels}")
    if not math.isfinite(float(scale)) or float(scale) < 0.0:
        raise ValueError(f"--channel-priority-scale must be finite and non-negative, got {scale!r}")
    payload = read_json(report_path)
    top_channels = payload.get("top_channels")
    if not isinstance(top_channels, list) or not top_channels:
        raise RuntimeError(f"{report_path}: missing non-empty top_channels")
    severity_by_channel: dict[int, float] = {}
    for row in top_channels:
        if not isinstance(row, dict):
            raise RuntimeError(f"{report_path}: top_channels row is not an object")
        channel = int(row.get("channel", -1))
        severity = float(row.get("severity", 0.0))
        if channel < 0 or channel >= channels:
            raise RuntimeError(f"{report_path}: channel {channel} outside [0,{channels})")
        if not math.isfinite(severity) or severity < 0.0:
            raise RuntimeError(f"{report_path}: invalid severity {severity!r} for channel {channel}")
        severity_by_channel[channel] = max(severity_by_channel.get(channel, 0.0), severity)
    max_severity = max(severity_by_channel.values())
    if max_severity <= 0.0:
        raise RuntimeError(f"{report_path}: top channel severities are all zero")
    weights = np.ones(channels, dtype=np.float32)
    for channel, severity in severity_by_channel.items():
        weights[channel] += float(scale) * float(severity / max_severity)
    summary = {
        "report": str(report_path),
        "channels": channels,
        "prioritized_channels": len(severity_by_channel),
        "scale": float(scale),
        "weight_min": float(weights.min()),
        "weight_max": float(weights.max()),
        "weight_mean": float(weights.mean()),
        "top_channels": [
            {"channel": int(channel), "severity": float(severity), "weight": float(weights[channel])}
            for channel, severity in sorted(severity_by_channel.items(), key=lambda item: (-item[1], item[0]))[:16]
        ],
    }
    return torch.as_tensor(weights, dtype=torch.float32, device=device), summary


def create_adapter_from_config(config: dict[str, Any], out_channels: int) -> LatentOutputAdapter:
    mode = str(config.get("mode") or "none")
    if mode == "none":
        raise RuntimeError("cannot create output adapter with mode=none")
    return LatentOutputAdapter(
        channels=out_channels,
        mode=mode,
        kernel_size=int(config.get("kernel_size") or 5),
        rank=int(config.get("rank") or 16),
        start_channel=int(config.get("start_channel") or 0),
        end_channel=int(config.get("end_channel") or out_channels),
    )


def output_adapter_slice_from_scope(
    *,
    model_config: dict[str, Any],
    out_channels: int,
    scope: str,
) -> tuple[int, int]:
    if scope == "all":
        return 0, int(out_channels)
    if scope not in {"factorized_envelope", "factorized_gain", "factorized_env_gain", "factorized_source"}:
        raise ValueError(f"unsupported output adapter scope: {scope!r}")
    factorized_config = factorized_base_config(model_config)
    if factorized_config is None:
        raise RuntimeError(
            f"--output-adapter-scope {scope} requires a factorized_token_context checkpoint"
        )
    envelope_channels = int(factorized_config.get("envelope_channels") or 0)
    gain_channels = int(factorized_config.get("gain_channels") or 0)
    start_channel = envelope_channels + gain_channels
    if envelope_channels <= 0 or gain_channels <= 0 or start_channel >= int(out_channels):
        raise RuntimeError(f"invalid factorized channel split in model config: {factorized_config!r}")
    if scope == "factorized_envelope":
        return 0, int(envelope_channels)
    if scope == "factorized_gain":
        return int(envelope_channels), int(start_channel)
    if scope == "factorized_env_gain":
        return 0, int(start_channel)
    return int(start_channel), int(out_channels)


def create_model_from_config(config: dict[str, Any]) -> nn.Module:
    architecture = str(config.get("architecture") or "frame_conv")
    if architecture == "calibrated":
        base_config = config.get("base_config")
        adapter_config = config.get("output_adapter")
        if not isinstance(base_config, dict):
            raise RuntimeError(f"calibrated model config missing base_config: {config!r}")
        if not isinstance(adapter_config, dict):
            raise RuntimeError(f"calibrated model config missing output_adapter: {config!r}")
        base = create_model_from_config(base_config)
        out_channels = int(base_config.get("out_channels") or 0)
        if out_channels <= 0:
            raise RuntimeError(f"calibrated base_config missing out_channels: {base_config!r}")
        return CalibratedLatentStudent(base, create_adapter_from_config(adapter_config, out_channels))
    if architecture == "source_branch_calibrated":
        base_config = config.get("base_config")
        branch_config = config.get("source_branch_config")
        if not isinstance(base_config, dict):
            raise RuntimeError(f"source-branch config missing base_config: {config!r}")
        if not isinstance(branch_config, dict):
            raise RuntimeError(f"source-branch config missing source_branch_config: {config!r}")
        base = create_model_from_config(base_config)
        source_branch = create_model_from_config(branch_config)
        start_channel = int(config.get("start_channel") or 0)
        end_channel = int(config.get("end_channel") or 0)
        return SourceBranchCalibratedLatentStudent(
            base,
            source_branch,
            start_channel=start_channel,
            end_channel=end_channel,
        )
    vocab_size = int(config.get("vocab_size") or 0)
    hidden = int(config.get("hidden") or 0)
    depth = int(config.get("depth") or 0)
    kernel_size = int(config.get("kernel_size") or 0)
    out_channels = int(config.get("out_channels") or 0)
    if vocab_size <= 0 or hidden <= 0 or depth <= 0 or kernel_size <= 0 or out_channels <= 0:
        raise RuntimeError(f"invalid model config: {config!r}")
    if architecture == "frame_conv":
        return LatentStudent(
            vocab_size=vocab_size,
            hidden=hidden,
            depth=depth,
            kernel_size=kernel_size,
            out_channels=out_channels,
        )
    if architecture == "token_context":
        token_depth = int(config.get("token_depth") or 0)
        return ContextualLatentStudent(
            vocab_size=vocab_size,
            hidden=hidden,
            depth=depth,
            token_depth=token_depth,
            kernel_size=kernel_size,
            out_channels=out_channels,
        )
    if architecture == "separable_token_context":
        token_depth = int(config.get("token_depth") or 0)
        return SeparableContextualLatentStudent(
            vocab_size=vocab_size,
            hidden=hidden,
            depth=depth,
            token_depth=token_depth,
            kernel_size=kernel_size,
            out_channels=out_channels,
        )
    if architecture == "factorized_token_context":
        token_depth = int(config.get("token_depth") or 0)
        envelope_channels = int(config.get("envelope_channels") or 16)
        gain_channels = int(config.get("gain_channels") or 4)
        return FactorizedContextualLatentStudent(
            vocab_size=vocab_size,
            hidden=hidden,
            depth=depth,
            token_depth=token_depth,
            kernel_size=kernel_size,
            out_channels=out_channels,
            envelope_channels=envelope_channels,
            gain_channels=gain_channels,
            head_depth=int(config.get("head_depth") or 0),
            head_kernel_size=int(config.get("head_kernel_size") or kernel_size),
        )
    raise RuntimeError(f"unsupported latent student architecture: {architecture}")


def predict_latent_tensor(model: nn.Module, features: ExpandedFeatures) -> torch.Tensor:
    if isinstance(model, CalibratedLatentStudent):
        return model(features)
    if isinstance(model, SourceBranchCalibratedLatentStudent):
        return model(features)
    if isinstance(model, ContextualLatentStudent):
        return model(features)
    if isinstance(model, SeparableContextualLatentStudent):
        return model(features)
    if isinstance(model, FactorizedContextualLatentStudent):
        return model(features)
    if isinstance(model, LatentStudent):
        return model(
            features.expanded_ids,
            features.frame_pos,
            features.token_pos,
            features.duration_pos,
        )
    raise RuntimeError(f"unsupported model class: {type(model).__name__}")


def add_output_adapter_to_model(
    *,
    model: nn.Module,
    model_config: dict[str, Any],
    mode: str,
    kernel_size: int,
    rank: int,
    scope: str,
    freeze_base: bool,
) -> tuple[nn.Module, dict[str, Any]]:
    if mode == "none":
        return model, model_config
    if isinstance(model, CalibratedLatentStudent) or str(model_config.get("architecture") or "") == "calibrated":
        raise RuntimeError("refusing to add a second output adapter to an already calibrated checkpoint")
    out_channels = int(model_config.get("out_channels") or 0)
    if out_channels <= 0:
        raise RuntimeError(f"cannot infer output channels from model config: {model_config!r}")
    start_channel, end_channel = output_adapter_slice_from_scope(
        model_config=model_config,
        out_channels=out_channels,
        scope=str(scope),
    )
    adapter_config = {
        "mode": str(mode),
        "scope": str(scope),
        "kernel_size": int(kernel_size),
        "rank": int(rank),
        "start_channel": int(start_channel),
        "end_channel": int(end_channel),
    }
    if freeze_base:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    calibrated = CalibratedLatentStudent(
        model,
        create_adapter_from_config(adapter_config, out_channels),
    )
    calibrated_config = {
        "architecture": "calibrated",
        "base_config": model_config,
        "output_adapter": adapter_config,
        "out_channels": out_channels,
        "vocab_size": int(model_config.get("vocab_size") or 0),
        "base_frozen": bool(freeze_base),
    }
    return calibrated, calibrated_config


def add_source_branch_to_model(
    *,
    model: nn.Module,
    model_config: dict[str, Any],
    architecture: str,
    scope: str,
    hidden: int,
    depth: int,
    token_depth: int,
    kernel_size: int,
    freeze_base: bool,
) -> tuple[nn.Module, dict[str, Any]]:
    if isinstance(model, CalibratedLatentStudent) or str(model_config.get("architecture") or "") == "calibrated":
        raise RuntimeError("add the source branch to a factorized base checkpoint, not to an output-adapted checkpoint")
    out_channels = int(model_config.get("out_channels") or 0)
    vocab_size = int(model_config.get("vocab_size") or 0)
    if out_channels <= 0 or vocab_size <= 0:
        raise RuntimeError(f"cannot infer source branch dimensions from model config: {model_config!r}")
    start_channel, end_channel = output_adapter_slice_from_scope(
        model_config=model_config,
        out_channels=out_channels,
        scope=str(scope),
    )
    existing_slices: list[tuple[int, int]] = []
    config_cursor: dict[str, Any] | None = model_config
    while isinstance(config_cursor, dict) and str(config_cursor.get("architecture") or "") == "source_branch_calibrated":
        existing_start = int(config_cursor.get("start_channel") or 0)
        existing_end = int(config_cursor.get("end_channel") or 0)
        if existing_start < existing_end:
            existing_slices.append((existing_start, existing_end))
        base_config = config_cursor.get("base_config")
        config_cursor = base_config if isinstance(base_config, dict) else None
    for existing_start, existing_end in existing_slices:
        if max(int(start_channel), existing_start) < min(int(end_channel), existing_end):
            raise RuntimeError(
                "refusing to add overlapping branch slice "
                f"{start_channel}:{end_channel}; existing slice {existing_start}:{existing_end}"
            )
    branch_channels = end_channel - start_channel
    branch_config = {
        "architecture": str(architecture),
        "vocab_size": int(vocab_size),
        "hidden": int(hidden),
        "depth": int(depth),
        "token_depth": int(token_depth),
        "kernel_size": int(kernel_size),
        "out_channels": int(branch_channels),
    }
    source_branch = create_model_from_config(branch_config)
    if freeze_base:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    wrapped = SourceBranchCalibratedLatentStudent(
        model,
        source_branch,
        start_channel=int(start_channel),
        end_channel=int(end_channel),
    )
    wrapped_config = {
        "architecture": "source_branch_calibrated",
        "base_config": model_config,
        "source_branch_config": branch_config,
        "source_branch_scope": str(scope),
        "out_channels": int(out_channels),
        "vocab_size": int(vocab_size),
        "start_channel": int(start_channel),
        "end_channel": int(end_channel),
        "base_frozen": bool(freeze_base),
    }
    return wrapped, wrapped_config


def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = load_torch_checkpoint_windows_safe(checkpoint_path, "latent student checkpoint")
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise RuntimeError(f"{checkpoint_path}: missing checkpoint config")
    model = create_model_from_config(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return model, config


def import_decoder_student_module() -> Any:
    module_path = ROOT / "tools" / "train_roota_piper_decoder_student.py"
    require_file(module_path, "decoder student module")
    module_name = "roota_decoder_student_module"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to create import spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "DecoderStudent"):
        raise RuntimeError(f"{module_path}: missing DecoderStudent")
    return module


def load_decoder_student_from_checkpoint(checkpoint_path: Path, device: torch.device) -> nn.Module:
    module = import_decoder_student_module()
    checkpoint = load_torch_checkpoint_windows_safe(checkpoint_path, "decoder-aware checkpoint")
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise RuntimeError(f"{checkpoint_path}: missing decoder checkpoint config")
    raw_channels = config.get("channels")
    if not isinstance(raw_channels, (list, tuple)) or len(raw_channels) != 4:
        raise RuntimeError(f"{checkpoint_path}: expected four decoder channels, got {raw_channels!r}")
    model = module.DecoderStudent(
        in_channels=int(config["in_channels"]),
        channels=tuple(int(value) for value in raw_channels),
        res_layers=int(config.get("res_layers", 1)),
        variant=str(config.get("variant") or "dense"),
        rank_ratio=float(config.get("rank_ratio") or 0.5),
        activation=str(config.get("activation") or "leaky_relu"),
        stage_affine=bool(config.get("stage_affine", False)),
        factorized_pre_rank=int(config.get("factorized_pre_rank", 0)),
        piper_res_factor_rank_ratio=float(config.get("piper_res_factor_rank_ratio", 0.0)),
        stage0_branches=tuple(int(value) for value in config.get("stage0_branches", [0, 1, 2])),
        stage1_branches=tuple(int(value) for value in config.get("stage1_branches", [0, 1, 2])),
        stage2_branches=tuple(int(value) for value in config.get("stage2_branches", [0, 1, 2])),
        post_filter_channels=int(config.get("post_filter_channels") or 0),
        post_filter_layers=int(config.get("post_filter_layers") or 0),
        post_filter_kernel=int(config.get("post_filter_kernel") or 9),
        post_filter_scale=float(config.get("post_filter_scale") or 0.25),
        istft_n_fft=int(config.get("istft_n_fft", 512)),
    )
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise RuntimeError(f"{checkpoint_path}: missing model_state_dict")
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def parse_decoder_feature_keys(value: str) -> list[str]:
    supported = {
        "pre",
        "up0_raw",
        "up0",
        "stage0_mix",
        "up1_raw",
        "up1",
        "stage1_mix",
        "up2_raw",
        "up2",
        "stage2_mix",
        "pre_tanh",
        "audio_pre_filter",
        "audio",
    }
    keys = [item.strip() for item in value.split(",") if item.strip()]
    if not keys:
        raise ValueError("--decoder-aware-feature-keys must contain at least one key")
    unknown = sorted(set(keys) - supported)
    if unknown:
        raise ValueError(f"unsupported --decoder-aware-feature-keys {unknown}; supported={sorted(supported)}")
    return list(dict.fromkeys(keys))


def decoder_feature_loss(
    student_features: dict[str, torch.Tensor],
    teacher_features: dict[str, torch.Tensor],
    feature_keys: list[str],
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for key in feature_keys:
        student = student_features.get(key)
        teacher = teacher_features.get(key)
        if student is None or teacher is None:
            raise RuntimeError(f"decoder feature loss missing key {key!r}")
        if student.ndim != 3 or teacher.ndim != 3:
            raise RuntimeError(f"{key}: expected 3D feature tensors, got {student.shape} and {teacher.shape}")
        if student.shape != teacher.shape:
            raise RuntimeError(f"{key}: decoder feature shape mismatch {student.shape} != {teacher.shape}")
        losses.append(F.l1_loss(student, teacher.detach()))
    if not losses:
        return next(iter(student_features.values())).new_tensor(0.0)
    total = losses[0]
    for loss in losses[1:]:
        total = total + loss
    return total / float(len(losses))


def check_model_compatible(
    *,
    config: dict[str, Any],
    required_vocab_size: int,
    out_channels: int,
) -> None:
    model_vocab_size = int(config.get("vocab_size") or 0)
    model_out_channels = int(config.get("out_channels") or 0)
    if model_vocab_size < required_vocab_size:
        raise RuntimeError(
            f"checkpoint vocab_size {model_vocab_size} cannot cover required vocab_size {required_vocab_size}"
        )
    if model_out_channels != out_channels:
        raise RuntimeError(f"checkpoint out_channels {model_out_channels} != required {out_channels}")


def decoder_target_cache_key(sample: ChunkSample, decoder_path: Path) -> str:
    require_file(sample.tensor_path, "tensor NPZ")
    require_file(decoder_path, "decoder ONNX")
    tensor_stat = sample.tensor_path.stat()
    decoder_stat = decoder_path.stat()
    target = np.ascontiguousarray(sample.target.astype(np.float32, copy=False))
    payload = {
        "row_id": sample.row_id,
        "chunk_index": int(sample.chunk_index),
        "tensor_path": str(sample.tensor_path.resolve()),
        "tensor_size": int(tensor_stat.st_size),
        "tensor_mtime_ns": int(tensor_stat.st_mtime_ns),
        "decoder_path": str(decoder_path.resolve()),
        "decoder_size": int(decoder_stat.st_size),
        "decoder_mtime_ns": int(decoder_stat.st_mtime_ns),
        "frames": int(sample.target.shape[0]),
        "channels": int(sample.target.shape[1]),
        "target_sha256": hashlib.sha256(target.tobytes()).hexdigest(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def load_cached_decoder_target(path: Path, expected_samples: int) -> np.ndarray | None:
    if not path.is_file():
        return None
    audio = np.asarray(np.load(path), dtype=np.float32).reshape(-1)
    if int(audio.size) != int(expected_samples):
        raise RuntimeError(f"{path}: cached target samples {audio.size} != expected {expected_samples}")
    if not np.isfinite(audio).all():
        raise RuntimeError(f"{path}: cached target contains non-finite values")
    return audio


def attach_decoder_targets(
    samples: list[ChunkSample],
    decoder_path: Path,
    cache_dir: Path | None = None,
) -> list[ChunkSample]:
    require_file(decoder_path, "decoder ONNX")
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    session = ort.InferenceSession(str(decoder_path), providers=["CPUExecutionProvider"])
    updated: list[ChunkSample] = []
    cache_hits = 0
    cache_misses = 0
    for index, sample in enumerate(samples, start=1):
        expected_samples = int(sample.target.shape[0]) * 256
        cache_path = None
        audio = None
        if cache_dir is not None:
            key = decoder_target_cache_key(sample, decoder_path)
            cache_path = cache_dir / f"{sample.row_id}_chunk{sample.chunk_index:02d}-{key}.npy"
            audio = load_cached_decoder_target(cache_path, expected_samples)
            if audio is not None:
                cache_hits += 1
        if audio is None:
            latent = sample.target.transpose(1, 0)[None, :, :].astype(np.float32, copy=False)
            audio = decode_latent(session, latent)
            if int(audio.size) != expected_samples:
                raise RuntimeError(
                    f"{sample.row_id} chunk {sample.chunk_index}: decoder target samples "
                    f"{audio.size} != expected {expected_samples}"
                )
            if cache_path is not None:
                np.save(cache_path, audio.astype(np.float32, copy=False))
            cache_misses += 1
        updated.append(replace(sample, decoder_target_audio=audio.astype(np.float32, copy=False)))
        if index == 1 or index % 100 == 0 or index == len(samples):
            print(
                "prepared decoder-aware target audio "
                f"{index}/{len(samples)} cache_hits={cache_hits} cache_misses={cache_misses}",
                flush=True,
            )
    return updated


def multi_resolution_stft_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise RuntimeError(f"STFT loss shape mismatch: {prediction.shape} != {target.shape}")
    if prediction.ndim != 3 or prediction.shape[1] != 1:
        raise RuntimeError(f"expected audio [batch,1,samples], got {prediction.shape}")
    pred = prediction.squeeze(1)
    true = target.squeeze(1)
    configs = ((512, 128), (1024, 256), (2048, 512))
    usable_configs = [item for item in configs if item[0] <= int(prediction.shape[-1])]
    if not usable_configs:
        raise RuntimeError(f"audio crop too short for STFT loss: {prediction.shape[-1]} samples")
    loss = prediction.new_tensor(0.0)
    for n_fft, hop_length in usable_configs:
        window = torch.hann_window(n_fft, device=prediction.device)
        pred_spec = torch.stft(
            pred,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=window,
            return_complex=True,
        )
        true_spec = torch.stft(
            true,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=window,
            return_complex=True,
        )
        loss = loss + F.l1_loss(torch.log1p(torch.abs(pred_spec)), torch.log1p(torch.abs(true_spec)))
    return loss / float(len(usable_configs))


def decoder_aware_loss_components(
    *,
    decoder_student: nn.Module,
    prediction: torch.Tensor,
    sample: ChunkSample,
    crop_frames: int,
    l1_weight: float,
    stft_weight: float,
    feature_weight: float,
    feature_keys: list[str],
) -> dict[str, torch.Tensor]:
    if sample.decoder_target_audio is None:
        raise RuntimeError(f"{sample.row_id} chunk {sample.chunk_index}: missing decoder-aware target audio")
    frames = int(prediction.shape[0])
    if frames <= 0:
        raise RuntimeError(f"{sample.row_id} chunk {sample.chunk_index}: empty prediction")
    crop = min(max(int(crop_frames), 1), frames)
    start_frame = 0 if frames == crop else random.randint(0, frames - crop)
    end_frame = start_frame + crop
    pred_crop = prediction[start_frame:end_frame].transpose(0, 1).unsqueeze(0)
    target_audio_np = sample.decoder_target_audio[start_frame * 256 : end_frame * 256]
    expected_audio = crop * 256
    if int(target_audio_np.size) != expected_audio:
        raise RuntimeError(
            f"{sample.row_id} chunk {sample.chunk_index}: target crop samples "
            f"{target_audio_np.size} != expected {expected_audio}"
        )
    target_audio = torch.as_tensor(
        target_audio_np,
        dtype=prediction.dtype,
        device=prediction.device,
    ).reshape(1, 1, -1)
    needs_features = feature_weight > 0.0
    if needs_features:
        decoded_value = decoder_student(pred_crop, return_features=True)
        if not isinstance(decoded_value, tuple) or len(decoded_value) != 2:
            raise RuntimeError("decoder student did not return (audio, features)")
        decoded_audio, student_features = decoded_value
        teacher_crop_np = sample.target[start_frame:end_frame].transpose(1, 0)[None, :, :]
        teacher_crop = torch.as_tensor(teacher_crop_np, dtype=prediction.dtype, device=prediction.device)
        with torch.no_grad():
            teacher_value = decoder_student(teacher_crop, return_features=True)
        if not isinstance(teacher_value, tuple) or len(teacher_value) != 2:
            raise RuntimeError("decoder student did not return teacher (audio, features)")
        _teacher_audio, teacher_features = teacher_value
        feature = decoder_feature_loss(student_features, teacher_features, feature_keys)
    else:
        decoded_audio = decoder_student(pred_crop)
        feature = prediction.new_tensor(0.0)
    if not isinstance(decoded_audio, torch.Tensor):
        raise RuntimeError("decoder student returned non-tensor output")
    if decoded_audio.shape != target_audio.shape:
        raise RuntimeError(f"decoder audio shape {decoded_audio.shape} != target {target_audio.shape}")
    l1 = F.l1_loss(decoded_audio, target_audio) if l1_weight > 0.0 else prediction.new_tensor(0.0)
    stft = multi_resolution_stft_loss(decoded_audio, target_audio) if stft_weight > 0.0 else prediction.new_tensor(0.0)
    return {
        "decoder_l1": l1,
        "decoder_stft": stft,
        "decoder_feature": feature,
        "decoder_total": float(l1_weight) * l1 + float(stft_weight) * stft + float(feature_weight) * feature,
    }


def source_filter_aware_enabled(args: argparse.Namespace) -> bool:
    return (
        float(args.source_filter_aware_spectral_weight) > 0.0
        or float(args.source_filter_aware_rms_weight) > 0.0
        or float(args.source_filter_aware_waveform_weight) > 0.0
    )


def load_source_filter_aware_target(
    sample: ChunkSample,
    *,
    target_audio_key: str,
    subframes_per_frame: int,
    gain_floor: float,
    gain_ceil: float,
    device: torch.device,
    dtype: torch.dtype,
    cache: dict[tuple[str, str, str, str], SourceFilterAwareTarget],
) -> SourceFilterAwareTarget:
    if sample.target_npz_path is None:
        raise RuntimeError(
            f"{sample.row_id} chunk {sample.chunk_index}: source-filter-aware loss requires target NPZ path"
        )
    key = (str(sample.target_npz_path), str(target_audio_key), str(device), str(dtype))
    cached = cache.get(key)
    if cached is not None:
        return cached
    with np.load(sample.target_npz_path) as target_npz:
        required = {"lpc_residual", "subframe_gain", target_audio_key}
        missing = required - set(target_npz.files)
        if missing:
            raise RuntimeError(f"{sample.target_npz_path}: missing source-filter-aware arrays {sorted(missing)}")
        residual = np.asarray(target_npz["lpc_residual"], dtype=np.float32).reshape(-1)
        subframe_gain = np.asarray(target_npz["subframe_gain"], dtype=np.float32)
        target_audio = np.asarray(target_npz[target_audio_key], dtype=np.float32).reshape(-1)
    frame_count = int(sample.target.shape[0])
    expected_samples = frame_count * SOURCE_FILTER_HOP_LENGTH
    if residual.shape != (expected_samples,):
        raise RuntimeError(
            f"{sample.target_npz_path}: lpc_residual samples {residual.shape} != ({expected_samples},)"
        )
    if target_audio.shape != (expected_samples,):
        raise RuntimeError(
            f"{sample.target_npz_path}: {target_audio_key} samples {target_audio.shape} != ({expected_samples},)"
        )
    if subframe_gain.shape != (frame_count, int(subframes_per_frame)):
        raise RuntimeError(
            f"{sample.target_npz_path}: subframe_gain {subframe_gain.shape} != "
            f"({frame_count}, {subframes_per_frame})"
        )
    if SOURCE_FILTER_HOP_LENGTH % int(subframes_per_frame) != 0:
        raise RuntimeError(
            f"hop {SOURCE_FILTER_HOP_LENGTH} is not divisible by subframes_per_frame={subframes_per_frame}"
        )
    subframe_size = SOURCE_FILTER_HOP_LENGTH // int(subframes_per_frame)
    clipped_gain = np.clip(subframe_gain, float(gain_floor), float(gain_ceil)).astype(np.float32)
    gain_signal = np.repeat(clipped_gain.reshape(-1), subframe_size).astype(np.float32)
    if gain_signal.shape != residual.shape:
        raise RuntimeError(f"{sample.target_npz_path}: gain signal shape {gain_signal.shape} != residual {residual.shape}")
    normalized_source = residual / np.maximum(gain_signal, float(gain_floor))
    for label, value in (
        ("normalized_source", normalized_source),
        ("target_audio", target_audio),
    ):
        if not np.isfinite(value).all():
            raise RuntimeError(f"{sample.target_npz_path}: non-finite {label}")
    loaded = SourceFilterAwareTarget(
        normalized_source=torch.as_tensor(normalized_source, dtype=dtype, device=device),
        target_audio=torch.as_tensor(target_audio, dtype=dtype, device=device),
        frame_count=frame_count,
    )
    cache[key] = loaded
    return loaded


def reflection_logits_to_lpc_torch(logits: torch.Tensor, *, reflection_radius: float) -> torch.Tensor:
    if logits.ndim != 2 or int(logits.shape[1]) <= 0:
        raise RuntimeError(f"expected reflection logits [frames, order], got {logits.shape}")
    reflection = torch.tanh(logits) * float(reflection_radius)
    current: torch.Tensor | None = None
    for index in range(int(reflection.shape[1])):
        k_value = reflection[:, index : index + 1]
        if current is None:
            current = k_value
        else:
            current = torch.cat([current + k_value * torch.flip(current, dims=[1]), k_value], dim=1)
    if current is None:
        raise RuntimeError("empty reflection coefficient sequence")
    return current


def lpc_response_magnitude(
    lpc_coeffs: torch.Tensor,
    *,
    n_fft: int,
    preemphasis: float,
) -> torch.Tensor:
    if lpc_coeffs.ndim != 2:
        raise RuntimeError(f"expected LPC coeffs [frames, order], got {lpc_coeffs.shape}")
    bins = int(n_fft) // 2 + 1
    freqs = torch.linspace(0.0, math.pi, bins, dtype=lpc_coeffs.dtype, device=lpc_coeffs.device)
    order = torch.arange(1, int(lpc_coeffs.shape[1]) + 1, dtype=lpc_coeffs.dtype, device=lpc_coeffs.device)
    angles = freqs[:, None] * order[None, :]
    cosines = torch.cos(angles)
    sines = torch.sin(angles)
    denom_real = 1.0 + lpc_coeffs @ cosines.transpose(0, 1)
    denom_imag = -(lpc_coeffs @ sines.transpose(0, 1))
    response = torch.rsqrt(torch.clamp(denom_real.square() + denom_imag.square(), min=1e-12))
    if float(preemphasis) != 0.0:
        coeff = float(preemphasis)
        deemph_real = 1.0 - coeff * torch.cos(freqs)
        deemph_imag = coeff * torch.sin(freqs)
        deemph = torch.rsqrt(torch.clamp(deemph_real.square() + deemph_imag.square(), min=1e-12))
        response = response * deemph.reshape(1, -1)
    return response


def source_filter_aware_loss_components(
    *,
    prediction: torch.Tensor,
    sample: ChunkSample,
    target_audio_key: str,
    subframes_per_frame: int,
    gain_floor: float,
    gain_ceil: float,
    reflection_radius: float,
    preemphasis: float,
    crop_frames: int,
    n_fft: int,
    spectral_weight: float,
    rms_weight: float,
    waveform_weight: float,
    target_cache: dict[tuple[str, str, str, str], SourceFilterAwareTarget],
) -> dict[str, torch.Tensor]:
    if prediction.ndim != 2 or int(prediction.shape[1]) < 16 + int(subframes_per_frame):
        raise RuntimeError(f"source-filter-aware prediction shape invalid: {prediction.shape}")
    if int(n_fft) < SOURCE_FILTER_HOP_LENGTH:
        raise RuntimeError(f"--source-filter-aware-n-fft must be >= {SOURCE_FILTER_HOP_LENGTH}, got {n_fft}")
    target = load_source_filter_aware_target(
        sample,
        target_audio_key=target_audio_key,
        subframes_per_frame=int(subframes_per_frame),
        gain_floor=float(gain_floor),
        gain_ceil=float(gain_ceil),
        device=prediction.device,
        dtype=prediction.dtype,
        cache=target_cache,
    )
    frames = int(prediction.shape[0])
    if frames != int(target.frame_count):
        raise RuntimeError(f"{sample.row_id} chunk {sample.chunk_index}: prediction frames {frames} != target {target.frame_count}")
    crop = min(max(int(crop_frames), 1), frames)
    start_frame = 0 if crop == frames else random.randint(0, frames - crop)
    end_frame = start_frame + crop
    start_sample = start_frame * SOURCE_FILTER_HOP_LENGTH
    end_sample = end_frame * SOURCE_FILTER_HOP_LENGTH

    pred_crop = prediction[start_frame:end_frame]
    lpc_coeffs = reflection_logits_to_lpc_torch(pred_crop[:, :16], reflection_radius=float(reflection_radius))
    log_gain = pred_crop[:, 16 : 16 + int(subframes_per_frame)]
    clipped_log_gain = torch.clamp(log_gain, min=math.log(float(gain_floor)), max=math.log(float(gain_ceil)))
    gains = torch.exp(clipped_log_gain)
    subframe_size = SOURCE_FILTER_HOP_LENGTH // int(subframes_per_frame)
    gain_signal = torch.repeat_interleave(gains.reshape(-1), subframe_size)
    normalized = target.normalized_source[start_sample:end_sample]
    if int(normalized.numel()) != int(gain_signal.numel()):
        raise RuntimeError(
            f"{sample.row_id} chunk {sample.chunk_index}: normalized source {normalized.numel()} != gain {gain_signal.numel()}"
        )
    residual = normalized * gain_signal
    residual_frames = residual.reshape(crop, SOURCE_FILTER_HOP_LENGTH)
    target_frames = target.target_audio[start_sample:end_sample].reshape(crop, SOURCE_FILTER_HOP_LENGTH)
    waveform = prediction.new_tensor(0.0)
    if float(waveform_weight) > 0.0:
        try:
            import torchaudio.functional as torchaudio_functional
        except Exception as exc:
            raise RuntimeError("source-filter-aware waveform loss requires torchaudio") from exc
        filter_dtype = torch.float64
        denominator = torch.cat(
            [
                torch.ones(crop, 1, dtype=filter_dtype, device=prediction.device),
                lpc_coeffs.to(dtype=filter_dtype),
            ],
            dim=1,
        )
        numerator = torch.zeros_like(denominator)
        numerator[:, 0] = 1.0
        emphasized = torchaudio_functional.lfilter(
            residual_frames.to(dtype=filter_dtype),
            denominator,
            numerator,
            clamp=False,
            batching=True,
        )
        if float(preemphasis) != 0.0:
            deemph_denominator = torch.tensor(
                [[1.0, -float(preemphasis)]],
                dtype=filter_dtype,
                device=prediction.device,
            ).repeat(crop, 1)
            deemph_numerator = torch.tensor(
                [[1.0, 0.0]],
                dtype=filter_dtype,
                device=prediction.device,
            ).repeat(crop, 1)
            pred_audio = torchaudio_functional.lfilter(
                emphasized,
                deemph_denominator,
                deemph_numerator,
                clamp=False,
                batching=True,
            )
        else:
            pred_audio = emphasized
        waveform = F.l1_loss(pred_audio, target_frames.to(dtype=filter_dtype)).to(dtype=prediction.dtype)
    window = torch.hann_window(SOURCE_FILTER_HOP_LENGTH, dtype=prediction.dtype, device=prediction.device)
    residual_spec = torch.fft.rfft(residual_frames * window.reshape(1, -1), n=int(n_fft), dim=-1)
    target_spec = torch.fft.rfft(target_frames * window.reshape(1, -1), n=int(n_fft), dim=-1)
    residual_mag = torch.abs(residual_spec)
    target_mag = torch.abs(target_spec)
    response_mag = lpc_response_magnitude(lpc_coeffs, n_fft=int(n_fft), preemphasis=float(preemphasis))
    pred_mag = residual_mag * response_mag
    spectral = (
        F.l1_loss(torch.log1p(pred_mag), torch.log1p(target_mag))
        if float(spectral_weight) > 0.0
        else prediction.new_tensor(0.0)
    )
    pred_rms = torch.sqrt(torch.mean(pred_mag.square(), dim=1) + 1e-12)
    target_rms = torch.sqrt(torch.mean(target_mag.square(), dim=1) + 1e-12)
    rms = F.l1_loss(torch.log1p(pred_rms), torch.log1p(target_rms)) if float(rms_weight) > 0.0 else prediction.new_tensor(0.0)
    return {
        "source_filter_spectral": spectral,
        "source_filter_rms": rms,
        "source_filter_waveform": waveform,
        "source_filter_total": (
            float(spectral_weight) * spectral
            + float(rms_weight) * rms
            + float(waveform_weight) * waveform
        ),
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    if args.eval_only and args.load_checkpoint is None:
        raise ValueError("--eval-only requires --load-checkpoint")
    for name in (
        "mse_weight",
        "norm_l1_weight",
        "delta_l1_weight",
        "channel_stat_weight",
        "channel_priority_weight",
        "channel_priority_delta_weight",
        "channel_priority_scale",
        "decoder_aware_l1_weight",
        "decoder_aware_stft_weight",
        "decoder_aware_feature_weight",
        "source_filter_aware_spectral_weight",
        "source_filter_aware_rms_weight",
        "source_filter_aware_waveform_weight",
        "envelope_norm_weight",
        "gain_norm_weight",
        "source_norm_weight",
        "source_delta_weight",
        "source_smooth_weight",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be a finite non-negative value, got {value!r}")
    if int(args.decoder_aware_crop_frames) <= 0:
        raise ValueError(f"--decoder-aware-crop-frames must be positive, got {args.decoder_aware_crop_frames!r}")
    if int(args.source_filter_aware_crop_frames) <= 0:
        raise ValueError(
            f"--source-filter-aware-crop-frames must be positive, got {args.source_filter_aware_crop_frames!r}"
        )
    if int(args.source_filter_aware_n_fft) < SOURCE_FILTER_HOP_LENGTH:
        raise ValueError(
            f"--source-filter-aware-n-fft must be >= {SOURCE_FILTER_HOP_LENGTH}, "
            f"got {args.source_filter_aware_n_fft!r}"
        )
    channel_priority_enabled = (
        float(args.channel_priority_weight) > 0.0
        or float(args.channel_priority_delta_weight) > 0.0
    )
    if channel_priority_enabled and args.channel_priority_report is None:
        raise ValueError("--channel-priority-report is required when channel priority loss weights are non-zero")
    if str(args.output_adapter) != "none" and args.load_checkpoint is None:
        raise ValueError("--output-adapter requires --load-checkpoint so the adapter has a stable base model")
    if args.freeze_loaded_base and args.load_checkpoint is None:
        raise ValueError("--freeze-loaded-base requires --load-checkpoint")
    if (
        args.freeze_factorized_trunk
        or args.freeze_factorized_envelope_head
        or args.freeze_factorized_gain_head
        or args.freeze_factorized_source_head
    ) and str(args.architecture) != "factorized_token_context":
        if args.load_checkpoint is None:
            raise ValueError("factorized freeze flags require --architecture factorized_token_context or a factorized checkpoint")
    if int(args.output_adapter_kernel_size) <= 0 or int(args.output_adapter_kernel_size) % 2 == 0:
        raise ValueError(
            "--output-adapter-kernel-size must be a positive odd integer, "
            f"got {args.output_adapter_kernel_size!r}"
        )
    if int(args.output_adapter_rank) <= 0:
        raise ValueError(f"--output-adapter-rank must be positive, got {args.output_adapter_rank!r}")
    if int(args.factorized_envelope_channels) <= 0:
        raise ValueError(
            f"--factorized-envelope-channels must be positive, got {args.factorized_envelope_channels!r}"
        )
    if int(args.factorized_gain_channels) <= 0:
        raise ValueError(f"--factorized-gain-channels must be positive, got {args.factorized_gain_channels!r}")
    if int(args.factorized_head_depth) < 0:
        raise ValueError(f"--factorized-head-depth must be non-negative, got {args.factorized_head_depth!r}")
    if int(args.factorized_head_kernel_size) <= 0 or int(args.factorized_head_kernel_size) % 2 == 0:
        raise ValueError(
            "--factorized-head-kernel-size must be a positive odd integer, "
            f"got {args.factorized_head_kernel_size!r}"
        )
    if not math.isfinite(float(args.output_adapter_lr_multiplier)) or float(args.output_adapter_lr_multiplier) <= 0.0:
        raise ValueError(
            "--output-adapter-lr-multiplier must be finite and positive, "
            f"got {args.output_adapter_lr_multiplier!r}"
        )
    decoder_aware_enabled = (
        float(args.decoder_aware_l1_weight) > 0.0
        or float(args.decoder_aware_stft_weight) > 0.0
        or float(args.decoder_aware_feature_weight) > 0.0
    )
    source_filter_enabled = source_filter_aware_enabled(args)
    source_branch_enabled = int(args.source_branch_hidden) > 0
    if source_branch_enabled:
        if args.load_checkpoint is None:
            raise ValueError("--source-branch-hidden requires --load-checkpoint")
        if str(args.output_adapter) != "none":
            raise ValueError("--source-branch-hidden cannot be combined with --output-adapter")
        for name in ("source_branch_depth", "source_branch_token_depth", "source_branch_kernel_size"):
            value = int(getattr(args, name))
            if value <= 0:
                raise ValueError(f"--{name.replace('_', '-')} must be positive, got {value!r}")
        if int(args.source_branch_kernel_size) % 2 == 0:
            raise ValueError(
                "--source-branch-kernel-size must be odd, "
                f"got {args.source_branch_kernel_size!r}"
            )
    decoder_aware_feature_keys = parse_decoder_feature_keys(str(args.decoder_aware_feature_keys))
    if decoder_aware_enabled and args.decoder_aware_checkpoint is None:
        raise ValueError("--decoder-aware-checkpoint is required when decoder-aware loss weights are non-zero")
    target_index = load_target_index(args.target_dir) if args.target_dir is not None else None
    target_summary = read_json(args.target_dir / "summary.json") if args.target_dir is not None else None
    eval_target_index = None
    if args.eval_target_dir is not None:
        eval_target_index = load_target_index(args.eval_target_dir)
    if args.target_dir is not None and args.eval_pack_dir is not None and args.eval_target_dir is None:
        raise ValueError("--eval-target-dir is required when --target-dir and --eval-pack-dir are both set")
    if args.target_dir is None and args.eval_target_dir is not None:
        raise ValueError("--eval-target-dir was provided without --target-dir")
    if target_index is not None and decoder_aware_enabled and str(args.target_key) != "generator_input":
        raise ValueError("decoder-aware waveform losses are only valid for generator_input targets")
    if source_filter_enabled:
        if target_index is None or target_summary is None:
            raise ValueError("--target-dir is required when source-filter-aware loss weights are non-zero")
        if str(target_summary.get("lpc_param") or "") != "reflection_logit":
            raise ValueError(
                "source-filter-aware loss currently requires reflection_logit LPC targets, "
                f"got {target_summary.get('lpc_param')!r}"
            )
        if int(target_summary.get("lpc_order") or 0) != 16:
            raise ValueError(
                f"source-filter-aware loss expects lpc_order=16, got {target_summary.get('lpc_order')!r}"
            )
        summary_subframes = int(target_summary.get("subframes_per_frame") or 0)
        if summary_subframes <= 0:
            raise ValueError(
                "source-filter-aware loss requires a positive subframes_per_frame, "
                f"got {target_summary.get('subframes_per_frame')!r}"
            )
        if SOURCE_FILTER_HOP_LENGTH % summary_subframes != 0:
            raise ValueError(
                "source-filter-aware loss requires hop divisible by subframes_per_frame, "
                f"got {target_summary.get('subframes_per_frame')!r}"
            )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rows, samples, train_vocab_size, out_channels = load_samples(
        args.pack_dir,
        target_index=target_index,
        target_key=str(args.target_key),
        target_duration_key=args.target_duration_key,
    )
    eval_rows: list[dict[str, Any]] | None = None
    eval_samples: list[ChunkSample] | None = None
    eval_vocab_size = 0
    if args.eval_pack_dir is not None:
        eval_rows, eval_samples, eval_vocab_size, eval_out_channels = load_samples(
            args.eval_pack_dir,
            target_index=eval_target_index,
            target_key=str(args.target_key),
            target_duration_key=args.target_duration_key,
        )
        if eval_out_channels != out_channels:
            raise RuntimeError(f"eval out_channels {eval_out_channels} != train out_channels {out_channels}")
    vocab_size = max(train_vocab_size, eval_vocab_size)
    device = pick_device(args.device)
    source_filter_config: dict[str, Any] | None = None
    if source_filter_enabled:
        if target_summary is None:
            raise RuntimeError("internal error: missing target summary for source-filter-aware loss")
        source_filter_subframes = int(target_summary["subframes_per_frame"])
        source_filter_rank = int(target_summary["phase_template_rank"])
        expected_channels = 16 + source_filter_subframes + source_filter_rank
        if int(out_channels) != expected_channels:
            raise RuntimeError(
                f"source-filter-aware expected {expected_channels} channels from summary, got {out_channels}"
            )
        source_filter_config = {
            "enabled": True,
            "target_audio_key": str(args.source_filter_aware_target_audio_key),
            "subframes_per_frame": source_filter_subframes,
            "phase_template_rank": source_filter_rank,
            "gain_floor": float(target_summary["residual_gain_floor"]),
            "gain_ceil": float(target_summary["residual_gain_ceil"]),
            "reflection_radius": float(target_summary["reflection_radius"]),
            "preemphasis": float(target_summary["preemphasis"]),
            "crop_frames": int(args.source_filter_aware_crop_frames),
            "n_fft": int(args.source_filter_aware_n_fft),
        }
    if decoder_aware_enabled:
        samples = attach_decoder_targets(samples, args.decoder, args.decoder_aware_target_cache_dir)
    latent_stats = compute_latent_channel_stats(samples, device)
    channel_priority, channel_priority_summary = load_channel_priority(
        args.channel_priority_report,
        channels=int(out_channels),
        scale=float(args.channel_priority_scale),
        device=device,
    )
    train_sample_weights = sample_weights_for(samples, str(args.sample_weight_mode))
    sample_weight_summary = summarize_sample_weights(samples, train_sample_weights)
    decoder_student: nn.Module | None = None
    if decoder_aware_enabled:
        decoder_student = load_decoder_student_from_checkpoint(args.decoder_aware_checkpoint, device)
    source_filter_target_cache: dict[tuple[str, str, str, str], SourceFilterAwareTarget] = {}

    if args.load_checkpoint is not None:
        model, model_config = load_model_from_checkpoint(args.load_checkpoint, device)
        check_model_compatible(config=model_config, required_vocab_size=vocab_size, out_channels=out_channels)
        if source_branch_enabled:
            model, model_config = add_source_branch_to_model(
                model=model,
                model_config=model_config,
                architecture=str(args.source_branch_architecture),
                scope=str(args.source_branch_scope),
                hidden=int(args.source_branch_hidden),
                depth=int(args.source_branch_depth),
                token_depth=int(args.source_branch_token_depth),
                kernel_size=int(args.source_branch_kernel_size),
                freeze_base=bool(args.freeze_loaded_base),
            )
        elif str(args.output_adapter) == "none":
            if bool(args.freeze_loaded_base):
                if isinstance(model, CalibratedLatentStudent):
                    for parameter in model.base.parameters():
                        parameter.requires_grad_(False)
                    model_config["base_frozen"] = True
                elif isinstance(model, SourceBranchCalibratedLatentStudent):
                    for parameter in model.base.parameters():
                        parameter.requires_grad_(False)
                    model_config["base_frozen"] = True
                else:
                    raise RuntimeError(
                        "--freeze-loaded-base with --output-adapter none requires a calibrated or "
                        "source_branch_calibrated checkpoint with an existing adapter/branch"
                    )
        else:
            model, model_config = add_output_adapter_to_model(
                model=model,
                model_config=model_config,
                mode=str(args.output_adapter),
                kernel_size=int(args.output_adapter_kernel_size),
                rank=int(args.output_adapter_rank),
                scope=str(args.output_adapter_scope),
                freeze_base=bool(args.freeze_loaded_base),
            )
        model.to(device)
    else:
        model_config = {
            "architecture": str(args.architecture),
            "vocab_size": vocab_size,
            "hidden": args.hidden,
            "depth": args.depth,
            "kernel_size": args.kernel_size,
            "out_channels": out_channels,
        }
        if args.architecture == "token_context":
            model_config["token_depth"] = int(args.token_depth)
        if args.architecture == "factorized_token_context":
            model_config["token_depth"] = int(args.token_depth)
            model_config["envelope_channels"] = int(args.factorized_envelope_channels)
            model_config["gain_channels"] = int(args.factorized_gain_channels)
            model_config["head_depth"] = int(args.factorized_head_depth)
            model_config["head_kernel_size"] = int(args.factorized_head_kernel_size)
            factorized_source_channels = (
                int(out_channels)
                - int(args.factorized_envelope_channels)
                - int(args.factorized_gain_channels)
            )
            if factorized_source_channels <= 0:
                raise RuntimeError(
                    "factorized source channels must be positive, "
                    f"got out_channels={out_channels}, "
                    f"envelope={args.factorized_envelope_channels}, "
                    f"gain={args.factorized_gain_channels}"
                )
            model_config["source_channels"] = int(factorized_source_channels)
        model = create_model_from_config(model_config).to(device)
    parameter_table = latent_parameter_table(model)
    print(json.dumps({"model_parameters": parameter_table}, ensure_ascii=False), flush=True)
    factorized_freezing = apply_factorized_freezing(
        model,
        freeze_trunk=bool(args.freeze_factorized_trunk),
        freeze_envelope_head=bool(args.freeze_factorized_envelope_head),
        freeze_gain_head=bool(args.freeze_factorized_gain_head),
        freeze_source_head=bool(args.freeze_factorized_source_head),
    )
    if any(bool(value) for value in factorized_freezing.values()):
        model_config["factorized_freezing"] = factorized_freezing
    logs: list[dict[str, Any]] = []
    optimizer_groups_report: list[dict[str, Any]] = []

    if not args.eval_only:
        optimizer_groups = trainable_optimizer_groups(
            model,
            lr=float(args.lr),
            adapter_lr_multiplier=float(args.output_adapter_lr_multiplier),
        )
        trainable_parameters = [
            parameter
            for group in optimizer_groups
            for parameter in group["params"]
        ]
        if not optimizer_groups or not trainable_parameters:
            raise RuntimeError("model has no trainable parameters")
        optimizer_groups_report = [
            {
                "name": str(group.get("name") or f"group_{index}"),
                "lr": float(group["lr"]),
                "parameters": int(sum(parameter.numel() for parameter in group["params"])),
            }
            for index, group in enumerate(optimizer_groups)
        ]
        optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=1e-5)
        # Latent adversarial de-smoothing (off unless --latent-adv-weight > 0).
        # Created lazily on first activation so the channel count comes from the
        # actual target latent; discarded after training (never checkpointed).
        latent_disc: LatentDiscriminator | None = None
        disc_optimizer: torch.optim.Optimizer | None = None
        for step in range(1, args.steps + 1):
            sample = random.choices(samples, weights=train_sample_weights, k=1)[0]
            features = expand_features(sample, device)
            target = target_tensor(sample, device)
            prediction = predict_latent_tensor(model, features)
            if prediction.shape != target.shape:
                raise RuntimeError(f"prediction shape {prediction.shape} != target shape {target.shape}")
            components = latent_loss_components(prediction, target, latent_stats, channel_priority)
            loss = weighted_latent_loss(
                components,
                mse_weight=float(args.mse_weight),
                norm_l1_weight=float(args.norm_l1_weight),
                delta_l1_weight=float(args.delta_l1_weight),
                channel_stat_weight=float(args.channel_stat_weight),
                channel_priority_weight=float(args.channel_priority_weight),
                channel_priority_delta_weight=float(args.channel_priority_delta_weight),
            )
            factorized_components = factorized_loss_components(prediction, target, latent_stats, model_config)
            factorized_total = weighted_factorized_loss(
                factorized_components,
                envelope_norm_weight=float(args.envelope_norm_weight),
                gain_norm_weight=float(args.gain_norm_weight),
                source_norm_weight=float(args.source_norm_weight),
                source_delta_weight=float(args.source_delta_weight),
                source_smooth_weight=float(args.source_smooth_weight),
            )
            loss = loss + factorized_total
            decoder_components: dict[str, torch.Tensor] | None = None
            source_filter_components: dict[str, torch.Tensor] | None = None
            if decoder_aware_enabled:
                if decoder_student is None:
                    raise RuntimeError("decoder-aware loss enabled without decoder student")
                decoder_components = decoder_aware_loss_components(
                    decoder_student=decoder_student,
                    prediction=prediction,
                    sample=sample,
                    crop_frames=int(args.decoder_aware_crop_frames),
                    l1_weight=float(args.decoder_aware_l1_weight),
                    stft_weight=float(args.decoder_aware_stft_weight),
                    feature_weight=float(args.decoder_aware_feature_weight),
                    feature_keys=decoder_aware_feature_keys,
                )
                loss = loss + decoder_components["decoder_total"]
            if source_filter_enabled:
                if source_filter_config is None:
                    raise RuntimeError("source-filter-aware loss enabled without source_filter_config")
                source_filter_components = source_filter_aware_loss_components(
                    prediction=prediction,
                    sample=sample,
                    target_audio_key=str(source_filter_config["target_audio_key"]),
                    subframes_per_frame=int(source_filter_config["subframes_per_frame"]),
                    gain_floor=float(source_filter_config["gain_floor"]),
                    gain_ceil=float(source_filter_config["gain_ceil"]),
                    reflection_radius=float(source_filter_config["reflection_radius"]),
                    preemphasis=float(source_filter_config["preemphasis"]),
                    crop_frames=int(source_filter_config["crop_frames"]),
                    n_fft=int(source_filter_config["n_fft"]),
                    spectral_weight=float(args.source_filter_aware_spectral_weight),
                    rms_weight=float(args.source_filter_aware_rms_weight),
                    waveform_weight=float(args.source_filter_aware_waveform_weight),
                    target_cache=source_filter_target_cache,
                )
                loss = loss + source_filter_components["source_filter_total"]
            latent_adv_g: torch.Tensor | None = None
            latent_disc_loss: torch.Tensor | None = None
            if float(args.latent_adv_weight) > 0.0 and step >= int(args.latent_adv_start_step):
                if latent_disc is None:
                    latent_disc = LatentDiscriminator(
                        channels=int(target.shape[1]),
                        hidden=int(args.latent_disc_hidden),
                        layers=int(args.latent_disc_layers),
                    ).to(device)
                    disc_optimizer = torch.optim.AdamW(
                        latent_disc.parameters(), lr=float(args.latent_adv_lr), weight_decay=1e-5
                    )
                assert disc_optimizer is not None
                # Discriminator step (hinge): real=teacher latent, fake=predicted (detached).
                for parameter in latent_disc.parameters():
                    parameter.requires_grad_(True)
                d_real = latent_disc(target.detach())
                d_fake = latent_disc(prediction.detach())
                latent_disc_loss = F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean()
                disc_optimizer.zero_grad(set_to_none=True)
                latent_disc_loss.backward()
                disc_optimizer.step()
                # Generator term: freeze D so the main backward only grads the student.
                for parameter in latent_disc.parameters():
                    parameter.requires_grad_(False)
                latent_adv_g = -latent_disc(prediction).mean()
                loss = loss + float(args.latent_adv_weight) * latent_adv_g
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=5.0).detach().cpu())
            optimizer.step()
            if step == 1 or step % args.log_interval == 0 or step == args.steps:
                log = {
                    "step": int(step),
                    "sample": f"{sample.row_id}_chunk{sample.chunk_index:02d}",
                    "loss": float(loss.detach().cpu()),
                    "l1": float(components["l1"].detach().cpu()),
                    "mse": float(components["mse"].detach().cpu()),
                    "norm_l1": float(components["norm_l1"].detach().cpu()),
                    "delta_l1": float(components["delta_l1"].detach().cpu()),
                    "channel_mean_l1": float(components["channel_mean_l1"].detach().cpu()),
                    "channel_std_l1": float(components["channel_std_l1"].detach().cpu()),
                    "channel_stat": float(components["channel_stat"].detach().cpu()),
                    "priority_norm_l1": float(components["priority_norm_l1"].detach().cpu()),
                    "priority_delta_l1": float(components["priority_delta_l1"].detach().cpu()),
                    "envelope_norm_l1": float(factorized_components["envelope_norm_l1"].detach().cpu()),
                    "gain_norm_l1": float(factorized_components["gain_norm_l1"].detach().cpu()),
                    "source_norm_l1": float(factorized_components["source_norm_l1"].detach().cpu()),
                    "source_delta_l1": float(factorized_components["source_delta_l1"].detach().cpu()),
                    "source_smooth_l1": float(factorized_components["source_smooth_l1"].detach().cpu()),
                    "factorized_total": float(factorized_total.detach().cpu()),
                    "grad_norm": grad_norm,
                }
                if decoder_components is not None:
                    log.update(
                        {
                            "decoder_l1": float(decoder_components["decoder_l1"].detach().cpu()),
                            "decoder_stft": float(decoder_components["decoder_stft"].detach().cpu()),
                            "decoder_feature": float(decoder_components["decoder_feature"].detach().cpu()),
                            "decoder_total": float(decoder_components["decoder_total"].detach().cpu()),
                        }
                    )
                if source_filter_components is not None:
                    log.update(
                        {
                            "source_filter_spectral": float(
                                source_filter_components["source_filter_spectral"].detach().cpu()
                            ),
                            "source_filter_rms": float(source_filter_components["source_filter_rms"].detach().cpu()),
                            "source_filter_waveform": float(
                                source_filter_components["source_filter_waveform"].detach().cpu()
                            ),
                            "source_filter_total": float(
                                source_filter_components["source_filter_total"].detach().cpu()
                            ),
                        }
                    )
                if latent_adv_g is not None and latent_disc_loss is not None:
                    log["latent_adv_g"] = float(latent_adv_g.detach().cpu())
                    log["latent_disc_loss"] = float(latent_disc_loss.detach().cpu())
                logs.append(log)
                print(json.dumps(log, ensure_ascii=False))

    train_eval_metrics = evaluate_latent(model, samples, device, latent_stats)
    eval_metrics = evaluate_latent(model, eval_samples, device, latent_stats) if eval_samples is not None else None
    saved_checkpoint: Path | None = None
    if not args.eval_only:
        saved_checkpoint = args.out_dir / "latent-student.pt"
        saved_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.cpu().state_dict(),
                "config": model_config,
                "train_args": vars(args),
                "latent_channel_stats": {
                    "mean": latent_stats.mean.squeeze(0).detach().cpu().tolist(),
                    "std": latent_stats.std.squeeze(0).detach().cpu().tolist(),
                },
                "train_eval_metrics": train_eval_metrics,
                "eval_metrics": eval_metrics,
            },
            saved_checkpoint,
        )
        model.to(device)

    if bool(args.skip_render):
        render_summary = {
            "skipped": True,
            "reason": "skip_render",
            "dashboard": None,
        }
    else:
        render_rows = eval_rows if eval_rows is not None else rows
        render_samples = eval_samples if eval_samples is not None else samples
        render_label = "heldout" if eval_rows is not None else "train"
        render_summary = render_dashboard(args, render_rows, render_samples, model, device, render_label)
    report = {
        "passed": True,
        "train_pack_dir": str(args.pack_dir),
        "eval_pack_dir": str(args.eval_pack_dir) if args.eval_pack_dir else None,
        "target_dir": str(args.target_dir) if args.target_dir else None,
        "eval_target_dir": str(args.eval_target_dir) if args.eval_target_dir else None,
        "target_key": str(args.target_key),
        "target_duration_key": str(args.target_duration_key) if args.target_duration_key is not None else None,
        "pack_dir": str(args.pack_dir),
        "decoder": str(args.decoder),
        "out_dir": str(args.out_dir),
        "checkpoint": str(saved_checkpoint) if saved_checkpoint else None,
        "loaded_checkpoint": str(args.load_checkpoint) if args.load_checkpoint else None,
        "eval_only": bool(args.eval_only),
        "device": str(device),
        "seed": int(args.seed),
        "steps": int(args.steps),
        "lr": float(args.lr),
        "sample_weighting": sample_weight_summary,
        "optimizer_groups": optimizer_groups_report,
        "loss_weights": {
            "mse": float(args.mse_weight),
            "norm_l1": float(args.norm_l1_weight),
            "delta_l1": float(args.delta_l1_weight),
            "channel_stat": float(args.channel_stat_weight),
            "channel_priority": float(args.channel_priority_weight),
            "channel_priority_delta": float(args.channel_priority_delta_weight),
            "decoder_aware_l1": float(args.decoder_aware_l1_weight),
            "decoder_aware_stft": float(args.decoder_aware_stft_weight),
            "decoder_aware_feature": float(args.decoder_aware_feature_weight),
            "source_filter_aware_spectral": float(args.source_filter_aware_spectral_weight),
            "source_filter_aware_rms": float(args.source_filter_aware_rms_weight),
            "source_filter_aware_waveform": float(args.source_filter_aware_waveform_weight),
            "envelope_norm": float(args.envelope_norm_weight),
            "gain_norm": float(args.gain_norm_weight),
            "source_norm": float(args.source_norm_weight),
            "source_delta": float(args.source_delta_weight),
            "source_smooth": float(args.source_smooth_weight),
        },
        "channel_priority": channel_priority_summary,
        "decoder_aware": {
            "enabled": bool(decoder_aware_enabled),
            "checkpoint": str(args.decoder_aware_checkpoint) if args.decoder_aware_checkpoint else None,
            "crop_frames": int(args.decoder_aware_crop_frames),
            "feature_keys": decoder_aware_feature_keys,
            "target_decoder": str(args.decoder),
            "target_cache_dir": (
                str(args.decoder_aware_target_cache_dir)
                if args.decoder_aware_target_cache_dir is not None
                else None
            ),
        },
        "source_filter_aware": source_filter_config
        if source_filter_config is not None
        else {
            "enabled": False,
            "target_audio_key": str(args.source_filter_aware_target_audio_key),
            "crop_frames": int(args.source_filter_aware_crop_frames),
            "n_fft": int(args.source_filter_aware_n_fft),
        },
        "train_rows": int(len(rows)),
        "train_chunks": int(len(samples)),
        "eval_rows": int(len(eval_rows)) if eval_rows is not None else 0,
        "eval_chunks": int(len(eval_samples)) if eval_samples is not None else 0,
        "rows": int(len(rows)),
        "chunks": int(len(samples)),
        "vocab_size": int(vocab_size),
        "train_vocab_size": int(train_vocab_size),
        "eval_vocab_size": int(eval_vocab_size),
        "out_channels": int(out_channels),
        "model_config": model_config,
        "parameter_table": parameter_table,
        "student_parameters": count_parameters(model),
        "trainable_parameters": int(sum(param.numel() for param in model.parameters() if param.requires_grad)),
        "factorized_freezing": factorized_freezing,
        "latent_channel_stats": {
            "channels": int(latent_stats.mean.numel()),
            "mean_abs_mean": float(torch.mean(torch.abs(latent_stats.mean)).detach().cpu()),
            "std_mean": float(torch.mean(latent_stats.std).detach().cpu()),
            "std_min": float(torch.min(latent_stats.std).detach().cpu()),
            "std_max": float(torch.max(latent_stats.std).detach().cpu()),
        },
        "logs": logs,
        "train_latent_eval": train_eval_metrics,
        "eval_latent_eval": eval_metrics,
        "latent_eval": train_eval_metrics,
        "render_summary": render_summary,
    }
    write_json(args.out_dir / "train-report.json", report)
    return report


@torch.no_grad()
def evaluate_latent(
    model: nn.Module,
    samples: list[ChunkSample],
    device: torch.device,
    stats: LatentChannelStats,
) -> dict[str, Any]:
    model.eval()
    l1_values: list[float] = []
    mse_values: list[float] = []
    cosine_values: list[float] = []
    norm_l1_values: list[float] = []
    delta_l1_values: list[float] = []
    channel_mean_l1_values: list[float] = []
    channel_std_l1_values: list[float] = []
    for sample in samples:
        features = expand_features(sample, device)
        target = target_tensor(sample, device)
        prediction = predict_latent_tensor(model, features)
        diff = prediction - target
        components = latent_loss_components(prediction, target, stats)
        l1_values.append(float(torch.mean(torch.abs(diff)).detach().cpu()))
        mse_values.append(float(torch.mean(torch.square(diff)).detach().cpu()))
        norm_l1_values.append(float(components["norm_l1"].detach().cpu()))
        delta_l1_values.append(float(components["delta_l1"].detach().cpu()))
        channel_mean_l1_values.append(float(components["channel_mean_l1"].detach().cpu()))
        channel_std_l1_values.append(float(components["channel_std_l1"].detach().cpu()))
        pred_f = prediction.reshape(-1)
        target_f = target.reshape(-1)
        denom = torch.linalg.vector_norm(pred_f) * torch.linalg.vector_norm(target_f)
        if float(denom.detach().cpu()) <= 0:
            raise RuntimeError(f"{sample.row_id} chunk {sample.chunk_index}: zero norm in latent cosine")
        cosine_values.append(float((torch.dot(pred_f, target_f) / denom).detach().cpu()))
    model.train()
    return {
        "mean_l1": float(np.mean(l1_values)),
        "max_l1": float(np.max(l1_values)),
        "mean_mse": float(np.mean(mse_values)),
        "max_mse": float(np.max(mse_values)),
        "mean_norm_l1": float(np.mean(norm_l1_values)),
        "max_norm_l1": float(np.max(norm_l1_values)),
        "mean_delta_l1": float(np.mean(delta_l1_values)),
        "max_delta_l1": float(np.max(delta_l1_values)),
        "mean_channel_mean_l1": float(np.mean(channel_mean_l1_values)),
        "mean_channel_std_l1": float(np.mean(channel_std_l1_values)),
        "mean_cosine": float(np.mean(cosine_values)),
        "min_cosine": float(np.min(cosine_values)),
    }


@torch.no_grad()
def predict_chunk(model: nn.Module, sample: ChunkSample, device: torch.device) -> np.ndarray:
    features = expand_features(sample, device)
    prediction = predict_latent_tensor(model, features)
    return prediction.transpose(0, 1).unsqueeze(0).detach().cpu().numpy().astype(np.float32)


def decode_latent(session: ort.InferenceSession, latent: np.ndarray) -> np.ndarray:
    input_names = [item.name for item in session.get_inputs()]
    if len(input_names) != 1:
        raise RuntimeError(f"decoder session must have exactly one input, got {input_names}")
    audio = np.asarray(session.run(None, {input_names[0]: latent})[0], dtype=np.float32).reshape(-1)
    if audio.size <= 0:
        raise RuntimeError("decoder returned empty audio")
    if not np.isfinite(audio).all():
        raise RuntimeError("decoder returned non-finite audio")
    return audio


def audio_rms(audio: np.ndarray) -> float:
    value = np.asarray(audio, dtype=np.float64).reshape(-1)
    return float(math.sqrt(float(np.mean(np.square(value)))))


def cosine_np(a: np.ndarray, b: np.ndarray) -> float:
    a_f = np.asarray(a, dtype=np.float64).reshape(-1)
    b_f = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(a_f) * np.linalg.norm(b_f))
    if denom <= 0:
        raise RuntimeError("cannot compute cosine for zero-norm arrays")
    return float(np.dot(a_f, b_f) / denom)


def html_audio_src(path: Path, base_dir: Path) -> str:
    try:
        return Path(os.path.relpath(path.resolve(), base_dir.resolve())).as_posix()
    except ValueError:
        return path.resolve().as_uri()


@torch.no_grad()
def render_dashboard(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    samples: list[ChunkSample],
    model: nn.Module,
    device: torch.device,
    dataset_label: str,
) -> dict[str, Any]:
    require_file(args.decoder, "decoder ONNX")
    session = ort.InferenceSession(str(args.decoder), providers=["CPUExecutionProvider"])
    input_names = [item.name for item in session.get_inputs()]
    if len(input_names) != 1:
        raise RuntimeError(f"decoder input mismatch: expected exactly one latent input, got {input_names}")

    sample_rate = int(rows[0].get("sample_rate") or 22050)
    silence = np.zeros(int(round(sample_rate * args.sentence_silence)), dtype=np.float32)
    by_row: dict[str, list[ChunkSample]] = {}
    for sample in samples:
        by_row.setdefault(sample.row_id, []).append(sample)

    audio_dir = args.out_dir / "audio"
    rendered: list[dict[str, Any]] = []
    compare_mean_abs: list[float] = []
    compare_cosine: list[float] = []
    for row in rows[: args.render_rows]:
        row_id = str(row["row_id"])
        row_chunks = sorted(by_row[row_id], key=lambda item: item.chunk_index)
        predicted_chunks: list[np.ndarray] = []
        for index, sample in enumerate(row_chunks):
            if index > 0:
                predicted_chunks.append(silence)
            latent = predict_chunk(model, sample, device)
            predicted_chunks.append(decode_latent(session, latent))
        predicted_audio = np.concatenate(predicted_chunks) if predicted_chunks else np.zeros(0, dtype=np.float32)
        if predicted_audio.size <= 0:
            raise RuntimeError(f"{row_id}: empty predicted audio")
        predicted_path = audio_dir / f"{row_id}_predicted.wav"
        write_wav(predicted_path, predicted_audio, sample_rate)

        teacher_path = Path(str(row["audio"]))
        teacher_audio, _ = read_wav_float32(teacher_path)
        n = min(int(teacher_audio.size), int(predicted_audio.size))
        mean_abs = float(np.mean(np.abs(predicted_audio[:n] - teacher_audio[:n])))
        cos = cosine_np(predicted_audio[:n], teacher_audio[:n])
        compare_mean_abs.append(mean_abs)
        compare_cosine.append(cos)
        rendered.append(
            {
                "row_id": row_id,
                "index": int(row["index"]),
                "text": row["text"],
                "teacher_audio": str(teacher_path),
                "predicted_audio": str(predicted_path),
                "teacher_audio_src": html_audio_src(teacher_path, args.out_dir),
                "predicted_audio_src": html_audio_src(predicted_path, args.out_dir),
                "teacher_seconds": float(teacher_audio.size / sample_rate),
                "predicted_seconds": float(predicted_audio.size / sample_rate),
                "teacher_rms": audio_rms(teacher_audio),
                "predicted_rms": audio_rms(predicted_audio),
                "compare_samples": n,
                "mean_abs_vs_teacher": mean_abs,
                "cosine_vs_teacher": cos,
            }
        )

    write_json(args.out_dir / "rendered-samples.json", rendered)
    (args.out_dir / "index.html").write_text(html_page(rendered, dataset_label), encoding="utf-8")
    return {
        "dataset_label": dataset_label,
        "rendered_rows": int(len(rendered)),
        "mean_abs_vs_teacher_mean": float(np.mean(compare_mean_abs)) if compare_mean_abs else None,
        "mean_abs_vs_teacher_max": float(np.max(compare_mean_abs)) if compare_mean_abs else None,
        "cosine_vs_teacher_mean": float(np.mean(compare_cosine)) if compare_cosine else None,
        "cosine_vs_teacher_min": float(np.min(compare_cosine)) if compare_cosine else None,
        "dashboard": str(args.out_dir / "index.html"),
    }


def read_wav_float32(path: Path) -> tuple[np.ndarray, int]:
    require_file(path, "WAV")
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1:
            raise RuntimeError(f"expected mono WAV: {path}")
        if wav.getsampwidth() != 2:
            raise RuntimeError(f"expected 16-bit WAV: {path}")
        sample_rate = wav.getframerate()
        audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32767.0
    if audio.size <= 0:
        raise RuntimeError(f"empty WAV: {path}")
    return audio, int(sample_rate)


def html_page(rendered: list[dict[str, Any]], dataset_label: str) -> str:
    title = f"Root A Latent Student - {dataset_label}"
    lines = [
        "<!doctype html>",
        '<meta charset="utf-8">',
        f"<title>{html.escape(title)}</title>",
        "<style>",
        "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:24px;line-height:1.35;color:#151515;background:#fafafa}",
        "table{border-collapse:collapse;width:100%;background:#fff}",
        "th,td{border:1px solid #ddd;padding:8px;vertical-align:top;font-size:13px}",
        "th{background:#f0f0f0;text-align:left}",
        "audio{width:230px}",
        ".text{font-size:16px;max-width:520px}",
        "</style>",
        f"<h1>{html.escape(title)}</h1>",
        "<p>This dashboard renders the selected evaluation split. If the label is heldout, these sentences were not used for latent-student training.</p>",
        "<table>",
        "<thead><tr><th>#</th><th>Text</th><th>Teacher</th><th>Predicted</th><th>Metrics</th></tr></thead>",
        "<tbody>",
    ]
    for row in rendered:
        lines.append(
            "<tr>"
            f"<td>{row['index']}</td>"
            f"<td class='text'>{html.escape(str(row['text']))}</td>"
            f"<td><audio controls src='{html.escape(str(row['teacher_audio_src']))}'></audio></td>"
            f"<td><audio controls src='{html.escape(str(row['predicted_audio_src']))}'></audio></td>"
            f"<td>mean abs {row['mean_abs_vs_teacher']:.5f}<br>cos {row['cosine_vs_teacher']:.5f}<br>"
            f"rms {row['predicted_rms']:.5f}</td>"
            "</tr>"
        )
    lines.extend(["</tbody>", "</table>"])
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = train(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
