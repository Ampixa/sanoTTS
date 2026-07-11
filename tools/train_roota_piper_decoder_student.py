#!/usr/bin/env python3
"""Train compact Root A decoder students against the verified Piper decoder.

This isolates the second half of the current Piper/VITS-native route:

    generator_input latent [192, frames] -> waveform [frames * 256]

The teacher target is produced by the already verified decoder ONNX cut.  This
script can also render the full current stack by feeding the small acoustic
latent student into the decoder student.
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
import torch.nn.functional as F
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))
from roota_fsd_blocks import FsdConvNeXtBlock, logmag_phase_synthesize

DEFAULT_PACK_DIR = (
    ROOT
    / "artifacts"
    / "sub10m-search"
    / "root-a-piper-vits"
    / "chitwan-medium-a1-32row-piper-native-pack-20260625"
)
DEFAULT_TEACHER_DECODER = (
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
    / "chitwan-medium-a4-decoder-student-smoke-20260625"
)
HOP_LENGTH = 256
DEFAULT_SIGNATURE_KEYS = "stage0_mix,stage1_mix,stage2_mix,pre_tanh,audio"
SIGNATURE_FEATURE_MAP = {
    "conv_pre": "pre",
    "up0_raw": "up0_raw",
    "stage0_mix": "stage0_mix",
    "up1_raw": "up1_raw",
    "stage1_mix": "stage1_mix",
    "up2_raw": "up2_raw",
    "stage2_mix": "stage2_mix",
    "up3_raw": "up3_raw",
    "stage3_mix": "stage3_mix",
    "pre_tanh": "pre_tanh",
    "audio": "audio",
}


@dataclass(frozen=True)
class ChunkSample:
    row_id: str
    row_index: int
    text: str
    chunk_index: int
    phoneme_ids: np.ndarray
    durations: np.ndarray
    latent: np.ndarray
    teacher_audio: np.ndarray
    teacher_pre: np.ndarray
    teacher_up0: np.ndarray
    tensor_path: Path
    teacher_signatures: dict[str, np.ndarray] | None = None
    student_latent: np.ndarray | None = None
    lrc_pred_code: np.ndarray | None = None
    oracle_latent: np.ndarray | None = None
    oracle_audio: np.ndarray | None = None


class LeakyReluActivation(nn.Module):
    def __init__(self, negative_slope: float = 0.1) -> None:
        super().__init__()
        if negative_slope <= 0.0:
            raise ValueError(f"negative_slope must be positive, got {negative_slope}")
        self.negative_slope = float(negative_slope)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(x, negative_slope=self.negative_slope)


class SnakeActivation(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        self.log_alpha = nn.Parameter(torch.zeros(channels, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise RuntimeError(f"expected activation input [batch, channels, time], got {x.shape}")
        if x.shape[1] != self.log_alpha.numel():
            raise RuntimeError(
                f"activation channel mismatch: input has {x.shape[1]}, activation has {self.log_alpha.numel()}"
            )
        alpha = torch.exp(self.log_alpha).view(1, -1, 1).clamp(min=1e-4, max=100.0)
        return x + torch.sin(alpha * x).square() / alpha


def make_activation(name: str, channels: int, *, negative_slope: float = 0.1) -> nn.Module:
    if name == "leaky_relu":
        return LeakyReluActivation(negative_slope=negative_slope)
    if name == "snake":
        return SnakeActivation(channels)
    raise ValueError(f"unsupported activation: {name}")


class ResidualUnit(nn.Module):
    def __init__(self, channels: int, dilation: int = 1, activation: str = "leaky_relu") -> None:
        super().__init__()
        self.act0 = make_activation(activation, channels)
        self.act1 = make_activation(activation, channels)
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
        self.scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.act0(x)
        residual = self.conv1(residual)
        residual = self.act1(residual)
        residual = self.conv2(residual)
        return x + self.scale * residual


class HifiGanResidualBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int,
        dilations: tuple[int, int],
        activation: str = "leaky_relu",
    ) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")
        if len(dilations) != 2 or any(dilation <= 0 for dilation in dilations):
            raise ValueError(f"dilations must contain two positive integers, got {dilations}")
        self.act0 = make_activation(activation, channels)
        self.act1 = make_activation(activation, channels)
        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=dilations[0] * (kernel_size // 2),
            dilation=dilations[0],
        )
        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=dilations[1] * (kernel_size // 2),
            dilation=dilations[1],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.act0(x)
        residual = self.conv1(residual)
        residual = self.act1(residual)
        residual = self.conv2(residual)
        return x + residual


class HifiGanResidualBank(nn.Module):
    def __init__(self, channels: int, activation: str = "leaky_relu") -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                HifiGanResidualBlock(channels, kernel_size=3, dilations=(1, 2), activation=activation),
                HifiGanResidualBlock(channels, kernel_size=5, dilations=(2, 6), activation=activation),
                HifiGanResidualBlock(channels, kernel_size=7, dilations=(3, 12), activation=activation),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch_sum = None
        for block in self.blocks:
            value = block(x)
            branch_sum = value if branch_sum is None else branch_sum + value
        if branch_sum is None:
            raise RuntimeError("HiFi-GAN residual bank has no branches")
        return branch_sum / float(len(self.blocks))


class PiperResidualBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int,
        dilations: tuple[int, int],
        activation: str = "leaky_relu",
    ) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")
        if len(dilations) != 2 or any(dilation <= 0 for dilation in dilations):
            raise ValueError(f"dilations must contain two positive integers, got {dilations}")
        self.act0 = make_activation(activation, channels)
        self.act1 = make_activation(activation, channels)
        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=dilations[0] * (kernel_size // 2),
            dilation=dilations[0],
        )
        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=dilations[1] * (kernel_size // 2),
            dilation=dilations[1],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(self.act0(x)) + x
        y = self.conv2(self.act1(y)) + y
        return y


class PiperFactorizedResidualBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int,
        dilations: tuple[int, int],
        factor_rank: int,
        activation: str = "leaky_relu",
    ) -> None:
        super().__init__()
        if factor_rank <= 0:
            raise ValueError(f"factor_rank must be positive, got {factor_rank}")
        if factor_rank > channels:
            raise ValueError(f"factor_rank {factor_rank} exceeds channels {channels}")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")
        if len(dilations) != 2 or any(dilation <= 0 for dilation in dilations):
            raise ValueError(f"dilations must contain two positive integers, got {dilations}")
        self.act0 = make_activation(activation, channels)
        self.act1 = make_activation(activation, channels)
        self.conv1 = FactorizedConv1d(
            channels,
            channels,
            kernel_size,
            rank=factor_rank,
            dilation=dilations[0],
        )
        self.conv2 = FactorizedConv1d(
            channels,
            channels,
            kernel_size,
            rank=factor_rank,
            dilation=dilations[1],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(self.act0(x)) + x
        y = self.conv2(self.act1(y)) + y
        return y


class PiperResidualBank(nn.Module):
    def __init__(
        self,
        channels: int,
        activation: str = "leaky_relu",
        branch_indices: tuple[int, ...] = (0, 1, 2),
        factor_rank_ratio: float = 0.0,
        scale_mode: str = "kept",
    ) -> None:
        super().__init__()
        if factor_rank_ratio < 0.0:
            raise ValueError(f"factor_rank_ratio must be non-negative, got {factor_rank_ratio}")
        if scale_mode not in {"kept", "teacher"}:
            raise ValueError(f"scale_mode must be 'kept' or 'teacher', got {scale_mode!r}")
        branch_specs = (
            (3, (1, 2)),
            (5, (2, 6)),
            (7, (3, 12)),
        )
        if not branch_indices:
            raise ValueError("Piper residual bank must keep at least one branch")
        if any(index < 0 or index >= len(branch_specs) for index in branch_indices):
            raise ValueError(f"invalid Piper residual branch indices: {branch_indices}")
        if len(set(branch_indices)) != len(branch_indices):
            raise ValueError(f"duplicate Piper residual branch indices: {branch_indices}")
        self.source_branch_indices = tuple(int(index) for index in branch_indices)
        factor_rank = 0
        if factor_rank_ratio > 0.0:
            factor_rank = max(4, int(round(channels * factor_rank_ratio)))
            factor_rank = min(channels, factor_rank)
        self.blocks = nn.ModuleList(
            [
                (
                    PiperFactorizedResidualBlock(
                        channels,
                        kernel_size=branch_specs[index][0],
                        dilations=branch_specs[index][1],
                        factor_rank=factor_rank,
                        activation=activation,
                    )
                    if factor_rank > 0
                    else PiperResidualBlock(
                        channels,
                        kernel_size=branch_specs[index][0],
                        dilations=branch_specs[index][1],
                        activation=activation,
                    )
                )
                for index in self.source_branch_indices
            ]
        )
        self.scale_divisor = len(self.blocks) if scale_mode == "kept" else len(branch_specs)
        if self.scale_divisor <= 0:
            raise RuntimeError("Piper residual bank has invalid scale divisor")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch_sum = None
        for block in self.blocks:
            value = block(x)
            branch_sum = value if branch_sum is None else branch_sum + value
        if branch_sum is None:
            raise RuntimeError("Piper residual bank has no branches")
        return branch_sum / float(self.scale_divisor)


class VitsResidualBlock1(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int,
        dilations: tuple[int, ...] = (1, 3, 5),
        activation: str = "leaky_relu",
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")
        if not dilations or any(dilation <= 0 for dilation in dilations):
            raise ValueError(f"dilations must be positive integers, got {dilations}")
        self.convs1 = nn.ModuleList(
            [
                nn.Sequential(
                    make_activation(activation, channels),
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        padding=dilation * (kernel_size // 2),
                        dilation=dilation,
                    ),
                )
                for dilation in dilations
            ]
        )
        self.convs2 = nn.ModuleList(
            [
                nn.Sequential(
                    make_activation(activation, channels),
                    nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2),
                )
                for _ in dilations
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv1, conv2 in zip(self.convs1, self.convs2, strict=True):
            residual = conv1(x)
            residual = conv2(residual)
            x = residual + x
        return x


class VitsFactorizedResidualBlock1(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int,
        dilations: tuple[int, ...] = (1, 3, 5),
        factor_rank: int,
        activation: str = "leaky_relu",
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if factor_rank <= 0:
            raise ValueError(f"factor_rank must be positive, got {factor_rank}")
        if factor_rank > channels:
            raise ValueError(f"factor_rank {factor_rank} exceeds channels {channels}")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")
        if not dilations or any(dilation <= 0 for dilation in dilations):
            raise ValueError(f"dilations must be positive integers, got {dilations}")
        self.convs1 = nn.ModuleList(
            [
                nn.Sequential(
                    make_activation(activation, channels),
                    FactorizedConv1d(
                        channels,
                        channels,
                        kernel_size,
                        rank=factor_rank,
                        dilation=dilation,
                    ),
                )
                for dilation in dilations
            ]
        )
        self.convs2 = nn.ModuleList(
            [
                nn.Sequential(
                    make_activation(activation, channels),
                    FactorizedConv1d(
                        channels,
                        channels,
                        kernel_size,
                        rank=factor_rank,
                    ),
                )
                for _ in dilations
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv1, conv2 in zip(self.convs1, self.convs2, strict=True):
            residual = conv1(x)
            residual = conv2(residual)
            x = residual + x
        return x


class VitsResidualBank(nn.Module):
    def __init__(
        self,
        channels: int,
        activation: str = "leaky_relu",
        branch_indices: tuple[int, ...] = (0, 1, 2),
        factor_rank_ratio: float = 0.0,
        scale_mode: str = "kept",
    ) -> None:
        super().__init__()
        if factor_rank_ratio < 0.0:
            raise ValueError(f"factor_rank_ratio must be non-negative, got {factor_rank_ratio}")
        if scale_mode not in {"kept", "teacher"}:
            raise ValueError(f"scale_mode must be 'kept' or 'teacher', got {scale_mode!r}")
        branch_kernels = (3, 7, 11)
        if not branch_indices:
            raise ValueError("VITS residual bank must keep at least one branch")
        if any(index < 0 or index >= len(branch_kernels) for index in branch_indices):
            raise ValueError(f"invalid VITS residual branch indices: {branch_indices}")
        if len(set(branch_indices)) != len(branch_indices):
            raise ValueError(f"duplicate VITS residual branch indices: {branch_indices}")
        self.source_branch_indices = tuple(int(index) for index in branch_indices)
        factor_rank = 0
        if factor_rank_ratio > 0.0:
            factor_rank = max(4, int(round(channels * factor_rank_ratio)))
            factor_rank = min(channels, factor_rank)
        self.blocks = nn.ModuleList(
            [
                (
                    VitsFactorizedResidualBlock1(
                        channels,
                        kernel_size=branch_kernels[index],
                        factor_rank=factor_rank,
                        activation=activation,
                    )
                    if factor_rank > 0
                    else VitsResidualBlock1(
                        channels,
                        kernel_size=branch_kernels[index],
                        activation=activation,
                    )
                )
                for index in self.source_branch_indices
            ]
        )
        self.scale_divisor = len(self.blocks) if scale_mode == "kept" else len(branch_kernels)
        if self.scale_divisor <= 0:
            raise RuntimeError("VITS residual bank has invalid scale divisor")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch_sum = None
        for block in self.blocks:
            value = block(x)
            branch_sum = value if branch_sum is None else branch_sum + value
        if branch_sum is None:
            raise RuntimeError("VITS residual bank has no branches")
        return branch_sum / float(self.scale_divisor)


class DepthwiseSeparableConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class SeparableResidualUnit(nn.Module):
    def __init__(self, channels: int, dilation: int = 1, activation: str = "leaky_relu") -> None:
        super().__init__()
        self.act0 = make_activation(activation, channels)
        self.act1 = make_activation(activation, channels)
        self.conv1 = DepthwiseSeparableConv1d(channels, channels, 3, dilation=dilation)
        self.conv2 = DepthwiseSeparableConv1d(channels, channels, 3)
        self.scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.act0(x)
        residual = self.conv1(residual)
        residual = self.act1(residual)
        residual = self.conv2(residual)
        return x + self.scale * residual


class MultiReceptiveResidualUnit(nn.Module):
    def __init__(self, channels: int, dilation: int = 1, activation: str = "leaky_relu") -> None:
        super().__init__()
        self.act0 = make_activation(activation, channels)
        self.act1 = make_activation(activation, channels)
        self.branches = nn.ModuleList(
            [
                DepthwiseSeparableConv1d(channels, channels, 3, dilation=dilation),
                DepthwiseSeparableConv1d(channels, channels, 5, dilation=dilation),
                DepthwiseSeparableConv1d(channels, channels, 7, dilation=dilation),
            ]
        )
        self.mix = nn.Conv1d(channels, channels, 1)
        self.scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.act0(x)
        branch_sum = None
        for branch in self.branches:
            value = branch(residual)
            branch_sum = value if branch_sum is None else branch_sum + value
        if branch_sum is None:
            raise RuntimeError("multi-receptive residual has no branches")
        residual = self.mix(self.act1(branch_sum / float(len(self.branches))))
        return x + self.scale * residual


class SeparableUpsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int) -> None:
        super().__init__()
        self.depthwise = nn.ConvTranspose1d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class FactorizedUpsample(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        *,
        rank_ratio: float,
    ) -> None:
        super().__init__()
        if rank_ratio <= 0:
            raise ValueError(f"rank_ratio must be positive, got {rank_ratio}")
        rank = max(8, int(round(min(in_channels, out_channels) * rank_ratio)))
        self.reduce = nn.Conv1d(in_channels, rank, 1)
        self.upsample = nn.ConvTranspose1d(rank, rank, kernel_size, stride=stride, padding=padding)
        self.expand = nn.Conv1d(rank, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.reduce(x)
        x = self.upsample(x)
        return self.expand(x)


class FactorizedConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        rank: int,
        padding: int | None = None,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError(f"channels must be positive, got in={in_channels}, out={out_channels}")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")
        if dilation <= 0:
            raise ValueError(f"dilation must be positive, got {dilation}")
        if padding is None:
            padding = dilation * (kernel_size // 2)
        self.rank = int(rank)
        self.reduce = nn.Conv1d(in_channels, self.rank, kernel_size, padding=padding, dilation=dilation)
        self.expand = nn.Conv1d(self.rank, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.expand(self.reduce(x))


class ResizeConvUpsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, kernel_size: int = 7) -> None:
        super().__init__()
        if stride <= 1:
            raise ValueError(f"stride must be greater than 1, got {stride}")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")
        self.stride = int(stride)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=self.stride, mode="nearest")
        return self.conv(x)


class WaveformPostFilter(nn.Module):
    def __init__(
        self,
        *,
        channels: int,
        layers: int,
        kernel_size: int,
        residual_scale: float,
        activation: str,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"post-filter channels must be positive, got {channels}")
        if layers <= 0:
            raise ValueError(f"post-filter layers must be positive, got {layers}")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(f"post-filter kernel must be a positive odd integer, got {kernel_size}")
        if residual_scale <= 0.0:
            raise ValueError(f"post-filter residual scale must be positive, got {residual_scale}")
        self.residual_scale = float(residual_scale)
        self.in_conv = nn.Conv1d(1, channels, kernel_size, padding=kernel_size // 2)
        units: list[nn.Module] = []
        for index in range(layers):
            units.append(ResidualUnit(channels, dilation=1 + index, activation=activation))
        self.units = nn.Sequential(*units)
        self.out_conv = nn.Conv1d(channels, 1, kernel_size, padding=kernel_size // 2)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        residual = self.in_conv(audio)
        residual = self.units(residual)
        residual = self.out_conv(residual)
        return torch.tanh(audio + self.residual_scale * residual)


class PreTanhContextRepair(nn.Module):
    def __init__(
        self,
        *,
        pre_tanh_channels: int,
        context_channels: int,
        channels: int,
        layers: int,
        kernel_size: int,
        residual_scale: float,
        activation: str,
    ) -> None:
        super().__init__()
        if pre_tanh_channels <= 0:
            raise ValueError(f"pre-tanh channels must be positive, got {pre_tanh_channels}")
        if context_channels <= 0:
            raise ValueError(f"context channels must be positive, got {context_channels}")
        if channels <= 0:
            raise ValueError(f"pre-tanh repair channels must be positive, got {channels}")
        if layers <= 0:
            raise ValueError(f"pre-tanh repair layers must be positive, got {layers}")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(f"pre-tanh repair kernel must be a positive odd integer, got {kernel_size}")
        if residual_scale <= 0.0:
            raise ValueError(f"pre-tanh repair residual scale must be positive, got {residual_scale}")
        self.residual_scale = float(residual_scale)
        self.pre_proj = nn.Conv1d(pre_tanh_channels, channels, kernel_size, padding=kernel_size // 2)
        self.context_proj = nn.Conv1d(context_channels, channels, 1)
        self.act = make_activation(activation, channels)
        self.units = nn.Sequential(
            *[ResidualUnit(channels, dilation=1 + index, activation=activation) for index in range(layers)]
        )
        self.out_conv = nn.Conv1d(channels, pre_tanh_channels, kernel_size, padding=kernel_size // 2)

    def forward(self, pre_tanh: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if pre_tanh.ndim != 3:
            raise RuntimeError(f"expected pre_tanh [batch, channels, samples], got {pre_tanh.shape}")
        if context.ndim != 3:
            raise RuntimeError(f"expected context [batch, channels, samples], got {context.shape}")
        if context.shape[0] != pre_tanh.shape[0]:
            raise RuntimeError(f"context batch {context.shape[0]} != pre_tanh batch {pre_tanh.shape[0]}")
        if context.shape[-1] != pre_tanh.shape[-1]:
            context = F.interpolate(context, size=int(pre_tanh.shape[-1]), mode="nearest")
        hidden = self.pre_proj(pre_tanh) + self.context_proj(context)
        hidden = self.units(self.act(hidden))
        residual = self.out_conv(hidden)
        return pre_tanh + self.residual_scale * residual


class ChannelAffine1d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        self.scale = nn.Parameter(torch.ones(1, channels, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise RuntimeError(f"expected [batch, channels, frames], got {x.shape}")
        if x.shape[1] != self.scale.shape[1]:
            raise RuntimeError(f"channel affine expected {self.scale.shape[1]} channels, got {x.shape[1]}")
        return x * self.scale + self.bias


class StageProjectionBottleneck(nn.Module):
    def __init__(self, channels: int, bottleneck: int) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if bottleneck <= 0:
            raise ValueError(f"bottleneck must be positive, got {bottleneck}")
        self.reduce = nn.Conv1d(channels, bottleneck, 1)
        self.expand = nn.Conv1d(bottleneck, channels, 1)
        nn.init.zeros_(self.expand.weight)
        if self.expand.bias is not None:
            nn.init.zeros_(self.expand.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.expand(self.reduce(x))


class PeriodDiscriminator(nn.Module):
    def __init__(self, period: int, channels: tuple[int, ...]) -> None:
        super().__init__()
        if period <= 1:
            raise ValueError(f"period must be greater than 1, got {period}")
        if not channels:
            raise ValueError("period discriminator channels must not be empty")
        if any(channel <= 0 for channel in channels):
            raise ValueError(f"period discriminator channels must be positive, got {channels}")
        self.period = int(period)
        layers: list[nn.Module] = []
        in_channels = 1
        for out_channels in channels:
            layers.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=(5, 1),
                    stride=(3, 1),
                    padding=(2, 0),
                )
            )
            in_channels = out_channels
        self.layers = nn.ModuleList(layers)
        self.post = nn.Conv2d(in_channels, 1, kernel_size=(3, 1), padding=(1, 0))

    def forward(self, audio: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if audio.ndim != 3 or audio.shape[1] != 1:
            raise RuntimeError(f"expected discriminator audio [batch, 1, samples], got {audio.shape}")
        samples = int(audio.shape[-1])
        pad = (self.period - (samples % self.period)) % self.period
        if pad:
            audio = F.pad(audio, (0, pad))
        frames = int(audio.shape[-1]) // self.period
        x = audio.reshape(audio.shape[0], 1, frames, self.period)
        features: list[torch.Tensor] = []
        for layer in self.layers:
            x = F.leaky_relu(layer(x), negative_slope=0.1)
            features.append(x)
        score = self.post(x)
        features.append(score)
        return score, features


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods: tuple[int, ...], channels: tuple[int, ...]) -> None:
        super().__init__()
        if not periods:
            raise ValueError("at least one discriminator period is required")
        if len(set(periods)) != len(periods):
            raise ValueError(f"duplicate discriminator periods are not allowed: {periods}")
        self.periods = periods
        self.discriminators = nn.ModuleList([PeriodDiscriminator(period, channels) for period in periods])

    def forward(self, audio: torch.Tensor) -> tuple[list[torch.Tensor], list[list[torch.Tensor]]]:
        scores: list[torch.Tensor] = []
        features: list[list[torch.Tensor]] = []
        for discriminator in self.discriminators:
            score, feature = discriminator(audio)
            scores.append(score)
            features.append(feature)
        return scores, features


class LrcEncoder(nn.Module):
    """Training-only z->c encoder for the latent re-contract racer."""

    def __init__(self, *, in_channels: int, hidden: int, code_dim: int) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if hidden <= 0:
            raise ValueError(f"hidden must be positive, got {hidden}")
        if code_dim <= 0:
            raise ValueError(f"code_dim must be positive, got {code_dim}")
        self.in_channels = int(in_channels)
        self.hidden = int(hidden)
        self.code_dim = int(code_dim)
        self.net = nn.Sequential(
            nn.Conv1d(self.in_channels, self.hidden, 1),
            nn.GELU(),
            nn.Conv1d(self.hidden, self.code_dim, 1),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 3:
            raise RuntimeError(f"expected LRC encoder input [batch, channels, frames], got {latent.shape}")
        if int(latent.shape[1]) != self.in_channels:
            raise RuntimeError(f"LRC encoder expected {self.in_channels} channels, got {latent.shape[1]}")
        return self.net(latent)


class DecoderStudent(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        channels: tuple[int, ...],
        res_layers: int,
        variant: str = "dense",
        rank_ratio: float = 0.5,
        activation: str = "leaky_relu",
        stage_affine: bool = False,
        factorized_pre_rank: int = 0,
        piper_res_factor_rank_ratio: float = 0.0,
        res_bank_scale_mode: str = "kept",
        stage0_branches: tuple[int, ...] = (0, 1, 2),
        stage1_branches: tuple[int, ...] = (0, 1, 2),
        stage2_branches: tuple[int, ...] = (0, 1, 2),
        stage3_branches: tuple[int, ...] = (0, 1, 2),
        post_filter_channels: int = 0,
        post_filter_layers: int = 0,
        post_filter_kernel: int = 9,
        post_filter_scale: float = 0.25,
        pre_tanh_repair_channels: int = 0,
        pre_tanh_repair_layers: int = 0,
        pre_tanh_repair_kernel: int = 7,
        pre_tanh_repair_scale: float = 0.15,
        istft_n_fft: int = 512,
        fsd_dim: int = 72,
        fsd_blocks: int = 5,
        fsd_film_rank: int = 12,
        fsd_head_rank: int = 48,
        stage_projection_bottlenecks: tuple[int, ...] = (),
    ) -> None:
        super().__init__()
        if variant not in {
            "dense",
            "separable",
            "lowrank",
            "multires",
            "polyphase",
            "resizeconv",
            "hifiganlite",
            "piperlite",
            "piperlite4",
            "piperfold",
            "piperphase",
            "istft",
            "apnetlite",
            "fsd",
            "lrc",
            "pb",
        }:
            raise ValueError(f"unsupported decoder variant: {variant}")
        if activation not in {"leaky_relu", "snake"}:
            raise ValueError(f"unsupported decoder activation: {activation}")
        if post_filter_channels < 0:
            raise ValueError(f"post_filter_channels must be non-negative, got {post_filter_channels}")
        if post_filter_layers < 0:
            raise ValueError(f"post_filter_layers must be non-negative, got {post_filter_layers}")
        if (post_filter_channels == 0) != (post_filter_layers == 0):
            raise ValueError("post_filter_channels and post_filter_layers must both be zero or both be positive")
        if pre_tanh_repair_channels < 0:
            raise ValueError(f"pre_tanh_repair_channels must be non-negative, got {pre_tanh_repair_channels}")
        if pre_tanh_repair_layers < 0:
            raise ValueError(f"pre_tanh_repair_layers must be non-negative, got {pre_tanh_repair_layers}")
        if (pre_tanh_repair_channels == 0) != (pre_tanh_repair_layers == 0):
            raise ValueError(
                "pre_tanh_repair_channels and pre_tanh_repair_layers must both be zero or both be positive"
            )
        if pre_tanh_repair_kernel <= 0 or pre_tanh_repair_kernel % 2 == 0:
            raise ValueError(
                f"pre_tanh_repair_kernel must be a positive odd integer, got {pre_tanh_repair_kernel}"
            )
        if pre_tanh_repair_scale <= 0.0:
            raise ValueError(f"pre_tanh_repair_scale must be positive, got {pre_tanh_repair_scale}")
        self.variant = variant
        self.activation = activation
        self.stage_affine_enabled = bool(stage_affine)
        self.stage0_branches = tuple(int(index) for index in stage0_branches)
        self.stage1_branches = tuple(int(index) for index in stage1_branches)
        self.stage2_branches = tuple(int(index) for index in stage2_branches)
        self.stage3_branches = tuple(int(index) for index in stage3_branches)
        self.stage_projection_bottlenecks = tuple(int(value) for value in stage_projection_bottlenecks)
        stage_branch_sets = (
            ("stage0", self.stage0_branches),
            ("stage1", self.stage1_branches),
            ("stage2", self.stage2_branches),
            ("stage3", self.stage3_branches),
        )
        for label, branches in stage_branch_sets:
            if not branches:
                raise ValueError(f"--{label}-branches must keep at least one branch")
            if any(index < 0 or index > 2 for index in branches):
                raise ValueError(f"--{label}-branches must be branch indices 0,1,2, got {branches}")
            if len(set(branches)) != len(branches):
                raise ValueError(f"--{label}-branches contains duplicates: {branches}")
        if any(branches != (0, 1, 2) for _, branches in stage_branch_sets) and variant not in {
            "piperlite",
            "piperlite4",
            "pb",
        }:
            raise ValueError("--stage*-branches is currently supported only for --variant piperlite/piperlite4/pb")
        if self.stage_projection_bottlenecks:
            if variant != "pb":
                raise ValueError("--stage-projection-bottlenecks is currently supported only for --variant pb")
            if len(self.stage_projection_bottlenecks) != 3:
                raise ValueError(
                    "--stage-projection-bottlenecks must be empty or contain three comma-separated integers"
                )
            if any(value <= 0 for value in self.stage_projection_bottlenecks):
                raise ValueError(
                    f"--stage-projection-bottlenecks values must be positive, got {self.stage_projection_bottlenecks}"
                )
        self.factorized_pre_rank = int(factorized_pre_rank)
        if self.factorized_pre_rank < 0:
            raise ValueError(f"factorized_pre_rank must be non-negative, got {self.factorized_pre_rank}")
        self.piper_res_factor_rank_ratio = float(piper_res_factor_rank_ratio)
        if self.piper_res_factor_rank_ratio < 0.0:
            raise ValueError(
                f"piper_res_factor_rank_ratio must be non-negative, got {self.piper_res_factor_rank_ratio}"
            )
        self.res_bank_scale_mode = str(res_bank_scale_mode)
        if self.res_bank_scale_mode not in {"kept", "teacher"}:
            raise ValueError(f"res_bank_scale_mode must be 'kept' or 'teacher', got {self.res_bank_scale_mode!r}")
        self.istft_n_fft = int(istft_n_fft)
        self.fsd_dim = int(fsd_dim)
        self.fsd_block_count = int(fsd_blocks)
        self.fsd_film_rank = int(fsd_film_rank)
        self.fsd_head_rank = int(fsd_head_rank)
        expected_channel_count = 5 if variant == "piperlite4" else 4
        if len(channels) != expected_channel_count:
            raise ValueError(
                f"--variant {variant} expects {expected_channel_count} channel widths, got {channels}"
            )
        c0, c1, c2, c3 = channels[:4]
        c4 = channels[4] if variant == "piperlite4" else 0
        if self.factorized_pre_rank > 0 and variant not in {"piperlite", "pb"}:
            raise ValueError("--factorized-pre-rank currently supports --variant piperlite/pb only")
        if self.piper_res_factor_rank_ratio > 0.0 and variant not in {"piperlite", "piperlite4", "pb"}:
            raise ValueError("--piper-res-factor-rank-ratio currently supports --variant piperlite/piperlite4/pb only")
        if variant in {"istft", "apnetlite", "fsd", "lrc"}:
            if self.istft_n_fft <= 0 or self.istft_n_fft % 2 != 0:
                raise ValueError(f"istft_n_fft must be a positive even integer, got {self.istft_n_fft}")
        if variant in {"fsd", "lrc"}:
            if self.fsd_dim <= 0:
                raise ValueError(f"fsd_dim must be positive, got {self.fsd_dim}")
            if self.fsd_block_count <= 0:
                raise ValueError(f"fsd_blocks must be positive, got {self.fsd_block_count}")
            if self.fsd_film_rank <= 0:
                raise ValueError(f"fsd_film_rank must be positive, got {self.fsd_film_rank}")
            if self.fsd_head_rank <= 0:
                raise ValueError(f"fsd_head_rank must be positive, got {self.fsd_head_rank}")
            bins = self.istft_n_fft // 2 + 1
            self.pre = nn.Conv1d(in_channels, self.fsd_dim, 1)
            self.pre_affine = nn.Identity()
            self.stage0_affine = nn.Identity()
            self.stage1_affine = nn.Identity()
            self.stage2_affine = nn.Identity()
            self.stage3_affine = nn.Identity()
            self.act_pre = nn.Identity()
            self.fsd_blocks = nn.ModuleList(
                [
                    FsdConvNeXtBlock(
                        channels=self.fsd_dim,
                        in_channels=in_channels,
                        film_rank=self.fsd_film_rank,
                    )
                    for _ in range(self.fsd_block_count)
                ]
            )
            self.fsd_head_in = nn.Conv1d(self.fsd_dim, self.fsd_head_rank, 1)
            self.fsd_head_out = nn.Conv1d(self.fsd_head_rank, bins * 3, 1)
            self.post = nn.Identity()
            self.amp_head = None
            self.phase_head = None
            self.register_buffer("istft_window", torch.hann_window(self.istft_n_fft), persistent=False)
            self.up0 = None
            self.act_up0 = nn.Identity()
            self.up1 = None
            self.act_up1 = nn.Identity()
            self.up2 = None
            self.act_up2 = nn.Identity()
            self.up3 = None
            self.res0 = nn.Identity()
            self.res1 = nn.Identity()
            self.res2 = nn.Identity()
            self.res3 = None
            self.act_post = nn.Identity()
            self.pre_tanh_repair = nn.Identity()
            self.post_filter = nn.Identity()
            self.stage0_projection = nn.Identity()
            self.stage1_projection = nn.Identity()
            self.stage2_projection = nn.Identity()
            return
        if variant in {"istft", "apnetlite"}:
            self.pre = nn.Conv1d(in_channels, c0, 7, padding=3)
            self.pre_affine = ChannelAffine1d(c0) if stage_affine else nn.Identity()
            self.stage0_affine = ChannelAffine1d(c0) if stage_affine else nn.Identity()
            self.stage1_affine = nn.Identity()
            self.stage2_affine = nn.Identity()
            self.stage3_affine = nn.Identity()
            self.act_pre = make_activation(activation, c0)
            self.res0 = self._make_res_stack(c0, res_layers, "dense", activation)
            self.act_post = make_activation(activation, c0)
            if variant == "istft":
                self.post = nn.Conv1d(c0, self.istft_n_fft + 2, 3, padding=1)
                self.amp_head = None
                self.phase_head = None
            else:
                bins = self.istft_n_fft // 2 + 1
                self.post = nn.Identity()
                self.amp_head = nn.Conv1d(c0, bins, 3, padding=1)
                self.phase_head = nn.Conv1d(c0, bins * 2, 3, padding=1)
            self.register_buffer("istft_window", torch.hann_window(self.istft_n_fft), persistent=False)
            self.up0 = None
            self.act_up0 = nn.Identity()
            self.up1 = None
            self.act_up1 = nn.Identity()
            self.res1 = nn.Identity()
            self.up2 = None
            self.res2 = None
            self.up3 = None
            self.res3 = None
            self.stage0_projection = nn.Identity()
            self.stage1_projection = nn.Identity()
            self.stage2_projection = nn.Identity()
            self.pre_tanh_repair = nn.Identity()
            self.post_filter = nn.Identity()
            return
        self.pre = (
            FactorizedConv1d(in_channels, c0, 7, rank=self.factorized_pre_rank)
            if self.factorized_pre_rank > 0
            else self._make_pre(in_channels, c0, variant)
        )
        self.pre_affine = ChannelAffine1d(c0) if stage_affine else nn.Identity()
        self.stage0_affine = ChannelAffine1d(c1) if stage_affine else nn.Identity()
        self.stage1_affine = ChannelAffine1d(c2) if stage_affine else nn.Identity()
        self.stage2_affine = ChannelAffine1d(c3) if stage_affine and variant not in {"polyphase", "piperphase"} else nn.Identity()
        self.stage3_affine = ChannelAffine1d(c4) if stage_affine and variant == "piperlite4" else nn.Identity()
        self.act_pre = make_activation(activation, c0)
        self.up0 = self._make_upsample(c0, c1, 16, 8, 4, variant, rank_ratio)
        self.act_up0 = make_activation(activation, c1)
        self.res0 = self._make_res_stack(
            c1,
            res_layers,
            variant,
            activation,
            piper_branch_indices=self.stage0_branches,
            piper_res_factor_rank_ratio=self.piper_res_factor_rank_ratio,
            res_bank_scale_mode=self.res_bank_scale_mode,
        )
        self.up1 = self._make_upsample(c1, c2, 16, 8, 4, variant, rank_ratio)
        self.act_up1 = make_activation(activation, c2)
        self.res1 = self._make_res_stack(
            c2,
            res_layers,
            variant,
            activation,
            piper_branch_indices=self.stage1_branches,
            piper_res_factor_rank_ratio=self.piper_res_factor_rank_ratio,
            res_bank_scale_mode=self.res_bank_scale_mode,
        )
        if variant in {"polyphase", "piperphase"}:
            self.up2 = None
            self.res2 = None
            self.up3 = None
            self.res3 = None
            self.act_post = make_activation(activation, c2)
            self.post = nn.Conv1d(c2, 4, 7, padding=3)
        else:
            up2_kernel = 4 if variant == "piperlite4" else 8
            up2_stride = 2 if variant == "piperlite4" else 4
            up2_padding = 1 if variant == "piperlite4" else 2
            self.up2 = self._make_upsample(c2, c3, up2_kernel, up2_stride, up2_padding, variant, rank_ratio)
            if variant == "piperfold":
                self.res2 = nn.Identity()
            elif variant == "piperlite4":
                if res_layers != 1:
                    raise ValueError("piperlite4 expects --res-layers 1 because each stage already has three branches")
                self.res2 = nn.Sequential(
                    VitsResidualBank(
                        c3,
                        activation=activation,
                        branch_indices=self.stage2_branches,
                        factor_rank_ratio=self.piper_res_factor_rank_ratio,
                        scale_mode=self.res_bank_scale_mode,
                    )
                )
                self.act_up2 = make_activation(activation, c3)
                self.up3 = self._make_upsample(c3, c4, 4, 2, 1, variant, rank_ratio)
                self.res3 = nn.Sequential(
                    VitsResidualBank(
                        c4,
                        activation=activation,
                        branch_indices=self.stage3_branches,
                        factor_rank_ratio=self.piper_res_factor_rank_ratio,
                        scale_mode=self.res_bank_scale_mode,
                    )
                )
                self.act_post = make_activation(activation, c4, negative_slope=0.01)
                self.post = nn.Conv1d(c4, 1, 7, padding=3)
            elif variant in {"piperlite", "pb"} and self.stage2_branches != (0, 1, 2):
                if res_layers != 1:
                    raise ValueError(f"{variant} partial stage2 branches require --res-layers 1")
                self.res2 = nn.Sequential(
                    PiperResidualBank(
                        c3,
                        activation=activation,
                        branch_indices=self.stage2_branches,
                        factor_rank_ratio=self.piper_res_factor_rank_ratio,
                        scale_mode=self.res_bank_scale_mode,
                    )
                )
            else:
                self.res2 = self._make_res_stack(
                    c3,
                    res_layers,
                    variant,
                    activation,
                    piper_branch_indices=self.stage2_branches,
                    piper_res_factor_rank_ratio=self.piper_res_factor_rank_ratio,
                    res_bank_scale_mode=self.res_bank_scale_mode,
                )
            if variant != "piperlite4":
                self.up3 = None
                self.res3 = None
                self.act_up2 = nn.Identity()
                post_negative_slope = (
                    0.01 if variant in {"piperlite", "pb", "piperfold"} and activation == "leaky_relu" else 0.1
                )
                self.act_post = make_activation(activation, c3, negative_slope=post_negative_slope)
                self.post = nn.Conv1d(c3, 1, 7, padding=3)
        if variant in {"polyphase", "piperphase"}:
            repair_context_channels = c2
            repair_pre_tanh_channels = 4
        elif variant == "piperlite4":
            repair_context_channels = c4
            repair_pre_tanh_channels = 1
        else:
            repair_context_channels = c3
            repair_pre_tanh_channels = 1
        self.pre_tanh_repair = (
            PreTanhContextRepair(
                pre_tanh_channels=repair_pre_tanh_channels,
                context_channels=repair_context_channels,
                channels=pre_tanh_repair_channels,
                layers=pre_tanh_repair_layers,
                kernel_size=pre_tanh_repair_kernel,
                residual_scale=pre_tanh_repair_scale,
                activation=activation,
            )
            if pre_tanh_repair_channels > 0 and pre_tanh_repair_layers > 0
            else nn.Identity()
        )
        self.post_filter = (
            WaveformPostFilter(
                channels=post_filter_channels,
                layers=post_filter_layers,
                kernel_size=post_filter_kernel,
                residual_scale=post_filter_scale,
                activation=activation,
            )
            if post_filter_channels > 0 and post_filter_layers > 0
            else nn.Identity()
        )
        if variant == "pb" and self.stage_projection_bottlenecks:
            k0, k1, k2 = self.stage_projection_bottlenecks
            self.stage0_projection = StageProjectionBottleneck(c1, k0)
            self.stage1_projection = StageProjectionBottleneck(c2, k1)
            self.stage2_projection = StageProjectionBottleneck(c3, k2)
        else:
            self.stage0_projection = nn.Identity()
            self.stage1_projection = nn.Identity()
            self.stage2_projection = nn.Identity()

    @staticmethod
    def _make_pre(in_channels: int, out_channels: int, variant: str) -> nn.Module:
        if variant in {"separable", "multires", "polyphase"}:
            return DepthwiseSeparableConv1d(in_channels, out_channels, 7)
        return nn.Conv1d(in_channels, out_channels, 7, padding=3)

    @staticmethod
    def _make_upsample(
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        variant: str,
        rank_ratio: float,
    ) -> nn.Module:
        if variant in {"separable", "multires", "polyphase"}:
            return SeparableUpsample(in_channels, out_channels, kernel_size, stride, padding)
        if variant == "resizeconv":
            return ResizeConvUpsample(in_channels, out_channels, stride)
        if variant == "lowrank":
            return FactorizedUpsample(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                rank_ratio=rank_ratio,
            )
        return nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

    @staticmethod
    def _make_res_stack(
        channels: int,
        res_layers: int,
        variant: str,
        activation: str,
        *,
        piper_branch_indices: tuple[int, ...] = (0, 1, 2),
        piper_res_factor_rank_ratio: float = 0.0,
        res_bank_scale_mode: str = "kept",
    ) -> nn.Sequential:
        units: list[nn.Module] = []
        for index in range(res_layers):
            dilation = 1 + index
            if variant == "hifiganlite":
                if res_layers != 1:
                    raise ValueError("hifiganlite expects --res-layers 1 because each stage already has three branches")
                units.append(HifiGanResidualBank(channels, activation=activation))
            elif variant in {"piperlite", "pb", "piperfold", "piperphase"}:
                if res_layers != 1:
                    raise ValueError(f"{variant} expects --res-layers 1 because each stage already has three branches")
                units.append(
                    PiperResidualBank(
                        channels,
                        activation=activation,
                        branch_indices=piper_branch_indices,
                        factor_rank_ratio=piper_res_factor_rank_ratio,
                        scale_mode=res_bank_scale_mode,
                    )
                )
            elif variant == "piperlite4":
                if res_layers != 1:
                    raise ValueError("piperlite4 expects --res-layers 1 because each stage already has three branches")
                units.append(
                    VitsResidualBank(
                        channels,
                        activation=activation,
                        branch_indices=piper_branch_indices,
                        factor_rank_ratio=piper_res_factor_rank_ratio,
                        scale_mode=res_bank_scale_mode,
                    )
                )
            elif variant == "multires":
                units.append(MultiReceptiveResidualUnit(channels, dilation=dilation, activation=activation))
            elif variant in {"separable", "polyphase"}:
                units.append(SeparableResidualUnit(channels, dilation=dilation, activation=activation))
            else:
                units.append(ResidualUnit(channels, dilation=dilation, activation=activation))
        return nn.Sequential(*units)

    @staticmethod
    def _interleave_phases(phases: torch.Tensor) -> torch.Tensor:
        if phases.ndim != 3 or phases.shape[1] != 4:
            raise RuntimeError(f"expected polyphase output [batch, 4, time], got {phases.shape}")
        return phases.permute(0, 2, 1).reshape(phases.shape[0], 1, phases.shape[2] * 4)

    def _apply_pre_tanh_repair(self, pre_tanh: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if isinstance(self.pre_tanh_repair, nn.Identity):
            return pre_tanh
        return self.pre_tanh_repair(pre_tanh, context)

    def _istft_synthesize(self, spec_params: torch.Tensor, latent_frames: int) -> torch.Tensor:
        if spec_params.ndim != 3:
            raise RuntimeError(f"expected iSTFT params [batch, bins*2, frames], got {spec_params.shape}")
        expected_channels = self.istft_n_fft + 2
        if spec_params.shape[1] != expected_channels:
            raise RuntimeError(f"iSTFT params channel count {spec_params.shape[1]} != {expected_channels}")
        if spec_params.shape[2] != latent_frames:
            raise RuntimeError(f"iSTFT params frames {spec_params.shape[2]} != latent frames {latent_frames}")
        bins = self.istft_n_fft // 2 + 1
        real = spec_params[:, :bins, :]
        imag = spec_params[:, bins:, :]
        complex_spec = torch.complex(real, imag)
        audio = torch.istft(
            complex_spec,
            n_fft=self.istft_n_fft,
            hop_length=HOP_LENGTH,
            win_length=self.istft_n_fft,
            window=self.istft_window.to(device=spec_params.device, dtype=spec_params.dtype),
            center=True,
            length=int(latent_frames) * HOP_LENGTH,
        )
        return audio.unsqueeze(1)

    def _apnetlite_synthesize(
        self,
        amp_logits: torch.Tensor,
        phase_logits: torch.Tensor,
        latent_frames: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if amp_logits.ndim != 3:
            raise RuntimeError(f"expected AP amplitude logits [batch, bins, frames], got {amp_logits.shape}")
        if phase_logits.ndim != 3:
            raise RuntimeError(f"expected AP phase logits [batch, 2*bins, frames], got {phase_logits.shape}")
        bins = self.istft_n_fft // 2 + 1
        if amp_logits.shape[1] != bins:
            raise RuntimeError(f"AP amplitude bins {amp_logits.shape[1]} != expected {bins}")
        if phase_logits.shape[1] != bins * 2:
            raise RuntimeError(f"AP phase channels {phase_logits.shape[1]} != expected {bins * 2}")
        if amp_logits.shape[2] != latent_frames or phase_logits.shape[2] != latent_frames:
            raise RuntimeError(
                f"AP frame mismatch: amp={amp_logits.shape[2]}, phase={phase_logits.shape[2]}, latent={latent_frames}"
            )
        phase = phase_logits.reshape(phase_logits.shape[0], 2, bins, phase_logits.shape[2])
        phase = F.normalize(phase, dim=1, eps=1e-6)
        magnitude = F.softplus(amp_logits).clamp_min(1e-7)
        real = magnitude * phase[:, 0]
        imag = magnitude * phase[:, 1]
        complex_spec = torch.complex(real, imag)
        audio = torch.istft(
            complex_spec,
            n_fft=self.istft_n_fft,
            hop_length=HOP_LENGTH,
            win_length=self.istft_n_fft,
            window=self.istft_window.to(device=amp_logits.device, dtype=amp_logits.dtype),
            center=True,
            length=int(latent_frames) * HOP_LENGTH,
        ).unsqueeze(1)
        return audio, {
            "ap_amp_logits": amp_logits,
            "ap_log_amplitude": torch.log1p(magnitude),
            "ap_phase_unit": phase,
            "ap_real": real,
            "ap_imag": imag,
        }

    def _logmag_phase_synthesize(
        self,
        log_magnitude: torch.Tensor,
        phase_logits: torch.Tensor,
        latent_frames: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return logmag_phase_synthesize(
            log_magnitude,
            phase_logits,
            latent_frames=latent_frames,
            n_fft=self.istft_n_fft,
            window=self.istft_window.to(device=log_magnitude.device, dtype=log_magnitude.dtype),
            hop_length=HOP_LENGTH,
        )

    def forward(self, latent: torch.Tensor, return_features: bool = False) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if latent.ndim != 3:
            raise RuntimeError(f"expected latent [batch, channels, frames], got {latent.shape}")
        features: dict[str, torch.Tensor] = {}
        x = self.pre_affine(self.pre(latent))
        features["pre"] = x
        if self.variant in {"fsd", "lrc"}:
            for index, block in enumerate(self.fsd_blocks):
                x = block(x, latent)
                features[f"fsd_block{index}"] = x
            features["stage0_mix"] = x
            head_hidden = F.gelu(self.fsd_head_in(x))
            head_params = self.fsd_head_out(head_hidden)
            bins = self.istft_n_fft // 2 + 1
            log_magnitude = head_params[:, :bins, :]
            phase_logits = head_params[:, bins:, :]
            audio, ap_features = self._logmag_phase_synthesize(log_magnitude, phase_logits, int(latent.shape[-1]))
            features["pre_tanh"] = torch.cat([log_magnitude, phase_logits], dim=1)
            features.update(ap_features)
            features["audio_pre_filter"] = audio
            features["audio"] = audio
            if return_features:
                return audio, features
            return audio
        if self.variant in {"istft", "apnetlite"}:
            x = self.stage0_affine(self.res0(self.act_pre(x)))
            features["stage0_mix"] = x
            if self.variant == "istft":
                spec_params = self.post(self.act_post(x))
                features["pre_tanh"] = spec_params
                audio = self._istft_synthesize(spec_params, int(latent.shape[-1]))
            else:
                if self.amp_head is None or self.phase_head is None:
                    raise RuntimeError("apnetlite decoder missing amplitude/phase heads")
                ap_input = self.act_post(x)
                amp_logits = self.amp_head(ap_input)
                phase_logits = self.phase_head(ap_input)
                audio, ap_features = self._apnetlite_synthesize(amp_logits, phase_logits, int(latent.shape[-1]))
                features["pre_tanh"] = torch.cat([amp_logits, phase_logits], dim=1)
                features.update(ap_features)
            features["audio_pre_filter"] = audio
            features["audio"] = audio
            if return_features:
                return audio, features
            return audio
        x = self.up0(self.act_pre(x))
        features["up0_raw"] = x
        features["up0"] = x
        x = self.stage0_affine(self.res0(x))
        x = self.stage0_projection(x)
        features["stage0_mix"] = x
        x = self.up1(self.act_up0(x))
        features["up1_raw"] = x
        features["up1"] = x
        x = self.stage1_affine(self.res1(x))
        x = self.stage1_projection(x)
        features["stage1_mix"] = x
        if self.variant in {"polyphase", "piperphase"}:
            pre_tanh = self.post(self.act_post(x))
            features["pre_tanh_raw"] = pre_tanh
            pre_tanh = self._apply_pre_tanh_repair(pre_tanh, x)
            features["pre_tanh"] = pre_tanh
            phases = torch.tanh(pre_tanh)
            audio = self._interleave_phases(phases)
        else:
            if self.up2 is None or self.res2 is None:
                raise RuntimeError("non-polyphase decoder missing up2/res2")
            x = self.up2(self.act_up1(x))
            features["up2_raw"] = x
            features["up2"] = x
            x = self.stage2_affine(self.res2(x))
            x = self.stage2_projection(x)
            features["stage2_mix"] = x
            if self.variant == "piperlite4":
                if self.up3 is None or self.res3 is None:
                    raise RuntimeError("piperlite4 decoder missing up3/res3")
                x = self.up3(self.act_up2(x))
                features["up3_raw"] = x
                features["up3"] = x
                x = self.stage3_affine(self.res3(x))
                features["stage3_mix"] = x
            pre_tanh = self.post(self.act_post(x))
            features["pre_tanh_raw"] = pre_tanh
            pre_tanh = self._apply_pre_tanh_repair(pre_tanh, x)
            features["pre_tanh"] = pre_tanh
            audio = torch.tanh(pre_tanh)
        features["audio_pre_filter"] = audio
        audio = self.post_filter(audio)
        features["audio"] = audio
        if return_features:
            return audio, features
        return audio


def initialize_spectral_heads(
    model: DecoderStudent,
    *,
    mode: str,
    scale: float,
    ap_amp_bias: float,
    ap_phase_real_bias: float,
) -> dict[str, Any] | None:
    if mode == "default":
        return None
    if mode not in {"zero", "small"}:
        raise ValueError(f"unsupported spectral head init mode: {mode}")
    if model.variant not in {"istft", "apnetlite", "fsd", "lrc"}:
        raise RuntimeError(f"--spectral-head-init {mode} requires --variant istft, apnetlite, fsd, or lrc")
    if scale <= 0.0:
        raise ValueError(f"spectral head init scale must be positive, got {scale}")

    def init_conv(conv: nn.Conv1d, *, bias: float = 0.0) -> None:
        if mode == "zero":
            nn.init.zeros_(conv.weight)
        else:
            nn.init.normal_(conv.weight, mean=0.0, std=scale)
        if conv.bias is not None:
            nn.init.constant_(conv.bias, bias)

    with torch.no_grad():
        if model.variant == "istft":
            if not isinstance(model.post, nn.Conv1d):
                raise RuntimeError("iSTFT spectral head init expected model.post to be Conv1d")
            init_conv(model.post, bias=0.0)
            return {
                "mode": mode,
                "scale": float(scale),
                "variant": model.variant,
                "post_weight_std": float(model.post.weight.detach().float().std(unbiased=False).cpu()),
                "post_bias_mean": (
                    float(model.post.bias.detach().float().mean().cpu()) if model.post.bias is not None else None
                ),
            }
        if model.variant in {"fsd", "lrc"}:
            if not isinstance(model.fsd_head_out, nn.Conv1d):
                raise RuntimeError("FSD spectral head init expected model.fsd_head_out to be Conv1d")
            init_conv(model.fsd_head_out, bias=0.0)
            if model.fsd_head_out.bias is None:
                raise RuntimeError("FSD head unexpectedly has no bias")
            bins = model.istft_n_fft // 2 + 1
            model.fsd_head_out.bias[:bins].fill_(float(ap_amp_bias))
            model.fsd_head_out.bias[bins : bins * 2].fill_(float(ap_phase_real_bias))
            model.fsd_head_out.bias[bins * 2 :].zero_()
            return {
                "mode": mode,
                "scale": float(scale),
                "variant": model.variant,
                "ap_amp_bias": float(ap_amp_bias),
                "ap_phase_real_bias": float(ap_phase_real_bias),
                "head_out_weight_std": float(model.fsd_head_out.weight.detach().float().std(unbiased=False).cpu()),
                "head_out_bias_mean": float(model.fsd_head_out.bias.detach().float().mean().cpu()),
            }
        if model.amp_head is None or model.phase_head is None:
            raise RuntimeError("APNetLite spectral head init expected amplitude and phase heads")
        init_conv(model.amp_head, bias=ap_amp_bias)
        init_conv(model.phase_head, bias=0.0)
        if model.phase_head.bias is None:
            raise RuntimeError("APNetLite phase head unexpectedly has no bias")
        bins = model.istft_n_fft // 2 + 1
        model.phase_head.bias[:bins].fill_(float(ap_phase_real_bias))
        model.phase_head.bias[bins:].zero_()
        return {
            "mode": mode,
            "scale": float(scale),
            "variant": model.variant,
            "ap_amp_bias": float(ap_amp_bias),
            "ap_phase_real_bias": float(ap_phase_real_bias),
            "amp_weight_std": float(model.amp_head.weight.detach().float().std(unbiased=False).cpu()),
            "phase_weight_std": float(model.phase_head.weight.detach().float().std(unbiased=False).cpu()),
            "amp_bias_mean": (
                float(model.amp_head.bias.detach().float().mean().cpu()) if model.amp_head.bias is not None else None
            ),
            "phase_bias_real_mean": float(model.phase_head.bias[:bins].detach().float().mean().cpu()),
            "phase_bias_imag_mean": float(model.phase_head.bias[bins:].detach().float().mean().cpu()),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack-dir", type=Path, default=DEFAULT_PACK_DIR)
    parser.add_argument(
        "--eval-pack-dir",
        type=Path,
        default=None,
        help="Optional held-out Piper-native pack used only for evaluation and dashboard rendering.",
    )
    parser.add_argument(
        "--input-target-dir",
        type=Path,
        default=None,
        help=(
            "Optional frame-target directory whose manifest.jsonl supplies decoder input "
            "tensors keyed by row_id/chunk_index. This is used for mel-to-wave probes; "
            "when omitted, decoder input remains Piper generator_input."
        ),
    )
    parser.add_argument(
        "--eval-input-target-dir",
        type=Path,
        default=None,
        help="Held-out input-target directory matching --eval-pack-dir when --input-target-dir is used.",
    )
    parser.add_argument(
        "--input-target-key",
        default="log_mel",
        help="NPZ tensor key to read from --input-target-dir, for example log_mel, mfcc, rms, or log_energy.",
    )
    parser.add_argument(
        "--oracle-target-mix-prob",
        type=float,
        default=0.0,
        help=(
            "When paired target_audio_npy rows are loaded, probability that a crop trains on the original "
            "teacher latent -> teacher audio pair instead of the alternate input target. This preserves "
            "decoder-oracle behavior while giving limited exposure to deployed student latents."
        ),
    )
    parser.add_argument("--teacher-decoder", type=Path, default=DEFAULT_TEACHER_DECODER)
    parser.add_argument(
        "--acoustic-checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional acoustic latent checkpoint for full-stack dashboard rendering "
            "or acoustic-latent mix/residual training. If omitted, decoder training "
            "and decoder-oracle rendering still run, but no full-stack render is "
            "produced. Pass the language-matched checkpoint explicitly."
        ),
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--channels", type=str, default="168,84,42,21")
    parser.add_argument(
        "--variant",
        choices=(
            "dense",
            "separable",
            "lowrank",
            "multires",
            "polyphase",
            "resizeconv",
            "hifiganlite",
            "piperlite",
            "piperlite4",
            "piperfold",
            "piperphase",
            "istft",
            "apnetlite",
            "fsd",
            "lrc",
            "pb",
        ),
        default="dense",
    )
    parser.add_argument("--rank-ratio", type=float, default=0.5)
    parser.add_argument("--activation", choices=("leaky_relu", "snake"), default="leaky_relu")
    parser.add_argument("--res-layers", type=int, default=1)
    parser.add_argument(
        "--stage-affine",
        action="store_true",
        help=(
            "Add identity-initialized per-channel affine calibration after the decoder pre "
            "projection and each stage. This is intended for teacher-sliced Piper variants."
        ),
    )
    parser.add_argument(
        "--factorized-pre-rank",
        type=int,
        default=0,
        help=(
            "Replace the piperlite pre Conv1d with Conv1d(kernel)->Conv1d(1x1) "
            "rank factorization. With --teacher-init-checkpoint this is initialized "
            "by SVD of the selected teacher pre convolution."
        ),
    )
    parser.add_argument(
        "--piper-res-factor-rank-ratio",
        type=float,
        default=0.0,
        help=(
            "For piperlite/piperlite4, replace each Piper/VITS residual branch "
            "Conv1d with a Conv1d(kernel)->Conv1d(1x1) low-rank factorization. "
            "The rank is round(channels * ratio), clamped to [4, channels], and "
            "teacher init uses SVD of each selected branch convolution."
        ),
    )
    parser.add_argument(
        "--res-bank-scale-mode",
        choices=("kept", "teacher"),
        default="kept",
        help=(
            "Residual-bank divisor for piperlite/piperlite4 branch subsets. "
            "'kept' averages retained branches. 'teacher' keeps the original three-branch divisor."
        ),
    )
    parser.add_argument("--fsd-dim", type=int, default=72)
    parser.add_argument("--fsd-blocks", type=int, default=5)
    parser.add_argument("--fsd-film-rank", type=int, default=12)
    parser.add_argument("--fsd-head-rank", type=int, default=48)
    parser.add_argument("--lrc-code-dim", type=int, default=40)
    parser.add_argument("--lrc-encoder-hidden", type=int, default=64)
    parser.add_argument(
        "--lrc-pred-code-mix-prob",
        type=float,
        default=0.0,
        help=(
            "For --variant lrc only, probability per training crop of feeding the decoder "
            "the cached c-acoustic predicted code instead of the exact E(z) code."
        ),
    )
    parser.add_argument(
        "--lrc-pred-code-checkpoint",
        type=Path,
        default=None,
        help=(
            "For --variant lrc predicted-code training, c-acoustic latent-student checkpoint "
            "whose output channels match --lrc-code-dim."
        ),
    )
    parser.add_argument(
        "--lrc-pred-code-residual-prob",
        type=float,
        default=0.0,
        help=(
            "For --variant lrc only, probability on exact-code crops of adding scaled "
            "(exact_c - predicted_c) residual noise before the decoder."
        ),
    )
    parser.add_argument(
        "--lrc-pred-code-residual-max-scale",
        type=float,
        default=0.25,
        help="Maximum residual scale for --lrc-pred-code-residual-prob.",
    )
    parser.add_argument(
        "--stage-projection-bottlenecks",
        type=str,
        default="",
        help=(
            "For --variant pb, comma-separated k0,k1,k2 bottleneck widths for identity-initialized "
            "1x1 C->k->C projections after stages 0, 1, and 2. Empty disables them."
        ),
    )
    parser.add_argument(
        "--stage0-branches",
        type=str,
        default="0,1,2",
        help=(
            "Comma-separated Piper residual branch indices to keep in stage 0 for piperlite. "
            "Default keeps all branches. Branches are 0=kernel3, 1=kernel5, 2=kernel7."
        ),
    )
    parser.add_argument(
        "--stage1-branches",
        type=str,
        default="0,1,2",
        help=(
            "Comma-separated Piper residual branch indices to keep in stage 1 for piperlite. "
            "Default keeps all branches. Branches are 0=kernel3, 1=kernel5, 2=kernel7."
        ),
    )
    parser.add_argument(
        "--stage2-branches",
        type=str,
        default="0,1,2",
        help=(
            "Comma-separated Piper residual branch indices to keep in stage 2 for piperlite. "
            "Default keeps all branches. Branches are 0=kernel3, 1=kernel5, 2=kernel7."
        ),
    )
    parser.add_argument(
        "--stage3-branches",
        type=str,
        default="0,1,2",
        help=(
            "Comma-separated VITS residual branch indices to keep in stage 3 for piperlite4. "
            "Default keeps all branches. Branches are 0=kernel3, 1=kernel7, 2=kernel11."
        ),
    )
    parser.add_argument(
        "--teacher-init-checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional PyTorch Piper decoder parity checkpoint used to initialize a smaller "
            "piperlite, piperfold, or piperphase student by deterministic channel slicing before training."
        ),
    )
    parser.add_argument(
        "--teacher-init-method",
        choices=("first", "importance"),
        default="first",
        help=(
            "Channel selection method for --teacher-init-checkpoint. 'first' preserves the "
            "A75/A77 deterministic first-N slicing. 'importance' selects top-norm teacher "
            "channels per stage and copies the consistent channel subgraph."
        ),
    )
    parser.add_argument(
        "--init-decoder-checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional previous decoder-student.pt checkpoint used to initialize the "
            "student before a repair/fine-tune run. The saved decoder config must "
            "match the requested architecture exactly."
        ),
    )
    parser.add_argument(
        "--allow-new-post-filter-init",
        action="store_true",
        help=(
            "Allow --init-decoder-checkpoint to initialize the shared decoder body when the "
            "requested model adds a post-filter to a checkpoint that had no post-filter. "
            "Only post-filter config mismatches are allowed; the new post-filter remains "
            "randomly initialized."
        ),
    )
    parser.add_argument(
        "--allow-new-pre-tanh-repair-init",
        action="store_true",
        help=(
            "Allow --init-decoder-checkpoint to initialize the shared decoder body when the "
            "requested model adds a pre-tanh context repair branch to a checkpoint that had "
            "no such branch. Only pre-tanh repair config mismatches are allowed; the new "
            "repair branch remains randomly initialized."
        ),
    )
    parser.add_argument(
        "--allow-leaky-to-snake-init",
        action="store_true",
        help=(
            "Allow --init-decoder-checkpoint to initialize shared weights when the checkpoint "
            "uses leaky_relu activations and the requested model uses snake activations. "
            "Only Snake log_alpha parameters may be newly initialized."
        ),
    )
    parser.add_argument("--post-filter-channels", type=int, default=0)
    parser.add_argument("--post-filter-layers", type=int, default=0)
    parser.add_argument("--post-filter-kernel", type=int, default=9)
    parser.add_argument("--post-filter-scale", type=float, default=0.25)
    parser.add_argument("--pre-tanh-repair-channels", type=int, default=0)
    parser.add_argument("--pre-tanh-repair-layers", type=int, default=0)
    parser.add_argument("--pre-tanh-repair-kernel", type=int, default=7)
    parser.add_argument("--pre-tanh-repair-scale", type=float, default=0.15)
    parser.add_argument(
        "--freeze-decoder-body",
        action="store_true",
        help=(
            "Train only enabled repair modules while keeping the initialized decoder body fixed. "
            "Requires --post-filter-* or --pre-tanh-repair-*."
        ),
    )
    parser.add_argument(
        "--istft-n-fft",
        type=int,
        default=None,
        help=(
            "FFT size for spectral decoder variants. Effective default is 1024 for fsd and 512 otherwise. "
            "iSTFT outputs raw real/imag bins; apnetlite/fsd output amplitude/phase bins."
        ),
    )
    parser.add_argument(
        "--ap-amplitude-weight",
        type=float,
        default=0.0,
        help="Weight for apnetlite log-amplitude STFT supervision against teacher waveform.",
    )
    parser.add_argument(
        "--ap-phase-weight",
        type=float,
        default=0.0,
        help="Weight for apnetlite magnitude-weighted phase-unit supervision against teacher waveform.",
    )
    parser.add_argument(
        "--ap-complex-weight",
        type=float,
        default=0.0,
        help="Weight for apnetlite real/imag STFT supervision against teacher waveform.",
    )
    parser.add_argument(
        "--spectral-head-init",
        choices=("default", "zero", "small"),
        default="default",
        help=(
            "Explicit output-head initialization for istft/apnetlite/fsd/lrc variants. "
            "'default' keeps PyTorch init, 'zero' starts from silence, and 'small' "
            "uses small random weights. APNetLite/FSD also apply the AP amplitude/phase "
            "bias controls below."
        ),
    )
    parser.add_argument(
        "--spectral-head-init-scale",
        type=float,
        default=1e-3,
        help="Normal stddev used by --spectral-head-init small.",
    )
    parser.add_argument(
        "--ap-amp-init-bias",
        type=float,
        default=-6.0,
        help="Amplitude-head bias for APNetLite when --spectral-head-init is zero or small.",
    )
    parser.add_argument(
        "--ap-phase-real-init-bias",
        type=float,
        default=1.0,
        help=(
            "Initial real phase-unit bias for APNetLite when --spectral-head-init is "
            "zero or small. Imaginary phase bias is set to zero."
        ),
    )
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--crop-frames", type=int, default=64)
    parser.add_argument("--spectral-weight", type=float, default=0.5)
    parser.add_argument(
        "--stft-phase-weight",
        type=float,
        default=0.0,
        help="Weight for magnitude-weighted STFT phase-unit matching on generated waveform.",
    )
    parser.add_argument("--feature-hint-weight", type=float, default=0.0)
    parser.add_argument(
        "--feature-exact-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for exact L1 matching of selected teacher decoder feature channels. "
            "Uses the same teacher channel indices as --teacher-init-checkpoint."
        ),
    )
    parser.add_argument(
        "--feature-exact-keys",
        default="pre,up0",
        help="Comma-separated exact teacher feature keys to match. Supported: pre,up0.",
    )
    parser.add_argument(
        "--quiet-frame-weight",
        type=float,
        default=0.0,
        help="Weight for log-RMS matching on the quietest teacher waveform frames.",
    )
    parser.add_argument(
        "--quiet-delta-weight",
        type=float,
        default=0.0,
        help="Weight for first-difference log-RMS matching on quiet teacher frames.",
    )
    parser.add_argument(
        "--quiet-ceiling-weight",
        type=float,
        default=0.0,
        help="Weight for quiet-frame dB excess penalty when student RMS rises above teacher RMS.",
    )
    parser.add_argument(
        "--quiet-ceiling-margin-db",
        type=float,
        default=3.0,
        help="Allowed student-over-teacher quiet-frame RMS margin before quiet ceiling loss applies.",
    )
    parser.add_argument(
        "--click-delta-weight",
        type=float,
        default=0.0,
        help="Weight for top-k sample-derivative excess loss against the teacher waveform.",
    )
    parser.add_argument(
        "--click-delta-margin",
        type=float,
        default=0.015,
        help="Absolute derivative margin allowed above the scaled teacher derivative before click loss applies.",
    )
    parser.add_argument(
        "--click-delta-target-scale",
        type=float,
        default=1.25,
        help="Multiplier on teacher sample derivatives for the click-delta threshold.",
    )
    parser.add_argument(
        "--click-delta-topk-frac",
        type=float,
        default=0.005,
        help="Fraction of largest derivative excess samples to average for click-delta loss.",
    )
    parser.add_argument(
        "--quiet-sample-weight",
        type=float,
        default=0.0,
        help="Weight for sample-level amplitude excess loss on teacher-quiet samples.",
    )
    parser.add_argument(
        "--quiet-sample-quantile",
        type=float,
        default=0.10,
        help="Per-crop teacher absolute-amplitude quantile used to select quiet samples.",
    )
    parser.add_argument(
        "--quiet-sample-margin",
        type=float,
        default=0.0005,
        help="Allowed absolute-amplitude margin above the scaled teacher quiet samples.",
    )
    parser.add_argument(
        "--quiet-sample-target-scale",
        type=float,
        default=1.0,
        help="Scale applied to teacher absolute amplitude before sample-level quiet excess is penalized.",
    )
    parser.add_argument(
        "--high-band-excess-weight",
        type=float,
        default=0.0,
        help="Weight for penalizing student high-frequency spectral energy above the teacher ratio.",
    )
    parser.add_argument(
        "--high-band-excess-hz",
        type=float,
        default=10000.0,
        help="Frequency cutoff for high-band excess artifact loss.",
    )
    parser.add_argument(
        "--high-band-excess-margin-db",
        type=float,
        default=0.5,
        help="Allowed student-over-teacher high-band power-ratio margin before penalty applies.",
    )
    parser.add_argument(
        "--echo-tail-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for penalizing delayed teacher-correlated residual energy. "
            "This targets frame-hop robotic shadow artifacts."
        ),
    )
    parser.add_argument(
        "--echo-tail-min-ms",
        type=float,
        default=8.0,
        help="Minimum positive lag, in milliseconds, for echo-tail residual correlation loss.",
    )
    parser.add_argument(
        "--echo-tail-max-ms",
        type=float,
        default=40.0,
        help="Maximum positive lag, in milliseconds, for echo-tail residual correlation loss.",
    )
    parser.add_argument(
        "--echo-tail-lags",
        type=int,
        default=9,
        help="Number of evenly spaced lags to test for echo-tail residual correlation.",
    )
    parser.add_argument(
        "--echo-tail-margin",
        type=float,
        default=0.02,
        help="Allowed absolute residual/reference lag correlation before echo-tail loss applies.",
    )
    parser.add_argument(
        "--adv-weight",
        type=float,
        default=0.0,
        help="Training-only multi-period discriminator generator loss weight.",
    )
    parser.add_argument(
        "--adv-feature-weight",
        type=float,
        default=0.0,
        help="Training-only discriminator feature-matching loss weight.",
    )
    parser.add_argument(
        "--adv-delta-weight",
        type=float,
        default=0.0,
        help="Training-only generator loss weight for a discriminator on waveform first differences.",
    )
    parser.add_argument(
        "--adv-delta-feature-weight",
        type=float,
        default=0.0,
        help="Training-only feature-matching weight for the waveform first-difference discriminator.",
    )
    parser.add_argument(
        "--adv-start-step",
        type=int,
        default=1000,
        help="First step where adversarial decoder training is enabled.",
    )
    parser.add_argument("--adv-lr", type=float, default=2e-4)
    parser.add_argument(
        "--adv-periods",
        type=str,
        default="2,3,5,7,11",
        help="Comma-separated waveform periods for the multi-period discriminator.",
    )
    parser.add_argument(
        "--adv-channels",
        type=str,
        default="8,16,32,64",
        help="Comma-separated channel widths for each period discriminator.",
    )
    parser.add_argument(
        "--adv-gate-mode",
        choices=("none", "target-energy"),
        default="none",
        help="Optionally gate discriminator audio with a teacher-energy mask before adversarial losses.",
    )
    parser.add_argument(
        "--adv-gate-quantile",
        type=float,
        default=0.40,
        help="Teacher log-RMS quantile used as the voiced/quiet split for target-energy adversarial gating.",
    )
    parser.add_argument(
        "--adv-gate-sharpness",
        type=float,
        default=24.0,
        help="Sigmoid sharpness for target-energy adversarial gating in log-RMS space.",
    )
    parser.add_argument("--adv-gate-frame-size", type=int, default=1024)
    parser.add_argument("--adv-gate-frame-hop", type=int, default=256)
    parser.add_argument(
        "--quiet-frame-quantile",
        type=float,
        default=0.10,
        help="Per-sample teacher frame-power quantile used as the quiet-frame mask.",
    )
    parser.add_argument("--quiet-frame-size", type=int, default=1024)
    parser.add_argument("--quiet-frame-hop", type=int, default=256)
    parser.add_argument(
        "--signature-pack-dir",
        type=Path,
        default=None,
        help="Optional decoder activation-signature pack built by build_piper_vits_decoder_signature_pack.py.",
    )
    parser.add_argument(
        "--exact-feature-pack-dir",
        type=Path,
        default=None,
        help=(
            "Optional decoder feature pack containing *_exact tensors for --feature-exact-keys beyond pre/up0. "
            "When omitted, --signature-pack-dir is used for both pooled signatures and exact features."
        ),
    )
    parser.add_argument(
        "--signature-hint-weight",
        type=float,
        default=0.0,
        help="Weight for channel-agnostic pooled decoder activation signature loss.",
    )
    parser.add_argument(
        "--signature-temporal-weight",
        type=float,
        default=0.0,
        help="Weight for channel-agnostic temporal Gram loss on decoder activation signatures.",
    )
    parser.add_argument(
        "--signature-phase-weight",
        type=float,
        default=0.0,
        help="Weight for intra-latent-frame phase-binned decoder activation signature loss.",
    )
    parser.add_argument(
        "--signature-phase-bins",
        type=int,
        default=0,
        help="Number of phase bins expected in the signature pack when --signature-phase-weight is positive.",
    )
    parser.add_argument(
        "--signature-keys",
        default=DEFAULT_SIGNATURE_KEYS,
        help="Comma-separated signature labels to match, e.g. stage1_mix,stage2_mix,pre_tanh,audio.",
    )
    parser.add_argument(
        "--bottleneck-code-checkpoint",
        type=Path,
        default=None,
        help="Frozen activation-bottleneck encoders from probe_decoder_activation_bottleneck.py.",
    )
    parser.add_argument(
        "--bottleneck-code-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for matching student pooled activations to learned low-dimensional teacher activation codes. "
            "This is training-only and does not add deployment parameters."
        ),
    )
    parser.add_argument(
        "--bottleneck-code-keys",
        default="auto",
        help=(
            "Comma-separated signature keys to use from --bottleneck-code-checkpoint, or auto for every supported key."
        ),
    )
    parser.add_argument(
        "--acoustic-latent-mix-prob",
        type=float,
        default=0.0,
        help=(
            "Probability of training a crop from the acoustic-student latent instead of the "
            "teacher latent, while keeping the Piper decoder audio as target."
        ),
    )
    parser.add_argument(
        "--acoustic-latent-residual-prob",
        type=float,
        default=0.0,
        help=(
            "Probability of training a crop from a near-manifold interpolation "
            "teacher_latent + scale * (acoustic_student_latent - teacher_latent)."
        ),
    )
    parser.add_argument(
        "--acoustic-latent-residual-max-scale",
        type=float,
        default=0.25,
        help="Maximum interpolation scale for --acoustic-latent-residual-prob.",
    )
    parser.add_argument(
        "--paired-acoustic-residual-weight",
        type=float,
        default=0.0,
        help=(
            "Auxiliary robustness loss weight. When positive, each step also trains a paired "
            "batch from teacher_latent + scale * (acoustic_student_latent - teacher_latent), "
            "while the main batch remains controlled by --acoustic-latent-* options."
        ),
    )
    parser.add_argument(
        "--paired-acoustic-residual-max-scale",
        type=float,
        default=0.25,
        help="Maximum interpolation scale for --paired-acoustic-residual-weight.",
    )
    parser.add_argument("--log-interval", type=int, default=250)
    parser.add_argument("--render-rows", type=int, default=16)
    parser.add_argument("--sentence-silence", type=float, default=0.12)
    parser.add_argument("--assert-max-decoder-params", type=int, default=0)
    parser.add_argument("--seed", type=int, default=4242)
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


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    require_file(path, "file")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_signature_keys(value: str) -> list[str]:
    keys = [item.strip() for item in value.split(",") if item.strip()]
    if not keys:
        raise ValueError("--signature-keys must contain at least one key")
    unsupported = sorted(set(keys) - set(SIGNATURE_FEATURE_MAP))
    if unsupported:
        raise ValueError(f"unsupported signature keys {unsupported}; supported={sorted(SIGNATURE_FEATURE_MAP)}")
    return keys


def parse_optional_signature_keys(value: str, *, label: str) -> list[str] | None:
    text = value.strip()
    if text == "auto":
        return None
    keys = [item.strip() for item in text.split(",") if item.strip()]
    if not keys:
        raise ValueError(f"{label} must be auto or contain at least one key")
    unsupported = sorted(set(keys) - set(SIGNATURE_FEATURE_MAP))
    if unsupported:
        raise ValueError(f"unsupported {label} keys {unsupported}; supported={sorted(SIGNATURE_FEATURE_MAP)}")
    return keys


def parse_feature_exact_keys(value: str) -> list[str]:
    supported = {
        "pre",
        "up0",
        "up1_raw",
        "stage1_mix",
        "up2_raw",
        "stage2_mix",
        "up3_raw",
        "stage3_mix",
    }
    keys = [item.strip() for item in value.split(",") if item.strip()]
    if not keys:
        raise ValueError("--feature-exact-keys must contain at least one key")
    unsupported = sorted(set(keys) - supported)
    if unsupported:
        raise ValueError(f"unsupported feature exact keys {unsupported}; supported={sorted(supported)}")
    return keys


def parse_stage_branches(value: str, *, label: str) -> tuple[int, ...]:
    text = value.strip()
    if not text:
        raise ValueError(f"{label} must not be empty")
    branches: list[int] = []
    for raw in text.split(","):
        item = raw.strip()
        if not item:
            raise ValueError(f"{label} contains an empty item: {value!r}")
        try:
            branch = int(item)
        except ValueError as exc:
            raise ValueError(f"{label} item is not an integer: {item!r}") from exc
        if branch < 0 or branch > 2:
            raise ValueError(f"{label} branch must be 0, 1, or 2, got {branch}")
        branches.append(branch)
    if len(set(branches)) != len(branches):
        raise ValueError(f"{label} contains duplicate branches: {branches}")
    return tuple(branches)


def parse_stage_projection_bottlenecks(value: str) -> tuple[int, ...]:
    text = value.strip()
    if not text:
        return ()
    parts = parse_positive_int_tuple(text, label="--stage-projection-bottlenecks")
    if len(parts) != 3:
        raise ValueError(
            f"--stage-projection-bottlenecks must contain exactly three integers, got {parts}"
        )
    return parts


def resolve_existing_path(value: str, *, base_dir: Path) -> Path:
    raw = Path(value)
    candidates = [raw] if raw.is_absolute() else [Path.cwd() / raw, ROOT / raw, base_dir / raw]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"could not resolve path from {base_dir}: {value}")


def load_signature_index(signature_pack_dir: Path) -> dict[Path, Path]:
    require_dir(signature_pack_dir, "signature pack directory")
    manifest_path = signature_pack_dir / "decoder-signature-manifest.jsonl"
    require_file(manifest_path, "signature manifest")
    index: dict[Path, Path] = {}
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{manifest_path}:{line_no}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise RuntimeError(f"{manifest_path}:{line_no}: expected object")
            source = resolve_existing_path(str(row.get("source_tensor_npz") or ""), base_dir=signature_pack_dir)
            signature = resolve_existing_path(str(row.get("signature_npz") or ""), base_dir=signature_pack_dir)
            if source in index:
                raise RuntimeError(f"{manifest_path}:{line_no}: duplicate source tensor {source}")
            index[source] = signature
    if not index:
        raise RuntimeError(f"{manifest_path}: no signature rows")
    return index


def load_signature_tensors(
    path: Path,
    signature_keys: list[str],
    latent_frames: int,
    *,
    phase_bins: int,
    exact_feature_keys: list[str],
) -> dict[str, np.ndarray]:
    require_file(path, "signature NPZ")
    result: dict[str, np.ndarray] = {}
    with np.load(path) as tensors:
        stored_frames = np.asarray(tensors.get("latent_frames"), dtype=np.int64).reshape(-1)
        if stored_frames.size != 1 or int(stored_frames[0]) != latent_frames:
            raise RuntimeError(f"{path}: latent frame mismatch, expected {latent_frames}, got {stored_frames.tolist()}")
        for key in signature_keys:
            for suffix in ("mean", "logrms"):
                tensor_key = f"{key}_{suffix}"
                if tensor_key not in tensors:
                    raise RuntimeError(f"{path}: missing {tensor_key}")
                value = np.asarray(tensors[tensor_key], dtype=np.float32)
                if value.ndim != 2 or value.shape[0] <= 0 or value.shape[1] != latent_frames:
                    raise RuntimeError(f"{path}: invalid {tensor_key} shape {value.shape}")
                if not np.isfinite(value).all():
                    raise RuntimeError(f"{path}: non-finite {tensor_key}")
                result[tensor_key] = value
            if phase_bins > 0:
                for suffix in ("phase_mean", "phase_logrms"):
                    tensor_key = f"{key}_{suffix}"
                    if tensor_key not in tensors:
                        raise RuntimeError(f"{path}: missing {tensor_key}")
                    value = np.asarray(tensors[tensor_key], dtype=np.float32)
                    if (
                        value.ndim != 3
                        or value.shape[0] <= 0
                        or value.shape[1] != latent_frames
                        or value.shape[2] != phase_bins
                    ):
                        raise RuntimeError(
                            f"{path}: invalid {tensor_key} shape {value.shape}, "
                            f"expected [channels,{latent_frames},{phase_bins}]"
                        )
                    if not np.isfinite(value).all():
                        raise RuntimeError(f"{path}: non-finite {tensor_key}")
                    result[tensor_key] = value
        for key in exact_feature_keys:
            tensor_key = f"{key}_exact"
            if tensor_key not in tensors:
                raise RuntimeError(f"{path}: missing {tensor_key}")
            value = np.asarray(tensors[tensor_key], dtype=np.float32)
            if value.ndim != 2 or value.shape[0] <= 0 or value.shape[1] <= 0:
                raise RuntimeError(f"{path}: invalid {tensor_key} shape {value.shape}, expected [channels,time]")
            if value.shape[1] < latent_frames:
                raise RuntimeError(f"{path}: {tensor_key} time {value.shape[1]} < latent frames {latent_frames}")
            if not np.isfinite(value).all():
                raise RuntimeError(f"{path}: non-finite {tensor_key}")
            result[tensor_key] = value
    return result


def attach_signature_targets(
    samples: list[ChunkSample],
    signature_pack_dir: Path,
    signature_keys: list[str],
    *,
    phase_bins: int,
    exact_feature_keys: list[str],
) -> list[ChunkSample]:
    index = load_signature_index(signature_pack_dir)
    updated: list[ChunkSample] = []
    missing = 0
    for sample in samples:
        signature_path = index.get(sample.tensor_path.resolve())
        if signature_path is None:
            missing += 1
            continue
        signatures = load_signature_tensors(
            signature_path,
            signature_keys,
            int(sample.latent.shape[-1]),
            phase_bins=phase_bins,
            exact_feature_keys=exact_feature_keys,
        )
        merged_signatures = dict(sample.teacher_signatures or {})
        overlap = sorted(set(merged_signatures) & set(signatures))
        if overlap:
            raise RuntimeError(
                f"{signature_path}: refusing to overwrite existing teacher signature tensors {overlap}"
            )
        merged_signatures.update(signatures)
        updated.append(replace(sample, teacher_signatures=merged_signatures))
    if not updated:
        raise RuntimeError(f"{signature_pack_dir}: no signatures matched training samples")
    if missing:
        print(
            f"signature pack covers {len(updated)}/{len(samples)} train chunks; "
            "restricting signature-hint training to covered chunks",
            flush=True,
        )
    return updated


def filter_rows_to_complete_samples(rows: list[dict[str, Any]], samples: list[ChunkSample]) -> list[dict[str, Any]]:
    available = {(sample.row_id, sample.chunk_index) for sample in samples}
    filtered: list[dict[str, Any]] = []
    for row in rows:
        row_id = str(row.get("row_id") or "")
        chunks = row.get("chunks")
        if not row_id or not isinstance(chunks, list) or not chunks:
            continue
        required = {(row_id, int(chunk.get("chunk_index") or 0)) for chunk in chunks if isinstance(chunk, dict)}
        if required and required <= available:
            filtered.append(row)
    if not filtered:
        raise RuntimeError("no complete rows remain after filtering to signature-covered samples")
    return filtered


def parse_channels(value: str) -> tuple[int, ...]:
    parts = [int(item.strip()) for item in value.split(",") if item.strip()]
    if len(parts) not in {4, 5}:
        raise ValueError(f"--channels must contain four or five comma-separated integers, got {value!r}")
    if any(item <= 0 for item in parts):
        raise ValueError(f"--channels must be positive, got {parts}")
    if any(left < right for left, right in zip(parts, parts[1:], strict=False)):
        raise ValueError(f"--channels must be descending, got {parts}")
    return tuple(parts)


def parse_positive_int_tuple(value: str, *, label: str, min_value: int = 1) -> tuple[int, ...]:
    parts = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not parts:
        raise ValueError(f"{label} must contain at least one integer")
    if any(item < min_value for item in parts):
        raise ValueError(f"{label} values must be >= {min_value}, got {parts}")
    return tuple(parts)


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


def count_parameters(model: nn.Module) -> int:
    return int(sum(param.numel() for param in model.parameters()))


def load_torch_checkpoint(path: Path, label: str) -> dict[str, Any]:
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


def resolve_decoder_checkpoint_path(value: str, *, base_dir: Path) -> Path:
    raw = Path(value)
    candidates = [raw] if raw.is_absolute() else [base_dir / raw, Path.cwd() / raw, ROOT / raw]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"decoder checkpoint not found from {value!r}; tried {[str(path) for path in candidates]}")


def find_teacher_init_summary_in_decoder_chain(checkpoint_path: Path, *, max_depth: int = 16) -> dict[str, Any]:
    path = checkpoint_path
    visited: set[Path] = set()
    for depth in range(max_depth):
        resolved = path.resolve()
        if resolved in visited:
            raise RuntimeError(f"decoder init chain contains a cycle at {resolved}")
        visited.add(resolved)
        checkpoint = load_torch_checkpoint(path, f"decoder init chain checkpoint depth {depth}")
        config = checkpoint.get("config")
        if not isinstance(config, dict):
            raise RuntimeError(f"{path}: decoder checkpoint missing config dict")
        teacher_init = config.get("teacher_init")
        if isinstance(teacher_init, dict):
            result = dict(teacher_init)
            result["resolved_from_decoder_checkpoint"] = str(path)
            result["resolved_chain_depth"] = int(depth)
            return result
        next_path = config.get("init_decoder_checkpoint")
        if not next_path:
            break
        path = resolve_decoder_checkpoint_path(str(next_path), base_dir=path.parent)
    raise RuntimeError(f"no teacher_init metadata found in decoder init chain starting at {checkpoint_path}")


def checkpoint_state_dict(checkpoint: dict[str, Any], path: Path, *, key: str) -> dict[str, torch.Tensor]:
    raw_state = checkpoint.get(key)
    if not isinstance(raw_state, dict) or not raw_state:
        raise RuntimeError(f"{path}: missing non-empty {key!r}")
    state: dict[str, torch.Tensor] = {}
    for name, value in raw_state.items():
        if not isinstance(name, str) or not name:
            raise RuntimeError(f"{path}: invalid state_dict key {name!r}")
        try:
            tensor = torch.as_tensor(value).detach().cpu()
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"{path}: state_dict value {name!r} is not tensor-like") from exc
        if tensor.numel() <= 0:
            raise RuntimeError(f"{path}: state_dict tensor {name!r} is empty")
        if not torch.isfinite(tensor).all().item():
            raise RuntimeError(f"{path}: state_dict tensor {name!r} contains non-finite values")
        state[name] = tensor
    return state


def copy_sliced_parameter(
    target: torch.Tensor | None,
    state: dict[str, torch.Tensor],
    key: str,
    slices: tuple[slice, ...],
) -> None:
    if target is None:
        raise RuntimeError(f"target parameter for {key} is missing")
    if key not in state:
        raise RuntimeError(f"teacher-init checkpoint missing tensor {key!r}")
    source = state[key]
    if source.ndim != target.ndim:
        raise RuntimeError(f"{key}: source ndim {source.ndim} != target ndim {target.ndim}")
    if len(slices) != source.ndim:
        raise RuntimeError(f"{key}: got {len(slices)} slices for source ndim {source.ndim}")
    sliced = source[slices]
    if tuple(sliced.shape) != tuple(target.shape):
        raise RuntimeError(f"{key}: sliced source shape {tuple(sliced.shape)} != target shape {tuple(target.shape)}")
    target.copy_(sliced.to(device=target.device, dtype=target.dtype))


def select_tensor(
    source: torch.Tensor,
    selectors: tuple[slice | torch.Tensor, ...],
) -> torch.Tensor:
    if len(selectors) != source.ndim:
        raise RuntimeError(f"got {len(selectors)} selectors for source ndim {source.ndim}")
    selected = source
    for dim, selector in enumerate(selectors):
        if isinstance(selector, slice):
            selected = selected[(slice(None),) * dim + (selector,)]
            continue
        if not isinstance(selector, torch.Tensor):
            raise RuntimeError(f"selector for dim {dim} must be a slice or tensor, got {type(selector).__name__}")
        if selector.ndim != 1:
            raise RuntimeError(f"selector for dim {dim} must be 1-D, got {tuple(selector.shape)}")
        if selector.numel() <= 0:
            raise RuntimeError(f"selector for dim {dim} is empty")
        if selector.dtype != torch.long:
            raise RuntimeError(f"selector for dim {dim} must be torch.long, got {selector.dtype}")
        if int(selector.min().item()) < 0 or int(selector.max().item()) >= int(selected.shape[dim]):
            raise RuntimeError(
                f"selector for dim {dim} out of range for shape {tuple(selected.shape)}: "
                f"min={int(selector.min().item())}, max={int(selector.max().item())}"
            )
        selected = torch.index_select(selected, dim, selector)
    return selected


def copy_selected_parameter(
    target: torch.Tensor | None,
    state: dict[str, torch.Tensor],
    key: str,
    selectors: tuple[slice | torch.Tensor, ...],
) -> None:
    if target is None:
        raise RuntimeError(f"target parameter for {key} is missing")
    if key not in state:
        raise RuntimeError(f"teacher-init checkpoint missing tensor {key!r}")
    source = state[key]
    if source.ndim != target.ndim:
        raise RuntimeError(f"{key}: source ndim {source.ndim} != target ndim {target.ndim}")
    selected = select_tensor(source, selectors)
    if tuple(selected.shape) != tuple(target.shape):
        raise RuntimeError(f"{key}: selected source shape {tuple(selected.shape)} != target shape {tuple(target.shape)}")
    target.copy_(selected.to(device=target.device, dtype=target.dtype))


def add_channel_norm(score: torch.Tensor, tensor: torch.Tensor, dim: int) -> None:
    if tensor.shape[dim] != score.numel():
        raise RuntimeError(
            f"channel score length {score.numel()} does not match tensor shape {tuple(tensor.shape)} dim {dim}"
        )
    axes = tuple(axis for axis in range(tensor.ndim) if axis != dim)
    contribution = tensor.float().square().sum(dim=axes).sqrt()
    score.add_(contribution)


def top_importance_indices(score: torch.Tensor, count: int, label: str) -> torch.Tensor:
    if count <= 0:
        raise RuntimeError(f"{label}: count must be positive, got {count}")
    if count > score.numel():
        raise RuntimeError(f"{label}: count {count} exceeds score length {score.numel()}")
    if not torch.isfinite(score).all().item():
        raise RuntimeError(f"{label}: non-finite importance score")
    values, indices = torch.topk(score, k=count, largest=True, sorted=False)
    if not torch.isfinite(values).all().item():
        raise RuntimeError(f"{label}: non-finite selected importance score")
    return indices.sort().values.to(dtype=torch.long)


def piper_stage_importance_indices(
    state: dict[str, torch.Tensor],
    channels: tuple[int, int, int, int],
    teacher_channels: tuple[int, int, int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    c0, c1, c2, c3 = channels
    t0, t1, t2, t3 = teacher_channels
    scores = (
        torch.zeros(t0, dtype=torch.float32),
        torch.zeros(t1, dtype=torch.float32),
        torch.zeros(t2, dtype=torch.float32),
        torch.zeros(t3, dtype=torch.float32),
    )

    add_channel_norm(scores[0], state["conv_pre.weight"], 0)
    add_channel_norm(scores[0], state["conv_pre.bias"], 0)
    add_channel_norm(scores[0], state["ups.0.weight"], 0)

    up_specs = (
        ("ups.0", 1, 0, scores[1]),
        ("ups.1", 2, 1, scores[2]),
        ("ups.2", 3, 2, scores[3]),
    )
    for prefix, output_stage, input_stage, output_score in up_specs:
        add_channel_norm(output_score, state[f"{prefix}.weight"], 1)
        add_channel_norm(output_score, state[f"{prefix}.bias"], 0)
        add_channel_norm(scores[input_stage], state[f"{prefix}.weight"], 0)

    for stage_index, score in enumerate((scores[1], scores[2], scores[3])):
        for branch_index in range(3):
            for conv_index in range(2):
                prefix = f"stages.{stage_index}.branches.{branch_index}.convs.{conv_index}"
                add_channel_norm(score, state[f"{prefix}.weight"], 0)
                add_channel_norm(score, state[f"{prefix}.weight"], 1)
                add_channel_norm(score, state[f"{prefix}.bias"], 0)

    add_channel_norm(scores[3], state["conv_post.weight"], 1)

    return (
        top_importance_indices(scores[0], c0, "stage0/pre"),
        top_importance_indices(scores[1], c1, "stage1/up0"),
        top_importance_indices(scores[2], c2, "stage2/up1"),
        top_importance_indices(scores[3], c3, "stage3/up2"),
    )


def piper_vits_stage_importance_indices(
    state: dict[str, torch.Tensor],
    channels: tuple[int, int, int, int, int],
    teacher_channels: tuple[int, int, int, int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    c0, c1, c2, c3, c4 = channels
    t0, t1, t2, t3, t4 = teacher_channels
    scores = (
        torch.zeros(t0, dtype=torch.float32),
        torch.zeros(t1, dtype=torch.float32),
        torch.zeros(t2, dtype=torch.float32),
        torch.zeros(t3, dtype=torch.float32),
        torch.zeros(t4, dtype=torch.float32),
    )

    add_channel_norm(scores[0], state["conv_pre.weight"], 0)
    add_channel_norm(scores[0], state["conv_pre.bias"], 0)
    add_channel_norm(scores[0], state["ups.0.weight"], 0)

    up_specs = (
        ("ups.0", 0, scores[1]),
        ("ups.1", 1, scores[2]),
        ("ups.2", 2, scores[3]),
        ("ups.3", 3, scores[4]),
    )
    for prefix, input_stage, output_score in up_specs:
        add_channel_norm(scores[input_stage], state[f"{prefix}.weight"], 0)
        add_channel_norm(output_score, state[f"{prefix}.weight"], 1)
        add_channel_norm(output_score, state[f"{prefix}.bias"], 0)

    for stage_index, score in enumerate((scores[1], scores[2], scores[3], scores[4])):
        for branch_index in range(3):
            for group_name in ("convs1", "convs2"):
                for conv_index in range(3):
                    prefix = f"stages.{stage_index}.branches.{branch_index}.{group_name}.{conv_index}"
                    add_channel_norm(score, state[f"{prefix}.weight"], 0)
                    add_channel_norm(score, state[f"{prefix}.weight"], 1)
                    add_channel_norm(score, state[f"{prefix}.bias"], 0)

    add_channel_norm(scores[4], state["conv_post.weight"], 1)

    return (
        top_importance_indices(scores[0], c0, "stage0/pre"),
        top_importance_indices(scores[1], c1, "stage1/up0"),
        top_importance_indices(scores[2], c2, "stage2/up1"),
        top_importance_indices(scores[3], c3, "stage3/up2"),
        top_importance_indices(scores[4], c4, "stage4/up3"),
    )


def require_conv1d(module: nn.Module, name: str) -> nn.Conv1d:
    if not isinstance(module, nn.Conv1d):
        raise RuntimeError(f"teacher init requires {name} to be Conv1d, got {type(module).__name__}")
    return module


def init_factorized_conv1d_from_selected_weight(
    module: FactorizedConv1d,
    state: dict[str, torch.Tensor],
    weight_key: str,
    bias_key: str,
    selectors: tuple[slice | torch.Tensor, ...],
) -> None:
    if weight_key not in state:
        raise RuntimeError(f"teacher-init checkpoint missing tensor {weight_key!r}")
    if bias_key not in state:
        raise RuntimeError(f"teacher-init checkpoint missing tensor {bias_key!r}")
    selected = select_tensor(state[weight_key], selectors).float()
    if selected.ndim != 3:
        raise RuntimeError(f"{weight_key}: expected selected Conv1d weight [out,in,k], got {tuple(selected.shape)}")
    out_channels, in_channels, kernel_size = selected.shape
    if out_channels != module.expand.out_channels:
        raise RuntimeError(f"{weight_key}: selected out channels {out_channels} != expand out {module.expand.out_channels}")
    if in_channels != module.reduce.in_channels:
        raise RuntimeError(f"{weight_key}: selected in channels {in_channels} != reduce in {module.reduce.in_channels}")
    if kernel_size != module.reduce.kernel_size[0]:
        raise RuntimeError(f"{weight_key}: selected kernel {kernel_size} != reduce kernel {module.reduce.kernel_size[0]}")
    rank = int(module.rank)
    max_rank = min(out_channels, in_channels * kernel_size)
    if rank > max_rank:
        raise RuntimeError(f"{weight_key}: rank {rank} exceeds selected matrix max rank {max_rank}")
    matrix = selected.reshape(out_channels, in_channels * kernel_size)
    try:
        u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
    except RuntimeError as exc:
        raise RuntimeError(f"{weight_key}: SVD factorization failed") from exc
    u_r = u[:, :rank]
    s_r = s[:rank]
    vh_r = vh[:rank, :]
    reduce_weight = vh_r.reshape(rank, in_channels, kernel_size)
    expand_weight = u_r * s_r.unsqueeze(0)
    selected_bias = select_tensor(state[bias_key], (selectors[0],)).float()
    if selected_bias.shape != module.expand.bias.shape:
        raise RuntimeError(
            f"{bias_key}: selected bias shape {tuple(selected_bias.shape)} != expand bias {tuple(module.expand.bias.shape)}"
        )
    module.reduce.weight.copy_(reduce_weight.to(device=module.reduce.weight.device, dtype=module.reduce.weight.dtype))
    if module.reduce.bias is None:
        raise RuntimeError("factorized pre reduce bias is unexpectedly missing")
    module.reduce.bias.zero_()
    module.expand.weight.copy_(
        expand_weight[:, :, None].to(device=module.expand.weight.device, dtype=module.expand.weight.dtype)
    )
    if module.expand.bias is None:
        raise RuntimeError("factorized pre expand bias is unexpectedly missing")
    module.expand.bias.copy_(selected_bias.to(device=module.expand.bias.device, dtype=module.expand.bias.dtype))


def require_conv_transpose1d(module: nn.Module, name: str) -> nn.ConvTranspose1d:
    if not isinstance(module, nn.ConvTranspose1d):
        raise RuntimeError(f"teacher init requires {name} to be ConvTranspose1d, got {type(module).__name__}")
    return module


def require_piper_bank(module: nn.Module, name: str) -> PiperResidualBank:
    if not isinstance(module, nn.Sequential):
        raise RuntimeError(f"teacher init requires {name} to be Sequential, got {type(module).__name__}")
    if len(module) != 1 or not isinstance(module[0], PiperResidualBank):
        raise RuntimeError(f"teacher init requires {name} to contain exactly one PiperResidualBank")
    return module[0]


def require_vits_bank(module: nn.Module, name: str) -> VitsResidualBank:
    if not isinstance(module, nn.Sequential):
        raise RuntimeError(f"teacher init requires {name} to be Sequential, got {type(module).__name__}")
    if len(module) != 1 or not isinstance(module[0], VitsResidualBank):
        raise RuntimeError(f"teacher init requires {name} to contain exactly one VitsResidualBank")
    return module[0]


def require_sequential_conv_module(module: nn.Module, name: str) -> nn.Conv1d | FactorizedConv1d:
    if not isinstance(module, nn.Sequential):
        raise RuntimeError(f"teacher init requires {name} to be Sequential, got {type(module).__name__}")
    if len(module) != 2 or not isinstance(module[1], (nn.Conv1d, FactorizedConv1d)):
        raise RuntimeError(f"teacher init requires {name} to contain activation + Conv1d/FactorizedConv1d")
    return module[1]


def initialize_piperlite4_from_teacher(
    model: DecoderStudent,
    checkpoint_path: Path,
    channels: tuple[int, ...],
    method: str,
) -> dict[str, Any]:
    if model.variant != "piperlite4":
        raise RuntimeError("initialize_piperlite4_from_teacher requires --variant piperlite4")
    if model.activation != "leaky_relu":
        raise RuntimeError("--teacher-init-checkpoint for piperlite4 currently expects --activation leaky_relu")
    if method not in {"first", "importance"}:
        raise RuntimeError(f"unsupported piperlite4 teacher init method: {method}")
    if len(channels) != 5:
        raise RuntimeError(f"piperlite4 teacher init expects five student channel widths, got {channels}")
    checkpoint = load_torch_checkpoint(checkpoint_path, "teacher-init checkpoint")
    config = checkpoint.get("config") if isinstance(checkpoint.get("config"), dict) else {}
    teacher_channels_raw = config.get("channels")
    if not isinstance(teacher_channels_raw, list) or len(teacher_channels_raw) != 5:
        raise RuntimeError(f"{checkpoint_path}: expected teacher config.channels with five entries, got {teacher_channels_raw!r}")
    teacher_channels = tuple(int(value) for value in teacher_channels_raw)
    if tuple(config.get("upsample_strides") or ()) != (8, 8, 2, 2):
        raise RuntimeError(f"{checkpoint_path}: piperlite4 requires Ryan-style upsample strides 8,8,2,2")
    if str(config.get("residual_layout") or "") != "vits_resblock1":
        raise RuntimeError(f"{checkpoint_path}: piperlite4 requires residual_layout vits_resblock1")
    if any(student > teacher for student, teacher in zip(channels, teacher_channels, strict=True)):
        raise RuntimeError(f"student channels {channels} exceed teacher channels {teacher_channels}")
    state = checkpoint_state_dict(checkpoint, checkpoint_path, key="state_dict")
    c0, c1, c2, c3, c4 = channels
    if method == "first":
        channel_indices = (
            torch.arange(c0, dtype=torch.long),
            torch.arange(c1, dtype=torch.long),
            torch.arange(c2, dtype=torch.long),
            torch.arange(c3, dtype=torch.long),
            torch.arange(c4, dtype=torch.long),
        )
    else:
        channel_indices = piper_vits_stage_importance_indices(state, channels, teacher_channels)
    idx0, idx1, idx2, idx3, idx4 = channel_indices
    copied = 0
    with torch.no_grad():
        pre = require_conv1d(model.pre, "pre")
        copy_selected_parameter(pre.weight, state, "conv_pre.weight", (idx0, slice(None), slice(None)))
        copy_selected_parameter(pre.bias, state, "conv_pre.bias", (idx0,))
        copied += 2

        upsample_specs = (
            (require_conv_transpose1d(model.up0, "up0"), "ups.0", idx0, idx1),
            (require_conv_transpose1d(model.up1, "up1"), "ups.1", idx1, idx2),
            (require_conv_transpose1d(model.up2, "up2"), "ups.2", idx2, idx3),
            (require_conv_transpose1d(model.up3, "up3"), "ups.3", idx3, idx4),
        )
        for module, prefix, in_indices, out_indices in upsample_specs:
            copy_selected_parameter(module.weight, state, f"{prefix}.weight", (in_indices, out_indices, slice(None)))
            copy_selected_parameter(module.bias, state, f"{prefix}.bias", (out_indices,))
            copied += 2

        residual_specs = (
            (require_vits_bank(model.res0, "res0"), 0, idx1),
            (require_vits_bank(model.res1, "res1"), 1, idx2),
            (require_vits_bank(model.res2, "res2"), 2, idx3),
            (require_vits_bank(model.res3, "res3"), 3, idx4),
        )
        for bank, stage_index, indices in residual_specs:
            if not bank.blocks:
                raise RuntimeError(f"stage {stage_index}: expected at least one VITS residual branch")
            for local_branch_index, block in enumerate(bank.blocks):
                branch_index = int(bank.source_branch_indices[local_branch_index])
                for group_name, conv_group in (("convs1", block.convs1), ("convs2", block.convs2)):
                    for conv_index, conv_module in enumerate(conv_group):
                        conv = require_sequential_conv_module(
                            conv_module,
                            f"stage {stage_index} branch {branch_index} {group_name}.{conv_index}",
                        )
                        weight_key = f"stages.{stage_index}.branches.{branch_index}.{group_name}.{conv_index}.weight"
                        bias_key = f"stages.{stage_index}.branches.{branch_index}.{group_name}.{conv_index}.bias"
                        if isinstance(conv, FactorizedConv1d):
                            init_factorized_conv1d_from_selected_weight(
                                conv,
                                state,
                                weight_key,
                                bias_key,
                                (indices, indices, slice(None)),
                            )
                        else:
                            copy_selected_parameter(conv.weight, state, weight_key, (indices, indices, slice(None)))
                            copy_selected_parameter(conv.bias, state, bias_key, (indices,))
                        copied += 2

        post = require_conv1d(model.post, "post")
        copy_selected_parameter(post.weight, state, "conv_post.weight", (slice(None), idx4, slice(None)))
        copied += 1
        if post.bias is not None:
            post.bias.zero_()
        for name, parameter in model.named_parameters():
            if not torch.isfinite(parameter).all().item():
                raise RuntimeError(f"non-finite parameter after piperlite4 teacher init: {name}")
    return {
        "checkpoint": str(checkpoint_path),
        "method": method,
        "teacher_channels": list(teacher_channels),
        "student_channels": list(channels),
        "selected_channel_indices": [indices.tolist() for indices in channel_indices],
        "copied_tensors": int(copied),
        "zeroed_extra_tensors": int(1),
        "post_filter_random_init": bool(not isinstance(model.post_filter, nn.Identity)),
        "source_parameter_count": int(sum(tensor.numel() for tensor in state.values())),
        "source_architecture": str(config.get("architecture") or ""),
        "source_residual_layout": str(config.get("residual_layout") or ""),
        "source_upsample_strides": list(config.get("upsample_strides") or []),
    }


def initialize_piperlite_from_teacher(
    model: DecoderStudent,
    checkpoint_path: Path,
    channels: tuple[int, ...],
    method: str,
) -> dict[str, Any]:
    if model.variant == "piperlite4":
        return initialize_piperlite4_from_teacher(model, checkpoint_path, channels, method)
    if model.variant not in {"piperlite", "pb", "piperfold", "piperphase"}:
        raise RuntimeError(
            "--teacher-init-checkpoint only supports --variant piperlite, pb, piperlite4, piperfold, or piperphase"
        )
    if model.activation != "leaky_relu":
        raise RuntimeError("--teacher-init-checkpoint currently expects --activation leaky_relu")
    has_post_filter = not isinstance(model.post_filter, nn.Identity)
    teacher_channels = (256, 128, 64, 32)
    if len(channels) != 4:
        raise RuntimeError(f"{model.variant} teacher init expects four student channel widths, got {channels}")
    if any(student > teacher for student, teacher in zip(channels, teacher_channels, strict=True)):
        raise RuntimeError(f"student channels {channels} exceed teacher channels {teacher_channels}")
    if method not in {"first", "importance"}:
        raise RuntimeError(f"unsupported teacher init method: {method}")
    checkpoint = load_torch_checkpoint(checkpoint_path, "teacher-init checkpoint")
    state = checkpoint_state_dict(checkpoint, checkpoint_path, key="state_dict")
    copied = 0
    c0, c1, c2, c3 = channels
    if method == "first":
        channel_indices = (
            torch.arange(c0, dtype=torch.long),
            torch.arange(c1, dtype=torch.long),
            torch.arange(c2, dtype=torch.long),
            torch.arange(c3, dtype=torch.long),
        )
    else:
        channel_indices = piper_stage_importance_indices(state, channels, teacher_channels)
    idx0, idx1, idx2, idx3 = channel_indices
    with torch.no_grad():
        if isinstance(model.pre, FactorizedConv1d):
            init_factorized_conv1d_from_selected_weight(
                model.pre,
                state,
                "conv_pre.weight",
                "conv_pre.bias",
                (idx0, slice(None), slice(None)),
            )
        else:
            pre = require_conv1d(model.pre, "pre")
            copy_selected_parameter(pre.weight, state, "conv_pre.weight", (idx0, slice(None), slice(None)))
            copy_selected_parameter(pre.bias, state, "conv_pre.bias", (idx0,))
        copied += 2

        up0 = require_conv_transpose1d(model.up0, "up0")
        up1 = require_conv_transpose1d(model.up1, "up1")
        upsample_specs = [
            (up0, "ups.0", idx0, idx1),
            (up1, "ups.1", idx1, idx2),
        ]
        if model.variant in {"piperlite", "pb", "piperfold"}:
            if model.up2 is None:
                raise RuntimeError(f"{model.variant} teacher init requires up2")
            up2 = require_conv_transpose1d(model.up2, "up2")
            upsample_specs.append((up2, "ups.2", idx2, idx3))
        for module, prefix, in_indices, out_indices in upsample_specs:
            copy_selected_parameter(
                module.weight,
                state,
                f"{prefix}.weight",
                (in_indices, out_indices, slice(None)),
            )
            copy_selected_parameter(module.bias, state, f"{prefix}.bias", (out_indices,))
            copied += 2

        residual_specs = (
            (require_piper_bank(model.res0, "res0"), 0, idx1),
            (require_piper_bank(model.res1, "res1"), 1, idx2),
        )
        if model.variant in {"piperlite", "pb"}:
            if model.res2 is None:
                raise RuntimeError(f"{model.variant} teacher init requires res2")
            residual_specs = residual_specs + ((require_piper_bank(model.res2, "res2"), 2, idx3),)
        for spec in residual_specs:
            bank, stage_index, indices = spec
            if not bank.blocks:
                raise RuntimeError(f"stage {stage_index}: expected at least one Piper residual branch")
            for local_branch_index, block in enumerate(bank.blocks):
                branch_index = int(bank.source_branch_indices[local_branch_index])
                conv_specs = (
                    (block.conv1, 0),
                    (block.conv2, 1),
                )
                for conv, conv_index in conv_specs:
                    weight_key = f"stages.{stage_index}.branches.{branch_index}.convs.{conv_index}.weight"
                    bias_key = f"stages.{stage_index}.branches.{branch_index}.convs.{conv_index}.bias"
                    if isinstance(conv, FactorizedConv1d):
                        init_factorized_conv1d_from_selected_weight(
                            conv,
                            state,
                            weight_key,
                            bias_key,
                            (indices, indices, slice(None)),
                        )
                    elif isinstance(conv, nn.Conv1d):
                        copy_selected_parameter(
                            conv.weight,
                            state,
                            weight_key,
                            (indices, indices, slice(None)),
                        )
                        copy_selected_parameter(
                            conv.bias,
                            state,
                            bias_key,
                            (indices,),
                        )
                    else:
                        raise RuntimeError(
                            f"stage {stage_index} branch {branch_index} conv {conv_index}: "
                            f"unsupported module for teacher init: {type(conv).__name__}"
                        )
                    copied += 2

        post = require_conv1d(model.post, "post")
        if model.variant in {"piperlite", "pb", "piperfold"}:
            copy_selected_parameter(post.weight, state, "conv_post.weight", (slice(None), idx3, slice(None)))
            copied += 1
        else:
            copied += 0
        zeroed = 0
        if post.bias is not None:
            post.bias.zero_()
            zeroed += 1
        for name, parameter in model.named_parameters():
            if not torch.isfinite(parameter).all().item():
                raise RuntimeError(f"non-finite parameter after teacher init: {name}")
    config = checkpoint.get("config") if isinstance(checkpoint.get("config"), dict) else {}
    return {
        "checkpoint": str(checkpoint_path),
        "method": method,
        "teacher_channels": list(teacher_channels),
        "student_channels": list(channels),
        "selected_channel_indices": [indices.tolist() for indices in channel_indices],
        "copied_tensors": int(copied),
        "zeroed_extra_tensors": int(zeroed),
        "post_filter_random_init": bool(has_post_filter),
        "folded_stage2_residual": bool(model.variant == "piperfold"),
        "phase_head_random_init": bool(model.variant == "piperphase"),
        "source_parameter_count": int(sum(tensor.numel() for tensor in state.values())),
        "source_architecture": str(config.get("architecture") or ""),
    }


def init_decoder_student_from_checkpoint(
    model: DecoderStudent,
    checkpoint_path: Path,
    *,
    expected_in_channels: int,
    expected_channels: tuple[int, int, int, int],
    expected_res_layers: int,
    expected_variant: str,
    expected_rank_ratio: float,
    expected_activation: str,
    expected_post_filter_channels: int,
    expected_post_filter_layers: int,
    expected_post_filter_kernel: int,
    expected_post_filter_scale: float,
    expected_pre_tanh_repair_channels: int,
    expected_pre_tanh_repair_layers: int,
    expected_pre_tanh_repair_kernel: int,
    expected_pre_tanh_repair_scale: float,
    expected_istft_n_fft: int,
    expected_stage_affine: bool,
    expected_factorized_pre_rank: int,
    expected_piper_res_factor_rank_ratio: float,
    expected_res_bank_scale_mode: str,
    expected_stage0_branches: tuple[int, ...],
    expected_stage1_branches: tuple[int, ...],
    expected_stage2_branches: tuple[int, ...],
    expected_stage3_branches: tuple[int, ...],
    expected_fsd_dim: int,
    expected_fsd_blocks: int,
    expected_fsd_film_rank: int,
    expected_fsd_head_rank: int,
    expected_stage_projection_bottlenecks: tuple[int, ...],
    allow_new_post_filter: bool,
    allow_new_pre_tanh_repair: bool,
    allow_leaky_to_snake: bool,
) -> dict[str, Any]:
    checkpoint = load_torch_checkpoint(checkpoint_path, "decoder init checkpoint")
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise RuntimeError(f"decoder init checkpoint missing config dict: {checkpoint_path}")
    expected: dict[str, Any] = {
        "in_channels": int(expected_in_channels),
        "channels": list(expected_channels),
        "res_layers": int(expected_res_layers),
        "variant": str(expected_variant),
        "rank_ratio": float(expected_rank_ratio),
        "activation": str(expected_activation),
        "stage_affine": bool(expected_stage_affine),
        "factorized_pre_rank": int(expected_factorized_pre_rank),
        "piper_res_factor_rank_ratio": float(expected_piper_res_factor_rank_ratio),
        "res_bank_scale_mode": str(expected_res_bank_scale_mode),
        "stage0_branches": list(expected_stage0_branches),
        "stage1_branches": list(expected_stage1_branches),
        "stage2_branches": list(expected_stage2_branches),
        "stage3_branches": list(expected_stage3_branches),
        "fsd_dim": int(expected_fsd_dim),
        "fsd_blocks": int(expected_fsd_blocks),
        "fsd_film_rank": int(expected_fsd_film_rank),
        "fsd_head_rank": int(expected_fsd_head_rank),
        "stage_projection_bottlenecks": list(expected_stage_projection_bottlenecks),
        "post_filter_channels": int(expected_post_filter_channels),
        "post_filter_layers": int(expected_post_filter_layers),
        "post_filter_kernel": int(expected_post_filter_kernel),
        "post_filter_scale": float(expected_post_filter_scale),
        "pre_tanh_repair_channels": int(expected_pre_tanh_repair_channels),
        "pre_tanh_repair_layers": int(expected_pre_tanh_repair_layers),
        "pre_tanh_repair_kernel": int(expected_pre_tanh_repair_kernel),
        "pre_tanh_repair_scale": float(expected_pre_tanh_repair_scale),
    }
    if expected_variant in {"istft", "apnetlite", "fsd", "lrc"} or "istft_n_fft" in config:
        expected["istft_n_fft"] = int(expected_istft_n_fft)
    mismatches: list[str] = []
    allowed_post_filter_mismatches: list[str] = []
    allowed_pre_tanh_repair_mismatches: list[str] = []
    allowed_activation_mismatches: list[str] = []
    post_filter_keys = {
        "post_filter_channels",
        "post_filter_layers",
        "post_filter_kernel",
        "post_filter_scale",
    }
    checkpoint_post_filter_channels = int(config.get("post_filter_channels") or 0)
    checkpoint_post_filter_layers = int(config.get("post_filter_layers") or 0)
    requested_post_filter = int(expected_post_filter_channels) > 0 and int(expected_post_filter_layers) > 0
    checkpoint_has_post_filter = checkpoint_post_filter_channels > 0 and checkpoint_post_filter_layers > 0
    pre_tanh_repair_keys = {
        "pre_tanh_repair_channels",
        "pre_tanh_repair_layers",
        "pre_tanh_repair_kernel",
        "pre_tanh_repair_scale",
    }
    checkpoint_pre_tanh_repair_channels = int(config.get("pre_tanh_repair_channels") or 0)
    checkpoint_pre_tanh_repair_layers = int(config.get("pre_tanh_repair_layers") or 0)
    requested_pre_tanh_repair = (
        int(expected_pre_tanh_repair_channels) > 0 and int(expected_pre_tanh_repair_layers) > 0
    )
    checkpoint_has_pre_tanh_repair = (
        checkpoint_pre_tanh_repair_channels > 0 and checkpoint_pre_tanh_repair_layers > 0
    )
    for key, expected_value in expected.items():
        if key in {"fsd_dim", "fsd_blocks", "fsd_film_rank", "fsd_head_rank"} and expected_variant not in {"fsd", "lrc"}:
            continue
        if key == "stage_projection_bottlenecks" and expected_variant != "pb" and not expected_value:
            continue
        if key not in config:
            if key == "stage_affine" and expected_value is False:
                continue
            if key == "factorized_pre_rank" and expected_value == 0:
                continue
            if key == "piper_res_factor_rank_ratio" and expected_value == 0.0:
                continue
            if key == "res_bank_scale_mode" and expected_value == "kept":
                continue
            if key in {"stage0_branches", "stage1_branches", "stage2_branches", "stage3_branches"} and expected_value == [0, 1, 2]:
                continue
            if key == "stage_projection_bottlenecks" and not expected_value:
                continue
            if key in post_filter_keys and not requested_post_filter and not checkpoint_has_post_filter:
                continue
            if key in pre_tanh_repair_keys and not requested_pre_tanh_repair and not checkpoint_has_pre_tanh_repair:
                continue
            if key in pre_tanh_repair_keys:
                if allow_new_pre_tanh_repair and requested_pre_tanh_repair and not checkpoint_has_pre_tanh_repair:
                    allowed_pre_tanh_repair_mismatches.append(f"{key}: missing, expected {expected_value!r}")
                    continue
                if expected_value in {0, 0.0}:
                    continue
            mismatches.append(f"{key}: missing, expected {expected_value!r}")
            continue
        actual_value = config[key]
        if key in {
            "channels",
            "stage0_branches",
            "stage1_branches",
            "stage2_branches",
            "stage3_branches",
            "stage_projection_bottlenecks",
        }:
            try:
                actual_normalized = [int(value) for value in actual_value]
            except (TypeError, ValueError) as exc:
                raise RuntimeError(f"{checkpoint_path}: config.{key} is invalid: {actual_value!r}") from exc
            if actual_normalized != expected_value:
                mismatches.append(f"{key}: got {actual_normalized!r}, expected {expected_value!r}")
        elif isinstance(expected_value, float):
            try:
                actual_float = float(actual_value)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(f"{checkpoint_path}: config.{key} is not numeric: {actual_value!r}") from exc
            if abs(actual_float - expected_value) > 1e-9:
                message = f"{key}: got {actual_float!r}, expected {expected_value!r}"
                if (
                    allow_new_post_filter
                    and key in post_filter_keys
                    and requested_post_filter
                    and not checkpoint_has_post_filter
                ):
                    allowed_post_filter_mismatches.append(message)
                elif (
                    allow_new_pre_tanh_repair
                    and key in pre_tanh_repair_keys
                    and requested_pre_tanh_repair
                    and not checkpoint_has_pre_tanh_repair
                ):
                    allowed_pre_tanh_repair_mismatches.append(message)
                else:
                    mismatches.append(message)
        elif isinstance(expected_value, int):
            try:
                actual_int = int(actual_value)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(f"{checkpoint_path}: config.{key} is not an integer: {actual_value!r}") from exc
            if actual_int != expected_value:
                message = f"{key}: got {actual_int!r}, expected {expected_value!r}"
                if (
                    allow_new_post_filter
                    and key in post_filter_keys
                    and requested_post_filter
                    and not checkpoint_has_post_filter
                ):
                    allowed_post_filter_mismatches.append(message)
                elif (
                    allow_new_pre_tanh_repair
                    and key in pre_tanh_repair_keys
                    and requested_pre_tanh_repair
                    and not checkpoint_has_pre_tanh_repair
                ):
                    allowed_pre_tanh_repair_mismatches.append(message)
                else:
                    mismatches.append(message)
        else:
            actual_str = str(actual_value)
            if actual_str != expected_value:
                message = f"{key}: got {actual_str!r}, expected {expected_value!r}"
                if (
                    allow_leaky_to_snake
                    and key == "activation"
                    and actual_str == "leaky_relu"
                    and expected_value == "snake"
                ):
                    allowed_activation_mismatches.append(message)
                else:
                    mismatches.append(message)
    if mismatches:
        joined = "; ".join(mismatches)
        raise RuntimeError(f"decoder init checkpoint config mismatch for {checkpoint_path}: {joined}")
    state = checkpoint_state_dict(checkpoint, checkpoint_path, key="model_state_dict")
    if allowed_post_filter_mismatches or allowed_pre_tanh_repair_mismatches or allowed_activation_mismatches:
        load_result = model.load_state_dict(state, strict=False)
        missing_keys = sorted(load_result.missing_keys)
        unexpected_keys = sorted(load_result.unexpected_keys)
        disallowed_missing = []
        for key in missing_keys:
            is_allowed_post_filter = bool(allowed_post_filter_mismatches) and key.startswith("post_filter.")
            is_allowed_pre_tanh_repair = (
                bool(allowed_pre_tanh_repair_mismatches) and key.startswith("pre_tanh_repair.")
            )
            is_allowed_snake_param = bool(allowed_activation_mismatches) and key.endswith(".log_alpha")
            if not is_allowed_post_filter and not is_allowed_pre_tanh_repair and not is_allowed_snake_param:
                disallowed_missing.append(key)
        if disallowed_missing or unexpected_keys:
            raise RuntimeError(
                f"failed to load decoder init checkpoint with only allowed new parameters missing: "
                f"{checkpoint_path}; missing={missing_keys}; unexpected={unexpected_keys}"
            )
    else:
        try:
            load_result = model.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(f"failed to load decoder init checkpoint strictly: {checkpoint_path}") from exc
        missing_keys = sorted(load_result.missing_keys)
        unexpected_keys = sorted(load_result.unexpected_keys)
    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter).all().item():
            raise RuntimeError(f"non-finite parameter after decoder checkpoint init: {name}")
    return {
        "checkpoint": str(checkpoint_path),
        "decoder_parameters": int(checkpoint.get("decoder_parameters") or count_parameters(model)),
        "matched_config": expected,
        "allowed_post_filter_mismatches": allowed_post_filter_mismatches,
        "allowed_pre_tanh_repair_mismatches": allowed_pre_tanh_repair_mismatches,
        "allowed_activation_mismatches": allowed_activation_mismatches,
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
    }


def decode_teacher(session: ort.InferenceSession, latent: np.ndarray) -> np.ndarray:
    input_names = [item.name for item in session.get_inputs()]
    if len(input_names) != 1:
        raise RuntimeError(f"teacher decoder session must have exactly one input, got {input_names}")
    audio = np.asarray(session.run(None, {input_names[0]: latent})[0], dtype=np.float32).reshape(-1)
    if audio.size <= 0:
        raise RuntimeError("teacher decoder returned empty audio")
    if not np.isfinite(audio).all():
        raise RuntimeError("teacher decoder returned non-finite audio")
    return audio


def decoder_cache_dir(pack_dir: Path, teacher_decoder: Path) -> Path:
    """Stable cache path for teacher-decoder chunk audio."""
    stat = teacher_decoder.stat()
    key_source = f"{teacher_decoder.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
    key = hashlib.sha1(key_source.encode("utf-8")).hexdigest()[:16]
    return pack_dir / ".decoder-audio-cache" / key


def chunk_audio_from_row_cache(
    row: dict[str, Any],
    chunk_offsets: dict[int, tuple[int, int]],
    chunk_index: int,
    expected_audio: int,
    expected_row_audio: int,
    row_audio_cache: dict[str, np.ndarray],
) -> np.ndarray | None:
    audio_path_text = str(row.get("audio") or "")
    if not audio_path_text:
        return None
    audio_path = Path(audio_path_text)
    if not audio_path.is_file():
        return None
    cache_key = str(audio_path)
    if cache_key not in row_audio_cache:
        audio, sample_rate = read_wav_float32(audio_path)
        expected_sample_rate = int(row.get("sample_rate") or 0)
        if expected_sample_rate and int(sample_rate) != expected_sample_rate:
            raise RuntimeError(
                f"{audio_path}: sample rate {sample_rate} != row sample_rate {expected_sample_rate}"
            )
        row_audio_cache[cache_key] = audio
    row_audio = row_audio_cache[cache_key]
    if int(row_audio.size) != int(expected_row_audio):
        return None
    offset = chunk_offsets.get(chunk_index)
    if offset is None:
        return None
    audio_start, audio_end = offset
    if audio_end > int(row_audio.size):
        return None
    chunk_audio = np.asarray(row_audio[audio_start:audio_end], dtype=np.float32).reshape(-1)
    if int(chunk_audio.size) != expected_audio:
        return None
    if not np.isfinite(chunk_audio).all():
        raise RuntimeError(f"{audio_path}: non-finite row-audio slice for chunk {chunk_index}")
    return chunk_audio


def load_cached_or_decode_teacher(
    session: ort.InferenceSession,
    latent: np.ndarray,
    cache_path: Path,
) -> np.ndarray:
    if cache_path.is_file():
        audio = np.asarray(np.load(cache_path), dtype=np.float32).reshape(-1)
        if audio.size <= 0:
            raise RuntimeError(f"{cache_path}: cached teacher audio is empty")
        if not np.isfinite(audio).all():
            raise RuntimeError(f"{cache_path}: cached teacher audio is non-finite")
        return audio
    audio = decode_teacher(session, latent)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with tmp_path.open("wb") as handle:
        np.save(handle, audio.astype(np.float32, copy=False))
    tmp_path.replace(cache_path)
    return audio


def load_input_target_index(target_dir: Path) -> dict[tuple[str, int], dict[str, Any]]:
    require_dir(target_dir, "input target directory")
    manifest_path = target_dir / "manifest.jsonl"
    require_file(manifest_path, "input target manifest")
    index: dict[tuple[str, int], dict[str, Any]] = {}
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{manifest_path}:{line_no}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise RuntimeError(f"{manifest_path}:{line_no}: expected object")
            row_id = str(row.get("row_id") or "")
            if not row_id:
                raise RuntimeError(f"{manifest_path}:{line_no}: missing row_id")
            try:
                chunk_index = int(row.get("chunk_index"))
            except (TypeError, ValueError) as exc:
                raise RuntimeError(f"{manifest_path}:{line_no}: invalid chunk_index") from exc
            target_npz = Path(str(row.get("target_npz") or ""))
            require_file(target_npz, "input target NPZ")
            key = (row_id, chunk_index)
            if key in index:
                raise RuntimeError(f"{manifest_path}:{line_no}: duplicate target for {row_id} chunk {chunk_index}")
            index[key] = row
    if not index:
        raise RuntimeError(f"{manifest_path}: no input targets")
    return index


def load_input_target_tensor(
    input_target_index: dict[tuple[str, int], dict[str, Any]],
    *,
    row_id: str,
    chunk_index: int,
    target_key: str,
    expected_frames: int | None,
    source_tensor_path: Path,
) -> np.ndarray:
    key = (row_id, int(chunk_index))
    row = input_target_index.get(key)
    if row is None:
        raise RuntimeError(f"{source_tensor_path}: no input target for {row_id} chunk {chunk_index}")
    target_npz = Path(str(row.get("target_npz") or ""))
    require_file(target_npz, "input target NPZ")
    with np.load(target_npz) as tensors:
        if target_key not in tensors.files:
            raise RuntimeError(f"{target_npz}: missing input target key {target_key!r}; available={tensors.files}")
        value = np.asarray(tensors[target_key], dtype=np.float32)
    if value.ndim == 1:
        value = value.reshape(-1, 1)
    if value.ndim != 2:
        raise RuntimeError(f"{target_npz}:{target_key}: expected [frames, channels], got {value.shape}")
    if expected_frames is not None and int(value.shape[0]) != int(expected_frames):
        raise RuntimeError(
            f"{target_npz}:{target_key}: frame count {value.shape[0]} != decoder frames {expected_frames}"
        )
    if int(value.shape[1]) <= 0:
        raise RuntimeError(f"{target_npz}:{target_key}: channel count must be positive, got {value.shape[1]}")
    if not np.isfinite(value).all():
        raise RuntimeError(f"{target_npz}:{target_key}: non-finite values")
    return np.transpose(value, (1, 0)).reshape(1, int(value.shape[1]), int(value.shape[0])).astype(np.float32)


def load_input_target_audio(
    row: dict[str, Any],
    *,
    expected_frames: int,
    source_tensor_path: Path,
) -> np.ndarray | None:
    raw_path = row.get("target_audio_npy")
    if raw_path is None or str(raw_path) == "":
        return None
    audio_path = Path(str(raw_path))
    require_file(audio_path, "input target audio NPY")
    audio = np.asarray(np.load(audio_path), dtype=np.float32).reshape(-1)
    expected_audio = int(expected_frames) * HOP_LENGTH
    if int(audio.size) != expected_audio:
        raise RuntimeError(
            f"{audio_path}: audio samples {audio.size} != expected {expected_audio} "
            f"for {expected_frames} latent frames from {source_tensor_path}"
        )
    if not np.isfinite(audio).all():
        raise RuntimeError(f"{audio_path}: non-finite target audio")
    return audio


def load_samples(
    pack_dir: Path,
    teacher_decoder: Path,
    *,
    input_target_index: dict[tuple[str, int], dict[str, Any]] | None = None,
    input_target_key: str = "log_mel",
) -> tuple[list[dict[str, Any]], list[ChunkSample], int]:
    require_dir(pack_dir, "pack directory")
    require_file(teacher_decoder, "teacher decoder ONNX")
    rows = read_json(pack_dir / "rows.json")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"{pack_dir / 'rows.json'} must contain a non-empty list")
    session: ort.InferenceSession | None = None
    teacher_cache_root = decoder_cache_dir(pack_dir, teacher_decoder)
    row_audio_cache: dict[str, np.ndarray] = {}

    samples: list[ChunkSample] = []
    in_channels = 0
    for row in rows:
        row_id = str(row.get("row_id") or "")
        text = str(row.get("text") or "")
        chunks = row.get("chunks")
        if not row_id or not isinstance(chunks, list) or not chunks:
            raise RuntimeError(f"invalid row metadata: {row!r}")
        chunk_offsets: dict[int, tuple[int, int]] = {}
        offset_samples = 0
        for chunk in chunks:
            chunk_index = int(chunk.get("chunk_index") or 0)
            chunk_audio_samples = int(chunk.get("audio_samples") or 0)
            if chunk_audio_samples > 0:
                chunk_offsets[chunk_index] = (offset_samples, offset_samples + chunk_audio_samples)
                offset_samples += chunk_audio_samples
        expected_row_audio = int(offset_samples)
        for chunk in chunks:
            tensor_path = Path(str(chunk.get("tensor_npz") or ""))
            require_file(tensor_path, "tensor NPZ")
            with np.load(tensor_path) as tensors:
                missing = {
                    "phoneme_ids",
                    "w_ceil",
                    "generator_input",
                } - set(tensors.files)
                if missing:
                    raise RuntimeError(f"{tensor_path}: missing tensors {sorted(missing)}")
                phoneme_ids = np.asarray(tensors["phoneme_ids"], dtype=np.int64).reshape(-1)
                durations = np.rint(np.asarray(tensors["w_ceil"], dtype=np.float32).reshape(-1)).astype(np.int64)
                teacher_latent = np.asarray(tensors["generator_input"], dtype=np.float32)
                # Stripped packs omit teacher decoder features; they are only
                # consumed by the pre/up0 feature-exact losses, so substitute
                # shape-valid zero placeholders when absent.
                if "generator_conv_pre" in tensors.files:
                    teacher_pre = np.asarray(tensors["generator_conv_pre"], dtype=np.float32)
                else:
                    teacher_pre = np.zeros((1, 1, teacher_latent.shape[2]), dtype=np.float32)
                if "generator_first_upsample" in tensors.files:
                    teacher_up0 = np.asarray(tensors["generator_first_upsample"], dtype=np.float32)
                else:
                    teacher_up0 = np.zeros((1, 1, teacher_latent.shape[2] * 8), dtype=np.float32)
            if (
                teacher_latent.ndim != 3
                or teacher_latent.shape[0] != 1
                or teacher_latent.shape[1] <= 0
                or teacher_latent.shape[2] <= 0
            ):
                raise RuntimeError(f"{tensor_path}: invalid generator_input shape {teacher_latent.shape}")
            if not np.isfinite(teacher_latent).all():
                raise RuntimeError(f"{tensor_path}: non-finite teacher latent")
            if teacher_pre.shape[0] != 1 or teacher_pre.shape[2] != teacher_latent.shape[2]:
                raise RuntimeError(f"{tensor_path}: invalid generator_conv_pre shape {teacher_pre.shape}")
            if teacher_up0.shape[0] != 1 or teacher_up0.shape[2] != teacher_latent.shape[2] * 8:
                raise RuntimeError(f"{tensor_path}: invalid generator_first_upsample shape {teacher_up0.shape}")
            if not np.isfinite(teacher_pre).all() or not np.isfinite(teacher_up0).all():
                raise RuntimeError(f"{tensor_path}: non-finite decoder teacher features")
            if durations.shape[0] != phoneme_ids.shape[0]:
                raise RuntimeError(f"{tensor_path}: duration/id length mismatch")
            if int(durations.sum()) != int(teacher_latent.shape[2]):
                raise RuntimeError(
                    f"{tensor_path}: duration sum {durations.sum()} != latent frames {teacher_latent.shape[2]}"
                )
            input_target_row: dict[str, Any] | None = None
            if input_target_index is not None:
                input_key = (row_id, int(chunk.get("chunk_index") or 0))
                input_target_row = input_target_index.get(input_key)
                if input_target_row is None:
                    raise RuntimeError(f"{tensor_path}: no input target for {row_id} chunk {input_key[1]}")
            latent = (
                load_input_target_tensor(
                    input_target_index,
                    row_id=row_id,
                    chunk_index=int(chunk.get("chunk_index") or 0),
                    target_key=str(input_target_key),
                    expected_frames=(
                        None
                        if input_target_row is not None and input_target_row.get("target_audio_npy")
                        else int(teacher_latent.shape[2])
                    ),
                    source_tensor_path=tensor_path,
                )
                if input_target_index is not None
                else teacher_latent
            )
            if latent.ndim != 3 or latent.shape[0] != 1 or latent.shape[1] <= 0 or latent.shape[2] <= 0:
                raise RuntimeError(f"{tensor_path}: invalid decoder input shape {latent.shape}")
            has_paired_target_audio = input_target_row is not None and bool(input_target_row.get("target_audio_npy"))
            if not has_paired_target_audio and int(latent.shape[2]) != int(teacher_latent.shape[2]):
                raise RuntimeError(
                    f"{tensor_path}: decoder input frames {latent.shape[2]} != teacher frames {teacher_latent.shape[2]}"
                )
            if not np.isfinite(latent).all():
                raise RuntimeError(f"{tensor_path}: non-finite decoder input")
            target_audio = (
                load_input_target_audio(
                    input_target_row,
                    expected_frames=int(latent.shape[2]),
                    source_tensor_path=tensor_path,
                )
                if input_target_row is not None
                else None
            )
            oracle_audio: np.ndarray | None = None
            if target_audio is not None:
                oracle_expected_audio = int(teacher_latent.shape[2]) * HOP_LENGTH
                oracle_audio = chunk_audio_from_row_cache(
                    row=row,
                    chunk_offsets=chunk_offsets,
                    chunk_index=int(chunk.get("chunk_index") or 0),
                    expected_audio=oracle_expected_audio,
                    expected_row_audio=expected_row_audio,
                    row_audio_cache=row_audio_cache,
                )
                if oracle_audio is None:
                    if session is None:
                        session = ort.InferenceSession(str(teacher_decoder), providers=["CPUExecutionProvider"])
                        input_names = [item.name for item in session.get_inputs()]
                        if len(input_names) != 1:
                            raise RuntimeError(
                                f"teacher decoder input mismatch: expected one latent input, got {input_names}"
                            )
                    cache_name = f"{row_id}_chunk{int(chunk.get('chunk_index') or 0):02d}.npy"
                    oracle_audio = load_cached_or_decode_teacher(
                        session=session,
                        latent=teacher_latent,
                        cache_path=teacher_cache_root / cache_name,
                    )
                if int(oracle_audio.size) != oracle_expected_audio:
                    raise RuntimeError(
                        f"{tensor_path}: oracle teacher audio {oracle_audio.size} != {oracle_expected_audio}"
                    )
            expected_audio = int(latent.shape[2]) * HOP_LENGTH
            teacher_audio = target_audio
            if teacher_audio is None:
                expected_audio = int(teacher_latent.shape[2]) * HOP_LENGTH
                teacher_audio = chunk_audio_from_row_cache(
                    row=row,
                    chunk_offsets=chunk_offsets,
                    chunk_index=int(chunk.get("chunk_index") or 0),
                    expected_audio=expected_audio,
                    expected_row_audio=expected_row_audio,
                    row_audio_cache=row_audio_cache,
                )
            if teacher_audio is None:
                if session is None:
                    session = ort.InferenceSession(str(teacher_decoder), providers=["CPUExecutionProvider"])
                    input_names = [item.name for item in session.get_inputs()]
                    if len(input_names) != 1:
                        raise RuntimeError(
                            f"teacher decoder input mismatch: expected one latent input, got {input_names}"
                        )
                cache_name = f"{row_id}_chunk{int(chunk.get('chunk_index') or 0):02d}.npy"
                teacher_audio = load_cached_or_decode_teacher(
                    session=session,
                    latent=teacher_latent,
                    cache_path=teacher_cache_root / cache_name,
                )
            if int(teacher_audio.size) != expected_audio:
                raise RuntimeError(f"{tensor_path}: teacher audio {teacher_audio.size} != {expected_audio}")
            if in_channels and int(latent.shape[1]) != in_channels:
                raise RuntimeError(
                    f"{tensor_path}: decoder input channels {latent.shape[1]} != previous {in_channels}"
                )
            in_channels = int(latent.shape[1])
            samples.append(
                ChunkSample(
                    row_id=row_id,
                    row_index=int(row.get("index") or len(samples) + 1),
                    text=text,
                    chunk_index=int(chunk.get("chunk_index") or 0),
                    phoneme_ids=phoneme_ids,
                    durations=durations,
                    latent=latent,
                    teacher_audio=teacher_audio,
                    teacher_pre=teacher_pre,
                    teacher_up0=teacher_up0,
                    tensor_path=tensor_path,
                    oracle_latent=teacher_latent if target_audio is not None else None,
                    oracle_audio=oracle_audio,
                )
            )
    if not samples:
        raise RuntimeError("pack produced no chunk samples")
    return rows, samples, in_channels


def crop_batch(
    samples: list[ChunkSample],
    *,
    batch_size: int,
    crop_frames: int,
    include_base_teacher_features: bool,
    oracle_target_mix_prob: float,
    acoustic_latent_mix_prob: float,
    acoustic_latent_residual_prob: float,
    acoustic_latent_residual_max_scale: float,
    lrc_pred_code_mix_prob: float,
    lrc_pred_code_residual_prob: float,
    lrc_pred_code_residual_max_scale: float,
    signature_keys: list[str],
    signature_phase_enabled: bool,
    exact_feature_keys: list[str],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    eligible = [
        sample
        for sample in samples
        if int(sample.latent.shape[2]) >= crop_frames
        or (sample.oracle_latent is not None and int(sample.oracle_latent.shape[2]) >= crop_frames)
    ]
    if not eligible:
        raise RuntimeError(f"no samples have at least {crop_frames} frames")
    latents: list[np.ndarray] = []
    audios: list[np.ndarray] = []
    teacher_pre: list[np.ndarray] = []
    teacher_up0: list[np.ndarray] = []
    lrc_pred_codes: list[np.ndarray] = []
    lrc_pred_code_masks: list[float] = []
    lrc_pred_code_residual_masks: list[float] = []
    lrc_pred_code_residual_scales: list[float] = []
    exact_feature_values: dict[str, list[np.ndarray]] = {key: [] for key in exact_feature_keys}
    signature_suffixes = ["mean", "logrms"]
    if signature_phase_enabled:
        signature_suffixes.extend(["phase_mean", "phase_logrms"])
    signature_values: dict[str, list[np.ndarray]] = {
        f"{key}_{suffix}": [] for key in signature_keys for suffix in signature_suffixes
    }
    for _ in range(batch_size):
        sample = random.choice(eligible)
        use_oracle_target = (
            sample.oracle_latent is not None
            and sample.oracle_audio is not None
            and int(sample.oracle_latent.shape[2]) >= crop_frames
            and random.random() < oracle_target_mix_prob
        )
        if use_oracle_target:
            base_latent = sample.oracle_latent
            target_audio = sample.oracle_audio
        else:
            if int(sample.latent.shape[2]) < crop_frames:
                if sample.oracle_latent is None or sample.oracle_audio is None:
                    raise RuntimeError(
                        f"{sample.row_id} chunk {sample.chunk_index}: sample shorter than crop and no oracle fallback"
                    )
                base_latent = sample.oracle_latent
                target_audio = sample.oracle_audio
                use_oracle_target = True
            else:
                base_latent = sample.latent
                target_audio = sample.teacher_audio
        source_latent = base_latent
        use_hard_acoustic = sample.student_latent is not None and random.random() < acoustic_latent_mix_prob
        use_residual_acoustic = (
            sample.student_latent is not None
            and not use_hard_acoustic
            and random.random() < acoustic_latent_residual_prob
        )
        if use_hard_acoustic or use_residual_acoustic:
            if sample.student_latent is None:
                raise RuntimeError(f"{sample.row_id} chunk {sample.chunk_index}: missing acoustic student latent")
            if sample.student_latent.shape != sample.latent.shape:
                raise RuntimeError(
                    f"{sample.row_id} chunk {sample.chunk_index}: acoustic latent shape "
                    f"{sample.student_latent.shape} != teacher latent shape {sample.latent.shape}"
                )
            if use_hard_acoustic:
                source_latent = sample.student_latent
            else:
                scale = random.random() * acoustic_latent_residual_max_scale
                source_latent = sample.latent + scale * (sample.student_latent - sample.latent)
        lrc_pred_enabled = lrc_pred_code_mix_prob > 0.0 or lrc_pred_code_residual_prob > 0.0
        use_lrc_pred_code = False
        use_lrc_pred_code_residual = False
        lrc_pred_code_residual_scale = 0.0
        if lrc_pred_enabled:
            if sample.lrc_pred_code is None:
                raise RuntimeError(f"{sample.row_id} chunk {sample.chunk_index}: missing cached LRC predicted code")
            if sample.lrc_pred_code.ndim != 3 or sample.lrc_pred_code.shape[0] != 1:
                raise RuntimeError(
                    f"{sample.row_id} chunk {sample.chunk_index}: invalid cached LRC predicted code "
                    f"shape {sample.lrc_pred_code.shape}"
                )
            if int(sample.lrc_pred_code.shape[2]) != int(base_latent.shape[2]):
                raise RuntimeError(
                    f"{sample.row_id} chunk {sample.chunk_index}: LRC predicted code frames "
                    f"{sample.lrc_pred_code.shape[2]} != latent frames {base_latent.shape[2]}"
                )
            use_lrc_pred_code = random.random() < lrc_pred_code_mix_prob
            use_lrc_pred_code_residual = (not use_lrc_pred_code) and random.random() < lrc_pred_code_residual_prob
            if use_lrc_pred_code_residual:
                lrc_pred_code_residual_scale = random.random() * lrc_pred_code_residual_max_scale
        frames = int(base_latent.shape[2])
        start = random.randint(0, frames - crop_frames)
        end = start + crop_frames
        audio_start = start * HOP_LENGTH
        audio_end = end * HOP_LENGTH
        latents.append(source_latent[:, :, start:end])
        audios.append(target_audio[audio_start:audio_end].reshape(1, -1))
        if lrc_pred_enabled:
            if sample.lrc_pred_code is None:
                raise RuntimeError(f"{sample.row_id} chunk {sample.chunk_index}: missing cached LRC predicted code")
            lrc_pred_codes.append(sample.lrc_pred_code[:, :, start:end])
            lrc_pred_code_masks.append(1.0 if use_lrc_pred_code else 0.0)
            lrc_pred_code_residual_masks.append(1.0 if use_lrc_pred_code_residual else 0.0)
            lrc_pred_code_residual_scales.append(float(lrc_pred_code_residual_scale))
        if include_base_teacher_features:
            teacher_pre.append(sample.teacher_pre[:, :, start:end])
            teacher_up0.append(sample.teacher_up0[:, :, start * 8 : end * 8])
        if exact_feature_keys:
            if sample.teacher_signatures is None:
                raise RuntimeError(f"{sample.row_id} chunk {sample.chunk_index}: missing exact teacher feature pack")
            for key in exact_feature_keys:
                tensor_key = f"{key}_exact"
                value = sample.teacher_signatures.get(tensor_key)
                if value is None:
                    raise RuntimeError(f"{sample.row_id} chunk {sample.chunk_index}: missing {tensor_key}")
                if value.ndim != 2:
                    raise RuntimeError(
                        f"{sample.row_id} chunk {sample.chunk_index}: invalid {tensor_key} rank {value.ndim}"
                    )
                feature_time = int(value.shape[1])
                feature_start = int(math.floor(start * feature_time / frames))
                feature_end = int(math.floor(end * feature_time / frames))
                if feature_end <= feature_start:
                    raise RuntimeError(
                        f"{sample.row_id} chunk {sample.chunk_index}: empty exact feature slice for {tensor_key} "
                        f"start={feature_start} end={feature_end} feature_time={feature_time} frames={frames}"
                    )
                exact_feature_values[key].append(value[:, feature_start:feature_end])
        if signature_keys:
            if sample.teacher_signatures is None:
                raise RuntimeError(f"{sample.row_id} chunk {sample.chunk_index}: missing teacher signatures")
            for key in signature_keys:
                for suffix in signature_suffixes:
                    tensor_key = f"{key}_{suffix}"
                    value = sample.teacher_signatures.get(tensor_key)
                    if value is None:
                        raise RuntimeError(f"{sample.row_id} chunk {sample.chunk_index}: missing {tensor_key}")
                    if value.ndim == 2:
                        signature_values[tensor_key].append(value[:, start:end])
                    elif value.ndim == 3:
                        signature_values[tensor_key].append(value[:, start:end, :])
                    else:
                        raise RuntimeError(
                            f"{sample.row_id} chunk {sample.chunk_index}: "
                            f"invalid {tensor_key} rank {value.ndim}"
                        )
    latent_tensor = torch.as_tensor(np.concatenate(latents, axis=0), dtype=torch.float32, device=device)
    audio_tensor = torch.as_tensor(np.stack(audios, axis=0), dtype=torch.float32, device=device)
    hint_tensors: dict[str, torch.Tensor] = {}
    if include_base_teacher_features:
        hint_tensors.update(
            {
                "pre": torch.as_tensor(np.concatenate(teacher_pre, axis=0), dtype=torch.float32, device=device),
                "up0": torch.as_tensor(np.concatenate(teacher_up0, axis=0), dtype=torch.float32, device=device),
            }
        )
    if lrc_pred_codes:
        hint_tensors["_lrc_pred_code"] = torch.as_tensor(
            np.concatenate(lrc_pred_codes, axis=0),
            dtype=torch.float32,
            device=device,
        )
        hint_tensors["_lrc_pred_code_mask"] = torch.as_tensor(
            np.asarray(lrc_pred_code_masks, dtype=np.float32).reshape(-1, 1, 1),
            dtype=torch.float32,
            device=device,
        )
        hint_tensors["_lrc_pred_code_residual_mask"] = torch.as_tensor(
            np.asarray(lrc_pred_code_residual_masks, dtype=np.float32).reshape(-1, 1, 1),
            dtype=torch.float32,
            device=device,
        )
        hint_tensors["_lrc_pred_code_residual_scale"] = torch.as_tensor(
            np.asarray(lrc_pred_code_residual_scales, dtype=np.float32).reshape(-1, 1, 1),
            dtype=torch.float32,
            device=device,
        )
    for key, values in exact_feature_values.items():
        if not values:
            continue
        first_shape = values[0].shape
        if any(value.shape != first_shape for value in values):
            shapes = [tuple(value.shape) for value in values]
            raise RuntimeError(f"exact feature crop shapes for {key} do not match within batch: {shapes}")
        hint_tensors[key] = torch.as_tensor(np.stack(values, axis=0), dtype=torch.float32, device=device)
    signature_tensors = {
        key: torch.as_tensor(np.stack(values, axis=0), dtype=torch.float32, device=device)
        for key, values in signature_values.items()
        if values
    }
    return latent_tensor, audio_tensor, hint_tensors, signature_tensors


def attach_acoustic_latents(
    module: Any,
    model: torch.nn.Module,
    samples: list[ChunkSample],
    device: torch.device,
) -> list[ChunkSample]:
    updated: list[ChunkSample] = []
    for index, sample in enumerate(samples, start=1):
        student_latent = predict_acoustic_latent(module, model, sample, device)
        if student_latent.shape != sample.latent.shape:
            raise RuntimeError(
                f"{sample.row_id} chunk {sample.chunk_index}: acoustic latent shape "
                f"{student_latent.shape} != teacher latent shape {sample.latent.shape}"
            )
        updated.append(replace(sample, student_latent=student_latent.astype(np.float32, copy=False)))
        if index == 1 or index % 50 == 0 or index == len(samples):
            print(f"precomputed acoustic latents {index}/{len(samples)}", flush=True)
    return updated


def multi_resolution_stft_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = prediction.squeeze(1)
    true = target.squeeze(1)
    loss = prediction.new_tensor(0.0)
    configs = ((512, 128), (1024, 256), (2048, 512))
    for n_fft, hop_length in configs:
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
    return loss / float(len(configs))


def multi_resolution_stft_phase_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = prediction.squeeze(1)
    true = target.squeeze(1)
    loss = prediction.new_tensor(0.0)
    configs = ((512, 128), (1024, 256), (2048, 512))
    for n_fft, hop_length in configs:
        window = torch.hann_window(n_fft, device=prediction.device, dtype=prediction.dtype)
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
        pred_mag = torch.abs(pred_spec).clamp_min(1e-7)
        true_mag = torch.abs(true_spec).clamp_min(1e-7)
        pred_phase = torch.stack((pred_spec.real / pred_mag, pred_spec.imag / pred_mag), dim=1)
        true_phase = torch.stack((true_spec.real / true_mag, true_spec.imag / true_mag), dim=1)
        weight = torch.log1p(true_mag)
        weight = weight / weight.detach().amax(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        weight = weight.unsqueeze(1)
        loss = loss + torch.sum(torch.abs(pred_phase - true_phase) * weight) / torch.sum(weight).clamp_min(1.0)
    return loss / float(len(configs))


def crop_or_pad_frames(value: torch.Tensor, target_frames: int) -> torch.Tensor:
    if value.ndim < 1:
        raise RuntimeError(f"expected tensor with a time axis, got {value.shape}")
    if target_frames <= 0:
        raise ValueError(f"target_frames must be positive, got {target_frames}")
    frames = int(value.shape[-1])
    if frames == target_frames:
        return value
    if frames > target_frames:
        start = max(0, (frames - target_frames) // 2)
        return value[..., start : start + target_frames]
    pad_left = (target_frames - frames) // 2
    pad_right = target_frames - frames - pad_left
    return F.pad(value, (pad_left, pad_right))


def target_stft_for_decoder(
    target: torch.Tensor,
    *,
    n_fft: int,
    target_frames: int,
) -> torch.Tensor:
    if target.ndim != 3 or target.shape[1] != 1:
        raise RuntimeError(f"expected target audio [batch, 1, samples], got {target.shape}")
    window = torch.hann_window(n_fft, device=target.device, dtype=target.dtype)
    spec = torch.stft(
        target.squeeze(1),
        n_fft=n_fft,
        hop_length=HOP_LENGTH,
        win_length=n_fft,
        window=window,
        return_complex=True,
        center=True,
    )
    return crop_or_pad_frames(spec, target_frames)


def apnetlite_losses(
    student_features: dict[str, torch.Tensor],
    target: torch.Tensor,
    *,
    n_fft: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    required = ("ap_log_amplitude", "ap_phase_unit", "ap_real", "ap_imag")
    missing = [key for key in required if key not in student_features]
    if missing:
        raise RuntimeError(f"apnetlite losses require missing feature keys: {missing}")
    student_log_amp = student_features["ap_log_amplitude"]
    student_phase = student_features["ap_phase_unit"]
    student_real = student_features["ap_real"]
    student_imag = student_features["ap_imag"]
    if student_log_amp.ndim != 3:
        raise RuntimeError(f"expected ap_log_amplitude [batch, bins, frames], got {student_log_amp.shape}")
    if student_phase.ndim != 4 or student_phase.shape[1] != 2:
        raise RuntimeError(f"expected ap_phase_unit [batch, 2, bins, frames], got {student_phase.shape}")
    target_spec = target_stft_for_decoder(target, n_fft=n_fft, target_frames=int(student_log_amp.shape[-1]))
    target_mag = torch.abs(target_spec).clamp_min(1e-7)
    target_log_amp = torch.log1p(target_mag)
    if target_log_amp.shape != student_log_amp.shape:
        raise RuntimeError(f"AP amplitude shape mismatch: {student_log_amp.shape} vs {target_log_amp.shape}")
    amplitude = F.l1_loss(student_log_amp, target_log_amp)

    target_phase = torch.stack((target_spec.real / target_mag, target_spec.imag / target_mag), dim=1)
    if target_phase.shape != student_phase.shape:
        raise RuntimeError(f"AP phase shape mismatch: {student_phase.shape} vs {target_phase.shape}")
    phase_weight = (target_log_amp / target_log_amp.detach().amax(dim=(1, 2), keepdim=True).clamp_min(1e-6)).unsqueeze(1)
    phase = torch.sum(torch.abs(student_phase - target_phase) * phase_weight) / torch.sum(phase_weight).clamp_min(1.0)

    if student_real.shape != target_spec.real.shape or student_imag.shape != target_spec.imag.shape:
        raise RuntimeError(
            "AP complex shape mismatch: "
            f"real {student_real.shape}/{target_spec.real.shape}, imag {student_imag.shape}/{target_spec.imag.shape}"
        )
    complex_loss = F.l1_loss(student_real, target_spec.real) + F.l1_loss(student_imag, target_spec.imag)
    return amplitude, phase, complex_loss


def framed_log_rms(audio: torch.Tensor, frame_size: int, frame_hop: int) -> torch.Tensor:
    return torch.log1p(framed_rms(audio, frame_size, frame_hop))


def framed_rms(audio: torch.Tensor, frame_size: int, frame_hop: int) -> torch.Tensor:
    if audio.ndim != 3 or audio.shape[1] != 1:
        raise RuntimeError(f"expected audio tensor [batch, 1, samples], got {audio.shape}")
    if frame_size <= 0:
        raise ValueError(f"frame_size must be positive, got {frame_size}")
    if frame_hop <= 0:
        raise ValueError(f"frame_hop must be positive, got {frame_hop}")
    if audio.shape[-1] < frame_size:
        raise RuntimeError(f"audio has {audio.shape[-1]} samples, shorter than frame_size {frame_size}")
    power = F.avg_pool1d(audio.square(), kernel_size=frame_size, stride=frame_hop)
    return torch.sqrt(power.clamp_min(1e-12))


def quiet_frame_mask(target_log_rms: torch.Tensor, quantile: float) -> torch.Tensor:
    if target_log_rms.ndim != 3 or target_log_rms.shape[1] != 1:
        raise RuntimeError(f"expected target frame tensor [batch, 1, frames], got {target_log_rms.shape}")
    if not (0.0 < quantile <= 1.0):
        raise ValueError(f"quiet quantile must be in (0, 1], got {quantile}")
    frames = int(target_log_rms.shape[-1])
    if frames <= 0:
        raise RuntimeError("target frame tensor has no frames")
    k = max(1, min(frames, int(math.ceil(float(quantile) * frames))))
    threshold = torch.sort(target_log_rms.detach(), dim=-1).values[..., k - 1 : k]
    return (target_log_rms.detach() <= threshold).to(dtype=target_log_rms.dtype)


def masked_l1(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.shape != mask.shape:
        raise RuntimeError(f"masked L1 shape mismatch: {prediction.shape}, {target.shape}, {mask.shape}")
    numerator = torch.sum(torch.abs(prediction - target) * mask)
    denominator = torch.sum(mask).clamp_min(1.0)
    return numerator / denominator


def quiet_frame_losses(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    quantile: float,
    frame_size: int,
    frame_hop: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    target_log_rms = framed_log_rms(target, frame_size, frame_hop)
    prediction_log_rms = framed_log_rms(prediction, frame_size, frame_hop)
    mask = quiet_frame_mask(target_log_rms, quantile)
    quiet_rms = masked_l1(prediction_log_rms, target_log_rms, mask)

    prediction_delta = prediction[..., 1:] - prediction[..., :-1]
    target_delta = target[..., 1:] - target[..., :-1]
    target_delta_log_rms = framed_log_rms(target_delta, frame_size, frame_hop)
    prediction_delta_log_rms = framed_log_rms(prediction_delta, frame_size, frame_hop)
    delta_mask = mask[..., : target_delta_log_rms.shape[-1]]
    quiet_delta = masked_l1(prediction_delta_log_rms, target_delta_log_rms, delta_mask)
    return quiet_rms, quiet_delta


def quiet_ceiling_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    quantile: float,
    frame_size: int,
    frame_hop: int,
    margin_db: float,
) -> torch.Tensor:
    if margin_db < 0.0:
        raise ValueError(f"quiet ceiling margin must be non-negative, got {margin_db}")
    target_rms = framed_rms(target, frame_size, frame_hop)
    prediction_rms = framed_rms(prediction, frame_size, frame_hop)
    mask = quiet_frame_mask(torch.log1p(target_rms), quantile)
    db_scale = 20.0 / math.log(10.0)
    excess_db = db_scale * (torch.log(prediction_rms + 1e-8) - torch.log(target_rms + 1e-8)) - float(margin_db)
    return torch.sum(F.relu(excess_db) * mask) / torch.sum(mask).clamp_min(1.0) / 20.0


def click_delta_excess_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    margin: float,
    target_scale: float,
    topk_frac: float,
) -> torch.Tensor:
    if margin < 0.0:
        raise ValueError(f"click delta margin must be non-negative, got {margin}")
    if target_scale < 0.0:
        raise ValueError(f"click delta target scale must be non-negative, got {target_scale}")
    if not (0.0 < topk_frac <= 1.0):
        raise ValueError(f"click delta top-k fraction must be in (0, 1], got {topk_frac}")
    if prediction.shape != target.shape:
        raise RuntimeError(f"click delta shape mismatch {prediction.shape} vs {target.shape}")
    if prediction.shape[-1] < 2:
        return prediction.new_tensor(0.0)
    prediction_delta = torch.abs(prediction[..., 1:] - prediction[..., :-1])
    target_delta = torch.abs(target[..., 1:] - target[..., :-1])
    threshold = target_delta * float(target_scale) + float(margin)
    excess = F.relu(prediction_delta - threshold)
    flat = excess.reshape(excess.shape[0], -1)
    k = max(1, int(math.ceil(float(flat.shape[-1]) * float(topk_frac))))
    topk = torch.topk(flat, k=k, dim=-1, largest=True).values
    return torch.mean(topk.square())


def quiet_sample_excess_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    quantile: float,
    margin: float,
    target_scale: float,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise RuntimeError(f"quiet sample shape mismatch {prediction.shape} vs {target.shape}")
    if not (0.0 < quantile <= 1.0):
        raise ValueError(f"quiet sample quantile must be in (0, 1], got {quantile}")
    if margin < 0.0:
        raise ValueError(f"quiet sample margin must be non-negative, got {margin}")
    if target_scale < 0.0:
        raise ValueError(f"quiet sample target scale must be non-negative, got {target_scale}")
    target_abs = torch.abs(target.detach())
    flat_target = target_abs.reshape(target_abs.shape[0], -1)
    k = max(1, int(math.ceil(float(flat_target.shape[-1]) * float(quantile))))
    threshold = torch.topk(flat_target, k=k, dim=-1, largest=False).values[..., -1:]
    threshold = threshold.reshape(target_abs.shape[0], 1, 1)
    mask = (target_abs <= threshold).to(dtype=prediction.dtype)
    excess = F.relu(torch.abs(prediction) - target_abs * float(target_scale) - float(margin))
    return torch.sum(excess * mask) / torch.sum(mask).clamp_min(1.0)


def high_band_excess_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    sample_rate: int,
    high_band_hz: float,
    margin_db: float,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise RuntimeError(f"high-band shape mismatch {prediction.shape} vs {target.shape}")
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")
    if high_band_hz <= 0.0:
        raise ValueError(f"high_band_hz must be positive, got {high_band_hz}")
    nyquist = float(sample_rate) / 2.0
    if high_band_hz >= nyquist:
        return prediction.new_tensor(0.0)
    if margin_db < 0.0:
        raise ValueError(f"margin_db must be non-negative, got {margin_db}")
    n_fft = 1024
    hop_length = 256
    window = torch.hann_window(n_fft, device=prediction.device, dtype=prediction.dtype)
    pred_spec = torch.stft(
        prediction.squeeze(1),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        return_complex=True,
    )
    target_spec = torch.stft(
        target.squeeze(1),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        return_complex=True,
    )
    freqs = torch.linspace(0.0, nyquist, n_fft // 2 + 1, device=prediction.device, dtype=prediction.dtype)
    high_mask = (freqs >= float(high_band_hz)).reshape(1, -1, 1).to(dtype=prediction.dtype)
    pred_power = pred_spec.abs().square()
    target_power = target_spec.abs().square()
    pred_ratio = torch.sum(pred_power * high_mask, dim=(1, 2)) / torch.sum(pred_power, dim=(1, 2)).clamp_min(1e-10)
    target_ratio = torch.sum(target_power * high_mask, dim=(1, 2)) / torch.sum(target_power, dim=(1, 2)).clamp_min(1e-10)
    margin_ratio = 10.0 ** (float(margin_db) / 10.0)
    excess_log_ratio = torch.log(pred_ratio.clamp_min(1e-10)) - torch.log((target_ratio * margin_ratio).clamp_min(1e-10))
    return torch.mean(F.relu(excess_log_ratio).square())


def echo_tail_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    sample_rate: int,
    min_lag_ms: float,
    max_lag_ms: float,
    lag_count: int,
    margin: float,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise RuntimeError(f"echo-tail shape mismatch {prediction.shape} vs {target.shape}")
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")
    if min_lag_ms <= 0.0:
        raise ValueError(f"min_lag_ms must be positive, got {min_lag_ms}")
    if max_lag_ms < min_lag_ms:
        raise ValueError(f"max_lag_ms must be >= min_lag_ms, got {max_lag_ms} < {min_lag_ms}")
    if lag_count <= 0:
        raise ValueError(f"lag_count must be positive, got {lag_count}")
    if margin < 0.0:
        raise ValueError(f"margin must be non-negative, got {margin}")
    if prediction.shape[-1] < 4:
        return prediction.new_tensor(0.0)

    residual = prediction - target
    residual = residual - residual.mean(dim=-1, keepdim=True)
    reference = target.detach() - target.detach().mean(dim=-1, keepdim=True)
    lag_values = torch.linspace(
        float(min_lag_ms),
        float(max_lag_ms),
        steps=int(lag_count),
        device=prediction.device,
        dtype=prediction.dtype,
    )
    eps = prediction.new_tensor(1e-8)
    penalties: list[torch.Tensor] = []
    for lag_ms in lag_values:
        lag = int(round(float(lag_ms.detach().cpu()) * float(sample_rate) / 1000.0))
        lag = max(1, lag)
        if lag >= prediction.shape[-1] - 1:
            continue
        tail_error = residual[..., lag:]
        original = reference[..., :-lag]
        numerator = torch.mean(tail_error * original, dim=-1)
        denominator = torch.sqrt(
            torch.mean(tail_error.square(), dim=-1).clamp_min(eps)
            * torch.mean(original.square(), dim=-1).clamp_min(eps)
        )
        correlation = numerator / denominator.clamp_min(eps)
        penalties.append(F.relu(torch.abs(correlation) - float(margin)).square().mean())
    if not penalties:
        return prediction.new_tensor(0.0)
    return torch.stack(penalties).mean()


def target_energy_gate(
    target: torch.Tensor,
    *,
    quantile: float,
    sharpness: float,
    frame_size: int,
    frame_hop: int,
) -> torch.Tensor:
    if target.ndim != 3 or target.shape[1] != 1:
        raise RuntimeError(f"expected target audio [batch, 1, samples], got {target.shape}")
    if not (0.0 < quantile < 1.0):
        raise ValueError(f"adversarial gate quantile must be in (0, 1), got {quantile}")
    if sharpness <= 0.0:
        raise ValueError(f"adversarial gate sharpness must be positive, got {sharpness}")
    target_log_rms = framed_log_rms(target.detach(), frame_size, frame_hop)
    frames = int(target_log_rms.shape[-1])
    if frames <= 0:
        raise RuntimeError("adversarial gate target has no frames")
    k = max(1, min(frames, int(math.ceil(float(quantile) * frames))))
    threshold = torch.sort(target_log_rms, dim=-1).values[..., k - 1 : k]
    gate_frames = torch.sigmoid((target_log_rms - threshold) * float(sharpness))
    return F.interpolate(gate_frames, size=int(target.shape[-1]), mode="linear", align_corners=False)


def discriminator_lsgan_loss(real_scores: list[torch.Tensor], fake_scores: list[torch.Tensor]) -> torch.Tensor:
    if len(real_scores) != len(fake_scores):
        raise RuntimeError(f"discriminator score length mismatch: {len(real_scores)} vs {len(fake_scores)}")
    if not real_scores:
        raise RuntimeError("discriminator produced no scores")
    losses: list[torch.Tensor] = []
    for real_score, fake_score in zip(real_scores, fake_scores, strict=True):
        losses.append(torch.mean((real_score - 1.0).square()) + torch.mean(fake_score.square()))
    total = losses[0]
    for loss in losses[1:]:
        total = total + loss
    return total / float(len(losses))


def generator_lsgan_loss(fake_scores: list[torch.Tensor]) -> torch.Tensor:
    if not fake_scores:
        raise RuntimeError("discriminator produced no fake scores")
    total = torch.mean((fake_scores[0] - 1.0).square())
    for fake_score in fake_scores[1:]:
        total = total + torch.mean((fake_score - 1.0).square())
    return total / float(len(fake_scores))


def discriminator_feature_matching_loss(
    real_features: list[list[torch.Tensor]],
    fake_features: list[list[torch.Tensor]],
) -> torch.Tensor:
    if len(real_features) != len(fake_features):
        raise RuntimeError(f"feature map stack mismatch: {len(real_features)} vs {len(fake_features)}")
    losses: list[torch.Tensor] = []
    for real_stack, fake_stack in zip(real_features, fake_features, strict=True):
        if len(real_stack) != len(fake_stack):
            raise RuntimeError(f"feature map layer mismatch: {len(real_stack)} vs {len(fake_stack)}")
        for real_feature, fake_feature in zip(real_stack, fake_stack, strict=True):
            if real_feature.shape != fake_feature.shape:
                raise RuntimeError(f"feature map shape mismatch: {real_feature.shape} vs {fake_feature.shape}")
            losses.append(F.l1_loss(fake_feature, real_feature.detach()))
    if not losses:
        return fake_features[0][0].new_tensor(0.0)
    total = losses[0]
    for loss in losses[1:]:
        total = total + loss
    return total / float(len(losses))


def first_difference_discriminator_audio(
    prediction: torch.Tensor,
    target: torch.Tensor,
    gate: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if prediction.shape != target.shape:
        raise RuntimeError(f"first-difference discriminator shape mismatch {prediction.shape} vs {target.shape}")
    if prediction.ndim != 3 or prediction.shape[1] != 1:
        raise RuntimeError(f"expected discriminator audio [batch, 1, samples], got {prediction.shape}")
    if int(prediction.shape[-1]) < 2:
        raise RuntimeError("first-difference discriminator requires at least two waveform samples")
    prediction_delta = prediction[..., 1:] - prediction[..., :-1]
    target_delta = target.detach()[..., 1:] - target.detach()[..., :-1]
    if gate is not None:
        if gate.shape != prediction.shape:
            raise RuntimeError(f"first-difference gate shape mismatch {gate.shape} vs {prediction.shape}")
        delta_gate = 0.5 * (gate.detach()[..., 1:] + gate.detach()[..., :-1])
        prediction_delta = prediction_delta * delta_gate
        target_delta = target_delta * delta_gate
    return prediction_delta, target_delta


def set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def standardized_time_series(value: torch.Tensor) -> torch.Tensor:
    mean = value.mean(dim=1, keepdim=True)
    std = value.std(dim=1, keepdim=True).clamp_min(1e-4)
    return (value - mean) / std


def channel_moment_hint_loss(
    student_features: dict[str, torch.Tensor],
    teacher_features: dict[str, torch.Tensor],
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for key in ("pre", "up0"):
        if key not in student_features or key not in teacher_features:
            continue
        student = student_features[key]
        teacher = teacher_features[key]
        if student.ndim != 3 or teacher.ndim != 3:
            raise RuntimeError(f"{key}: expected 3D feature tensors, got {student.shape} and {teacher.shape}")
        if student.shape[0] != teacher.shape[0] or student.shape[2] != teacher.shape[2]:
            raise RuntimeError(f"{key}: feature shape mismatch {student.shape} vs {teacher.shape}")
        student_mean = standardized_time_series(student.mean(dim=1))
        teacher_mean = standardized_time_series(teacher.mean(dim=1))
        student_rms = standardized_time_series(torch.log1p(torch.sqrt(torch.mean(student.square(), dim=1) + 1e-6)))
        teacher_rms = standardized_time_series(torch.log1p(torch.sqrt(torch.mean(teacher.square(), dim=1) + 1e-6)))
        losses.append(F.l1_loss(student_mean, teacher_mean))
        losses.append(F.l1_loss(student_rms, teacher_rms))
    if not losses:
        return next(iter(student_features.values())).new_tensor(0.0)
    total = losses[0]
    for loss in losses[1:]:
        total = total + loss
    return total / float(len(losses))


def selected_teacher_feature_loss(
    student_features: dict[str, torch.Tensor],
    teacher_features: dict[str, torch.Tensor],
    selected_channel_indices: dict[str, torch.Tensor],
    feature_keys: list[str],
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for key in feature_keys:
        student = student_features.get(key)
        teacher = teacher_features.get(key)
        selector = selected_channel_indices.get(key)
        if student is None or teacher is None:
            raise RuntimeError(f"missing exact feature inputs for {key}")
        if student.ndim != 3 or teacher.ndim != 3:
            raise RuntimeError(f"{key}: expected 3D feature tensors, got {student.shape} and {teacher.shape}")
        if student.shape[0] != teacher.shape[0] or student.shape[2] != teacher.shape[2]:
            raise RuntimeError(f"{key}: feature shape mismatch {student.shape} vs {teacher.shape}")
        if teacher.shape[1] == student.shape[1]:
            selected_teacher = teacher.detach()
        else:
            if selector is None:
                raise RuntimeError(f"{key}: missing selector for full teacher feature with shape {teacher.shape}")
            if selector.ndim != 1:
                raise RuntimeError(f"{key}: selector must be 1-D, got {tuple(selector.shape)}")
            if selector.numel() != student.shape[1]:
                raise RuntimeError(f"{key}: selector length {selector.numel()} != student channels {student.shape[1]}")
            if int(selector.min().item()) < 0 or int(selector.max().item()) >= int(teacher.shape[1]):
                raise RuntimeError(
                    f"{key}: selector out of range for teacher channels {teacher.shape[1]}: "
                    f"min={int(selector.min().item())}, max={int(selector.max().item())}"
                )
            selected_teacher = torch.index_select(teacher, 1, selector.to(device=teacher.device)).detach()
        if selected_teacher.shape != student.shape:
            raise RuntimeError(f"{key}: selected teacher shape {selected_teacher.shape} != student shape {student.shape}")
        losses.append(F.l1_loss(student, selected_teacher))
    if not losses:
        return next(iter(student_features.values())).new_tensor(0.0)
    total = losses[0]
    for loss in losses[1:]:
        total = total + loss
    return total / float(len(losses))


def pooled_activation_signature(value: torch.Tensor, latent_frames: int) -> tuple[torch.Tensor, torch.Tensor]:
    if value.ndim != 3:
        raise RuntimeError(f"expected feature [batch, channels, time], got {value.shape}")
    if latent_frames <= 0:
        raise ValueError(f"latent_frames must be positive, got {latent_frames}")
    pooled_mean = F.adaptive_avg_pool1d(value, latent_frames)
    pooled_logrms = torch.log1p(torch.sqrt(F.adaptive_avg_pool1d(value.square(), latent_frames) + 1e-12))
    return pooled_mean, pooled_logrms


def channel_summary(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if value.ndim != 3:
        raise RuntimeError(f"expected signature tensor [batch, channels, frames], got {value.shape}")
    channel_mean = standardized_time_series(value.mean(dim=1))
    channel_std = standardized_time_series(value.std(dim=1, unbiased=False))
    return channel_mean, channel_std


def decoder_signature_hint_loss(
    student_features: dict[str, torch.Tensor],
    teacher_signatures: dict[str, torch.Tensor],
    signature_keys: list[str],
    latent_frames: int,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for key in signature_keys:
        feature_key = SIGNATURE_FEATURE_MAP[key]
        student = student_features.get(feature_key)
        teacher_mean = teacher_signatures.get(f"{key}_mean")
        teacher_logrms = teacher_signatures.get(f"{key}_logrms")
        if student is None or teacher_mean is None or teacher_logrms is None:
            raise RuntimeError(f"missing signature inputs for {key}")
        student_mean, student_logrms = pooled_activation_signature(student, latent_frames)
        for student_value, teacher_value in ((student_mean, teacher_mean), (student_logrms, teacher_logrms)):
            if teacher_value.ndim != 3 or teacher_value.shape[0] != student_value.shape[0]:
                raise RuntimeError(f"{key}: teacher signature shape mismatch {teacher_value.shape}")
            if teacher_value.shape[-1] != latent_frames:
                raise RuntimeError(f"{key}: teacher signature frames {teacher_value.shape[-1]} != {latent_frames}")
            student_channel_mean, student_channel_std = channel_summary(student_value)
            teacher_channel_mean, teacher_channel_std = channel_summary(teacher_value)
            losses.append(F.l1_loss(student_channel_mean, teacher_channel_mean))
            losses.append(F.l1_loss(student_channel_std, teacher_channel_std))
    if not losses:
        return next(iter(student_features.values())).new_tensor(0.0)
    total = losses[0]
    for loss in losses[1:]:
        total = total + loss
    return total / float(len(losses))


def temporal_cosine_gram(value: torch.Tensor) -> torch.Tensor:
    if value.ndim != 3:
        raise RuntimeError(f"expected signature tensor [batch, channels, frames], got {value.shape}")
    series = value.transpose(1, 2)
    series = series - series.mean(dim=-1, keepdim=True)
    series = F.normalize(series, dim=-1, eps=1e-6)
    return torch.matmul(series, series.transpose(1, 2))


def decoder_signature_temporal_loss(
    student_features: dict[str, torch.Tensor],
    teacher_signatures: dict[str, torch.Tensor],
    signature_keys: list[str],
    latent_frames: int,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for key in signature_keys:
        feature_key = SIGNATURE_FEATURE_MAP[key]
        student = student_features.get(feature_key)
        teacher_mean = teacher_signatures.get(f"{key}_mean")
        teacher_logrms = teacher_signatures.get(f"{key}_logrms")
        if student is None or teacher_mean is None or teacher_logrms is None:
            raise RuntimeError(f"missing temporal signature inputs for {key}")
        student_mean, student_logrms = pooled_activation_signature(student, latent_frames)
        for student_value, teacher_value in ((student_mean, teacher_mean), (student_logrms, teacher_logrms)):
            if teacher_value.ndim != 3 or teacher_value.shape[0] != student_value.shape[0]:
                raise RuntimeError(f"{key}: teacher temporal signature shape mismatch {teacher_value.shape}")
            if teacher_value.shape[-1] != latent_frames:
                raise RuntimeError(f"{key}: teacher temporal signature frames {teacher_value.shape[-1]} != {latent_frames}")
            losses.append(F.l1_loss(temporal_cosine_gram(student_value), temporal_cosine_gram(teacher_value)))
    if not losses:
        return next(iter(student_features.values())).new_tensor(0.0)
    total = losses[0]
    for loss in losses[1:]:
        total = total + loss
    return total / float(len(losses))


def phase_activation_signature(value: torch.Tensor, latent_frames: int, phase_bins: int) -> tuple[torch.Tensor, torch.Tensor]:
    if value.ndim != 3:
        raise RuntimeError(f"expected feature [batch, channels, time], got {value.shape}")
    if latent_frames <= 0:
        raise ValueError(f"latent_frames must be positive, got {latent_frames}")
    if phase_bins <= 0:
        raise ValueError(f"phase_bins must be positive, got {phase_bins}")
    target_frames = latent_frames * phase_bins
    pooled_mean = F.adaptive_avg_pool1d(value, target_frames)
    pooled_logrms = torch.log1p(torch.sqrt(F.adaptive_avg_pool1d(value.square(), target_frames) + 1e-12))
    shape = (value.shape[0], value.shape[1], latent_frames, phase_bins)
    return pooled_mean.reshape(shape), pooled_logrms.reshape(shape)


def phase_channel_summary(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if value.ndim != 4:
        raise RuntimeError(f"expected phase signature [batch, channels, frames, bins], got {value.shape}")
    channel_mean = value.mean(dim=1)
    channel_std = value.std(dim=1, unbiased=False)
    channel_mean = channel_mean - channel_mean.mean(dim=(1, 2), keepdim=True)
    channel_std = channel_std - channel_std.mean(dim=(1, 2), keepdim=True)
    channel_mean = channel_mean / channel_mean.std(dim=(1, 2), keepdim=True, unbiased=False).clamp_min(1e-6)
    channel_std = channel_std / channel_std.std(dim=(1, 2), keepdim=True, unbiased=False).clamp_min(1e-6)
    return channel_mean, channel_std


class FrozenBottleneckCodebook(nn.Module):
    def __init__(self, entries: list[dict[str, Any]], device: torch.device) -> None:
        super().__init__()
        self.entries = entries
        self.encoders = nn.ModuleDict()
        for entry in entries:
            signature_key = str(entry["signature_key"])
            channels = int(entry["channels"])
            bottleneck = int(entry["bottleneck"])
            encoder = nn.Linear(channels, bottleneck)
            encoder.load_state_dict(entry["encoder_state_dict"])
            encoder.eval()
            encoder.requires_grad_(False)
            self.encoders[signature_key] = encoder
        self.to(device)
        self.eval()

    @property
    def signature_keys(self) -> list[str]:
        return [str(entry["signature_key"]) for entry in self.entries]

    @property
    def parameter_count(self) -> int:
        return count_parameters(self)


def normalize_bottleneck_signature_key(raw_key: str) -> str:
    key = raw_key.strip()
    if key.endswith("_mean"):
        key = key[: -len("_mean")]
    if not key:
        raise ValueError(f"invalid empty bottleneck signature key from {raw_key!r}")
    if key not in SIGNATURE_FEATURE_MAP:
        raise ValueError(
            f"unsupported bottleneck signature key {key!r} from {raw_key!r}; "
            f"supported={sorted(SIGNATURE_FEATURE_MAP)}"
        )
    return key


def load_bottleneck_codebook(
    checkpoint_path: Path,
    requested_keys: list[str] | None,
    device: torch.device,
) -> FrozenBottleneckCodebook:
    require_file(checkpoint_path, "bottleneck code checkpoint")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"{checkpoint_path}: expected checkpoint object, got {type(checkpoint).__name__}")
    if checkpoint.get("format") != "roota_decoder_activation_bottlenecks_v1":
        raise RuntimeError(
            f"{checkpoint_path}: unsupported bottleneck checkpoint format {checkpoint.get('format')!r}"
        )
    raw_encoders = checkpoint.get("encoders")
    if not isinstance(raw_encoders, dict) or not raw_encoders:
        raise RuntimeError(f"{checkpoint_path}: missing non-empty encoders mapping")
    available_by_signature: dict[str, dict[str, Any]] = {}
    for raw_key, raw_entry in raw_encoders.items():
        if not isinstance(raw_entry, dict):
            raise RuntimeError(f"{checkpoint_path}: encoder entry {raw_key!r} is not an object")
        signature_key = normalize_bottleneck_signature_key(str(raw_key))
        entry_key = normalize_bottleneck_signature_key(str(raw_entry.get("key", raw_key)))
        if entry_key != signature_key:
            raise RuntimeError(
                f"{checkpoint_path}: encoder key mismatch {raw_key!r}: entry key {entry_key!r} != {signature_key!r}"
            )
        channels = int(raw_entry.get("channels", 0))
        bottleneck = int(raw_entry.get("bottleneck", 0))
        if channels <= 0 or bottleneck <= 0 or bottleneck > channels:
            raise RuntimeError(
                f"{checkpoint_path}: invalid bottleneck shape for {raw_key!r}: "
                f"channels={channels}, bottleneck={bottleneck}"
            )
        state = raw_entry.get("encoder_state_dict")
        if not isinstance(state, dict):
            raise RuntimeError(f"{checkpoint_path}: encoder {raw_key!r} missing encoder_state_dict")
        available_by_signature[signature_key] = {
            "raw_key": str(raw_key),
            "signature_key": signature_key,
            "feature_key": SIGNATURE_FEATURE_MAP[signature_key],
            "channels": channels,
            "bottleneck": bottleneck,
            "encoder_state_dict": state,
        }
    selected_keys = requested_keys if requested_keys is not None else sorted(available_by_signature)
    if not selected_keys:
        raise RuntimeError(f"{checkpoint_path}: no bottleneck code keys selected")
    missing = sorted(set(selected_keys) - set(available_by_signature))
    if missing:
        raise RuntimeError(
            f"{checkpoint_path}: requested bottleneck keys missing {missing}; "
            f"available={sorted(available_by_signature)}"
        )
    entries = [available_by_signature[key] for key in selected_keys]
    return FrozenBottleneckCodebook(entries, device)


def decoder_bottleneck_code_loss(
    student_features: dict[str, torch.Tensor],
    teacher_signatures: dict[str, torch.Tensor],
    codebook: FrozenBottleneckCodebook,
    latent_frames: int,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for entry in codebook.entries:
        signature_key = str(entry["signature_key"])
        feature_key = str(entry["feature_key"])
        student = student_features.get(feature_key)
        teacher_mean = teacher_signatures.get(f"{signature_key}_mean")
        if student is None or teacher_mean is None:
            raise RuntimeError(f"missing bottleneck-code inputs for {signature_key}")
        if teacher_mean.ndim != 3:
            raise RuntimeError(f"{signature_key}: expected teacher mean [batch, channels, frames], got {teacher_mean.shape}")
        if int(teacher_mean.shape[-1]) != latent_frames:
            raise RuntimeError(
                f"{signature_key}: teacher signature frames {teacher_mean.shape[-1]} != {latent_frames}"
            )
        student_mean, _student_logrms = pooled_activation_signature(student, latent_frames)
        expected_channels = int(entry["channels"])
        bottleneck = int(entry["bottleneck"])
        if int(teacher_mean.shape[1]) != expected_channels:
            raise RuntimeError(
                f"{signature_key}: teacher channels {teacher_mean.shape[1]} != encoder channels {expected_channels}"
            )
        if int(student_mean.shape[1]) != bottleneck:
            raise RuntimeError(
                f"{signature_key}: student feature {feature_key} pooled channels {student_mean.shape[1]} "
                f"!= bottleneck {bottleneck}"
            )
        encoder = codebook.encoders[signature_key]
        teacher_frames = teacher_mean.transpose(1, 2).contiguous()
        with torch.no_grad():
            teacher_code = encoder(teacher_frames).transpose(1, 2).contiguous()
        student_frames = student_mean.transpose(1, 2).reshape(-1, bottleneck)
        teacher_frames_code = teacher_code.transpose(1, 2).reshape(-1, bottleneck)
        losses.append(F.l1_loss(student_mean, teacher_code))
        losses.append(
            1.0
            - F.cosine_similarity(
                student_frames,
                teacher_frames_code,
                dim=1,
                eps=1e-6,
            ).mean()
        )
    if not losses:
        return next(iter(student_features.values())).new_tensor(0.0)
    total = losses[0]
    for loss in losses[1:]:
        total = total + loss
    return total / float(len(losses))


def decoder_signature_phase_loss(
    student_features: dict[str, torch.Tensor],
    teacher_signatures: dict[str, torch.Tensor],
    signature_keys: list[str],
    latent_frames: int,
    phase_bins: int,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for key in signature_keys:
        feature_key = SIGNATURE_FEATURE_MAP[key]
        student = student_features.get(feature_key)
        teacher_mean = teacher_signatures.get(f"{key}_phase_mean")
        teacher_logrms = teacher_signatures.get(f"{key}_phase_logrms")
        if student is None or teacher_mean is None or teacher_logrms is None:
            raise RuntimeError(f"missing phase signature inputs for {key}")
        student_mean, student_logrms = phase_activation_signature(student, latent_frames, phase_bins)
        for student_value, teacher_value in ((student_mean, teacher_mean), (student_logrms, teacher_logrms)):
            if teacher_value.ndim != 4 or teacher_value.shape[0] != student_value.shape[0]:
                raise RuntimeError(f"{key}: teacher phase signature shape mismatch {teacher_value.shape}")
            if teacher_value.shape[-2] != latent_frames:
                raise RuntimeError(f"{key}: teacher phase frames {teacher_value.shape[-2]} != {latent_frames}")
            if teacher_value.shape[-1] != phase_bins:
                raise RuntimeError(f"{key}: teacher phase bins {teacher_value.shape[-1]} != {phase_bins}")
            student_channel_mean, student_channel_std = phase_channel_summary(student_value)
            teacher_channel_mean, teacher_channel_std = phase_channel_summary(teacher_value)
            losses.append(F.l1_loss(student_channel_mean, teacher_channel_mean))
            losses.append(F.l1_loss(student_channel_std, teacher_channel_std))
    if not losses:
        return next(iter(student_features.values())).new_tensor(0.0)
    total = losses[0]
    for loss in losses[1:]:
        total = total + loss
    return total / float(len(losses))


def import_latent_student_module() -> Any:
    script = ROOT / "tools" / "train_roota_piper_latent_student.py"
    require_file(script, "acoustic training script")
    module_name = "roota_latent_student_mod"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_acoustic_model(checkpoint_path: Path, device: torch.device) -> tuple[Any, torch.nn.Module]:
    require_file(checkpoint_path, "acoustic checkpoint")
    module = import_latent_student_module()
    model, _config = module.load_model_from_checkpoint(checkpoint_path, device)
    model.eval()
    return module, model


def load_lrc_pred_code_model(checkpoint_path: Path, device: torch.device) -> tuple[Any, torch.nn.Module, dict[str, Any]]:
    require_file(checkpoint_path, "LRC predicted-code acoustic checkpoint")
    module = import_latent_student_module()
    model, config = module.load_model_from_checkpoint(checkpoint_path, device)
    if not isinstance(config, dict):
        raise RuntimeError(f"{checkpoint_path}: missing c-acoustic config")
    model.eval()
    return module, model, config


@torch.no_grad()
def predict_acoustic_latent(module: Any, model: torch.nn.Module, sample: ChunkSample, device: torch.device) -> np.ndarray:
    shim = module.ChunkSample(
        row_id=sample.row_id,
        row_index=sample.row_index,
        text=sample.text,
        chunk_index=sample.chunk_index,
        phoneme_ids=sample.phoneme_ids,
        durations=sample.durations,
        target=np.zeros((int(sample.latent.shape[2]), int(sample.latent.shape[1])), dtype=np.float32),
        tensor_path=sample.tensor_path,
        audio_samples=int(sample.teacher_audio.size),
    )
    latent = module.predict_chunk(model, shim, device)
    if latent.shape != sample.latent.shape:
        raise RuntimeError(f"acoustic prediction shape {latent.shape} != teacher latent {sample.latent.shape}")
    return latent


def lrc_pred_code_checkpoint_metadata(checkpoint_path: Path) -> dict[str, Any]:
    require_file(checkpoint_path, "LRC predicted-code acoustic checkpoint")
    stat = checkpoint_path.stat()
    return {
        "path": str(checkpoint_path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": sha256_file(checkpoint_path),
    }


def lrc_pred_code_cache_dir(
    out_dir: Path,
    checkpoint_metadata: dict[str, Any],
    code_dim: int,
) -> tuple[Path, str]:
    key = hashlib.sha1(
        stable_json(
            {
                "format": "roota_lrc_pred_code_cache_v1",
                "checkpoint": checkpoint_metadata,
                "code_dim": int(code_dim),
            }
        ).encode("utf-8")
    ).hexdigest()[:16]
    return out_dir / "lrc-pred-code-cache" / f"{key}-c{int(code_dim)}", key


def safe_cache_stem(value: str) -> str:
    stem = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)
    return stem[:80] or "row"


def lrc_pred_code_sample_metadata(
    sample: ChunkSample,
    *,
    checkpoint_key: str,
    code_dim: int,
) -> dict[str, Any]:
    stat = sample.tensor_path.stat()
    phoneme_ids = np.ascontiguousarray(sample.phoneme_ids.astype(np.int64, copy=False))
    durations = np.ascontiguousarray(sample.durations.astype(np.int64, copy=False))
    return {
        "format": "roota_lrc_pred_code_cache_v1",
        "checkpoint_key": str(checkpoint_key),
        "row_id": str(sample.row_id),
        "row_index": int(sample.row_index),
        "chunk_index": int(sample.chunk_index),
        "tensor_path": str(sample.tensor_path.resolve()),
        "tensor_size": int(stat.st_size),
        "tensor_mtime_ns": int(stat.st_mtime_ns),
        "frames": int(sample.latent.shape[2]),
        "source_channels": int(sample.latent.shape[1]),
        "code_dim": int(code_dim),
        "phoneme_sha256": hashlib.sha256(phoneme_ids.tobytes()).hexdigest(),
        "durations_sha256": hashlib.sha256(durations.tobytes()).hexdigest(),
    }


def lrc_pred_code_cache_path(cache_dir: Path, sample: ChunkSample, metadata: dict[str, Any]) -> Path:
    key = hashlib.sha1(stable_json(metadata).encode("utf-8")).hexdigest()[:20]
    return cache_dir / f"{safe_cache_stem(sample.row_id)}_chunk{int(sample.chunk_index):02d}_{key}.npz"


def load_cached_lrc_pred_code(
    cache_path: Path,
    *,
    expected_metadata: dict[str, Any],
    expected_shape: tuple[int, int, int],
) -> np.ndarray | None:
    if not cache_path.is_file():
        return None
    expected_metadata_json = stable_json(expected_metadata)
    with np.load(cache_path) as tensors:
        if "metadata_json" not in tensors.files or "lrc_pred_code" not in tensors.files:
            return None
        metadata_json = str(np.asarray(tensors["metadata_json"]).reshape(()).item())
        if metadata_json != expected_metadata_json:
            return None
        code = np.asarray(tensors["lrc_pred_code"], dtype=np.float32)
    if tuple(code.shape) != expected_shape:
        raise RuntimeError(f"{cache_path}: cached LRC predicted code shape {code.shape} != {expected_shape}")
    if not np.isfinite(code).all():
        raise RuntimeError(f"{cache_path}: cached LRC predicted code contains non-finite values")
    return np.ascontiguousarray(code, dtype=np.float32)


def write_lrc_pred_code_cache(cache_path: Path, code: np.ndarray, metadata: dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with tmp_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            lrc_pred_code=np.ascontiguousarray(code, dtype=np.float32),
            metadata_json=np.asarray(stable_json(metadata)),
        )
    tmp_path.replace(cache_path)


@torch.no_grad()
def predict_lrc_pred_code(
    module: Any,
    model: torch.nn.Module,
    sample: ChunkSample,
    *,
    code_dim: int,
    device: torch.device,
) -> np.ndarray:
    frames = int(sample.latent.shape[2])
    shim = module.ChunkSample(
        row_id=sample.row_id,
        row_index=sample.row_index,
        text=sample.text,
        chunk_index=sample.chunk_index,
        phoneme_ids=sample.phoneme_ids,
        durations=sample.durations,
        target=np.zeros((frames, int(code_dim)), dtype=np.float32),
        tensor_path=sample.tensor_path,
        audio_samples=int(sample.teacher_audio.size),
    )
    features = module.expand_features(shim, device)
    prediction = module.predict_latent_tensor(model, features)
    if prediction.ndim != 2:
        raise RuntimeError(f"c-acoustic prediction must be [T, C], got {prediction.shape}")
    if int(prediction.shape[0]) != frames or int(prediction.shape[1]) != int(code_dim):
        raise RuntimeError(
            f"{sample.row_id} chunk {sample.chunk_index}: c-acoustic prediction shape "
            f"{tuple(prediction.shape)} != ({frames}, {int(code_dim)})"
        )
    code = prediction.transpose(0, 1).unsqueeze(0).detach().cpu().numpy().astype(np.float32)
    if not np.isfinite(code).all():
        raise RuntimeError(f"{sample.row_id} chunk {sample.chunk_index}: c-acoustic prediction is non-finite")
    return np.ascontiguousarray(code, dtype=np.float32)


def attach_lrc_pred_code_cache(
    module: Any,
    model: torch.nn.Module,
    samples: list[ChunkSample],
    *,
    device: torch.device,
    out_dir: Path,
    checkpoint_path: Path,
    code_dim: int,
) -> tuple[list[ChunkSample], dict[str, Any]]:
    checkpoint_metadata = lrc_pred_code_checkpoint_metadata(checkpoint_path)
    cache_dir, checkpoint_key = lrc_pred_code_cache_dir(out_dir, checkpoint_metadata, int(code_dim))
    updated: list[ChunkSample] = []
    created = 0
    reused = 0
    for index, sample in enumerate(samples, start=1):
        metadata = lrc_pred_code_sample_metadata(
            sample,
            checkpoint_key=checkpoint_key,
            code_dim=int(code_dim),
        )
        cache_path = lrc_pred_code_cache_path(cache_dir, sample, metadata)
        expected_shape = (1, int(code_dim), int(sample.latent.shape[2]))
        code = load_cached_lrc_pred_code(
            cache_path,
            expected_metadata=metadata,
            expected_shape=expected_shape,
        )
        status = "reused"
        if code is None:
            code = predict_lrc_pred_code(module, model, sample, code_dim=int(code_dim), device=device)
            if tuple(code.shape) != expected_shape:
                raise RuntimeError(
                    f"{sample.row_id} chunk {sample.chunk_index}: predicted code shape "
                    f"{code.shape} != {expected_shape}"
                )
            write_lrc_pred_code_cache(cache_path, code, metadata)
            created += 1
            status = "created"
        else:
            reused += 1
        updated.append(replace(sample, lrc_pred_code=code))
        if index == 1 or index % 50 == 0 or index == len(samples):
            print(
                json.dumps(
                    {
                        "lrc_pred_code_cache_progress": {
                            "status": status,
                            "index": int(index),
                            "total": int(len(samples)),
                            "created": int(created),
                            "reused": int(reused),
                            "cache_dir": str(cache_dir),
                        }
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    summary = {
        "format": "roota_lrc_pred_code_cache_v1",
        "cache_dir": str(cache_dir),
        "checkpoint": str(checkpoint_path),
        "checkpoint_key": checkpoint_key,
        "checkpoint_sha256": str(checkpoint_metadata["sha256"]),
        "code_dim": int(code_dim),
        "samples": int(len(samples)),
        "created": int(created),
        "reused": int(reused),
    }
    write_json(cache_dir / "summary.json", summary)
    print(json.dumps({"lrc_pred_code_cache": summary}, ensure_ascii=False), flush=True)
    return updated, summary


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


def cosine_np_or_none(a: np.ndarray, b: np.ndarray) -> float | None:
    a_f = np.asarray(a, dtype=np.float64).reshape(-1)
    b_f = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(a_f) * np.linalg.norm(b_f))
    if denom <= 0:
        return None
    return float(np.dot(a_f, b_f) / denom)


def html_audio_src(path: Path, base_dir: Path) -> str:
    try:
        return Path(os.path.relpath(path.resolve(), base_dir.resolve())).as_posix()
    except ValueError:
        return path.resolve().as_uri()


@torch.no_grad()
def decode_with_student(
    model: DecoderStudent,
    latent: np.ndarray,
    device: torch.device,
    lrc_encoder: LrcEncoder | None = None,
) -> np.ndarray:
    latent_tensor = torch.as_tensor(latent, dtype=torch.float32, device=device)
    if lrc_encoder is not None:
        latent_tensor = lrc_encoder(latent_tensor)
    audio = model(latent_tensor).squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)
    if audio.size <= 0:
        raise RuntimeError("decoder student returned empty audio")
    if not np.isfinite(audio).all():
        raise RuntimeError("decoder student returned non-finite audio")
    return audio


@torch.no_grad()
def evaluate_chunks(
    model: DecoderStudent,
    samples: list[ChunkSample],
    device: torch.device,
    lrc_encoder: LrcEncoder | None = None,
) -> dict[str, float]:
    model.eval()
    if lrc_encoder is not None:
        lrc_encoder.eval()
    l1_values: list[float] = []
    cosine_values: list[float] = []
    rms_ratios: list[float] = []
    for sample in samples:
        predicted = decode_with_student(model, sample.latent, device, lrc_encoder=lrc_encoder)
        teacher = sample.teacher_audio
        if predicted.shape != teacher.shape:
            raise RuntimeError(f"{sample.row_id} chunk {sample.chunk_index}: audio shape mismatch")
        l1_values.append(float(np.mean(np.abs(predicted - teacher))))
        cosine_values.append(cosine_np(predicted, teacher))
        teacher_rms = audio_rms(teacher)
        rms_ratios.append(audio_rms(predicted) / teacher_rms if teacher_rms > 0 else 0.0)
    model.train()
    if lrc_encoder is not None:
        lrc_encoder.train()
    return {
        "mean_l1": float(np.mean(l1_values)),
        "max_l1": float(np.max(l1_values)),
        "mean_cosine": float(np.mean(cosine_values)),
        "min_cosine": float(np.min(cosine_values)),
        "mean_rms_ratio": float(np.mean(rms_ratios)),
    }


@torch.no_grad()
def render_dashboard(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    samples: list[ChunkSample],
    model: DecoderStudent,
    device: torch.device,
    dataset_label: str,
    lrc_encoder: LrcEncoder | None = None,
) -> dict[str, Any]:
    sample_rate = int(rows[0].get("sample_rate") or 22050)
    silence = np.zeros(int(round(sample_rate * args.sentence_silence)), dtype=np.float32)
    by_row: dict[str, list[ChunkSample]] = {}
    for sample in samples:
        by_row.setdefault(sample.row_id, []).append(sample)

    acoustic_module = None
    acoustic_model = None
    if args.acoustic_checkpoint:
        acoustic_module, acoustic_model = load_acoustic_model(args.acoustic_checkpoint, device)

    audio_dir = args.out_dir / "audio"
    rendered: list[dict[str, Any]] = []
    oracle_cosines: list[float] = []
    stack_cosines: list[float] = []
    for row in rows[: args.render_rows]:
        row_id = str(row["row_id"])
        row_chunks = sorted(by_row[row_id], key=lambda item: item.chunk_index)
        oracle_parts: list[np.ndarray] = []
        stack_parts: list[np.ndarray] = []
        for index, sample in enumerate(row_chunks):
            if index > 0:
                oracle_parts.append(silence)
                if acoustic_model is not None and acoustic_module is not None:
                    stack_parts.append(silence)
            oracle_parts.append(decode_with_student(model, sample.latent, device, lrc_encoder=lrc_encoder))
            if acoustic_model is not None and acoustic_module is not None:
                acoustic_latent = predict_acoustic_latent(acoustic_module, acoustic_model, sample, device)
                stack_parts.append(decode_with_student(model, acoustic_latent, device, lrc_encoder=lrc_encoder))
        oracle_audio = np.concatenate(oracle_parts) if oracle_parts else np.zeros(0, dtype=np.float32)
        stack_audio = np.concatenate(stack_parts) if stack_parts else np.zeros(0, dtype=np.float32)
        teacher_path = Path(str(row["audio"]))
        teacher_audio, _ = read_wav_float32(teacher_path)

        oracle_path = audio_dir / f"{row_id}_decoder_oracle.wav"
        stack_path = audio_dir / f"{row_id}_acoustic_decoder_stack.wav"
        write_wav(oracle_path, oracle_audio, sample_rate)
        if stack_audio.size > 0:
            write_wav(stack_path, stack_audio, sample_rate)

        n_oracle = min(int(teacher_audio.size), int(oracle_audio.size))
        oracle_cos = cosine_np_or_none(oracle_audio[:n_oracle], teacher_audio[:n_oracle])
        oracle_l1 = float(np.mean(np.abs(oracle_audio[:n_oracle] - teacher_audio[:n_oracle])))
        if oracle_cos is not None:
            oracle_cosines.append(oracle_cos)
        stack_cos = None
        stack_l1 = None
        if stack_audio.size > 0:
            n_stack = min(int(teacher_audio.size), int(stack_audio.size))
            stack_cos = cosine_np_or_none(stack_audio[:n_stack], teacher_audio[:n_stack])
            stack_l1 = float(np.mean(np.abs(stack_audio[:n_stack] - teacher_audio[:n_stack])))
            if stack_cos is not None:
                stack_cosines.append(stack_cos)
        rendered.append(
            {
                "row_id": row_id,
                "index": int(row["index"]),
                "text": row["text"],
                "teacher_audio": str(teacher_path),
                "oracle_audio": str(oracle_path),
                "stack_audio": str(stack_path) if stack_audio.size > 0 else None,
                "teacher_audio_src": html_audio_src(teacher_path, args.out_dir),
                "oracle_audio_src": html_audio_src(oracle_path, args.out_dir),
                "stack_audio_src": html_audio_src(stack_path, args.out_dir) if stack_audio.size > 0 else None,
                "oracle_l1": oracle_l1,
                "oracle_cosine": oracle_cos,
                "stack_l1": stack_l1,
                "stack_cosine": stack_cos,
                "teacher_rms": audio_rms(teacher_audio),
                "oracle_rms": audio_rms(oracle_audio),
                "stack_rms": audio_rms(stack_audio) if stack_audio.size > 0 else None,
            }
        )
    write_json(args.out_dir / "rendered-samples.json", rendered)
    (args.out_dir / "index.html").write_text(html_page(rendered, dataset_label), encoding="utf-8")
    return {
        "dataset_label": dataset_label,
        "rendered_rows": int(len(rendered)),
        "oracle_cosine_mean": float(np.mean(oracle_cosines)) if oracle_cosines else None,
        "oracle_cosine_min": float(np.min(oracle_cosines)) if oracle_cosines else None,
        "stack_cosine_mean": float(np.mean(stack_cosines)) if stack_cosines else None,
        "stack_cosine_min": float(np.min(stack_cosines)) if stack_cosines else None,
        "dashboard": str(args.out_dir / "index.html"),
    }


def html_page(rendered: list[dict[str, Any]], dataset_label: str) -> str:
    title = f"Root A Decoder Student - {dataset_label}"
    lines = [
        "<!doctype html>",
        '<meta charset="utf-8">',
        f"<title>{html.escape(title)}</title>",
        "<style>",
        "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:24px;line-height:1.35;color:#151515;background:#fafafa}",
        "table{border-collapse:collapse;width:100%;background:#fff}",
        "th,td{border:1px solid #ddd;padding:8px;vertical-align:top;font-size:13px}",
        "th{background:#f0f0f0;text-align:left}",
        "audio{width:220px}",
        ".text{font-size:16px;max-width:520px}",
        "</style>",
        f"<h1>{html.escape(title)}</h1>",
        "<p>This dashboard renders the selected evaluation split. If the label is heldout, these sentences were not used for decoder-student training.</p>",
        "<table>",
        "<thead><tr><th>#</th><th>Text</th><th>Teacher</th><th>Oracle latent -> decoder student</th><th>Acoustic student -> decoder student</th><th>Metrics</th></tr></thead>",
        "<tbody>",
    ]
    for row in rendered:
        stack_audio = row.get("stack_audio")
        stack_cell = ""
        if stack_audio:
            stack_cell = f"<audio controls src='{html.escape(str(row['stack_audio_src']))}'></audio>"
        lines.append(
            "<tr>"
            f"<td>{row['index']}</td>"
            f"<td class='text'>{html.escape(str(row['text']))}</td>"
            f"<td><audio controls src='{html.escape(str(row['teacher_audio_src']))}'></audio></td>"
            f"<td><audio controls src='{html.escape(str(row['oracle_audio_src']))}'></audio></td>"
            f"<td>{stack_cell}</td>"
            f"<td>oracle L1 {row['oracle_l1']:.5f}<br>oracle cos {row['oracle_cosine']:.5f}<br>"
            f"stack L1 {row['stack_l1'] if row['stack_l1'] is not None else 'n/a'}<br>"
            f"stack cos {row['stack_cosine'] if row['stack_cosine'] is not None else 'n/a'}</td>"
            "</tr>"
        )
    lines.extend(["</tbody>", "</table>"])
    return "\n".join(lines) + "\n"


def train(args: argparse.Namespace) -> dict[str, Any]:
    if args.istft_n_fft is None:
        args.istft_n_fft = 1024 if str(args.variant) in {"fsd", "lrc"} else 512
    if not (0.0 <= float(args.acoustic_latent_mix_prob) <= 1.0):
        raise ValueError(f"--acoustic-latent-mix-prob must be in [0, 1], got {args.acoustic_latent_mix_prob}")
    if not (0.0 <= float(args.acoustic_latent_residual_prob) <= 1.0):
        raise ValueError(
            f"--acoustic-latent-residual-prob must be in [0, 1], got {args.acoustic_latent_residual_prob}"
        )
    if not (0.0 <= float(args.acoustic_latent_residual_max_scale) <= 1.0):
        raise ValueError(
            "--acoustic-latent-residual-max-scale must be in [0, 1], "
            f"got {args.acoustic_latent_residual_max_scale}"
        )
    if not (0.0 <= float(args.lrc_pred_code_mix_prob) <= 1.0):
        raise ValueError(f"--lrc-pred-code-mix-prob must be in [0, 1], got {args.lrc_pred_code_mix_prob}")
    if not (0.0 <= float(args.lrc_pred_code_residual_prob) <= 1.0):
        raise ValueError(
            f"--lrc-pred-code-residual-prob must be in [0, 1], got {args.lrc_pred_code_residual_prob}"
        )
    if not (0.0 <= float(args.lrc_pred_code_residual_max_scale) <= 1.0):
        raise ValueError(
            "--lrc-pred-code-residual-max-scale must be in [0, 1], "
            f"got {args.lrc_pred_code_residual_max_scale}"
        )
    uses_lrc_pred_codes = (
        float(args.lrc_pred_code_mix_prob) > 0.0 or float(args.lrc_pred_code_residual_prob) > 0.0
    )
    if (uses_lrc_pred_codes or args.lrc_pred_code_checkpoint is not None) and str(args.variant) != "lrc":
        raise RuntimeError("--lrc-pred-code-* options require --variant lrc")
    if uses_lrc_pred_codes and args.lrc_pred_code_checkpoint is None:
        raise RuntimeError(
            "--lrc-pred-code-mix-prob/--lrc-pred-code-residual-prob require --lrc-pred-code-checkpoint"
        )
    if float(args.paired_acoustic_residual_weight) < 0.0:
        raise ValueError(
            f"--paired-acoustic-residual-weight must be non-negative, got {args.paired_acoustic_residual_weight}"
        )
    if not (0.0 <= float(args.paired_acoustic_residual_max_scale) <= 1.0):
        raise ValueError(
            "--paired-acoustic-residual-max-scale must be in [0, 1], "
            f"got {args.paired_acoustic_residual_max_scale}"
        )
    if float(args.acoustic_latent_mix_prob) > 0.0 and float(args.acoustic_latent_residual_prob) > 0.0:
        raise ValueError("--acoustic-latent-mix-prob and --acoustic-latent-residual-prob are mutually exclusive")
    if float(args.signature_hint_weight) < 0.0:
        raise ValueError(f"--signature-hint-weight must be non-negative, got {args.signature_hint_weight}")
    if float(args.signature_temporal_weight) < 0.0:
        raise ValueError(f"--signature-temporal-weight must be non-negative, got {args.signature_temporal_weight}")
    if float(args.signature_phase_weight) < 0.0:
        raise ValueError(f"--signature-phase-weight must be non-negative, got {args.signature_phase_weight}")
    if float(args.bottleneck_code_weight) < 0.0:
        raise ValueError(f"--bottleneck-code-weight must be non-negative, got {args.bottleneck_code_weight}")
    if int(args.signature_phase_bins) < 0:
        raise ValueError(f"--signature-phase-bins must be non-negative, got {args.signature_phase_bins}")
    if float(args.signature_phase_weight) > 0.0 and int(args.signature_phase_bins) <= 0:
        raise ValueError("--signature-phase-weight requires --signature-phase-bins > 0")
    if float(args.stft_phase_weight) < 0.0:
        raise ValueError(f"--stft-phase-weight must be non-negative, got {args.stft_phase_weight}")
    if float(args.feature_exact_weight) < 0.0:
        raise ValueError(f"--feature-exact-weight must be non-negative, got {args.feature_exact_weight}")
    if float(args.quiet_frame_weight) < 0.0:
        raise ValueError(f"--quiet-frame-weight must be non-negative, got {args.quiet_frame_weight}")
    if float(args.quiet_delta_weight) < 0.0:
        raise ValueError(f"--quiet-delta-weight must be non-negative, got {args.quiet_delta_weight}")
    if float(args.quiet_ceiling_weight) < 0.0:
        raise ValueError(f"--quiet-ceiling-weight must be non-negative, got {args.quiet_ceiling_weight}")
    if float(args.quiet_ceiling_margin_db) < 0.0:
        raise ValueError(f"--quiet-ceiling-margin-db must be non-negative, got {args.quiet_ceiling_margin_db}")
    if args.teacher_init_checkpoint is not None and args.init_decoder_checkpoint is not None:
        raise RuntimeError("--teacher-init-checkpoint and --init-decoder-checkpoint are mutually exclusive")
    has_alt_input_targets = args.input_target_dir is not None
    if not (0.0 <= float(args.oracle_target_mix_prob) <= 1.0):
        raise ValueError(f"--oracle-target-mix-prob must be in [0, 1], got {args.oracle_target_mix_prob}")
    if args.eval_input_target_dir is not None and not has_alt_input_targets:
        raise RuntimeError("--eval-input-target-dir requires --input-target-dir")
    if has_alt_input_targets and args.eval_pack_dir is not None and args.eval_input_target_dir is None:
        raise RuntimeError("--eval-pack-dir with --input-target-dir requires --eval-input-target-dir")
    if has_alt_input_targets and not str(args.input_target_key).strip():
        raise RuntimeError("--input-target-key must be non-empty")
    if has_alt_input_targets and args.teacher_init_checkpoint is not None:
        raise RuntimeError("--input-target-dir cannot be combined with --teacher-init-checkpoint")
    if has_alt_input_targets and (
        float(args.acoustic_latent_mix_prob) > 0.0
        or float(args.acoustic_latent_residual_prob) > 0.0
        or float(args.paired_acoustic_residual_weight) > 0.0
        or uses_lrc_pred_codes
    ):
        raise RuntimeError(
            "--input-target-dir cannot be combined with Piper-latent or LRC predicted-code mix/residual training"
        )
    if float(args.click_delta_weight) < 0.0:
        raise ValueError(f"--click-delta-weight must be non-negative, got {args.click_delta_weight}")
    if float(args.click_delta_margin) < 0.0:
        raise ValueError(f"--click-delta-margin must be non-negative, got {args.click_delta_margin}")
    if float(args.click_delta_target_scale) < 0.0:
        raise ValueError(f"--click-delta-target-scale must be non-negative, got {args.click_delta_target_scale}")
    if not (0.0 < float(args.click_delta_topk_frac) <= 1.0):
        raise ValueError(f"--click-delta-topk-frac must be in (0, 1], got {args.click_delta_topk_frac}")
    if float(args.quiet_sample_weight) < 0.0:
        raise ValueError(f"--quiet-sample-weight must be non-negative, got {args.quiet_sample_weight}")
    if not (0.0 < float(args.quiet_sample_quantile) <= 1.0):
        raise ValueError(f"--quiet-sample-quantile must be in (0, 1], got {args.quiet_sample_quantile}")
    if float(args.quiet_sample_margin) < 0.0:
        raise ValueError(f"--quiet-sample-margin must be non-negative, got {args.quiet_sample_margin}")
    if float(args.quiet_sample_target_scale) < 0.0:
        raise ValueError(
            f"--quiet-sample-target-scale must be non-negative, got {args.quiet_sample_target_scale}"
        )
    if float(args.echo_tail_weight) < 0.0:
        raise ValueError(f"--echo-tail-weight must be non-negative, got {args.echo_tail_weight}")
    if float(args.echo_tail_min_ms) <= 0.0:
        raise ValueError(f"--echo-tail-min-ms must be positive, got {args.echo_tail_min_ms}")
    if float(args.echo_tail_max_ms) < float(args.echo_tail_min_ms):
        raise ValueError(
            f"--echo-tail-max-ms must be >= --echo-tail-min-ms, got {args.echo_tail_max_ms} < {args.echo_tail_min_ms}"
        )
    if int(args.echo_tail_lags) <= 0:
        raise ValueError(f"--echo-tail-lags must be positive, got {args.echo_tail_lags}")
    if float(args.echo_tail_margin) < 0.0:
        raise ValueError(f"--echo-tail-margin must be non-negative, got {args.echo_tail_margin}")
    if float(args.adv_weight) < 0.0:
        raise ValueError(f"--adv-weight must be non-negative, got {args.adv_weight}")
    if float(args.adv_feature_weight) < 0.0:
        raise ValueError(f"--adv-feature-weight must be non-negative, got {args.adv_feature_weight}")
    if float(args.adv_delta_weight) < 0.0:
        raise ValueError(f"--adv-delta-weight must be non-negative, got {args.adv_delta_weight}")
    if float(args.adv_delta_feature_weight) < 0.0:
        raise ValueError(
            f"--adv-delta-feature-weight must be non-negative, got {args.adv_delta_feature_weight}"
        )
    if int(args.adv_start_step) < 1:
        raise ValueError(f"--adv-start-step must be >= 1, got {args.adv_start_step}")
    if float(args.adv_lr) <= 0.0:
        raise ValueError(f"--adv-lr must be positive, got {args.adv_lr}")
    if not (0.0 < float(args.adv_gate_quantile) < 1.0):
        raise ValueError(f"--adv-gate-quantile must be in (0, 1), got {args.adv_gate_quantile}")
    if float(args.adv_gate_sharpness) <= 0.0:
        raise ValueError(f"--adv-gate-sharpness must be positive, got {args.adv_gate_sharpness}")
    if int(args.adv_gate_frame_size) <= 1:
        raise ValueError(f"--adv-gate-frame-size must be greater than 1, got {args.adv_gate_frame_size}")
    if int(args.adv_gate_frame_hop) <= 0:
        raise ValueError(f"--adv-gate-frame-hop must be positive, got {args.adv_gate_frame_hop}")
    if int(args.post_filter_channels) < 0:
        raise ValueError(f"--post-filter-channels must be non-negative, got {args.post_filter_channels}")
    if int(args.post_filter_layers) < 0:
        raise ValueError(f"--post-filter-layers must be non-negative, got {args.post_filter_layers}")
    if (int(args.post_filter_channels) == 0) != (int(args.post_filter_layers) == 0):
        raise ValueError("--post-filter-channels and --post-filter-layers must both be zero or both be positive")
    if int(args.post_filter_kernel) <= 0 or int(args.post_filter_kernel) % 2 == 0:
        raise ValueError(f"--post-filter-kernel must be a positive odd integer, got {args.post_filter_kernel}")
    if float(args.post_filter_scale) <= 0.0:
        raise ValueError(f"--post-filter-scale must be positive, got {args.post_filter_scale}")
    if int(args.pre_tanh_repair_channels) < 0:
        raise ValueError(
            f"--pre-tanh-repair-channels must be non-negative, got {args.pre_tanh_repair_channels}"
        )
    if int(args.pre_tanh_repair_layers) < 0:
        raise ValueError(f"--pre-tanh-repair-layers must be non-negative, got {args.pre_tanh_repair_layers}")
    if (int(args.pre_tanh_repair_channels) == 0) != (int(args.pre_tanh_repair_layers) == 0):
        raise ValueError(
            "--pre-tanh-repair-channels and --pre-tanh-repair-layers must both be zero or both be positive"
        )
    if int(args.pre_tanh_repair_kernel) <= 0 or int(args.pre_tanh_repair_kernel) % 2 == 0:
        raise ValueError(
            f"--pre-tanh-repair-kernel must be a positive odd integer, got {args.pre_tanh_repair_kernel}"
        )
    if float(args.pre_tanh_repair_scale) <= 0.0:
        raise ValueError(f"--pre-tanh-repair-scale must be positive, got {args.pre_tanh_repair_scale}")
    if float(args.high_band_excess_weight) < 0.0:
        raise ValueError(f"--high-band-excess-weight must be non-negative, got {args.high_band_excess_weight}")
    if float(args.high_band_excess_hz) <= 0.0:
        raise ValueError(f"--high-band-excess-hz must be positive, got {args.high_band_excess_hz}")
    if float(args.high_band_excess_margin_db) < 0.0:
        raise ValueError(
            f"--high-band-excess-margin-db must be non-negative, got {args.high_band_excess_margin_db}"
        )
    if int(args.istft_n_fft) <= 0 or int(args.istft_n_fft) % 2 != 0:
        raise ValueError(f"--istft-n-fft must be a positive even integer, got {args.istft_n_fft}")
    if int(args.fsd_dim) <= 0:
        raise ValueError(f"--fsd-dim must be positive, got {args.fsd_dim}")
    if int(args.fsd_blocks) <= 0:
        raise ValueError(f"--fsd-blocks must be positive, got {args.fsd_blocks}")
    if int(args.fsd_film_rank) <= 0:
        raise ValueError(f"--fsd-film-rank must be positive, got {args.fsd_film_rank}")
    if int(args.fsd_head_rank) <= 0:
        raise ValueError(f"--fsd-head-rank must be positive, got {args.fsd_head_rank}")
    if int(args.lrc_code_dim) <= 0:
        raise ValueError(f"--lrc-code-dim must be positive, got {args.lrc_code_dim}")
    if int(args.lrc_encoder_hidden) <= 0:
        raise ValueError(f"--lrc-encoder-hidden must be positive, got {args.lrc_encoder_hidden}")
    if int(args.assert_max_decoder_params) < 0:
        raise ValueError(
            f"--assert-max-decoder-params must be non-negative, got {args.assert_max_decoder_params}"
        )
    if float(args.ap_amplitude_weight) < 0.0:
        raise ValueError(f"--ap-amplitude-weight must be non-negative, got {args.ap_amplitude_weight}")
    if float(args.ap_phase_weight) < 0.0:
        raise ValueError(f"--ap-phase-weight must be non-negative, got {args.ap_phase_weight}")
    if float(args.ap_complex_weight) < 0.0:
        raise ValueError(f"--ap-complex-weight must be non-negative, got {args.ap_complex_weight}")
    if float(args.spectral_head_init_scale) <= 0.0:
        raise ValueError(f"--spectral-head-init-scale must be positive, got {args.spectral_head_init_scale}")
    if str(args.spectral_head_init) != "default" and str(args.variant) not in {"istft", "apnetlite", "fsd", "lrc"}:
        raise RuntimeError("--spectral-head-init zero/small requires --variant istft, apnetlite, fsd, or lrc")
    if str(args.variant) in {"fsd", "lrc"} and float(args.ap_phase_weight) > 0.0:
        raise RuntimeError(f"--variant {args.variant} does not support --ap-phase-weight; use 0")
    if str(args.variant) != "apnetlite" and (
        float(args.ap_amplitude_weight) > 0.0
        or float(args.ap_phase_weight) > 0.0
        or float(args.ap_complex_weight) > 0.0
    ):
        raise RuntimeError("AP amplitude/phase/complex losses require --variant apnetlite")
    if str(args.variant) in {"istft", "apnetlite", "fsd", "lrc"} and (
        int(args.post_filter_channels) > 0 or int(args.post_filter_layers) > 0
    ):
        raise RuntimeError("--variant istft/apnetlite/fsd/lrc does not support waveform post-filter parameters")
    if str(args.variant) in {"istft", "apnetlite", "fsd", "lrc"} and (
        int(args.pre_tanh_repair_channels) > 0 or int(args.pre_tanh_repair_layers) > 0
    ):
        raise RuntimeError("--variant istft/apnetlite/fsd/lrc does not support pre-tanh repair parameters")
    if str(args.variant) in {"istft", "apnetlite", "fsd", "lrc"} and args.teacher_init_checkpoint is not None:
        raise RuntimeError("--variant istft/apnetlite/fsd/lrc cannot use --teacher-init-checkpoint")
    if str(args.variant) in {"istft", "apnetlite", "fsd", "lrc"} and (
        float(args.feature_hint_weight) > 0.0
        or float(args.feature_exact_weight) > 0.0
        or float(args.signature_hint_weight) > 0.0
        or float(args.signature_temporal_weight) > 0.0
        or float(args.signature_phase_weight) > 0.0
    ):
        raise RuntimeError("--variant istft/apnetlite/fsd/lrc does not support teacher feature/signature hint losses yet")
    if has_alt_input_targets and (
        float(args.feature_hint_weight) > 0.0
        or float(args.feature_exact_weight) > 0.0
        or float(args.signature_hint_weight) > 0.0
        or float(args.signature_temporal_weight) > 0.0
        or float(args.signature_phase_weight) > 0.0
        or float(args.bottleneck_code_weight) > 0.0
    ):
        raise RuntimeError("--input-target-dir does not support Piper decoder feature/signature hint losses")
    if args.init_decoder_checkpoint is not None and str(args.spectral_head_init) != "default":
        raise RuntimeError("--spectral-head-init zero/small cannot be combined with --init-decoder-checkpoint")
    if not (0.0 < float(args.quiet_frame_quantile) <= 1.0):
        raise ValueError(f"--quiet-frame-quantile must be in (0, 1], got {args.quiet_frame_quantile}")
    if int(args.quiet_frame_size) <= 1:
        raise ValueError(f"--quiet-frame-size must be greater than 1, got {args.quiet_frame_size}")
    if int(args.quiet_frame_hop) <= 0:
        raise ValueError(f"--quiet-frame-hop must be positive, got {args.quiet_frame_hop}")
    signature_keys = parse_signature_keys(str(args.signature_keys))
    requested_bottleneck_code_keys = parse_optional_signature_keys(
        str(args.bottleneck_code_keys),
        label="--bottleneck-code-keys",
    )
    feature_exact_keys = parse_feature_exact_keys(str(args.feature_exact_keys))
    packed_exact_feature_keys = [key for key in feature_exact_keys if key not in {"pre", "up0"}]
    stage0_branches = parse_stage_branches(str(args.stage0_branches), label="--stage0-branches")
    stage1_branches = parse_stage_branches(str(args.stage1_branches), label="--stage1-branches")
    stage2_branches = parse_stage_branches(str(args.stage2_branches), label="--stage2-branches")
    stage3_branches = parse_stage_branches(str(args.stage3_branches), label="--stage3-branches")
    stage_projection_bottlenecks = parse_stage_projection_bottlenecks(str(args.stage_projection_bottlenecks))
    if stage_projection_bottlenecks and str(args.variant) != "pb":
        raise RuntimeError("--stage-projection-bottlenecks requires --variant pb")
    adv_periods = parse_positive_int_tuple(str(args.adv_periods), label="--adv-periods", min_value=2)
    adv_channels = parse_positive_int_tuple(str(args.adv_channels), label="--adv-channels")
    uses_any_signature_loss = (
        float(args.signature_hint_weight) > 0.0
        or float(args.signature_temporal_weight) > 0.0
        or float(args.signature_phase_weight) > 0.0
        or float(args.bottleneck_code_weight) > 0.0
    )
    if float(args.bottleneck_code_weight) > 0.0 and args.bottleneck_code_checkpoint is None:
        raise RuntimeError("--bottleneck-code-weight requires --bottleneck-code-checkpoint")
    if uses_any_signature_loss and args.signature_pack_dir is None:
        raise RuntimeError(
            "--signature-hint-weight/--signature-temporal-weight/--signature-phase-weight/"
            "--bottleneck-code-weight requires --signature-pack-dir"
        )
    if (
        float(args.feature_exact_weight) > 0.0
        and args.teacher_init_checkpoint is None
        and args.init_decoder_checkpoint is None
    ):
        raise RuntimeError(
            "--feature-exact-weight requires --teacher-init-checkpoint or an --init-decoder-checkpoint chain "
            "with teacher_init metadata so selected teacher channels are defined"
        )
    exact_feature_pack_dir = args.exact_feature_pack_dir or args.signature_pack_dir
    if float(args.feature_exact_weight) > 0.0 and packed_exact_feature_keys and exact_feature_pack_dir is None:
        raise RuntimeError(
            "--feature-exact-keys beyond pre/up0 require --exact-feature-pack-dir, "
            "or --signature-pack-dir containing matching *_exact tensors"
        )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    channels = parse_channels(args.channels)
    device = pick_device(args.device)
    bottleneck_codebook: FrozenBottleneckCodebook | None = None
    bottleneck_code_signature_keys: list[str] = []
    if float(args.bottleneck_code_weight) > 0.0:
        if args.bottleneck_code_checkpoint is None:
            raise RuntimeError("internal error: bottleneck code checkpoint disappeared after validation")
        bottleneck_codebook = load_bottleneck_codebook(
            args.bottleneck_code_checkpoint,
            requested_bottleneck_code_keys,
            device,
        )
        bottleneck_code_signature_keys = bottleneck_codebook.signature_keys
        print(
            json.dumps(
                {
                    "bottleneck_code": {
                        "checkpoint": str(args.bottleneck_code_checkpoint),
                        "keys": bottleneck_code_signature_keys,
                        "training_only_parameters": bottleneck_codebook.parameter_count,
                    }
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    input_target_index = (
        load_input_target_index(args.input_target_dir) if args.input_target_dir is not None else None
    )
    eval_input_target_index = (
        load_input_target_index(args.eval_input_target_dir)
        if args.eval_input_target_dir is not None
        else None
    )
    rows, samples, in_channels = load_samples(
        args.pack_dir,
        args.teacher_decoder,
        input_target_index=input_target_index,
        input_target_key=str(args.input_target_key),
    )
    if not rows:
        raise RuntimeError(f"{args.pack_dir}: no training rows loaded")
    is_lrc = str(args.variant) == "lrc"
    lrc_input_channels = int(in_channels)
    model_in_channels = int(args.lrc_code_dim) if is_lrc else int(in_channels)
    lrc_encoder: LrcEncoder | None = None
    if is_lrc:
        if int(in_channels) != 192:
            raise RuntimeError(f"--variant lrc expects teacher generator_input with 192 channels, got {in_channels}")
        lrc_encoder = LrcEncoder(
            in_channels=int(in_channels),
            hidden=int(args.lrc_encoder_hidden),
            code_dim=int(args.lrc_code_dim),
        ).to(device)
    if float(args.oracle_target_mix_prob) > 0.0 and not any(
        sample.oracle_latent is not None and sample.oracle_audio is not None for sample in samples
    ):
        raise RuntimeError("--oracle-target-mix-prob requires paired input targets with target_audio_npy rows")
    sample_rate = int(rows[0].get("sample_rate") or 22050)
    if sample_rate <= 0:
        raise RuntimeError(f"{args.pack_dir}: invalid sample_rate {sample_rate}")
    if float(args.high_band_excess_weight) > 0.0 and float(args.high_band_excess_hz) >= float(sample_rate) / 2.0:
        raise ValueError(
            f"--high-band-excess-hz {args.high_band_excess_hz} must be below Nyquist for sample_rate {sample_rate}"
        )
    eval_rows: list[dict[str, Any]] | None = None
    eval_samples: list[ChunkSample] | None = None
    if args.eval_pack_dir is not None:
        eval_rows, eval_samples, eval_in_channels = load_samples(
            args.eval_pack_dir,
            args.teacher_decoder,
            input_target_index=eval_input_target_index,
            input_target_key=str(args.input_target_key),
        )
        if eval_in_channels != in_channels:
            raise RuntimeError(f"eval pack channel count {eval_in_channels} != train channel count {in_channels}")
    if uses_any_signature_loss and args.signature_pack_dir is not None:
        signature_target_keys = sorted(
            set(signature_keys if (
                float(args.signature_hint_weight) > 0.0
                or float(args.signature_temporal_weight) > 0.0
                or float(args.signature_phase_weight) > 0.0
            ) else [])
            | set(bottleneck_code_signature_keys)
        )
        samples = attach_signature_targets(
            samples,
            args.signature_pack_dir,
            signature_target_keys,
            phase_bins=int(args.signature_phase_bins) if float(args.signature_phase_weight) > 0.0 else 0,
            exact_feature_keys=[],
        )
        rows = filter_rows_to_complete_samples(rows, samples)
    if float(args.feature_exact_weight) > 0.0 and packed_exact_feature_keys:
        if exact_feature_pack_dir is None:
            raise RuntimeError("internal error: exact feature pack directory was not resolved")
        samples = attach_signature_targets(
            samples,
            exact_feature_pack_dir,
            [],
            phase_bins=0,
            exact_feature_keys=packed_exact_feature_keys,
        )
        rows = filter_rows_to_complete_samples(rows, samples)
    needs_acoustic_latents = (
        float(args.acoustic_latent_mix_prob) > 0.0
        or float(args.acoustic_latent_residual_prob) > 0.0
        or float(args.paired_acoustic_residual_weight) > 0.0
    )
    if uses_lrc_pred_codes and needs_acoustic_latents:
        raise RuntimeError("--lrc-pred-code-* cannot be combined with --acoustic-latent-* or paired acoustic residuals")
    if needs_acoustic_latents:
        if args.acoustic_checkpoint is None:
            raise RuntimeError(
                "--acoustic-latent-mix-prob/--acoustic-latent-residual-prob/"
                "--paired-acoustic-residual-weight requires --acoustic-checkpoint"
            )
        acoustic_module, acoustic_model = load_acoustic_model(args.acoustic_checkpoint, device)
        samples = attach_acoustic_latents(acoustic_module, acoustic_model, samples, device)
        acoustic_model.cpu()
        del acoustic_model
        if device.type == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()
    lrc_pred_code_cache_summary: dict[str, Any] | None = None
    if uses_lrc_pred_codes:
        if args.lrc_pred_code_checkpoint is None:
            raise RuntimeError("internal error: LRC predicted-code checkpoint disappeared after validation")
        lrc_pred_module, lrc_pred_model, lrc_pred_config = load_lrc_pred_code_model(
            args.lrc_pred_code_checkpoint,
            device,
        )
        lrc_pred_out_channels = int(lrc_pred_config.get("out_channels") or 0)
        if lrc_pred_out_channels != int(args.lrc_code_dim):
            raise RuntimeError(
                f"{args.lrc_pred_code_checkpoint}: c-acoustic out_channels {lrc_pred_out_channels} "
                f"!= --lrc-code-dim {int(args.lrc_code_dim)}"
            )
        samples, lrc_pred_code_cache_summary = attach_lrc_pred_code_cache(
            lrc_pred_module,
            lrc_pred_model,
            samples,
            device=device,
            out_dir=args.out_dir,
            checkpoint_path=args.lrc_pred_code_checkpoint,
            code_dim=int(args.lrc_code_dim),
        )
        lrc_pred_model.cpu()
        del lrc_pred_model
        if device.type == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()
    model = DecoderStudent(
        in_channels=model_in_channels,
        channels=channels,
        res_layers=args.res_layers,
        variant=args.variant,
        rank_ratio=args.rank_ratio,
        activation=args.activation,
        stage_affine=bool(args.stage_affine),
        factorized_pre_rank=int(args.factorized_pre_rank),
        piper_res_factor_rank_ratio=float(args.piper_res_factor_rank_ratio),
        res_bank_scale_mode=str(args.res_bank_scale_mode),
        stage0_branches=stage0_branches,
        stage1_branches=stage1_branches,
        stage2_branches=stage2_branches,
        stage3_branches=stage3_branches,
        post_filter_channels=int(args.post_filter_channels),
        post_filter_layers=int(args.post_filter_layers),
        post_filter_kernel=int(args.post_filter_kernel),
        post_filter_scale=float(args.post_filter_scale),
        pre_tanh_repair_channels=int(args.pre_tanh_repair_channels),
        pre_tanh_repair_layers=int(args.pre_tanh_repair_layers),
        pre_tanh_repair_kernel=int(args.pre_tanh_repair_kernel),
        pre_tanh_repair_scale=float(args.pre_tanh_repair_scale),
        istft_n_fft=int(args.istft_n_fft),
        fsd_dim=int(args.fsd_dim),
        fsd_blocks=int(args.fsd_blocks),
        fsd_film_rank=int(args.fsd_film_rank),
        fsd_head_rank=int(args.fsd_head_rank),
        stage_projection_bottlenecks=stage_projection_bottlenecks,
    ).to(device)
    spectral_head_init_summary = initialize_spectral_heads(
        model,
        mode=str(args.spectral_head_init),
        scale=float(args.spectral_head_init_scale),
        ap_amp_bias=float(args.ap_amp_init_bias),
        ap_phase_real_bias=float(args.ap_phase_real_init_bias),
    )
    if spectral_head_init_summary is not None:
        print(json.dumps({"spectral_head_init": spectral_head_init_summary}, ensure_ascii=False), flush=True)
    teacher_init_summary = None
    decoder_init_summary = None
    if args.teacher_init_checkpoint is not None:
        teacher_init_summary = initialize_piperlite_from_teacher(
            model,
            args.teacher_init_checkpoint,
            channels,
            str(args.teacher_init_method),
        )
        print(json.dumps({"teacher_init": teacher_init_summary}, ensure_ascii=False), flush=True)
    if args.init_decoder_checkpoint is not None:
        decoder_init_summary = init_decoder_student_from_checkpoint(
            model,
            args.init_decoder_checkpoint,
            expected_in_channels=model_in_channels,
            expected_channels=channels,
            expected_res_layers=int(args.res_layers),
            expected_variant=str(args.variant),
            expected_rank_ratio=float(args.rank_ratio),
            expected_activation=str(args.activation),
            expected_stage_affine=bool(args.stage_affine),
            expected_factorized_pre_rank=int(args.factorized_pre_rank),
            expected_piper_res_factor_rank_ratio=float(args.piper_res_factor_rank_ratio),
            expected_res_bank_scale_mode=str(args.res_bank_scale_mode),
            expected_stage0_branches=stage0_branches,
            expected_stage1_branches=stage1_branches,
            expected_stage2_branches=stage2_branches,
            expected_stage3_branches=stage3_branches,
            expected_fsd_dim=int(args.fsd_dim),
            expected_fsd_blocks=int(args.fsd_blocks),
            expected_fsd_film_rank=int(args.fsd_film_rank),
            expected_fsd_head_rank=int(args.fsd_head_rank),
            expected_stage_projection_bottlenecks=stage_projection_bottlenecks,
            expected_post_filter_channels=int(args.post_filter_channels),
            expected_post_filter_layers=int(args.post_filter_layers),
            expected_post_filter_kernel=int(args.post_filter_kernel),
            expected_post_filter_scale=float(args.post_filter_scale),
            expected_pre_tanh_repair_channels=int(args.pre_tanh_repair_channels),
            expected_pre_tanh_repair_layers=int(args.pre_tanh_repair_layers),
            expected_pre_tanh_repair_kernel=int(args.pre_tanh_repair_kernel),
            expected_pre_tanh_repair_scale=float(args.pre_tanh_repair_scale),
            expected_istft_n_fft=int(args.istft_n_fft),
            allow_new_post_filter=bool(args.allow_new_post_filter_init),
            allow_new_pre_tanh_repair=bool(args.allow_new_pre_tanh_repair_init),
            allow_leaky_to_snake=bool(args.allow_leaky_to_snake_init),
        )
        print(json.dumps({"decoder_init": decoder_init_summary}, ensure_ascii=False), flush=True)
        if lrc_encoder is not None:
            lrc_init_checkpoint = load_torch_checkpoint(args.init_decoder_checkpoint, "LRC decoder init checkpoint")
            lrc_state = lrc_init_checkpoint.get("lrc_encoder_state_dict")
            if not isinstance(lrc_state, dict):
                raise RuntimeError(
                    f"{args.init_decoder_checkpoint}: LRC continuation requires lrc_encoder_state_dict"
                )
            try:
                lrc_encoder.load_state_dict(lrc_state, strict=True)
            except RuntimeError as exc:
                raise RuntimeError(f"failed to load LRC encoder state from {args.init_decoder_checkpoint}") from exc
            for name, parameter in lrc_encoder.named_parameters():
                if not torch.isfinite(parameter).all().item():
                    raise RuntimeError(f"non-finite LRC encoder parameter after init: {name}")
            print(
                json.dumps(
                    {
                        "lrc_encoder_init": {
                            "checkpoint": str(args.init_decoder_checkpoint),
                            "training_only_parameters": count_parameters(lrc_encoder),
                        }
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    selected_feature_indices: dict[str, torch.Tensor] = {}
    feature_exact_teacher_init_summary: dict[str, Any] | None = None
    if float(args.feature_exact_weight) > 0.0:
        feature_exact_teacher_init_summary = (
            teacher_init_summary
            if teacher_init_summary is not None
            else find_teacher_init_summary_in_decoder_chain(args.init_decoder_checkpoint)
        )
        raw_indices = feature_exact_teacher_init_summary.get("selected_channel_indices")
        if not isinstance(raw_indices, list) or len(raw_indices) < 4:
            raise RuntimeError("teacher_init_summary missing selected_channel_indices entries for exact feature loss")
        selected_feature_indices = {
            "pre": torch.as_tensor(raw_indices[0], dtype=torch.long, device=device),
            "up0": torch.as_tensor(raw_indices[1], dtype=torch.long, device=device),
            "up1_raw": torch.as_tensor(raw_indices[2], dtype=torch.long, device=device),
            "stage1_mix": torch.as_tensor(raw_indices[2], dtype=torch.long, device=device),
            "up2_raw": torch.as_tensor(raw_indices[3], dtype=torch.long, device=device),
            "stage2_mix": torch.as_tensor(raw_indices[3], dtype=torch.long, device=device),
        }
        if len(raw_indices) >= 5:
            selected_feature_indices.update(
                {
                    "up3_raw": torch.as_tensor(raw_indices[4], dtype=torch.long, device=device),
                    "stage3_mix": torch.as_tensor(raw_indices[4], dtype=torch.long, device=device),
                }
            )
    if bool(args.freeze_decoder_body):
        if isinstance(model.post_filter, nn.Identity) and isinstance(model.pre_tanh_repair, nn.Identity):
            raise RuntimeError("--freeze-decoder-body requires an enabled post-filter or pre-tanh repair branch")
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith("post_filter.") or name.startswith("pre_tanh_repair."))
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    lrc_encoder_parameter_count = 0
    lrc_encoder_trainable_parameter_count = 0
    if lrc_encoder is not None:
        lrc_encoder_parameter_count = count_parameters(lrc_encoder)
        lrc_encoder_trainable_parameters = [
            parameter for parameter in lrc_encoder.parameters() if parameter.requires_grad
        ]
        lrc_encoder_trainable_parameter_count = int(
            sum(parameter.numel() for parameter in lrc_encoder_trainable_parameters)
        )
        trainable_parameters.extend(lrc_encoder_trainable_parameters)
    if not trainable_parameters:
        raise RuntimeError("decoder student has no trainable parameters")
    parameter_count = count_parameters(model)
    trainable_parameter_count = int(sum(parameter.numel() for parameter in trainable_parameters))
    decoder_trainable_parameter_count = int(trainable_parameter_count - lrc_encoder_trainable_parameter_count)
    if int(args.assert_max_decoder_params) > 0 and parameter_count > int(args.assert_max_decoder_params):
        raise ValueError(
            f"decoder parameter count {parameter_count} exceeds --assert-max-decoder-params "
            f"{int(args.assert_max_decoder_params)}"
        )
    if str(args.variant) in {"fsd", "lrc", "pb"} or int(args.assert_max_decoder_params) > 0:
        print(
            json.dumps(
                {
                    "decoder_parameter_count": {
                        "variant": str(args.variant),
                        "decoder_parameters": int(parameter_count),
                        "decoder_trainable_parameters": int(decoder_trainable_parameter_count),
                        "lrc_encoder_training_only_parameters": int(lrc_encoder_parameter_count),
                        "lrc_encoder_trainable_parameters": int(lrc_encoder_trainable_parameter_count),
                        "assert_max_decoder_params": int(args.assert_max_decoder_params),
                    }
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.lr, weight_decay=1e-5)
    discriminator: MultiPeriodDiscriminator | None = None
    discriminator_optimizer: torch.optim.Optimizer | None = None
    discriminator_parameter_count = 0
    if float(args.adv_weight) > 0.0 or float(args.adv_feature_weight) > 0.0:
        discriminator = MultiPeriodDiscriminator(adv_periods, adv_channels).to(device)
        discriminator_parameter_count = count_parameters(discriminator)
        discriminator_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=float(args.adv_lr), weight_decay=1e-5)
    delta_discriminator: MultiPeriodDiscriminator | None = None
    delta_discriminator_optimizer: torch.optim.Optimizer | None = None
    delta_discriminator_parameter_count = 0
    if float(args.adv_delta_weight) > 0.0 or float(args.adv_delta_feature_weight) > 0.0:
        delta_discriminator = MultiPeriodDiscriminator(adv_periods, adv_channels).to(device)
        delta_discriminator_parameter_count = count_parameters(delta_discriminator)
        delta_discriminator_optimizer = torch.optim.AdamW(
            delta_discriminator.parameters(),
            lr=float(args.adv_lr),
            weight_decay=1e-5,
        )
    logs: list[dict[str, Any]] = []
    lrc_pred_code_total_samples = 0
    lrc_pred_code_mixed_samples = 0
    lrc_pred_code_residual_samples = 0
    for step in range(1, args.steps + 1):
        uses_signature_targets = (
            float(args.signature_hint_weight) > 0.0
            or float(args.signature_temporal_weight) > 0.0
            or float(args.signature_phase_weight) > 0.0
            or float(args.bottleneck_code_weight) > 0.0
        )
        batch_signature_keys = sorted(
            set(
                signature_keys
                if (
                    float(args.signature_hint_weight) > 0.0
                    or float(args.signature_temporal_weight) > 0.0
                    or float(args.signature_phase_weight) > 0.0
                )
                else []
            )
            | set(bottleneck_code_signature_keys if float(args.bottleneck_code_weight) > 0.0 else [])
        )
        needs_ap_losses = (
            float(args.ap_amplitude_weight) > 0.0
            or float(args.ap_phase_weight) > 0.0
            or float(args.ap_complex_weight) > 0.0
        )
        needs_base_teacher_features = (
            float(args.feature_hint_weight) > 0.0
            or (
                float(args.feature_exact_weight) > 0.0
                and any(key in {"pre", "up0"} for key in feature_exact_keys)
            )
        )
        latent, target, teacher_features, teacher_signatures = crop_batch(
            samples,
            batch_size=args.batch_size,
            crop_frames=args.crop_frames,
            include_base_teacher_features=needs_base_teacher_features,
            oracle_target_mix_prob=float(args.oracle_target_mix_prob),
            acoustic_latent_mix_prob=float(args.acoustic_latent_mix_prob),
            acoustic_latent_residual_prob=float(args.acoustic_latent_residual_prob),
            acoustic_latent_residual_max_scale=float(args.acoustic_latent_residual_max_scale),
            lrc_pred_code_mix_prob=float(args.lrc_pred_code_mix_prob),
            lrc_pred_code_residual_prob=float(args.lrc_pred_code_residual_prob),
            lrc_pred_code_residual_max_scale=float(args.lrc_pred_code_residual_max_scale),
            signature_keys=batch_signature_keys if uses_signature_targets else [],
            signature_phase_enabled=float(args.signature_phase_weight) > 0.0,
            exact_feature_keys=packed_exact_feature_keys if float(args.feature_exact_weight) > 0.0 else [],
            device=device,
        )
        lrc_pred_code_tensor = teacher_features.pop("_lrc_pred_code", None)
        lrc_pred_code_mask = teacher_features.pop("_lrc_pred_code_mask", None)
        lrc_pred_code_residual_mask = teacher_features.pop("_lrc_pred_code_residual_mask", None)
        lrc_pred_code_residual_scale = teacher_features.pop("_lrc_pred_code_residual_scale", None)
        lrc_pred_code_batch_total = 0
        lrc_pred_code_batch_mixed = 0
        lrc_pred_code_batch_residual = 0
        if lrc_pred_code_tensor is not None:
            if lrc_pred_code_mask is None or lrc_pred_code_residual_mask is None or lrc_pred_code_residual_scale is None:
                raise RuntimeError("internal error: incomplete LRC predicted-code crop metadata")
            lrc_pred_code_batch_total = int(lrc_pred_code_tensor.shape[0])
            lrc_pred_code_batch_mixed = int(torch.sum(lrc_pred_code_mask > 0.5).detach().cpu().item())
            lrc_pred_code_batch_residual = int(torch.sum(lrc_pred_code_residual_mask > 0.5).detach().cpu().item())
            lrc_pred_code_total_samples += lrc_pred_code_batch_total
            lrc_pred_code_mixed_samples += lrc_pred_code_batch_mixed
            lrc_pred_code_residual_samples += lrc_pred_code_batch_residual
        needs_features = (
            float(args.feature_hint_weight) > 0.0
            or float(args.feature_exact_weight) > 0.0
            or float(args.signature_hint_weight) > 0.0
            or float(args.signature_temporal_weight) > 0.0
            or float(args.signature_phase_weight) > 0.0
            or float(args.bottleneck_code_weight) > 0.0
            or needs_ap_losses
        )
        if lrc_encoder is not None:
            exact_decoder_latent = lrc_encoder(latent)
            decoder_latent = exact_decoder_latent
            if lrc_pred_code_tensor is not None:
                if (
                    lrc_pred_code_mask is None
                    or lrc_pred_code_residual_mask is None
                    or lrc_pred_code_residual_scale is None
                ):
                    raise RuntimeError("internal error: incomplete LRC predicted-code tensors")
                if lrc_pred_code_tensor.shape != exact_decoder_latent.shape:
                    raise RuntimeError(
                        f"LRC predicted code shape {lrc_pred_code_tensor.shape} "
                        f"!= exact code shape {exact_decoder_latent.shape}"
                    )
                hard_mask = lrc_pred_code_mask > 0.5
                residual_mask = lrc_pred_code_residual_mask > 0.5
                residual_decoder_latent = exact_decoder_latent + lrc_pred_code_residual_scale * (
                    exact_decoder_latent - lrc_pred_code_tensor
                )
                decoder_latent = torch.where(residual_mask, residual_decoder_latent, exact_decoder_latent)
                decoder_latent = torch.where(hard_mask, lrc_pred_code_tensor, decoder_latent)
        else:
            if lrc_pred_code_tensor is not None:
                raise RuntimeError("internal error: LRC predicted-code tensor produced for non-LRC model")
            decoder_latent = latent
        if needs_features:
            prediction_value = model(decoder_latent, return_features=True)
            if not isinstance(prediction_value, tuple):
                raise RuntimeError("expected decoder to return features")
            prediction, student_features = prediction_value
            hint = (
                channel_moment_hint_loss(student_features, teacher_features)
                if args.feature_hint_weight > 0
                else prediction.new_tensor(0.0)
            )
            feature_exact = (
                selected_teacher_feature_loss(
                    student_features,
                    teacher_features,
                    selected_feature_indices,
                    feature_exact_keys,
                )
                if float(args.feature_exact_weight) > 0.0
                else prediction.new_tensor(0.0)
            )
            signature_hint = (
                decoder_signature_hint_loss(student_features, teacher_signatures, signature_keys, int(latent.shape[-1]))
                if args.signature_hint_weight > 0
                else prediction.new_tensor(0.0)
            )
            signature_temporal = (
                decoder_signature_temporal_loss(
                    student_features,
                    teacher_signatures,
                    signature_keys,
                    int(latent.shape[-1]),
                )
                if args.signature_temporal_weight > 0
                else prediction.new_tensor(0.0)
            )
            signature_phase = (
                decoder_signature_phase_loss(
                    student_features,
                    teacher_signatures,
                    signature_keys,
                    int(latent.shape[-1]),
                    int(args.signature_phase_bins),
                )
                if args.signature_phase_weight > 0
                else prediction.new_tensor(0.0)
            )
            bottleneck_code = (
                decoder_bottleneck_code_loss(
                    student_features,
                    teacher_signatures,
                    bottleneck_codebook,
                    int(latent.shape[-1]),
                )
                if float(args.bottleneck_code_weight) > 0.0 and bottleneck_codebook is not None
                else prediction.new_tensor(0.0)
            )
            if needs_ap_losses:
                ap_amplitude, ap_phase, ap_complex = apnetlite_losses(
                    student_features,
                    target,
                    n_fft=int(args.istft_n_fft),
                )
            else:
                ap_amplitude = prediction.new_tensor(0.0)
                ap_phase = prediction.new_tensor(0.0)
                ap_complex = prediction.new_tensor(0.0)
        else:
            prediction_value = model(decoder_latent)
            if isinstance(prediction_value, tuple):
                raise RuntimeError("decoder returned features unexpectedly")
            prediction = prediction_value
            hint = prediction.new_tensor(0.0)
            feature_exact = prediction.new_tensor(0.0)
            signature_hint = prediction.new_tensor(0.0)
            signature_temporal = prediction.new_tensor(0.0)
            signature_phase = prediction.new_tensor(0.0)
            bottleneck_code = prediction.new_tensor(0.0)
            ap_amplitude = prediction.new_tensor(0.0)
            ap_phase = prediction.new_tensor(0.0)
            ap_complex = prediction.new_tensor(0.0)
        if prediction.shape != target.shape:
            raise RuntimeError(f"prediction shape {prediction.shape} != target shape {target.shape}")
        waveform_l1 = F.l1_loss(prediction, target)
        spectral = multi_resolution_stft_loss(prediction, target) if args.spectral_weight > 0 else prediction.new_tensor(0.0)
        stft_phase = (
            multi_resolution_stft_phase_loss(prediction, target)
            if args.stft_phase_weight > 0
            else prediction.new_tensor(0.0)
        )
        if args.quiet_frame_weight > 0 or args.quiet_delta_weight > 0:
            quiet_frame, quiet_delta = quiet_frame_losses(
                prediction,
                target,
                quantile=float(args.quiet_frame_quantile),
                frame_size=int(args.quiet_frame_size),
                frame_hop=int(args.quiet_frame_hop),
            )
        else:
            quiet_frame = prediction.new_tensor(0.0)
            quiet_delta = prediction.new_tensor(0.0)
        quiet_ceiling = (
            quiet_ceiling_loss(
                prediction,
                target,
                quantile=float(args.quiet_frame_quantile),
                frame_size=int(args.quiet_frame_size),
                frame_hop=int(args.quiet_frame_hop),
                margin_db=float(args.quiet_ceiling_margin_db),
            )
            if args.quiet_ceiling_weight > 0
            else prediction.new_tensor(0.0)
        )
        click_delta = (
            click_delta_excess_loss(
                prediction,
                target,
                margin=float(args.click_delta_margin),
                target_scale=float(args.click_delta_target_scale),
                topk_frac=float(args.click_delta_topk_frac),
            )
            if args.click_delta_weight > 0
            else prediction.new_tensor(0.0)
        )
        quiet_sample = (
            quiet_sample_excess_loss(
                prediction,
                target,
                quantile=float(args.quiet_sample_quantile),
                margin=float(args.quiet_sample_margin),
                target_scale=float(args.quiet_sample_target_scale),
            )
            if args.quiet_sample_weight > 0
            else prediction.new_tensor(0.0)
        )
        high_band_excess = (
            high_band_excess_loss(
                prediction,
                target,
                sample_rate=sample_rate,
                high_band_hz=float(args.high_band_excess_hz),
                margin_db=float(args.high_band_excess_margin_db),
            )
            if float(args.high_band_excess_weight) > 0.0
            else prediction.new_tensor(0.0)
        )
        echo_tail = (
            echo_tail_loss(
                prediction,
                target,
                sample_rate=sample_rate,
                min_lag_ms=float(args.echo_tail_min_ms),
                max_lag_ms=float(args.echo_tail_max_ms),
                lag_count=int(args.echo_tail_lags),
                margin=float(args.echo_tail_margin),
            )
            if float(args.echo_tail_weight) > 0.0
            else prediction.new_tensor(0.0)
        )
        paired_acoustic = prediction.new_tensor(0.0)
        paired_acoustic_waveform_l1 = prediction.new_tensor(0.0)
        paired_acoustic_spectral = prediction.new_tensor(0.0)
        paired_acoustic_stft_phase = prediction.new_tensor(0.0)
        paired_acoustic_echo_tail = prediction.new_tensor(0.0)
        if float(args.paired_acoustic_residual_weight) > 0.0:
            paired_latent, paired_target, _paired_features, _paired_signatures = crop_batch(
                samples,
                batch_size=args.batch_size,
                crop_frames=args.crop_frames,
                include_base_teacher_features=False,
                oracle_target_mix_prob=0.0,
                acoustic_latent_mix_prob=0.0,
                acoustic_latent_residual_prob=1.0,
                acoustic_latent_residual_max_scale=float(args.paired_acoustic_residual_max_scale),
                lrc_pred_code_mix_prob=0.0,
                lrc_pred_code_residual_prob=0.0,
                lrc_pred_code_residual_max_scale=0.0,
                signature_keys=[],
                signature_phase_enabled=False,
                exact_feature_keys=[],
                device=device,
            )
            paired_decoder_latent = lrc_encoder(paired_latent) if lrc_encoder is not None else paired_latent
            paired_prediction_value = model(paired_decoder_latent)
            if isinstance(paired_prediction_value, tuple):
                raise RuntimeError("paired acoustic residual branch unexpectedly returned decoder features")
            paired_prediction = paired_prediction_value
            if paired_prediction.shape != paired_target.shape:
                raise RuntimeError(
                    f"paired prediction shape {paired_prediction.shape} != target shape {paired_target.shape}"
                )
            paired_acoustic_waveform_l1 = F.l1_loss(paired_prediction, paired_target)
            paired_acoustic_spectral = (
                multi_resolution_stft_loss(paired_prediction, paired_target)
                if args.spectral_weight > 0
                else paired_prediction.new_tensor(0.0)
            )
            paired_acoustic_stft_phase = (
                multi_resolution_stft_phase_loss(paired_prediction, paired_target)
                if args.stft_phase_weight > 0
                else paired_prediction.new_tensor(0.0)
            )
            paired_acoustic_echo_tail = (
                echo_tail_loss(
                    paired_prediction,
                    paired_target,
                    sample_rate=sample_rate,
                    min_lag_ms=float(args.echo_tail_min_ms),
                    max_lag_ms=float(args.echo_tail_max_ms),
                    lag_count=int(args.echo_tail_lags),
                    margin=float(args.echo_tail_margin),
                )
                if float(args.echo_tail_weight) > 0.0
                else paired_prediction.new_tensor(0.0)
            )
            paired_acoustic = (
                paired_acoustic_waveform_l1
                + float(args.spectral_weight) * paired_acoustic_spectral
                + float(args.stft_phase_weight) * paired_acoustic_stft_phase
                + float(args.echo_tail_weight) * paired_acoustic_echo_tail
            )
        adversarial_generator = prediction.new_tensor(0.0)
        adversarial_feature = prediction.new_tensor(0.0)
        adversarial_discriminator = prediction.new_tensor(0.0)
        adversarial_gate_mean = 1.0
        discriminator_grad_norm = 0.0
        adversarial_delta_generator = prediction.new_tensor(0.0)
        adversarial_delta_feature = prediction.new_tensor(0.0)
        adversarial_delta_discriminator = prediction.new_tensor(0.0)
        adversarial_delta_gate_mean = 1.0
        delta_discriminator_grad_norm = 0.0
        if discriminator is not None and discriminator_optimizer is not None and step >= int(args.adv_start_step):
            discriminator_prediction = prediction
            discriminator_target = target.detach()
            if args.adv_gate_mode == "target-energy":
                adversarial_gate = target_energy_gate(
                    target,
                    quantile=float(args.adv_gate_quantile),
                    sharpness=float(args.adv_gate_sharpness),
                    frame_size=int(args.adv_gate_frame_size),
                    frame_hop=int(args.adv_gate_frame_hop),
                )
                adversarial_gate_mean = float(adversarial_gate.detach().mean().cpu())
                discriminator_prediction = prediction * adversarial_gate
                discriminator_target = discriminator_target * adversarial_gate.detach()
            set_requires_grad(discriminator, True)
            discriminator_optimizer.zero_grad(set_to_none=True)
            real_scores, _real_features = discriminator(discriminator_target)
            fake_scores, _fake_features = discriminator(discriminator_prediction.detach())
            adversarial_discriminator = discriminator_lsgan_loss(real_scores, fake_scores)
            if not torch.isfinite(adversarial_discriminator):
                raise RuntimeError(f"non-finite discriminator loss at step {step}")
            adversarial_discriminator.backward()
            discriminator_grad_norm = float(
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=5.0).detach().cpu()
            )
            discriminator_optimizer.step()

            set_requires_grad(discriminator, False)
            fake_scores_for_generator, fake_features_for_generator = discriminator(discriminator_prediction)
            _real_scores_for_generator, real_features_for_generator = discriminator(discriminator_target)
            adversarial_generator = generator_lsgan_loss(fake_scores_for_generator)
            adversarial_feature = discriminator_feature_matching_loss(
                real_features_for_generator,
                fake_features_for_generator,
            )
            set_requires_grad(discriminator, True)
            if not torch.isfinite(adversarial_generator):
                raise RuntimeError(f"non-finite generator adversarial loss at step {step}")
            if not torch.isfinite(adversarial_feature):
                raise RuntimeError(f"non-finite adversarial feature loss at step {step}")
        if (
            delta_discriminator is not None
            and delta_discriminator_optimizer is not None
            and step >= int(args.adv_start_step)
        ):
            delta_gate = None
            if args.adv_gate_mode == "target-energy":
                delta_gate = target_energy_gate(
                    target,
                    quantile=float(args.adv_gate_quantile),
                    sharpness=float(args.adv_gate_sharpness),
                    frame_size=int(args.adv_gate_frame_size),
                    frame_hop=int(args.adv_gate_frame_hop),
                )
                adversarial_delta_gate_mean = float(delta_gate.detach().mean().cpu())
            delta_prediction, delta_target = first_difference_discriminator_audio(prediction, target, delta_gate)
            set_requires_grad(delta_discriminator, True)
            delta_discriminator_optimizer.zero_grad(set_to_none=True)
            delta_real_scores, _delta_real_features = delta_discriminator(delta_target)
            delta_fake_scores, _delta_fake_features = delta_discriminator(delta_prediction.detach())
            adversarial_delta_discriminator = discriminator_lsgan_loss(delta_real_scores, delta_fake_scores)
            if not torch.isfinite(adversarial_delta_discriminator):
                raise RuntimeError(f"non-finite delta discriminator loss at step {step}")
            adversarial_delta_discriminator.backward()
            delta_discriminator_grad_norm = float(
                torch.nn.utils.clip_grad_norm_(delta_discriminator.parameters(), max_norm=5.0).detach().cpu()
            )
            delta_discriminator_optimizer.step()

            set_requires_grad(delta_discriminator, False)
            delta_fake_scores_for_generator, delta_fake_features_for_generator = delta_discriminator(delta_prediction)
            _delta_real_scores_for_generator, delta_real_features_for_generator = delta_discriminator(delta_target)
            adversarial_delta_generator = generator_lsgan_loss(delta_fake_scores_for_generator)
            adversarial_delta_feature = discriminator_feature_matching_loss(
                delta_real_features_for_generator,
                delta_fake_features_for_generator,
            )
            set_requires_grad(delta_discriminator, True)
            if not torch.isfinite(adversarial_delta_generator):
                raise RuntimeError(f"non-finite delta generator adversarial loss at step {step}")
            if not torch.isfinite(adversarial_delta_feature):
                raise RuntimeError(f"non-finite delta adversarial feature loss at step {step}")
        loss = (
            waveform_l1
            + float(args.spectral_weight) * spectral
            + float(args.stft_phase_weight) * stft_phase
            + float(args.feature_hint_weight) * hint
            + float(args.feature_exact_weight) * feature_exact
            + float(args.signature_hint_weight) * signature_hint
            + float(args.signature_temporal_weight) * signature_temporal
            + float(args.signature_phase_weight) * signature_phase
            + float(args.bottleneck_code_weight) * bottleneck_code
            + float(args.ap_amplitude_weight) * ap_amplitude
            + float(args.ap_phase_weight) * ap_phase
            + float(args.ap_complex_weight) * ap_complex
            + float(args.quiet_frame_weight) * quiet_frame
            + float(args.quiet_delta_weight) * quiet_delta
            + float(args.quiet_ceiling_weight) * quiet_ceiling
            + float(args.click_delta_weight) * click_delta
            + float(args.quiet_sample_weight) * quiet_sample
            + float(args.high_band_excess_weight) * high_band_excess
            + float(args.echo_tail_weight) * echo_tail
            + float(args.paired_acoustic_residual_weight) * paired_acoustic
            + float(args.adv_weight) * adversarial_generator
            + float(args.adv_feature_weight) * adversarial_feature
            + float(args.adv_delta_weight) * adversarial_delta_generator
            + float(args.adv_delta_feature_weight) * adversarial_delta_feature
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=5.0).detach().cpu())
        optimizer.step()
        if step == 1 or step % args.log_interval == 0 or step == args.steps:
            log = {
                "step": int(step),
                "loss": float(loss.detach().cpu()),
                "waveform_l1": float(waveform_l1.detach().cpu()),
                "spectral": float(spectral.detach().cpu()),
                "stft_phase": float(stft_phase.detach().cpu()),
                "feature_hint": float(hint.detach().cpu()),
                "feature_exact": float(feature_exact.detach().cpu()),
                "signature_hint": float(signature_hint.detach().cpu()),
                "signature_temporal": float(signature_temporal.detach().cpu()),
                "signature_phase": float(signature_phase.detach().cpu()),
                "bottleneck_code": float(bottleneck_code.detach().cpu()),
                "ap_amplitude": float(ap_amplitude.detach().cpu()),
                "ap_phase": float(ap_phase.detach().cpu()),
                "ap_complex": float(ap_complex.detach().cpu()),
                "quiet_frame": float(quiet_frame.detach().cpu()),
                "quiet_delta": float(quiet_delta.detach().cpu()),
                "quiet_ceiling": float(quiet_ceiling.detach().cpu()),
                "click_delta": float(click_delta.detach().cpu()),
                "quiet_sample": float(quiet_sample.detach().cpu()),
                "high_band_excess": float(high_band_excess.detach().cpu()),
                "echo_tail": float(echo_tail.detach().cpu()),
                "paired_acoustic": float(paired_acoustic.detach().cpu()),
                "paired_acoustic_waveform_l1": float(paired_acoustic_waveform_l1.detach().cpu()),
                "paired_acoustic_spectral": float(paired_acoustic_spectral.detach().cpu()),
                "paired_acoustic_stft_phase": float(paired_acoustic_stft_phase.detach().cpu()),
                "paired_acoustic_echo_tail": float(paired_acoustic_echo_tail.detach().cpu()),
                "adversarial_generator": float(adversarial_generator.detach().cpu()),
                "adversarial_feature": float(adversarial_feature.detach().cpu()),
                "adversarial_discriminator": float(adversarial_discriminator.detach().cpu()),
                "adversarial_gate_mean": adversarial_gate_mean,
                "adversarial_delta_generator": float(adversarial_delta_generator.detach().cpu()),
                "adversarial_delta_feature": float(adversarial_delta_feature.detach().cpu()),
                "adversarial_delta_discriminator": float(adversarial_delta_discriminator.detach().cpu()),
                "adversarial_delta_gate_mean": adversarial_delta_gate_mean,
                "grad_norm": grad_norm,
                "discriminator_grad_norm": discriminator_grad_norm,
                "delta_discriminator_grad_norm": delta_discriminator_grad_norm,
            }
            if uses_lrc_pred_codes:
                log.update(
                    {
                        "lrc_pred_code_batch_total": int(lrc_pred_code_batch_total),
                        "lrc_pred_code_mixed": int(lrc_pred_code_batch_mixed),
                        "lrc_pred_code_mix_fraction": (
                            float(lrc_pred_code_batch_mixed) / float(lrc_pred_code_batch_total)
                            if lrc_pred_code_batch_total
                            else 0.0
                        ),
                        "lrc_pred_code_residual": int(lrc_pred_code_batch_residual),
                        "lrc_pred_code_residual_fraction": (
                            float(lrc_pred_code_batch_residual) / float(lrc_pred_code_batch_total)
                            if lrc_pred_code_batch_total
                            else 0.0
                        ),
                        "lrc_pred_code_cumulative_total": int(lrc_pred_code_total_samples),
                        "lrc_pred_code_cumulative_mixed": int(lrc_pred_code_mixed_samples),
                        "lrc_pred_code_cumulative_mix_fraction": (
                            float(lrc_pred_code_mixed_samples) / float(lrc_pred_code_total_samples)
                            if lrc_pred_code_total_samples
                            else 0.0
                        ),
                        "lrc_pred_code_cumulative_residual": int(lrc_pred_code_residual_samples),
                        "lrc_pred_code_cumulative_residual_fraction": (
                            float(lrc_pred_code_residual_samples) / float(lrc_pred_code_total_samples)
                            if lrc_pred_code_total_samples
                            else 0.0
                        ),
                    }
                )
            logs.append(log)
            print(json.dumps(log, ensure_ascii=False), flush=True)

    # Full-length chunk evaluation can exceed MPS kernel size limits; it is
    # informational, so never let it destroy a finished training run.
    try:
        train_chunk_eval = evaluate_chunks(model, samples, device, lrc_encoder=lrc_encoder)
        eval_chunk_eval = (
            evaluate_chunks(model, eval_samples, device, lrc_encoder=lrc_encoder) if eval_samples is not None else None
        )
    except NotImplementedError as exc:
        print(json.dumps({"chunk_eval_skipped": f"{type(exc).__name__}: {exc}"}), flush=True)
        train_chunk_eval = None
        eval_chunk_eval = None
    checkpoint = args.out_dir / "decoder-student.pt"
    lrc_encoder_checkpoint = args.out_dir / "lrc-encoder.pt" if lrc_encoder is not None else None
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    lrc_encoder_state_dict = lrc_encoder.cpu().state_dict() if lrc_encoder is not None else None
    lrc_pred_code_checkpoint_reload: dict[str, Any] | None = None
    torch.save(
        {
            "model_state_dict": model.cpu().state_dict(),
            "lrc_encoder_state_dict": lrc_encoder_state_dict,
            "config": {
                "in_channels": model_in_channels,
                "source_in_channels": lrc_input_channels,
                "channels": channels,
                "res_layers": int(args.res_layers),
                "variant": str(args.variant),
                "lrc_code_dim": int(args.lrc_code_dim),
                "lrc_encoder_hidden": int(args.lrc_encoder_hidden),
                "rank_ratio": float(args.rank_ratio),
                "activation": str(args.activation),
                "stage_affine": bool(args.stage_affine),
                "factorized_pre_rank": int(args.factorized_pre_rank),
                "piper_res_factor_rank_ratio": float(args.piper_res_factor_rank_ratio),
                "res_bank_scale_mode": str(args.res_bank_scale_mode),
                "stage0_branches": list(stage0_branches),
                "stage1_branches": list(stage1_branches),
                "stage2_branches": list(stage2_branches),
                "stage3_branches": list(stage3_branches),
                "fsd_dim": int(args.fsd_dim),
                "fsd_blocks": int(args.fsd_blocks),
                "fsd_film_rank": int(args.fsd_film_rank),
                "fsd_head_rank": int(args.fsd_head_rank),
                "stage_projection_bottlenecks": list(stage_projection_bottlenecks),
                "teacher_init_checkpoint": str(args.teacher_init_checkpoint) if args.teacher_init_checkpoint else None,
                "teacher_init_method": str(args.teacher_init_method),
                "teacher_init": teacher_init_summary,
                "init_decoder_checkpoint": str(args.init_decoder_checkpoint) if args.init_decoder_checkpoint else None,
                "decoder_init": decoder_init_summary,
                "post_filter_channels": int(args.post_filter_channels),
                "post_filter_layers": int(args.post_filter_layers),
                "post_filter_kernel": int(args.post_filter_kernel),
                "post_filter_scale": float(args.post_filter_scale),
                "pre_tanh_repair_channels": int(args.pre_tanh_repair_channels),
                "pre_tanh_repair_layers": int(args.pre_tanh_repair_layers),
                "pre_tanh_repair_kernel": int(args.pre_tanh_repair_kernel),
                "pre_tanh_repair_scale": float(args.pre_tanh_repair_scale),
                "istft_n_fft": int(args.istft_n_fft),
                "ap_amplitude_weight": float(args.ap_amplitude_weight),
                "ap_phase_weight": float(args.ap_phase_weight),
                "ap_complex_weight": float(args.ap_complex_weight),
                "spectral_head_init": str(args.spectral_head_init),
                "spectral_head_init_scale": float(args.spectral_head_init_scale),
                "ap_amp_init_bias": float(args.ap_amp_init_bias),
                "ap_phase_real_init_bias": float(args.ap_phase_real_init_bias),
                "spectral_head_init_summary": spectral_head_init_summary,
                "feature_exact_weight": float(args.feature_exact_weight),
                "feature_exact_keys": feature_exact_keys,
                "feature_exact_teacher_init": feature_exact_teacher_init_summary,
                "signature_pack_dir": str(args.signature_pack_dir) if args.signature_pack_dir else None,
                "exact_feature_pack_dir": str(args.exact_feature_pack_dir) if args.exact_feature_pack_dir else None,
                "signature_hint_weight": float(args.signature_hint_weight),
                "signature_temporal_weight": float(args.signature_temporal_weight),
                "signature_phase_weight": float(args.signature_phase_weight),
                "signature_phase_bins": int(args.signature_phase_bins),
                "signature_keys": signature_keys,
                "bottleneck_code_checkpoint": str(args.bottleneck_code_checkpoint)
                if args.bottleneck_code_checkpoint
                else None,
                "bottleneck_code_weight": float(args.bottleneck_code_weight),
                "bottleneck_code_keys": bottleneck_code_signature_keys,
                "bottleneck_code_training_only_parameters": (
                    int(bottleneck_codebook.parameter_count) if bottleneck_codebook is not None else 0
                ),
                "echo_tail_weight": float(args.echo_tail_weight),
                "echo_tail_min_ms": float(args.echo_tail_min_ms),
                "echo_tail_max_ms": float(args.echo_tail_max_ms),
                "echo_tail_lags": int(args.echo_tail_lags),
                "echo_tail_margin": float(args.echo_tail_margin),
                "paired_acoustic_residual_weight": float(args.paired_acoustic_residual_weight),
                "paired_acoustic_residual_max_scale": float(args.paired_acoustic_residual_max_scale),
                "lrc_pred_code_mix_prob": float(args.lrc_pred_code_mix_prob),
                "lrc_pred_code_checkpoint": (
                    str(args.lrc_pred_code_checkpoint) if args.lrc_pred_code_checkpoint else None
                ),
                "lrc_pred_code_residual_prob": float(args.lrc_pred_code_residual_prob),
                "lrc_pred_code_residual_max_scale": float(args.lrc_pred_code_residual_max_scale),
                "lrc_pred_code_cache": lrc_pred_code_cache_summary,
            },
            "train_args": vars(args),
            "train_chunk_eval": train_chunk_eval,
            "eval_chunk_eval": eval_chunk_eval,
            "chunk_eval": train_chunk_eval,
            "decoder_parameters": parameter_count,
            "lrc_encoder_training_only_parameters": int(lrc_encoder_parameter_count),
            "adversarial_discriminator_parameters": discriminator_parameter_count,
            "adversarial_delta_discriminator_parameters": delta_discriminator_parameter_count,
        },
        checkpoint,
    )
    if uses_lrc_pred_codes:
        reloaded_checkpoint = load_torch_checkpoint(checkpoint, "saved decoder checkpoint")
        reloaded_state = reloaded_checkpoint.get("model_state_dict")
        reloaded_lrc_state = reloaded_checkpoint.get("lrc_encoder_state_dict")
        if not isinstance(reloaded_state, dict):
            raise RuntimeError(f"{checkpoint}: saved checkpoint reload missing model_state_dict")
        if lrc_encoder is not None and not isinstance(reloaded_lrc_state, dict):
            raise RuntimeError(f"{checkpoint}: saved checkpoint reload missing lrc_encoder_state_dict")
        reloaded_tensor_count = 0
        for name, tensor in reloaded_state.items():
            if torch.is_tensor(tensor):
                if not torch.isfinite(tensor).all().item():
                    raise RuntimeError(f"{checkpoint}: non-finite tensor after reload: model_state_dict.{name}")
                reloaded_tensor_count += 1
        reloaded_lrc_tensor_count = 0
        if isinstance(reloaded_lrc_state, dict):
            for name, tensor in reloaded_lrc_state.items():
                if torch.is_tensor(tensor):
                    if not torch.isfinite(tensor).all().item():
                        raise RuntimeError(
                            f"{checkpoint}: non-finite tensor after reload: lrc_encoder_state_dict.{name}"
                        )
                    reloaded_lrc_tensor_count += 1
        lrc_pred_code_checkpoint_reload = {
            "checkpoint": str(checkpoint),
            "ok": True,
            "model_state_tensors": int(reloaded_tensor_count),
            "lrc_encoder_state_tensors": int(reloaded_lrc_tensor_count),
        }
        print(json.dumps({"checkpoint_reload": lrc_pred_code_checkpoint_reload}, ensure_ascii=False), flush=True)
    model.to(device)
    if lrc_encoder is not None:
        lrc_encoder.to(device)
        if lrc_encoder_checkpoint is None or lrc_encoder_state_dict is None:
            raise RuntimeError("internal LRC encoder checkpoint state was not prepared")
        torch.save(
            {
                "model_state_dict": lrc_encoder_state_dict,
                "config": {
                    "architecture": "lrc_encoder",
                    "in_channels": int(lrc_input_channels),
                    "hidden": int(args.lrc_encoder_hidden),
                    "code_dim": int(args.lrc_code_dim),
                },
                "training_only": True,
                "params": int(lrc_encoder_parameter_count),
                "decoder_checkpoint": str(checkpoint),
                "train_args": vars(args),
            },
            lrc_encoder_checkpoint,
        )
    render_rows = eval_rows if eval_rows is not None else rows
    render_samples = eval_samples if eval_samples is not None else samples
    render_label = "heldout" if eval_rows is not None else "train"
    render_summary = render_dashboard(args, render_rows, render_samples, model, device, render_label, lrc_encoder=lrc_encoder)
    report = {
        "passed": True,
        "train_pack_dir": str(args.pack_dir),
        "eval_pack_dir": str(args.eval_pack_dir) if args.eval_pack_dir else None,
        "input_target_dir": str(args.input_target_dir) if args.input_target_dir else None,
        "eval_input_target_dir": str(args.eval_input_target_dir) if args.eval_input_target_dir else None,
        "input_target_key": str(args.input_target_key),
        "oracle_target_mix_prob": float(args.oracle_target_mix_prob),
        "pack_dir": str(args.pack_dir),
        "teacher_decoder": str(args.teacher_decoder),
        "acoustic_checkpoint": str(args.acoustic_checkpoint) if args.acoustic_checkpoint else None,
        "out_dir": str(args.out_dir),
        "checkpoint": str(checkpoint),
        "device": str(device),
        "seed": int(args.seed),
        "steps": int(args.steps),
        "lr": float(args.lr),
        "batch_size": int(args.batch_size),
        "crop_frames": int(args.crop_frames),
        "channels": list(channels),
        "variant": str(args.variant),
        "rank_ratio": float(args.rank_ratio),
        "activation": str(args.activation),
        "stage_affine": bool(args.stage_affine),
        "factorized_pre_rank": int(args.factorized_pre_rank),
        "piper_res_factor_rank_ratio": float(args.piper_res_factor_rank_ratio),
        "res_bank_scale_mode": str(args.res_bank_scale_mode),
        "stage0_branches": list(stage0_branches),
        "stage1_branches": list(stage1_branches),
        "stage2_branches": list(stage2_branches),
        "stage3_branches": list(stage3_branches),
        "fsd_dim": int(args.fsd_dim),
        "fsd_blocks": int(args.fsd_blocks),
        "fsd_film_rank": int(args.fsd_film_rank),
        "fsd_head_rank": int(args.fsd_head_rank),
        "lrc_code_dim": int(args.lrc_code_dim),
        "lrc_encoder_hidden": int(args.lrc_encoder_hidden),
        "stage_projection_bottlenecks": list(stage_projection_bottlenecks),
        "teacher_init_checkpoint": str(args.teacher_init_checkpoint) if args.teacher_init_checkpoint else None,
        "teacher_init_method": str(args.teacher_init_method),
        "teacher_init": teacher_init_summary,
        "init_decoder_checkpoint": str(args.init_decoder_checkpoint) if args.init_decoder_checkpoint else None,
        "decoder_init": decoder_init_summary,
        "post_filter_channels": int(args.post_filter_channels),
        "post_filter_layers": int(args.post_filter_layers),
        "post_filter_kernel": int(args.post_filter_kernel),
        "post_filter_scale": float(args.post_filter_scale),
        "pre_tanh_repair_channels": int(args.pre_tanh_repair_channels),
        "pre_tanh_repair_layers": int(args.pre_tanh_repair_layers),
        "pre_tanh_repair_kernel": int(args.pre_tanh_repair_kernel),
        "pre_tanh_repair_scale": float(args.pre_tanh_repair_scale),
        "freeze_decoder_body": bool(args.freeze_decoder_body),
        "istft_n_fft": int(args.istft_n_fft),
        "ap_amplitude_weight": float(args.ap_amplitude_weight),
        "ap_phase_weight": float(args.ap_phase_weight),
        "ap_complex_weight": float(args.ap_complex_weight),
        "spectral_head_init": str(args.spectral_head_init),
        "spectral_head_init_scale": float(args.spectral_head_init_scale),
        "ap_amp_init_bias": float(args.ap_amp_init_bias),
        "ap_phase_real_init_bias": float(args.ap_phase_real_init_bias),
        "spectral_head_init_summary": spectral_head_init_summary,
        "res_layers": int(args.res_layers),
        "spectral_weight": float(args.spectral_weight),
        "stft_phase_weight": float(args.stft_phase_weight),
        "feature_hint_weight": float(args.feature_hint_weight),
        "feature_exact_weight": float(args.feature_exact_weight),
        "feature_exact_keys": feature_exact_keys,
        "feature_exact_teacher_init": feature_exact_teacher_init_summary,
        "exact_feature_pack_dir": str(args.exact_feature_pack_dir) if args.exact_feature_pack_dir else None,
        "quiet_frame_weight": float(args.quiet_frame_weight),
        "quiet_delta_weight": float(args.quiet_delta_weight),
        "quiet_ceiling_weight": float(args.quiet_ceiling_weight),
        "quiet_ceiling_margin_db": float(args.quiet_ceiling_margin_db),
        "click_delta_weight": float(args.click_delta_weight),
        "click_delta_margin": float(args.click_delta_margin),
        "click_delta_target_scale": float(args.click_delta_target_scale),
        "click_delta_topk_frac": float(args.click_delta_topk_frac),
        "quiet_sample_weight": float(args.quiet_sample_weight),
        "quiet_sample_quantile": float(args.quiet_sample_quantile),
        "quiet_sample_margin": float(args.quiet_sample_margin),
        "quiet_sample_target_scale": float(args.quiet_sample_target_scale),
        "high_band_excess_weight": float(args.high_band_excess_weight),
        "high_band_excess_hz": float(args.high_band_excess_hz),
        "high_band_excess_margin_db": float(args.high_band_excess_margin_db),
        "echo_tail_weight": float(args.echo_tail_weight),
        "echo_tail_min_ms": float(args.echo_tail_min_ms),
        "echo_tail_max_ms": float(args.echo_tail_max_ms),
        "echo_tail_lags": int(args.echo_tail_lags),
        "echo_tail_margin": float(args.echo_tail_margin),
        "adv_weight": float(args.adv_weight),
        "adv_feature_weight": float(args.adv_feature_weight),
        "adv_delta_weight": float(args.adv_delta_weight),
        "adv_delta_feature_weight": float(args.adv_delta_feature_weight),
        "adv_start_step": int(args.adv_start_step),
        "adv_lr": float(args.adv_lr),
        "adv_periods": list(adv_periods),
        "adv_channels": list(adv_channels),
        "adv_gate_mode": str(args.adv_gate_mode),
        "adv_gate_quantile": float(args.adv_gate_quantile),
        "adv_gate_sharpness": float(args.adv_gate_sharpness),
        "adv_gate_frame_size": int(args.adv_gate_frame_size),
        "adv_gate_frame_hop": int(args.adv_gate_frame_hop),
        "quiet_frame_quantile": float(args.quiet_frame_quantile),
        "quiet_frame_size": int(args.quiet_frame_size),
        "quiet_frame_hop": int(args.quiet_frame_hop),
        "signature_pack_dir": str(args.signature_pack_dir) if args.signature_pack_dir else None,
        "signature_hint_weight": float(args.signature_hint_weight),
        "signature_temporal_weight": float(args.signature_temporal_weight),
        "signature_phase_weight": float(args.signature_phase_weight),
        "signature_phase_bins": int(args.signature_phase_bins),
        "signature_keys": signature_keys,
        "bottleneck_code_checkpoint": str(args.bottleneck_code_checkpoint) if args.bottleneck_code_checkpoint else None,
        "bottleneck_code_weight": float(args.bottleneck_code_weight),
        "bottleneck_code_keys": bottleneck_code_signature_keys,
        "bottleneck_code_training_only_parameters": (
            int(bottleneck_codebook.parameter_count) if bottleneck_codebook is not None else 0
        ),
        "acoustic_latent_mix_prob": float(args.acoustic_latent_mix_prob),
        "acoustic_latent_residual_prob": float(args.acoustic_latent_residual_prob),
        "acoustic_latent_residual_max_scale": float(args.acoustic_latent_residual_max_scale),
        "paired_acoustic_residual_weight": float(args.paired_acoustic_residual_weight),
        "paired_acoustic_residual_max_scale": float(args.paired_acoustic_residual_max_scale),
        "lrc_pred_code_mix_prob": float(args.lrc_pred_code_mix_prob),
        "lrc_pred_code_checkpoint": str(args.lrc_pred_code_checkpoint) if args.lrc_pred_code_checkpoint else None,
        "lrc_pred_code_residual_prob": float(args.lrc_pred_code_residual_prob),
        "lrc_pred_code_residual_max_scale": float(args.lrc_pred_code_residual_max_scale),
        "lrc_pred_code_cache": lrc_pred_code_cache_summary,
        "lrc_pred_code_mix_summary": {
            "enabled": bool(uses_lrc_pred_codes),
            "samples_seen": int(lrc_pred_code_total_samples),
            "mixed_samples": int(lrc_pred_code_mixed_samples),
            "mixed_fraction": (
                float(lrc_pred_code_mixed_samples) / float(lrc_pred_code_total_samples)
                if lrc_pred_code_total_samples
                else 0.0
            ),
            "residual_samples": int(lrc_pred_code_residual_samples),
            "residual_fraction": (
                float(lrc_pred_code_residual_samples) / float(lrc_pred_code_total_samples)
                if lrc_pred_code_total_samples
                else 0.0
            ),
        },
        "checkpoint_reload": lrc_pred_code_checkpoint_reload,
        "train_rows": int(len(rows)),
        "train_chunks": int(len(samples)),
        "eval_rows": int(len(eval_rows)) if eval_rows is not None else 0,
        "eval_chunks": int(len(eval_samples)) if eval_samples is not None else 0,
        "rows": int(len(rows)),
        "chunks": int(len(samples)),
        "in_channels": int(model_in_channels),
        "source_in_channels": int(lrc_input_channels),
        "assert_max_decoder_params": int(args.assert_max_decoder_params),
        "decoder_parameters": int(parameter_count),
        "decoder_trainable_parameters": int(decoder_trainable_parameter_count),
        "lrc_encoder_checkpoint": str(lrc_encoder_checkpoint) if lrc_encoder_checkpoint is not None else None,
        "lrc_encoder_training_only_parameters": int(lrc_encoder_parameter_count),
        "lrc_encoder_trainable_parameters": int(lrc_encoder_trainable_parameter_count),
        "adversarial_discriminator_parameters": int(discriminator_parameter_count),
        "adversarial_delta_discriminator_parameters": int(delta_discriminator_parameter_count),
        "logs": logs,
        "train_chunk_eval": train_chunk_eval,
        "eval_chunk_eval": eval_chunk_eval,
        "chunk_eval": train_chunk_eval,
        "render_summary": render_summary,
    }
    write_json(args.out_dir / "train-report.json", report)
    return report


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = train(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
