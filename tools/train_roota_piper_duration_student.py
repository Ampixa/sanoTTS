#!/usr/bin/env python3
"""Train a tiny Piper-ID duration predictor for Root A standalone probes."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_PACK = (
    ROOT
    / "artifacts"
    / "sub10m-search"
    / "root-a-piper-vits"
    / "chitwan-medium-a1-mod10-train2048-max12-acoustic-pack-20260625"
)
DEFAULT_EVAL_PACK = (
    ROOT
    / "artifacts"
    / "sub10m-search"
    / "root-a-piper-vits"
    / "chitwan-medium-a1-mod10-heldout64-max12-piper-native-pack-20260625"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "artifacts"
    / "sub10m-search"
    / "root-a-piper-vits"
    / "chitwan-medium-a5-duration-student-h64d3-20260625"
)
DEFAULT_PAUSE_TOKEN_IDS = (4, 8, 10, 11, 12, 13)


@dataclass(frozen=True)
class DurationSample:
    row_id: str
    row_index: int
    text: str
    chunk_index: int
    phoneme_ids: np.ndarray
    durations: np.ndarray
    tensor_path: Path


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

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_f = mask.unsqueeze(1).to(x.dtype)
        return (x + self.scale * self.net(x * mask_f)) * mask_f


class DurationStudent(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        hidden: int,
        depth: int,
        kernel_size: int,
        max_tokens: int,
    ) -> None:
        super().__init__()
        if vocab_size <= 0 or hidden <= 0 or depth <= 0 or kernel_size <= 0 or max_tokens <= 0:
            raise ValueError(
                f"invalid DurationStudent config: vocab_size={vocab_size}, hidden={hidden}, "
                f"depth={depth}, kernel_size={kernel_size}, max_tokens={max_tokens}"
            )
        self.embedding = nn.Embedding(vocab_size, hidden)
        self.input_proj = nn.Conv1d(hidden + 3, hidden, 1)
        self.blocks = nn.ModuleList([ResidualConvBlock(hidden, kernel_size) for _ in range(depth)])
        self.output = nn.Conv1d(hidden, 1, 1)
        self.max_tokens = int(max_tokens)

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if ids.ndim != 2 or mask.ndim != 2:
            raise RuntimeError(f"expected 2D ids/mask, got {ids.shape} and {mask.shape}")
        if ids.shape != mask.shape:
            raise RuntimeError(f"id/mask shape mismatch: {ids.shape} vs {mask.shape}")
        batch, tokens = ids.shape
        if tokens <= 0:
            raise RuntimeError("empty token batch")
        positions = torch.linspace(0.0, 1.0, tokens, device=ids.device).unsqueeze(0).expand(batch, tokens)
        lengths = mask.sum(dim=1).clamp_min(1).to(dtype=torch.float32)
        length_hint = (torch.log1p(lengths) / math.log1p(float(self.max_tokens))).unsqueeze(1).expand(batch, tokens)
        valid_hint = mask.to(dtype=torch.float32)
        x = self.embedding(ids).transpose(1, 2)
        features = torch.stack([positions, length_hint, valid_hint], dim=1)
        x = self.input_proj(torch.cat([x, features], dim=1))
        for block in self.blocks:
            x = block(x, mask)
        log_duration = self.output(x).squeeze(1)
        return log_duration * mask.to(dtype=log_duration.dtype)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack-dir", type=Path, default=DEFAULT_TRAIN_PACK)
    parser.add_argument("--eval-pack-dir", type=Path, default=DEFAULT_EVAL_PACK)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--total-weight", type=float, default=0.35)
    parser.add_argument(
        "--target-duration-scale",
        type=float,
        default=1.0,
        help="Scale source w_ceil duration targets before training/evaluation.",
    )
    parser.add_argument(
        "--pause-preserve-scale",
        type=float,
        default=0.0,
        help=(
            "If positive, use at least this target scale for pause tokens and the following "
            "pause-preserve-window tokens. Disabled by default."
        ),
    )
    parser.add_argument(
        "--pause-preserve-window",
        type=int,
        default=0,
        help="Number of tokens after each pause token to protect with pause-preserve-scale.",
    )
    parser.add_argument(
        "--pause-token-ids",
        type=str,
        default=",".join(str(token_id) for token_id in DEFAULT_PAUSE_TOKEN_IDS),
        help="Comma-separated Piper phoneme IDs treated as pause punctuation.",
    )
    parser.add_argument(
        "--long-preserve-threshold",
        type=float,
        default=0.0,
        help=(
            "If positive, source durations at or above this frame count use at least "
            "--long-preserve-scale. This preserves teacher-important long spans while "
            "allowing ordinary tokens to stay compressed."
        ),
    )
    parser.add_argument(
        "--long-preserve-scale",
        type=float,
        default=0.0,
        help="Minimum target scale for tokens selected by --long-preserve-threshold. Disabled when <= 0.",
    )
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=6262)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--max-duration", type=int, default=80)
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


def parse_token_ids(raw: str) -> tuple[int, ...]:
    token_ids: list[int] = []
    for part in str(raw).split(","):
        item = part.strip()
        if not item:
            continue
        try:
            token_id = int(item)
        except ValueError as exc:
            raise ValueError(f"invalid token id in --pause-token-ids: {item!r}") from exc
        if token_id < 0:
            raise ValueError(f"pause token ids must be non-negative, got {token_id}")
        token_ids.append(token_id)
    return tuple(sorted(set(token_ids)))


def scale_durations(
    phoneme_ids: np.ndarray,
    source_durations: np.ndarray,
    *,
    target_duration_scale: float,
    pause_preserve_scale: float,
    pause_preserve_window: int,
    pause_token_ids: tuple[int, ...],
    long_preserve_threshold: float = 0.0,
    long_preserve_scale: float = 0.0,
) -> np.ndarray:
    if phoneme_ids.shape != source_durations.shape:
        raise RuntimeError(f"duration/id length mismatch before scaling: {phoneme_ids.shape} vs {source_durations.shape}")
    scales = np.full(source_durations.shape, float(target_duration_scale), dtype=np.float32)
    if pause_preserve_scale > 0.0 and pause_token_ids:
        if pause_preserve_window < 0:
            raise ValueError(f"pause_preserve_window must be non-negative, got {pause_preserve_window}")
        pause_mask = np.isin(phoneme_ids, np.asarray(pause_token_ids, dtype=np.int64))
        for pause_index in np.flatnonzero(pause_mask):
            end = min(int(pause_index) + int(pause_preserve_window) + 1, int(source_durations.size))
            scales[pause_index:end] = np.maximum(scales[pause_index:end], float(pause_preserve_scale))
    if long_preserve_threshold > 0.0 and long_preserve_scale > 0.0:
        long_mask = source_durations.astype(np.float32) >= float(long_preserve_threshold)
        scales[long_mask] = np.maximum(scales[long_mask], float(long_preserve_scale))
    durations = np.rint(source_durations.astype(np.float32) * scales).astype(np.int64)
    return np.maximum(durations, 1)


def load_samples(
    pack_dir: Path,
    *,
    target_duration_scale: float = 1.0,
    pause_preserve_scale: float = 0.0,
    pause_preserve_window: int = 0,
    pause_token_ids: tuple[int, ...] = DEFAULT_PAUSE_TOKEN_IDS,
    long_preserve_threshold: float = 0.0,
    long_preserve_scale: float = 0.0,
) -> tuple[list[dict[str, Any]], list[DurationSample], int, int]:
    if not math.isfinite(target_duration_scale) or target_duration_scale <= 0.0:
        raise ValueError(f"target_duration_scale must be finite and positive, got {target_duration_scale}")
    if not math.isfinite(pause_preserve_scale) or pause_preserve_scale < 0.0:
        raise ValueError(f"pause_preserve_scale must be finite and non-negative, got {pause_preserve_scale}")
    if pause_preserve_window < 0:
        raise ValueError(f"pause_preserve_window must be non-negative, got {pause_preserve_window}")
    if not math.isfinite(long_preserve_threshold) or long_preserve_threshold < 0.0:
        raise ValueError(f"long_preserve_threshold must be finite and non-negative, got {long_preserve_threshold}")
    if not math.isfinite(long_preserve_scale) or long_preserve_scale < 0.0:
        raise ValueError(f"long_preserve_scale must be finite and non-negative, got {long_preserve_scale}")
    require_dir(pack_dir, "pack directory")
    rows = read_json(pack_dir / "rows.json")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"{pack_dir / 'rows.json'} must contain a non-empty list")
    samples: list[DurationSample] = []
    max_id = 0
    max_tokens = 0
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
            with np.load(tensor_path) as tensors:
                missing = {"phoneme_ids", "w_ceil"} - set(tensors.files)
                if missing:
                    raise RuntimeError(f"{tensor_path}: missing tensors {sorted(missing)}")
                phoneme_ids = np.asarray(tensors["phoneme_ids"], dtype=np.int64).reshape(-1)
                source_durations = np.asarray(tensors["w_ceil"], dtype=np.float32).reshape(-1)
                durations = scale_durations(
                    phoneme_ids,
                    source_durations,
                    target_duration_scale=target_duration_scale,
                    pause_preserve_scale=pause_preserve_scale,
                    pause_preserve_window=pause_preserve_window,
                    pause_token_ids=pause_token_ids,
                    long_preserve_threshold=long_preserve_threshold,
                    long_preserve_scale=long_preserve_scale,
                )
            if phoneme_ids.size <= 0:
                raise RuntimeError(f"{tensor_path}: empty phoneme_ids")
            if phoneme_ids.shape != durations.shape:
                raise RuntimeError(f"{tensor_path}: duration/id length mismatch")
            if np.any(durations <= 0):
                raise RuntimeError(f"{tensor_path}: non-positive duration")
            max_id = max(max_id, int(phoneme_ids.max()))
            max_tokens = max(max_tokens, int(phoneme_ids.size))
            samples.append(
                DurationSample(
                    row_id=row_id,
                    row_index=row_index,
                    text=text,
                    chunk_index=int(chunk.get("chunk_index") or 0),
                    phoneme_ids=phoneme_ids,
                    durations=durations,
                    tensor_path=tensor_path,
                )
            )
    if not samples:
        raise RuntimeError(f"{pack_dir}: produced no duration samples")
    return rows, samples, max_id + 1, max_tokens


def pad_batch(samples: list[DurationSample], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_len = max(int(sample.phoneme_ids.size) for sample in samples)
    ids = torch.zeros((len(samples), max_len), dtype=torch.long)
    durations = torch.ones((len(samples), max_len), dtype=torch.float32)
    mask = torch.zeros((len(samples), max_len), dtype=torch.bool)
    for index, sample in enumerate(samples):
        count = int(sample.phoneme_ids.size)
        ids[index, :count] = torch.as_tensor(sample.phoneme_ids, dtype=torch.long)
        durations[index, :count] = torch.as_tensor(sample.durations, dtype=torch.float32)
        mask[index, :count] = True
    return ids.to(device), durations.to(device), mask.to(device)


def count_parameters(model: nn.Module) -> int:
    return int(sum(param.numel() for param in model.parameters()))


def duration_loss(
    pred_log: torch.Tensor,
    target_duration: torch.Tensor,
    mask: torch.Tensor,
    total_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mask_f = mask.to(dtype=pred_log.dtype)
    target_log = torch.log(target_duration.clamp_min(1.0))
    token_loss = torch.nn.functional.smooth_l1_loss(
        pred_log[mask],
        target_log[mask],
        beta=0.25,
    )
    pred_frames = torch.exp(pred_log).clamp_min(1.0) * mask_f
    target_frames = target_duration * mask_f
    total_loss = torch.mean(
        torch.square(
            torch.log(pred_frames.sum(dim=1).clamp_min(1.0))
            - torch.log(target_frames.sum(dim=1).clamp_min(1.0))
        )
    )
    loss = token_loss + float(total_weight) * total_loss
    return loss, token_loss, total_loss


@torch.no_grad()
def predict_durations(
    model: DurationStudent,
    ids: torch.Tensor,
    mask: torch.Tensor,
    *,
    max_duration: int,
    length_scale: float = 1.0,
) -> torch.Tensor:
    pred_log = model(ids, mask)
    pred = torch.round(torch.exp(pred_log).clamp_min(1.0) * float(length_scale))
    pred = torch.clamp(pred, min=1.0, max=float(max_duration)).to(dtype=torch.long)
    return pred.masked_fill(~mask, 0)


@torch.no_grad()
def evaluate(
    model: DurationStudent,
    samples: list[DurationSample],
    device: torch.device,
    *,
    batch_size: int,
    max_duration: int,
) -> dict[str, Any]:
    model.eval()
    token_abs_errors: list[float] = []
    token_exact: list[float] = []
    frame_ratios: list[float] = []
    abs_log_frame_ratios: list[float] = []
    abs_frame_errors: list[int] = []
    pred_frame_totals: list[int] = []
    target_frame_totals: list[int] = []
    pred_duration_values: list[int] = []
    target_duration_values: list[int] = []

    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        ids, durations, mask = pad_batch(batch, device)
        pred = predict_durations(model, ids, mask, max_duration=max_duration)
        pred_cpu = pred.cpu().numpy()
        target_cpu = durations.to(dtype=torch.long).cpu().numpy()
        mask_cpu = mask.cpu().numpy()
        for index in range(len(batch)):
            valid = mask_cpu[index]
            p = pred_cpu[index][valid].astype(np.int64)
            t = target_cpu[index][valid].astype(np.int64)
            pred_sum = int(p.sum())
            target_sum = int(t.sum())
            token_abs_errors.extend(np.abs(p - t).astype(np.float64).tolist())
            token_exact.extend((p == t).astype(np.float64).tolist())
            frame_ratios.append(float(pred_sum / max(target_sum, 1)))
            abs_log_frame_ratios.append(abs(math.log(max(pred_sum, 1) / max(target_sum, 1))))
            abs_frame_errors.append(abs(pred_sum - target_sum))
            pred_frame_totals.append(pred_sum)
            target_frame_totals.append(target_sum)
            pred_duration_values.extend(p.tolist())
            target_duration_values.extend(t.tolist())

    model.train()
    token_abs = np.asarray(token_abs_errors, dtype=np.float64)
    exact = np.asarray(token_exact, dtype=np.float64)
    ratios = np.asarray(frame_ratios, dtype=np.float64)
    log_ratios = np.asarray(abs_log_frame_ratios, dtype=np.float64)
    abs_frames = np.asarray(abs_frame_errors, dtype=np.float64)
    return {
        "samples": int(len(samples)),
        "tokens": int(token_abs.size),
        "token_mae": float(token_abs.mean()),
        "token_exact": float(exact.mean()),
        "frame_ratio_mean": float(ratios.mean()),
        "frame_ratio_median": float(np.median(ratios)),
        "frame_ratio_min": float(ratios.min()),
        "frame_ratio_max": float(ratios.max()),
        "abs_log_frame_ratio_mean": float(log_ratios.mean()),
        "abs_frame_error_mean": float(abs_frames.mean()),
        "abs_frame_error_median": float(np.median(abs_frames)),
        "pred_frame_total": int(sum(pred_frame_totals)),
        "target_frame_total": int(sum(target_frame_totals)),
        "pred_target_total_ratio": float(sum(pred_frame_totals) / max(sum(target_frame_totals), 1)),
        "pred_duration_hist": histogram(pred_duration_values),
        "target_duration_hist": histogram(target_duration_values),
    }


def histogram(values: list[int], max_key: int = 12) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(int(value)) if int(value) <= max_key else f">{max_key}"
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (item[0].startswith(">"), int(item[0][1:] if item[0].startswith(">") else item[0]))))


def save_checkpoint(path: Path, model: DurationStudent, config: dict[str, Any], args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model_state_dict": model.cpu().state_dict(),
            "config": config,
            "train_args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "params": count_parameters(model),
        },
        tmp,
    )
    tmp.replace(path)


def load_model_from_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[DurationStudent, dict[str, Any]]:
    require_file(checkpoint_path, "duration checkpoint")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise RuntimeError(f"{checkpoint_path}: missing config")
    model = DurationStudent(
        vocab_size=int(config["vocab_size"]),
        hidden=int(config["hidden"]),
        depth=int(config["depth"]),
        kernel_size=int(config["kernel_size"]),
        max_tokens=int(config["max_tokens"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, config


def main() -> None:
    args = parse_args()
    if args.steps <= 0:
        raise ValueError(f"--steps must be positive, got {args.steps}")
    if args.batch_size <= 0:
        raise ValueError(f"--batch-size must be positive, got {args.batch_size}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    pause_token_ids = parse_token_ids(args.pause_token_ids)

    train_rows, train_samples, train_vocab_size, train_max_tokens = load_samples(
        args.pack_dir,
        target_duration_scale=args.target_duration_scale,
        pause_preserve_scale=args.pause_preserve_scale,
        pause_preserve_window=args.pause_preserve_window,
        pause_token_ids=pause_token_ids,
        long_preserve_threshold=args.long_preserve_threshold,
        long_preserve_scale=args.long_preserve_scale,
    )
    eval_rows, eval_samples, eval_vocab_size, eval_max_tokens = load_samples(
        args.eval_pack_dir,
        target_duration_scale=args.target_duration_scale,
        pause_preserve_scale=args.pause_preserve_scale,
        pause_preserve_window=args.pause_preserve_window,
        pause_token_ids=pause_token_ids,
        long_preserve_threshold=args.long_preserve_threshold,
        long_preserve_scale=args.long_preserve_scale,
    )
    vocab_size = max(train_vocab_size, eval_vocab_size)
    max_tokens = max(train_max_tokens, eval_max_tokens)
    device = pick_device(args.device)
    config = {
        "architecture": "duration_conv",
        "vocab_size": int(vocab_size),
        "hidden": int(args.hidden),
        "depth": int(args.depth),
        "kernel_size": int(args.kernel_size),
        "max_tokens": int(max_tokens),
        "max_duration": int(args.max_duration),
        "target_duration_scale": float(args.target_duration_scale),
        "pause_preserve_scale": float(args.pause_preserve_scale),
        "pause_preserve_window": int(args.pause_preserve_window),
        "pause_token_ids": list(pause_token_ids),
        "long_preserve_threshold": float(args.long_preserve_threshold),
        "long_preserve_scale": float(args.long_preserve_scale),
    }
    model = DurationStudent(
        vocab_size=vocab_size,
        hidden=args.hidden,
        depth=args.depth,
        kernel_size=args.kernel_size,
        max_tokens=max_tokens,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.98))
    rng = random.Random(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        args.out_dir / "config.json",
        {
            "model_config": config,
            "params": count_parameters(model),
            "train_pack": str(args.pack_dir),
            "eval_pack": str(args.eval_pack_dir),
            "train_rows": len(train_rows),
            "train_chunks": len(train_samples),
            "eval_rows": len(eval_rows),
            "eval_chunks": len(eval_samples),
            "device": str(device),
            "target_duration_scale": float(args.target_duration_scale),
            "pause_preserve_scale": float(args.pause_preserve_scale),
            "pause_preserve_window": int(args.pause_preserve_window),
            "pause_token_ids": list(pause_token_ids),
            "long_preserve_threshold": float(args.long_preserve_threshold),
            "long_preserve_scale": float(args.long_preserve_scale),
        },
    )

    logs: list[dict[str, Any]] = []
    started = time.time()
    print(
        json.dumps(
            {
                "event": "start",
                "params": count_parameters(model),
                "train_chunks": len(train_samples),
                "eval_chunks": len(eval_samples),
                "vocab_size": vocab_size,
                "max_tokens": max_tokens,
                "device": str(device),
                "target_duration_scale": float(args.target_duration_scale),
                "pause_preserve_scale": float(args.pause_preserve_scale),
                "pause_preserve_window": int(args.pause_preserve_window),
                "pause_token_ids": list(pause_token_ids),
                "long_preserve_threshold": float(args.long_preserve_threshold),
                "long_preserve_scale": float(args.long_preserve_scale),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    for step in range(1, args.steps + 1):
        batch = [train_samples[rng.randrange(len(train_samples))] for _ in range(args.batch_size)]
        ids, durations, mask = pad_batch(batch, device)
        pred_log = model(ids, mask)
        loss, token_loss, total_loss = duration_loss(pred_log, durations, mask, args.total_weight)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip).detach().cpu())
        optimizer.step()

        if step == 1 or step % args.log_interval == 0 or step == args.steps:
            with torch.no_grad():
                pred = predict_durations(model, ids, mask, max_duration=args.max_duration)
                pred_sum = pred.sum(dim=1).to(dtype=torch.float32)
                target_sum = durations.sum(dim=1)
                batch_ratio = torch.mean(pred_sum / target_sum.clamp_min(1.0)).detach().cpu()
                batch_token_mae = torch.mean(torch.abs(pred.to(dtype=torch.float32)[mask] - durations[mask])).detach().cpu()
            log = {
                "step": int(step),
                "loss": float(loss.detach().cpu()),
                "token_loss": float(token_loss.detach().cpu()),
                "total_loss": float(total_loss.detach().cpu()),
                "grad_norm": grad_norm,
                "batch_frame_ratio": float(batch_ratio),
                "batch_token_mae": float(batch_token_mae),
                "speed": float(step / max(time.time() - started, 1e-6)),
            }
            logs.append(log)
            print(json.dumps(log, ensure_ascii=False), flush=True)

        if step % args.save_interval == 0 or step == args.steps:
            save_checkpoint(args.out_dir / f"duration-student-step{step}.pt", model, config, args)
            model.to(device)
            model.train()

    train_eval = evaluate(model, train_samples, device, batch_size=args.batch_size, max_duration=args.max_duration)
    heldout_eval = evaluate(model, eval_samples, device, batch_size=args.batch_size, max_duration=args.max_duration)
    checkpoint_path = args.out_dir / "duration-student.pt"
    save_checkpoint(checkpoint_path, model, config, args)
    model.to(device)
    report = {
        "passed": True,
        "checkpoint": str(checkpoint_path),
        "out_dir": str(args.out_dir),
        "params": count_parameters(model),
        "config": config,
        "steps": int(args.steps),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "total_weight": float(args.total_weight),
        "logs": logs,
        "train_eval": train_eval,
        "heldout_eval": heldout_eval,
    }
    write_json(args.out_dir / "train-report.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
