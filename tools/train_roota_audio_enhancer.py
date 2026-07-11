#!/usr/bin/env python3
"""Train a tiny residual waveform enhancer for Root A decoder artifacts."""

from __future__ import annotations

import argparse
import html
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch import nn


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class PairRow:
    index: int
    row_id: str
    text: str
    noisy_audio: Path
    teacher_audio: Path
    raw_student_audio: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--eval-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--crop-samples", type=int, default=32768)
    parser.add_argument("--hidden", type=int, default=48)
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--kernel-size", type=int, default=9)
    parser.add_argument("--output-scale", type=float, default=0.20)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--render-rows", type=int, default=24)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--l1-weight", type=float, default=1.0)
    parser.add_argument("--stft-weight", type=float, default=0.55)
    parser.add_argument("--high-band-weight", type=float, default=0.20)
    parser.add_argument("--delta-weight", type=float, default=0.12)
    parser.add_argument("--residual-weight", type=float, default=0.04)
    parser.add_argument("--frame-rms-weight", type=float, default=0.0)
    parser.add_argument("--quiet-weight", type=float, default=0.0)
    parser.add_argument("--quiet-rms-threshold", type=float, default=0.015)
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument(
        "--time-warp-target",
        action="store_true",
        help="Linearly warp the clean target to the noisy length for training crops.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"manifest not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: expected object")
            rows.append(value)
    if not rows:
        raise RuntimeError(f"{path}: no rows")
    return rows


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    candidates = [path] if path.is_absolute() else [Path.cwd() / path, ROOT / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"audio not found: {path_text}")


def coerce_pair_rows(path: Path) -> list[PairRow]:
    out: list[PairRow] = []
    for row_no, row in enumerate(read_jsonl(path), start=1):
        try:
            text = row["text"]
            noisy_audio = row["noisy_audio"]
            teacher_audio = row["teacher_audio"]
        except KeyError as exc:
            raise ValueError(f"{path}:{row_no}: missing required key {exc}") from exc
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{path}:{row_no}: text must be non-empty")
        if not isinstance(noisy_audio, str) or not isinstance(teacher_audio, str):
            raise ValueError(f"{path}:{row_no}: audio paths must be strings")
        raw_student_audio = row.get("raw_student_audio", noisy_audio)
        if not isinstance(raw_student_audio, str):
            raise ValueError(f"{path}:{row_no}: raw_student_audio must be a string")
        index_raw = row.get("index", row_no)
        index = int(index_raw) if isinstance(index_raw, (int, float, str)) and str(index_raw).isdigit() else row_no
        row_id = str(row.get("row_id") or f"row-{row_no:05d}")
        out.append(
            PairRow(
                index=index,
                row_id=row_id,
                text=text.strip(),
                noisy_audio=resolve_path(noisy_audio),
                teacher_audio=resolve_path(teacher_audio),
                raw_student_audio=resolve_path(raw_student_audio),
            )
        )
    return out


def pick_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("--device mps requested, but MPS is unavailable")
    return torch.device(device)


def read_audio(path: Path, expected_sr: int) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if sr != expected_sr:
        raise RuntimeError(f"{path}: sample rate {sr} != expected {expected_sr}")
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    if audio.ndim != 1:
        raise RuntimeError(f"{path}: expected mono/stereo audio, got {audio.shape}")
    audio = np.asarray(audio, dtype=np.float32)
    finite = np.isfinite(audio)
    if not bool(np.all(finite)):
        audio = np.where(finite, audio, 0.0).astype(np.float32)
    return np.clip(audio, -1.0, 1.0)


def write_audio(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    sf.write(path, np.clip(audio, -0.999, 0.999), sample_rate)


class PairDataset:
    def __init__(self, rows: list[PairRow], sample_rate: int, *, time_warp_target: bool) -> None:
        self.rows = rows
        self.sample_rate = sample_rate
        self.time_warp_target = bool(time_warp_target)
        self._cache: dict[Path, np.ndarray] = {}

    def audio(self, path: Path) -> np.ndarray:
        cached = self._cache.get(path)
        if cached is not None:
            return cached
        value = read_audio(path, self.sample_rate)
        self._cache[path] = value
        return value

    def sample_batch(self, batch_size: int, crop_samples: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        noisy_chunks: list[np.ndarray] = []
        clean_chunks: list[np.ndarray] = []
        for _ in range(batch_size):
            row = random.choice(self.rows)
            noisy = self.audio(row.noisy_audio)
            clean = self.audio(row.teacher_audio)
            if self.time_warp_target and clean.size > 1 and noisy.size > 1 and clean.size != noisy.size:
                clean = resample_to_length(clean, noisy.size)
            usable = min(noisy.size, clean.size)
            if usable <= 0:
                raise RuntimeError(f"{row.row_id}: empty audio")
            if usable < crop_samples:
                pad = crop_samples - usable
                noisy_crop = np.pad(noisy[:usable], (0, pad))
                clean_crop = np.pad(clean[:usable], (0, pad))
            else:
                start = random.randint(0, usable - crop_samples)
                noisy_crop = noisy[start : start + crop_samples]
                clean_crop = clean[start : start + crop_samples]
            noisy_chunks.append(noisy_crop.astype(np.float32))
            clean_chunks.append(clean_crop.astype(np.float32))
        noisy_tensor = torch.as_tensor(np.stack(noisy_chunks), dtype=torch.float32, device=device).unsqueeze(1)
        clean_tensor = torch.as_tensor(np.stack(clean_chunks), dtype=torch.float32, device=device).unsqueeze(1)
        return noisy_tensor, clean_tensor


def resample_to_length(audio: np.ndarray, length: int) -> np.ndarray:
    if length <= 0:
        raise ValueError(f"target length must be positive, got {length}")
    if audio.size == length:
        return audio.astype(np.float32, copy=False)
    if audio.size == 0:
        return np.zeros((length,), dtype=np.float32)
    if audio.size == 1:
        return np.full((length,), float(audio[0]), dtype=np.float32)
    old_x = np.linspace(0.0, 1.0, num=audio.size, dtype=np.float64)
    new_x = np.linspace(0.0, 1.0, num=length, dtype=np.float64)
    return np.interp(new_x, old_x, audio.astype(np.float64)).astype(np.float32)


class DepthwiseResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")
        if dilation <= 0:
            raise ValueError(f"dilation must be positive, got {dilation}")
        padding = dilation * (kernel_size // 2)
        self.norm = nn.GroupNorm(1, channels)
        self.depthwise = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation, groups=channels)
        self.pointwise = nn.Conv1d(channels, channels, 1)
        self.gate = nn.Conv1d(channels, channels, 1)
        self.scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = F.silu(self.norm(x))
        value = self.depthwise(value)
        value = torch.tanh(self.pointwise(value)) * torch.sigmoid(self.gate(value))
        return x + self.scale * value


class TinyResidualEnhancer(nn.Module):
    def __init__(self, hidden: int, depth: int, kernel_size: int, output_scale: float) -> None:
        super().__init__()
        if hidden <= 0:
            raise ValueError(f"hidden must be positive, got {hidden}")
        if depth <= 0:
            raise ValueError(f"depth must be positive, got {depth}")
        if output_scale <= 0.0:
            raise ValueError(f"output_scale must be positive, got {output_scale}")
        self.output_scale = float(output_scale)
        self.pre = nn.Conv1d(1, hidden, 7, padding=3)
        dilations = [1, 2, 4, 8, 16, 32]
        self.blocks = nn.ModuleList(
            [DepthwiseResidualBlock(hidden, kernel_size, dilations[index % len(dilations)]) for index in range(depth)]
        )
        self.post_norm = nn.GroupNorm(1, hidden)
        self.post = nn.Conv1d(hidden, 1, 7, padding=3)
        nn.init.zeros_(self.post.weight)
        nn.init.zeros_(self.post.bias)

    def forward(self, noisy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if noisy.ndim != 3 or noisy.shape[1] != 1:
            raise RuntimeError(f"expected [batch, 1, time], got {tuple(noisy.shape)}")
        value = self.pre(noisy)
        for block in self.blocks:
            value = block(value)
        residual = torch.tanh(self.post(F.silu(self.post_norm(value)))) * self.output_scale
        enhanced = torch.clamp(noisy + residual, -1.0, 1.0)
        return enhanced, residual


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def stft_mag(audio: torch.Tensor, n_fft: int) -> torch.Tensor:
    hop = n_fft // 4
    flat = audio.squeeze(1)
    window = torch.hann_window(n_fft, device=audio.device)
    spec = torch.stft(flat, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=window, return_complex=True)
    return torch.abs(spec)


def multi_resolution_stft_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for n_fft in (256, 512, 1024, 2048):
        if predicted.shape[-1] < n_fft or target.shape[-1] < n_fft:
            continue
        pred_mag = stft_mag(predicted, n_fft)
        target_mag = stft_mag(target, n_fft)
        losses.append(F.l1_loss(torch.log1p(pred_mag), torch.log1p(target_mag)))
    if not losses:
        return F.l1_loss(predicted, target)
    return torch.stack(losses).mean()


def high_band_loss(predicted: torch.Tensor, target: torch.Tensor, sample_rate: int, cutoff_hz: float = 6000.0) -> torch.Tensor:
    n_fft = 1024
    if predicted.shape[-1] < n_fft:
        return predicted.new_tensor(0.0)
    pred_mag = stft_mag(predicted, n_fft)
    target_mag = stft_mag(target, n_fft)
    freqs = torch.fft.rfftfreq(n_fft, d=1.0 / float(sample_rate)).to(predicted.device)
    mask = freqs >= cutoff_hz
    if not bool(torch.any(mask)):
        return predicted.new_tensor(0.0)
    return F.l1_loss(torch.log1p(pred_mag[:, mask, :]), torch.log1p(target_mag[:, mask, :]))


def delta_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_delta = predicted[..., 1:] - predicted[..., :-1]
    target_delta = target[..., 1:] - target[..., :-1]
    return F.l1_loss(pred_delta, target_delta)


def frame_rms(audio: torch.Tensor, frame_size: int = 512, hop_size: int = 128) -> torch.Tensor:
    if audio.ndim != 3 or audio.shape[1] != 1:
        raise RuntimeError(f"expected [batch, 1, time], got {tuple(audio.shape)}")
    if audio.shape[-1] < frame_size:
        rms = audio.square().mean(dim=-1, keepdim=True).add(1e-8).sqrt()
        return rms
    frames = audio.unfold(dimension=-1, size=frame_size, step=hop_size)
    return frames.square().mean(dim=-1).add(1e-8).sqrt()


def frame_rms_envelope_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_rms = frame_rms(predicted)
    target_rms = frame_rms(target)
    return F.l1_loss(torch.log(pred_rms + 1e-5), torch.log(target_rms + 1e-5))


def quiet_energy_loss(predicted: torch.Tensor, target: torch.Tensor, threshold: float) -> torch.Tensor:
    if threshold <= 0.0:
        return predicted.new_tensor(0.0)
    pred_rms = frame_rms(predicted)
    target_rms = frame_rms(target)
    quiet_mask = target_rms < float(threshold)
    if not bool(torch.any(quiet_mask)):
        return predicted.new_tensor(0.0)
    return pred_rms[quiet_mask].mean()


def load_checkpoint(path: Path, device: torch.device) -> tuple[TinyResidualEnhancer, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise RuntimeError(f"{path}: checkpoint missing config")
    model = TinyResidualEnhancer(
        hidden=int(config["hidden"]),
        depth=int(config["depth"]),
        kernel_size=int(config["kernel_size"]),
        output_scale=float(config["output_scale"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, config


@torch.no_grad()
def enhance_array(model: TinyResidualEnhancer, audio: np.ndarray, device: torch.device) -> np.ndarray:
    tensor = torch.as_tensor(audio, dtype=torch.float32, device=device).view(1, 1, -1)
    enhanced, _ = model(tensor)
    return enhanced.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)


def render_dashboard(report: dict[str, Any], out_path: Path) -> None:
    rows = report["rows"]
    cards: list[str] = []
    for row in rows:
        text = html.escape(str(row["text"]))
        teacher = html.escape(str(Path(row["teacher_audio"]).name))
        raw = html.escape(str(Path(row["oracle_decoder_audio"]).name))
        enhanced = html.escape(str(Path(row["student_audio"]).name))
        cards.append(
            f"""
      <section class="card">
        <div class="row-head"><span>#{row['index']}</span><p>{text}</p></div>
        <div class="grid">
          <label>Teacher<audio controls src="audio/{teacher}"></audio></label>
          <label>Raw input<audio controls src="audio/{raw}"></audio></label>
          <label>Enhanced<audio controls src="audio/{enhanced}"></audio></label>
        </div>
      </section>"""
        )
    body = "\n".join(cards)
    out_path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Root A Tiny Enhancer</title>
  <style>
    body {{ margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f6f3ea; color: #17191c; }}
    main {{ width: min(1120px, calc(100vw - 32px)); margin: 0 auto; padding: 28px 0 48px; }}
    header {{ border-bottom: 1px solid #d8d2c4; padding-bottom: 16px; margin-bottom: 20px; }}
    h1 {{ margin: 0; font-size: 36px; letter-spacing: 0; }}
    .meta {{ color: #635f57; margin: 8px 0 0; }}
    .card {{ background: #fff; border: 1px solid #d8d2c4; border-radius: 8px; padding: 14px; margin: 14px 0; box-shadow: 0 10px 26px rgba(30, 28, 23, .08); }}
    .row-head {{ display: grid; grid-template-columns: 54px 1fr; gap: 12px; align-items: start; }}
    .row-head span {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; color: #0f766e; }}
    .row-head p {{ margin: 0 0 12px; }}
    .grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }}
    label {{ font-weight: 650; font-size: 13px; color: #3f3b35; }}
    audio {{ display: block; width: 100%; margin-top: 6px; }}
    @media (max-width: 780px) {{ .grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
<main>
  <header>
    <h1>Root A Tiny Enhancer</h1>
    <p class="meta">Teacher, raw input, and enhanced output for the held-out render set.</p>
  </header>
{body}
</main>
</body>
</html>
""",
        encoding="utf-8",
    )


def train(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.crop_samples <= 0:
        raise ValueError("--crop-samples must be positive")
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    device = pick_device(args.device)
    train_rows = coerce_pair_rows(args.train_manifest)
    train_dataset = PairDataset(train_rows, int(args.sample_rate), time_warp_target=bool(args.time_warp_target))
    model = TinyResidualEnhancer(
        hidden=int(args.hidden),
        depth=int(args.depth),
        kernel_size=int(args.kernel_size),
        output_scale=float(args.output_scale),
    ).to(device)
    params = count_parameters(model)
    if params > 100_000:
        raise RuntimeError(f"enhancer has {params} parameters, above the 100k cap")
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-5)
    history: list[dict[str, float]] = []
    started = time.time()
    model.train()
    for step in range(1, int(args.steps) + 1):
        noisy, clean = train_dataset.sample_batch(int(args.batch_size), int(args.crop_samples), device)
        enhanced, residual = model(noisy)
        l1 = F.l1_loss(enhanced, clean)
        stft = multi_resolution_stft_loss(enhanced, clean)
        high = high_band_loss(enhanced, clean, int(args.sample_rate))
        delta = delta_loss(enhanced, clean)
        residual_penalty = torch.mean(torch.abs(residual))
        frame_rms_penalty = frame_rms_envelope_loss(enhanced, clean)
        quiet_penalty = quiet_energy_loss(enhanced, clean, float(args.quiet_rms_threshold))
        loss = (
            float(args.l1_weight) * l1
            + float(args.stft_weight) * stft
            + float(args.high_band_weight) * high
            + float(args.delta_weight) * delta
            + float(args.residual_weight) * residual_penalty
            + float(args.frame_rms_weight) * frame_rms_penalty
            + float(args.quiet_weight) * quiet_penalty
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % int(args.log_every) == 0 or step == int(args.steps):
            record = {
                "step": float(step),
                "loss": float(loss.detach().cpu()),
                "l1": float(l1.detach().cpu()),
                "stft": float(stft.detach().cpu()),
                "high_band": float(high.detach().cpu()),
                "delta": float(delta.detach().cpu()),
                "residual": float(residual_penalty.detach().cpu()),
                "frame_rms": float(frame_rms_penalty.detach().cpu()),
                "quiet": float(quiet_penalty.detach().cpu()),
            }
            history.append(record)
            print(json.dumps(record), flush=True)

    config = {
        "hidden": int(args.hidden),
        "depth": int(args.depth),
        "kernel_size": int(args.kernel_size),
        "output_scale": float(args.output_scale),
        "parameters": int(params),
        "sample_rate": int(args.sample_rate),
        "train_manifest": str(args.train_manifest),
        "eval_manifest": str(args.eval_manifest),
        "steps": int(args.steps),
        "batch_size": int(args.batch_size),
        "crop_samples": int(args.crop_samples),
        "lr": float(args.lr),
        "seed": int(args.seed),
        "loss_weights": {
            "l1": float(args.l1_weight),
            "stft": float(args.stft_weight),
            "high_band": float(args.high_band_weight),
            "delta": float(args.delta_weight),
            "residual": float(args.residual_weight),
            "frame_rms": float(args.frame_rms_weight),
            "quiet": float(args.quiet_weight),
        },
        "quiet_rms_threshold": float(args.quiet_rms_threshold),
        "time_warp_target": bool(args.time_warp_target),
        "elapsed_s": time.time() - started,
        "history": history,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.out_dir / "audio-enhancer.pt"
    torch.save({"model_state_dict": model.state_dict(), "config": config}, checkpoint_path)
    (args.out_dir / "training-summary.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return checkpoint_path, config


def render_eval(args: argparse.Namespace, checkpoint_path: Path, config: dict[str, Any]) -> Path:
    device = pick_device(args.device)
    model, _ = load_checkpoint(checkpoint_path, device)
    eval_rows = coerce_pair_rows(args.eval_manifest)
    if args.render_rows > 0:
        eval_rows = eval_rows[: int(args.render_rows)]
    if not eval_rows:
        raise RuntimeError("no eval rows selected")

    audio_dir = args.out_dir / "audio"
    report_rows: list[dict[str, Any]] = []
    for position, row in enumerate(eval_rows, start=1):
        raw = read_audio(row.noisy_audio, int(args.sample_rate))
        enhanced = enhance_array(model, raw, device)
        teacher = read_audio(row.teacher_audio, int(args.sample_rate))
        teacher_out = audio_dir / f"{position:05d}_{row.row_id}_teacher.wav"
        raw_out = audio_dir / f"{position:05d}_{row.row_id}_raw.wav"
        enhanced_out = audio_dir / f"{position:05d}_{row.row_id}_enhanced.wav"
        write_audio(teacher_out, teacher, int(args.sample_rate))
        write_audio(raw_out, raw, int(args.sample_rate))
        write_audio(enhanced_out, enhanced, int(args.sample_rate))
        report_rows.append(
            {
                "index": position,
                "source_index": row.index,
                "row_id": row.row_id,
                "text": row.text,
                "teacher_audio": str(teacher_out),
                "oracle_decoder_audio": str(raw_out),
                "student_audio": str(enhanced_out),
                "enhancer_input_audio": str(row.noisy_audio),
                "enhancer_parameters": int(config["parameters"]),
            }
        )
    report = {
        "label": "roota_tiny_audio_enhancer",
        "checkpoint": str(checkpoint_path),
        "config": config,
        "rows": report_rows,
        "lane_meaning": {
            "teacher_audio": "Piper teacher",
            "oracle_decoder_audio": "raw enhancer input",
            "student_audio": "enhanced output",
        },
    }
    report_path = args.out_dir / "render-scoreq-report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    render_dashboard(report, args.out_dir / "index.html")
    return report_path


def main() -> None:
    args = parse_args()
    checkpoint_path, config = train(args)
    report_path = render_eval(args, checkpoint_path, config)
    print(json.dumps({"checkpoint": str(checkpoint_path), "report": str(report_path), "parameters": config["parameters"]}, indent=2))


if __name__ == "__main__":
    main()
