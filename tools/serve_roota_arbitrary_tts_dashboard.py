#!/usr/bin/env python3
"""Serve a live arbitrary-text dashboard for the current Root A acoustic probe.

This is deliberately a probe surface, not a final TTS runtime. Piper provides
phoneme IDs; the duration and acoustic students predict Piper generator
latents; either the verified decoder cut or a compact decoder student renders
audio.
"""

from __future__ import annotations

import argparse
import html
import importlib.util
import json
import re
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import ModuleType
from typing import Any
from urllib.parse import unquote, urlparse

import numpy as np
import onnxruntime as ort
import torch
from scipy.signal import butter, sosfiltfilt


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ACOUSTIC_CHECKPOINT = (
    ROOT
    / "artifacts"
    / "sub10m-search"
    / "root-a-piper-vits"
    / "chitwan-medium-a16-acoustic-objective-v2-norm025-delta010-stat005-lr5e5-2000-20260625"
    / "latent-student.pt"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "artifacts"
    / "sub10m-search"
    / "root-a-piper-vits"
    / "live-arbitrary-tts-dashboard-20260625"
)
DEFAULT_DURATION_CHECKPOINT = (
    ROOT
    / "artifacts"
    / "sub10m-search"
    / "root-a-piper-vits"
    / "chitwan-medium-a5-duration-student-h64d3-4000-20260625"
    / "duration-student.pt"
)
DEFAULT_DECODER_STUDENT_CHECKPOINT = (
    ROOT
    / "artifacts"
    / "sub10m-search"
    / "root-a-piper-vits"
    / "chitwan-medium-a4-decoder-heldout-dense-1453k-featurehint08-5000-20260625"
    / "decoder-student.pt"
)


def load_local_module(name: str, path: Path) -> ModuleType:
    if not path.is_file():
        raise FileNotFoundError(f"module file not found: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


latent_mod = load_local_module("roota_latent_student", ROOT / "tools" / "train_roota_piper_latent_student.py")
pack_mod = load_local_module("roota_pack_builder", ROOT / "tools" / "build_piper_vits_roota_probe_pack.py")
duration_mod = load_local_module("roota_duration_student", ROOT / "tools" / "train_roota_piper_duration_student.py")
decoder_mod = load_local_module("roota_decoder_student", ROOT / "tools" / "train_roota_piper_decoder_student.py")
enhancer_mod = load_local_module("roota_audio_enhancer", ROOT / "tools" / "train_roota_audio_enhancer.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8877)
    parser.add_argument("--acoustic-checkpoint", type=Path, default=DEFAULT_ACOUSTIC_CHECKPOINT)
    parser.add_argument("--decoder", type=Path, default=latent_mod.DEFAULT_DECODER)
    parser.add_argument("--decoder-backend", choices=("onnx", "student"), default="student")
    parser.add_argument("--decoder-student-checkpoint", type=Path, default=DEFAULT_DECODER_STUDENT_CHECKPOINT)
    parser.add_argument(
        "--audio-enhancer-checkpoint",
        type=Path,
        default=None,
        help="Optional tiny residual audio enhancer applied after the student decoder.",
    )
    parser.add_argument(
        "--postprocess-gain",
        type=float,
        default=1.0,
        help="Deterministic gain applied to the enhanced lane after the student decoder/enhancer.",
    )
    parser.add_argument(
        "--postprocess-filter",
        default="none",
        help="Deterministic enhanced-lane filter: none, lpHZ, hpHZ, or bpLOW_HIGH.",
    )
    parser.add_argument("--duration-checkpoint", type=Path, default=DEFAULT_DURATION_CHECKPOINT)
    parser.add_argument("--duration-source", choices=("student", "oracle"), default="student")
    parser.add_argument("--duration-length-scale", type=float, default=1.16)
    parser.add_argument("--piper-model", type=Path, default=pack_mod.DEFAULT_MODEL)
    parser.add_argument("--piper-config", type=Path, default=pack_mod.DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--noise-scale", type=float, default=0.0)
    parser.add_argument("--length-scale", type=float, default=1.0)
    parser.add_argument("--noise-w", type=float, default=0.0)
    parser.add_argument("--sentence-silence", type=float, default=0.12)
    parser.add_argument(
        "--sibilant-inject-beta", type=float, default=0.0,
        help="restore sibilant hiss: inject beta*tea_std noise at /s ʃ z ʒ/ frames (0=off; kristin release uses 6.0)")
    parser.add_argument(
        "--sibilant-calib", type=Path, default=None,
        help="calib.npz from tools/calibrate_sibilant_noise.py (required when --sibilant-inject-beta>0)")
    parser.add_argument(
        "--text-chunking",
        choices=("none", "punctuation"),
        default="none",
        help="Optionally split arbitrary text before phonemization. Probe only; default preserves existing behavior.",
    )
    parser.add_argument("--dashboard-title", default="Root A live TTS bench")
    parser.add_argument(
        "--dashboard-subtitle",
        default="Arbitrary text through the current duration and acoustic students, with Piper as the reference.",
    )
    parser.add_argument(
        "--default-text",
        default="नमस्ते, यो परीक्षण वाक्य हो। हामी सानो ध्वनि मोडेल जाँच गर्दैछौं।",
    )
    return parser.parse_args()


CLAUSE_BOUNDARY_RE = re.compile(r"[^,;:.!?\n]+(?:[,;:.!?]+|\n+|$)")


def split_text_for_phonemizer(text: str, mode: str) -> list[str]:
    if mode == "none":
        return [text]
    if mode != "punctuation":
        raise ValueError(f"unsupported text chunking mode: {mode}")

    chunks: list[str] = []
    for match in CLAUSE_BOUNDARY_RE.finditer(text):
        chunk = " ".join(match.group(0).split())
        if chunk:
            chunks.append(chunk)
    if not chunks:
        stripped = text.strip()
        if not stripped:
            raise ValueError("text is empty")
        return [stripped]
    return chunks


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def response_json(handler: BaseHTTPRequestHandler, status: HTTPStatus, payload: dict[str, Any]) -> None:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status.value)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(data)


def response_html(handler: BaseHTTPRequestHandler, body: str) -> None:
    data = body.encode("utf-8")
    handler.send_response(HTTPStatus.OK.value)
    handler.send_header("Content-Type", "text/html; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(data)


def response_html_head(handler: BaseHTTPRequestHandler, body: str) -> None:
    data = body.encode("utf-8")
    handler.send_response(HTTPStatus.OK.value)
    handler.send_header("Content-Type", "text/html; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()


def response_file(handler: BaseHTTPRequestHandler, path: Path, content_type: str) -> None:
    data = path.read_bytes()
    handler.send_response(HTTPStatus.OK.value)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(data)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(data)


def response_no_content(handler: BaseHTTPRequestHandler) -> None:
    handler.send_response(HTTPStatus.NO_CONTENT.value)
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()


def safe_audio_path(audio_dir: Path, request_path: str) -> Path:
    parsed = urlparse(request_path)
    prefix = "/audio/"
    if not parsed.path.startswith(prefix):
        raise FileNotFoundError(request_path)
    raw_name = unquote(parsed.path[len(prefix) :])
    if "/" in raw_name or "\\" in raw_name or not raw_name.endswith(".wav"):
        raise FileNotFoundError(request_path)
    candidate = (audio_dir / raw_name).resolve()
    if candidate.parent != audio_dir.resolve():
        raise FileNotFoundError(request_path)
    if not candidate.is_file():
        raise FileNotFoundError(request_path)
    return candidate


def rms(audio: np.ndarray) -> float:
    audio_f = np.asarray(audio, dtype=np.float64).reshape(-1)
    if audio_f.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio_f))))


def apply_postprocess_filter(audio: np.ndarray, sample_rate: int, filter_spec: str) -> np.ndarray:
    audio_f = np.asarray(audio, dtype=np.float32).reshape(-1)
    filter_spec = str(filter_spec or "none").strip().lower()
    if filter_spec == "none":
        return audio_f
    if filter_spec.startswith("lp"):
        cutoff = float(filter_spec[2:])
        sos = butter(4, cutoff, btype="lowpass", fs=sample_rate, output="sos")
        return sosfiltfilt(sos, audio_f).astype(np.float32)
    if filter_spec.startswith("hp"):
        cutoff = float(filter_spec[2:])
        sos = butter(3, cutoff, btype="highpass", fs=sample_rate, output="sos")
        return sosfiltfilt(sos, audio_f).astype(np.float32)
    if filter_spec.startswith("bp"):
        parts = filter_spec[2:].split("_")
        if len(parts) != 2:
            raise ValueError(f"invalid bandpass filter spec: {filter_spec}")
        low, high = float(parts[0]), float(parts[1])
        sos = butter(3, [low, high], btype="bandpass", fs=sample_rate, output="sos")
        return sosfiltfilt(sos, audio_f).astype(np.float32)
    raise ValueError(f"unsupported postprocess filter: {filter_spec}")


@dataclass
class SynthChunk:
    chunk_index: int
    text: str
    phoneme_count: int
    phoneme_id_count: int
    frames: int
    oracle_frames: int
    duration_source: str
    teacher_samples: int
    oracle_samples: int
    student_samples: int
    w_ceil_sum: int
    student_rms: float
    oracle_rms: float
    teacher_rms: float


class DashboardState:
    def __init__(self, args: argparse.Namespace) -> None:
        require_file(args.acoustic_checkpoint, "acoustic checkpoint")
        if args.decoder_backend == "onnx":
            require_file(args.decoder, "decoder ONNX")
        elif args.decoder_backend == "student":
            require_file(args.decoder_student_checkpoint, "decoder student checkpoint")
        else:
            raise ValueError(f"unsupported decoder backend: {args.decoder_backend}")
        require_file(args.piper_model, "Piper ONNX")
        require_file(args.piper_config, "Piper config")
        if args.duration_source == "student":
            require_file(args.duration_checkpoint, "duration checkpoint")
        if args.audio_enhancer_checkpoint is not None:
            require_file(args.audio_enhancer_checkpoint, "audio enhancer checkpoint")
        if float(args.postprocess_gain) <= 0.0:
            raise ValueError("--postprocess-gain must be positive")

        self.args = args
        self.out_dir = args.out_dir
        self.audio_dir = self.out_dir / "audio"
        self.meta_dir = self.out_dir / "metadata"
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.meta_dir.mkdir(parents=True, exist_ok=True)

        self.device = latent_mod.pick_device(args.device)
        self.acoustic_model, self.acoustic_config = latent_mod.load_model_from_checkpoint(
            args.acoustic_checkpoint,
            self.device,
        )
        self.acoustic_model.eval()
        self.student_parameters = latent_mod.count_parameters(self.acoustic_model)
        self.duration_model = None
        self.duration_config: dict[str, Any] | None = None
        self.duration_parameters = 0
        if args.duration_source == "student":
            self.duration_model, self.duration_config = duration_mod.load_model_from_checkpoint(
                args.duration_checkpoint,
                self.device,
            )
            self.duration_model.eval()
            self.duration_parameters = duration_mod.count_parameters(self.duration_model)

        # Sibilant fricative-noise injection (restores /s ʃ z ʒ/ hiss the deterministic acoustic smooths away).
        self.sibilant_beta = float(getattr(args, "sibilant_inject_beta", 0.0) or 0.0)
        self.sibilant_tea_std: np.ndarray | None = None
        self.sibilant_ids: set[int] = set()
        self._sibilant_rng = np.random.default_rng(1234)
        if self.sibilant_beta > 0.0:
            calib_path = getattr(args, "sibilant_calib", None)
            if calib_path is None:
                raise RuntimeError("--sibilant-inject-beta>0 requires --sibilant-calib (from tools/calibrate_sibilant_noise.py)")
            cal = np.load(calib_path, allow_pickle=True)
            self.sibilant_tea_std = cal["tea_std"].astype(np.float32)[:, None]
            self.sibilant_ids = set(int(x) for x in cal["sib_ids"].tolist())

        self.decoder_backend = str(args.decoder_backend)
        self.decoder_session: ort.InferenceSession | None = None
        self.decoder_model = None
        self.decoder_config: dict[str, Any] | None = None
        self.decoder_parameters = 0
        if self.decoder_backend == "onnx":
            self.decoder_session = ort.InferenceSession(str(args.decoder), providers=["CPUExecutionProvider"])
        else:
            checkpoint = torch.load(args.decoder_student_checkpoint, map_location="cpu", weights_only=False)
            config_raw = checkpoint.get("config")
            if not isinstance(config_raw, dict):
                raise RuntimeError(f"decoder checkpoint missing config: {args.decoder_student_checkpoint}")
            required_config_keys = {"in_channels", "channels", "res_layers", "variant", "rank_ratio"}
            optional_config_keys = {
                "activation",
                "factorized_pre_rank",
                "piper_res_factor_rank_ratio",
                "post_filter_channels",
                "post_filter_layers",
                "post_filter_kernel",
                "post_filter_scale",
                "pre_tanh_repair_channels",
                "pre_tanh_repair_layers",
                "pre_tanh_repair_kernel",
                "pre_tanh_repair_scale",
                "stage0_branches",
                "stage1_branches",
                "stage2_branches",
                "stage3_branches",
                "res_bank_scale_mode",
                "stage_affine",
                "istft_n_fft",
                "fsd_dim",
                "fsd_blocks",
                "fsd_film_rank",
                "fsd_head_rank",
                "stage_projection_bottlenecks",
            }
            allowed_config_keys = required_config_keys | optional_config_keys
            self.decoder_config = {key: value for key, value in config_raw.items() if key in allowed_config_keys}
            missing_config_keys = sorted(required_config_keys - set(self.decoder_config))
            if missing_config_keys:
                raise RuntimeError(
                    f"decoder checkpoint config missing keys {missing_config_keys}: {args.decoder_student_checkpoint}"
                )
            self.decoder_model = decoder_mod.DecoderStudent(**self.decoder_config).to(self.device)
            self.decoder_model.load_state_dict(checkpoint["model_state_dict"])
            self.decoder_model.eval()
            self.decoder_parameters = decoder_mod.count_parameters(self.decoder_model)
            self.lrc_encoder = None
            if checkpoint.get("lrc_encoder_state_dict") is not None:
                self.lrc_encoder = decoder_mod.LrcEncoder(
                    in_channels=192,
                    hidden=int(config_raw.get("lrc_encoder_hidden", 64)),
                    code_dim=int(config_raw.get("lrc_code_dim", 40)),
                ).to(self.device)
                self.lrc_encoder.load_state_dict(checkpoint["lrc_encoder_state_dict"])
                self.lrc_encoder.eval()
        self.audio_enhancer = None
        self.audio_enhancer_config: dict[str, Any] | None = None
        self.audio_enhancer_parameters = 0
        if args.audio_enhancer_checkpoint is not None:
            self.audio_enhancer, self.audio_enhancer_config = enhancer_mod.load_checkpoint(
                args.audio_enhancer_checkpoint,
                self.device,
            )
            self.audio_enhancer.eval()
            self.audio_enhancer_parameters = int(self.audio_enhancer_config.get("parameters") or 0)
        source_model = pack_mod.onnx.load(str(args.piper_model))
        self.piper_tensor_names = pack_mod.resolve_tensor_outputs(
            source_model,
            pack_mod.ACOUSTIC_LOGICAL_OUTPUTS,
        )
        debug_model = pack_mod.make_debug_model(
            args.piper_model,
            list(dict.fromkeys(self.piper_tensor_names.values())),
        )
        self.piper_session = ort.InferenceSession(debug_model.SerializeToString(), providers=["CPUExecutionProvider"])
        self.voice = pack_mod.PiperVoice.load(args.piper_model, args.piper_config)
        self.sample_rate = int(self.voice.config.sample_rate)
        self.scales = [float(args.noise_scale), float(args.length_scale), float(args.noise_w)]
        sentence_silence = float(getattr(args, "sentence_silence", 0.0))
        self.silence = np.zeros(int(round(sentence_silence * self.sample_rate)), dtype=np.float32)
        self.lock = threading.Lock()

    @property
    def model_card(self) -> dict[str, Any]:
        decoder_path = self.args.decoder if self.decoder_backend == "onnx" else self.args.decoder_student_checkpoint
        return {
            "acoustic_checkpoint": str(self.args.acoustic_checkpoint),
            "duration_checkpoint": str(self.args.duration_checkpoint) if self.duration_model is not None else None,
            "decoder": str(decoder_path),
            "decoder_backend": self.decoder_backend,
            "audio_enhancer": str(self.args.audio_enhancer_checkpoint) if self.audio_enhancer is not None else None,
            "postprocess_gain": float(getattr(self.args, "postprocess_gain", 1.0)),
            "postprocess_filter": str(getattr(self.args, "postprocess_filter", "none")),
            "piper_model": str(self.args.piper_model),
            "piper_config": str(self.args.piper_config),
            "device": str(self.device),
            "sample_rate": self.sample_rate,
            "acoustic_parameters": self.student_parameters,
            "duration_parameters": self.duration_parameters,
            "decoder_parameters": self.decoder_parameters,
            "audio_enhancer_parameters": self.audio_enhancer_parameters,
            "front_half_parameters": self.student_parameters + self.duration_parameters,
            "student_parameters": (
                self.student_parameters
                + self.duration_parameters
                + self.decoder_parameters
                + self.audio_enhancer_parameters
            ),
            "acoustic_config": self.acoustic_config,
            "duration_config": self.duration_config,
            "decoder_config": self.decoder_config,
            "audio_enhancer_config": self.audio_enhancer_config,
            "duration_length_scale": float(self.args.duration_length_scale),
            "durations": (
                f"learned duration student, scale {float(self.args.duration_length_scale):.3f}"
                if self.duration_model is not None
                else "Piper/VITS w_ceil"
            ),
            "duration_source": str(self.args.duration_source),
            "text_chunking": str(getattr(self.args, "text_chunking", "none")),
            "sentence_silence_s": float(getattr(self.args, "sentence_silence", 0.0)),
            "frontend": "Piper/espeak-ng phoneme IDs",
            "dashboard_title": str(getattr(self.args, "dashboard_title", "Root A live TTS bench")),
            "dashboard_subtitle": str(getattr(self.args, "dashboard_subtitle", "")),
            "default_text": str(getattr(self.args, "default_text", "नमस्ते, यो छोटो परीक्षण हो।")),
        }

    @torch.no_grad()
    def decode_latent(self, latent: np.ndarray) -> np.ndarray:
        if self.decoder_backend == "onnx":
            if self.decoder_session is None:
                raise RuntimeError("ONNX decoder backend selected without an inference session")
            return latent_mod.decode_latent(self.decoder_session, latent)
        if self.decoder_model is None:
            raise RuntimeError("student decoder backend selected without a loaded model")
        encoder = getattr(self, "lrc_encoder", None)
        decoder_in = int(getattr(self.decoder_model, "in_channels", 0) or self.decoder_config.get("in_channels", 0))
        channel_axis = 0 if latent.ndim == 2 else 1
        if encoder is not None and decoder_in > 0 and latent.shape[channel_axis] != decoder_in:
            tensor = torch.from_numpy(np.ascontiguousarray(latent)).float()
            squeeze = tensor.ndim == 2
            if squeeze:
                tensor = tensor.unsqueeze(0)
            with torch.no_grad():
                encoded = encoder(tensor.to(self.device)).cpu()
            latent = (encoded.squeeze(0) if squeeze else encoded).numpy()
        return decoder_mod.decode_with_student(self.decoder_model, latent, self.device)

    def inject_sibilant_noise(self, latent: np.ndarray, phoneme_ids, durations) -> np.ndarray:
        """Add beta*tea_std Gaussian noise into the predicted latent at sibilant frames.

        A deterministic acoustic regresses the mean latent, collapsing /s ʃ z ʒ/ (broadband
        noise) to a whistly tone. Restoring the per-channel variance at those frames lets the
        decoder render proper hiss (it already renders high-variance teacher fricative latents
        cleanly). No-op unless --sibilant-inject-beta>0. Mutates + returns `latent`.
        """
        if self.sibilant_beta <= 0.0 or self.sibilant_tea_std is None:
            return latent
        ids = np.asarray(phoneme_ids).reshape(-1)
        durs = np.asarray(durations).reshape(-1)
        cum = np.concatenate([[0], np.cumsum(durs)]).astype(int)
        F = latent.shape[-1]
        for j, idv in enumerate(ids):
            if int(idv) in self.sibilant_ids and j + 1 < len(cum):
                x, y = int(cum[j]), int(cum[j + 1])
                if 0 <= x < y <= F:
                    noise = (self._sibilant_rng.standard_normal((self.sibilant_tea_std.shape[0], y - x)).astype(np.float32)
                             * self.sibilant_tea_std * self.sibilant_beta)
                    if latent.ndim == 3:
                        latent[0, :, x:y] += noise
                    else:
                        latent[:, x:y] += noise
        return latent

    @torch.no_grad()
    def enhance_audio(self, audio: np.ndarray) -> np.ndarray:
        enhanced = audio
        if self.audio_enhancer is None:
            enhanced = np.asarray(audio, dtype=np.float32)
        else:
            enhanced = enhancer_mod.enhance_array(self.audio_enhancer, audio, self.device)
        enhanced = apply_postprocess_filter(
            enhanced,
            self.sample_rate,
            str(getattr(self.args, "postprocess_filter", "none")),
        )
        enhanced = enhanced * float(getattr(self.args, "postprocess_gain", 1.0))
        return np.clip(enhanced, -0.98, 0.98).astype(np.float32)

    def predict_student_durations(
        self,
        phoneme_ids: list[int],
        oracle_durations: np.ndarray,
        requested_source: str,
        length_scale: float,
    ) -> tuple[np.ndarray, str]:
        if requested_source == "oracle" or self.duration_model is None:
            return np.asarray(oracle_durations, dtype=np.int64).reshape(-1), "oracle"
        if requested_source != "student":
            raise ValueError(f"unsupported duration_source: {requested_source}")
        ids = torch.as_tensor([phoneme_ids], dtype=torch.long, device=self.device)
        mask = torch.ones_like(ids, dtype=torch.bool, device=self.device)
        pred = duration_mod.predict_durations(
            self.duration_model,
            ids,
            mask,
            max_duration=int(self.duration_config.get("max_duration", 80)) if self.duration_config else 80,
            length_scale=float(length_scale),
        )
        durations = pred.squeeze(0).detach().cpu().numpy().astype(np.int64)
        if durations.shape != oracle_durations.shape:
            raise RuntimeError(f"duration shape mismatch: {durations.shape} vs {oracle_durations.shape}")
        if np.any(durations <= 0):
            raise RuntimeError("duration student produced non-positive durations")
        return durations, "student"

    def synthesize(self, text: str, *, duration_source: str | None = None, duration_length_scale: float | None = None) -> dict[str, Any]:
        clean_text = text.strip()
        if not clean_text:
            raise ValueError("text is empty")
        if len(clean_text) > 2000:
            raise ValueError("text is too long; keep this probe under 2000 characters")
        requested_duration_source = str(duration_source or self.args.duration_source)
        if requested_duration_source not in {"student", "oracle"}:
            raise ValueError("duration_source must be 'student' or 'oracle'")
        requested_length_scale = float(
            self.args.duration_length_scale if duration_length_scale is None else duration_length_scale
        )
        if not (0.5 <= requested_length_scale <= 2.0):
            raise ValueError("duration_length_scale must be between 0.5 and 2.0")

        started = time.time()
        render_id = f"{int(started)}-{uuid.uuid4().hex[:8]}"
        text_chunking = str(getattr(self.args, "text_chunking", "none"))
        text_chunks = split_text_for_phonemizer(clean_text, text_chunking)
        phoneme_chunks: list[tuple[str, list[Any]]] = []
        for text_chunk in text_chunks:
            sentence_phonemes = self.voice.phonemize(text_chunk)
            if not sentence_phonemes:
                raise RuntimeError(f"Piper produced no sentence phonemes for chunk: {text_chunk!r}")
            for phonemes in sentence_phonemes:
                phoneme_chunks.append((text_chunk, phonemes))
        if not phoneme_chunks:
            raise RuntimeError("Piper produced no sentence phonemes")

        teacher_parts: list[np.ndarray] = []
        oracle_parts: list[np.ndarray] = []
        student_parts: list[np.ndarray] = []
        chunk_summaries: list[SynthChunk] = []

        output_names = [output.name for output in self.piper_session.get_outputs()]
        with self.lock:
            for chunk_index, (chunk_text, phonemes) in enumerate(phoneme_chunks):
                if not phonemes:
                    raise RuntimeError(f"chunk {chunk_index}: empty phoneme list")
                if chunk_index > 0 and self.silence.size:
                    teacher_parts.append(self.silence)
                    oracle_parts.append(self.silence)
                    student_parts.append(self.silence)

                phoneme_ids = self.voice.phonemes_to_ids(phonemes)
                if not phoneme_ids:
                    raise RuntimeError(f"chunk {chunk_index}: empty phoneme ID list")

                feeds = {
                    "input": np.expand_dims(np.asarray(phoneme_ids, dtype=np.int64), 0),
                    "input_lengths": np.asarray([len(phoneme_ids)], dtype=np.int64),
                    "scales": np.asarray(self.scales, dtype=np.float32),
                }
                outputs = dict(zip(output_names, self.piper_session.run(output_names, feeds), strict=True))
                w_ceil_name = self.piper_tensor_names["w_ceil"]
                generator_input_name = self.piper_tensor_names["generator_input"]
                if "output" not in outputs or w_ceil_name not in outputs or generator_input_name not in outputs:
                    raise RuntimeError(f"chunk {chunk_index}: Piper debug output missing required tensors")

                teacher_audio = np.asarray(outputs["output"], dtype=np.float32).reshape(-1)
                w_ceil = np.asarray(outputs[w_ceil_name], dtype=np.float32).reshape(-1)
                oracle_durations = np.rint(w_ceil).astype(np.int64)
                if oracle_durations.shape[0] != len(phoneme_ids):
                    raise RuntimeError(
                        f"chunk {chunk_index}: duration/id length mismatch "
                        f"{oracle_durations.shape[0]} != {len(phoneme_ids)}"
                    )
                if np.any(oracle_durations < 0):
                    raise RuntimeError(f"chunk {chunk_index}: negative duration from Piper")
                durations, resolved_duration_source = self.predict_student_durations(
                    phoneme_ids,
                    oracle_durations,
                    requested_duration_source,
                    requested_length_scale,
                )
                frame_count = int(durations.sum())
                teacher_latent = np.asarray(outputs[generator_input_name], dtype=np.float32)
                teacher_frames = int(teacher_latent.shape[-1])
                if frame_count <= 0:
                    raise RuntimeError(f"chunk {chunk_index}: zero generated frames")
                if resolved_duration_source == "oracle" and frame_count != teacher_frames:
                    raise RuntimeError(
                        f"chunk {chunk_index}: duration sum {frame_count} != teacher latent frames {teacher_frames}"
                    )

                out_channels = int(self.acoustic_config.get("out_channels") or teacher_latent.shape[-2])
                sample = latent_mod.ChunkSample(
                    row_id=render_id,
                    row_index=1,
                    text=chunk_text,
                    chunk_index=chunk_index,
                    phoneme_ids=np.asarray(phoneme_ids, dtype=np.int64),
                    durations=durations,
                    target=np.zeros((frame_count, out_channels), dtype=np.float32),
                    tensor_path=Path("live"),
                    audio_samples=int(teacher_audio.size),
                )
                predicted_latent = latent_mod.predict_chunk(self.acoustic_model, sample, self.device)
                predicted_latent = self.inject_sibilant_noise(predicted_latent, phoneme_ids, durations)
                oracle_audio = self.decode_latent(teacher_latent)
                student_audio = self.decode_latent(predicted_latent)
                teacher_parts.append(teacher_audio)
                oracle_parts.append(oracle_audio)
                student_parts.append(student_audio)
                chunk_summaries.append(
                    SynthChunk(
                        chunk_index=chunk_index,
                        text=chunk_text,
                        phoneme_count=int(len(phonemes)),
                        phoneme_id_count=int(len(phoneme_ids)),
                        frames=frame_count,
                        oracle_frames=teacher_frames,
                        duration_source=resolved_duration_source,
                        teacher_samples=int(teacher_audio.size),
                        oracle_samples=int(oracle_audio.size),
                        student_samples=int(student_audio.size),
                        w_ceil_sum=int(frame_count),
                        student_rms=rms(student_audio),
                        oracle_rms=rms(oracle_audio),
                        teacher_rms=rms(teacher_audio),
                    )
                )

        teacher_full = np.concatenate(teacher_parts).astype(np.float32)
        oracle_full = np.concatenate(oracle_parts).astype(np.float32)
        student_full = np.concatenate(student_parts).astype(np.float32)
        enhanced_full = self.enhance_audio(student_full).astype(np.float32)
        student_path = self.audio_dir / f"{render_id}-student.wav"
        enhanced_path = self.audio_dir / f"{render_id}-enhanced.wav"
        oracle_path = self.audio_dir / f"{render_id}-oracle-decoder.wav"
        teacher_path = self.audio_dir / f"{render_id}-teacher.wav"
        latent_mod.write_wav(student_path, student_full, self.sample_rate)
        latent_mod.write_wav(enhanced_path, enhanced_full, self.sample_rate)
        latent_mod.write_wav(oracle_path, oracle_full, self.sample_rate)
        latent_mod.write_wav(teacher_path, teacher_full, self.sample_rate)

        elapsed_ms = int(round((time.time() - started) * 1000.0))
        metadata = {
            "id": render_id,
            "text": clean_text,
            "created_at_unix": started,
            "elapsed_ms": elapsed_ms,
            "student_audio": str(student_path),
            "enhanced_audio": str(enhanced_path),
            "oracle_decoder_audio": str(oracle_path),
            "teacher_audio": str(teacher_path),
            "student_duration_s": float(student_full.size / self.sample_rate),
            "enhanced_duration_s": float(enhanced_full.size / self.sample_rate),
            "oracle_decoder_duration_s": float(oracle_full.size / self.sample_rate),
            "teacher_duration_s": float(teacher_full.size / self.sample_rate),
            "student_rms": rms(student_full),
            "enhanced_rms": rms(enhanced_full),
            "oracle_decoder_rms": rms(oracle_full),
            "teacher_rms": rms(teacher_full),
            "duration_source": requested_duration_source,
            "duration_length_scale": requested_length_scale,
            "text_chunking": text_chunking,
            "text_chunks": text_chunks,
            "chunks": [chunk.__dict__ for chunk in chunk_summaries],
            "model": self.model_card,
        }
        meta_path = self.meta_dir / f"{render_id}.json"
        meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return {
            "ok": True,
            "id": render_id,
            "student_audio": f"/audio/{student_path.name}",
            "enhanced_audio": f"/audio/{enhanced_path.name}",
            "oracle_decoder_audio": f"/audio/{oracle_path.name}",
            "teacher_audio": f"/audio/{teacher_path.name}",
            "elapsed_ms": elapsed_ms,
            "student_duration_s": metadata["student_duration_s"],
            "enhanced_duration_s": metadata["enhanced_duration_s"],
            "oracle_decoder_duration_s": metadata["oracle_decoder_duration_s"],
            "teacher_duration_s": metadata["teacher_duration_s"],
            "student_rms": metadata["student_rms"],
            "enhanced_rms": metadata["enhanced_rms"],
            "oracle_decoder_rms": metadata["oracle_decoder_rms"],
            "teacher_rms": metadata["teacher_rms"],
            "duration_source": requested_duration_source,
            "duration_length_scale": requested_length_scale,
            "text_chunking": text_chunking,
            "text_chunks": text_chunks,
            "chunks": metadata["chunks"],
            "model": self.model_card,
        }


def index_html(model: dict[str, Any]) -> str:
    acoustic_config = model.get("acoustic_config", {})
    duration_length_scale = float(model.get("duration_length_scale", 1.0))
    duration_source = str(model.get("duration_source", "student"))
    decoder_backend = str(model.get("decoder_backend", "student"))
    oracle_decoder_label = (
        "Teacher latent -> student decoder"
        if decoder_backend == "student"
        else "Teacher latent -> ONNX decoder"
    )
    student_duration_selected = " selected" if duration_source == "student" else ""
    oracle_duration_selected = " selected" if duration_source == "oracle" else ""
    title = html.escape(str(model.get("dashboard_title") or "Root A live TTS bench"))
    subtitle = html.escape(
        str(
            model.get("dashboard_subtitle")
            or "Arbitrary text through the current duration and acoustic students, with Piper as the reference."
        )
    )
    default_text = html.escape(str(model.get("default_text") or ""))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{
      --ink: #17191c;
      --paper: #f7f5ef;
      --panel: #ffffff;
      --line: #d8d2c4;
      --muted: #68645c;
      --accent: #0f766e;
      --accent-ink: #f7fffb;
      --warn: #a53f2b;
      --code: #263238;
      --field: #fbfaf7;
      --shadow: 0 14px 40px rgba(30, 28, 23, 0.10);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background:
        linear-gradient(90deg, rgba(23,25,28,.035) 1px, transparent 1px),
        linear-gradient(0deg, rgba(23,25,28,.028) 1px, transparent 1px),
        var(--paper);
      background-size: 28px 28px;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
    }}
    main {{
      width: min(1180px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 28px 0 40px;
    }}
    header {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 18px;
      align-items: end;
      border-bottom: 1px solid var(--line);
      padding-bottom: 16px;
      margin-bottom: 20px;
    }}
    h1 {{
      margin: 0;
      font-size: clamp(28px, 4vw, 52px);
      line-height: 0.95;
      letter-spacing: 0;
      font-weight: 780;
    }}
    .sub {{
      margin: 10px 0 0;
      color: var(--muted);
      max-width: 760px;
      font-size: 15px;
    }}
    .chipbar {{
      display: flex;
      flex-wrap: wrap;
      justify-content: flex-end;
      gap: 8px;
      min-width: 250px;
    }}
    .chip {{
      border: 1px solid var(--line);
      background: rgba(255,255,255,.72);
      padding: 7px 9px;
      border-radius: 7px;
      font: 12px/1.1 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      color: var(--code);
      white-space: nowrap;
    }}
    .grid {{
      display: grid;
      grid-template-columns: minmax(0, 1.05fr) minmax(330px, .95fr);
      gap: 18px;
      align-items: start;
    }}
    section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }}
    .composer {{ padding: 18px; }}
    label {{
      display: block;
      color: var(--muted);
      font-size: 13px;
      font-weight: 680;
      margin-bottom: 8px;
    }}
    textarea {{
      width: 100%;
      min-height: 190px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: var(--field);
      color: var(--ink);
      padding: 14px;
      font: 18px/1.55 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      outline: none;
    }}
    textarea:focus, button:focus-visible {{
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(15,118,110,.18);
    }}
    .actions {{
      display: flex;
      gap: 10px;
      align-items: center;
      margin-top: 12px;
      flex-wrap: wrap;
    }}
    .control {{
      display: grid;
      gap: 5px;
      min-width: 150px;
    }}
    select, input[type="number"] {{
      min-height: 42px;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: var(--field);
      color: var(--ink);
      padding: 8px 10px;
      font: 14px/1.1 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    button {{
      appearance: none;
      border: 0;
      border-radius: 7px;
      background: var(--accent);
      color: var(--accent-ink);
      padding: 11px 16px;
      font-size: 14px;
      font-weight: 760;
      cursor: pointer;
      min-height: 42px;
    }}
    button:disabled {{ opacity: .55; cursor: wait; }}
    .status {{
      color: var(--muted);
      font-size: 13px;
      min-height: 20px;
    }}
    .status.error {{ color: var(--warn); }}
    .samples {{
      display: grid;
      gap: 12px;
      margin-top: 16px;
    }}
    .sample {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #fcfbf8;
    }}
    .sample h2 {{
      margin: 0 0 8px;
      font-size: 14px;
      letter-spacing: 0;
      display: flex;
      justify-content: space-between;
      gap: 8px;
    }}
    audio {{ width: 100%; height: 38px; }}
    .meter {{
      height: 46px;
      border: 1px solid var(--line);
      border-radius: 6px;
      margin-top: 9px;
      background:
        repeating-linear-gradient(90deg, rgba(15,118,110,.18) 0 2px, transparent 2px 11px),
        linear-gradient(180deg, #fff, #f4f1e8);
      overflow: hidden;
      position: relative;
    }}
    .meter::after {{
      content: "";
      position: absolute;
      inset: 0;
      transform: translateX(var(--meter-offset, -100%));
      background: linear-gradient(90deg, transparent, rgba(15,118,110,.30), transparent);
      transition: transform .5s ease;
    }}
    .side {{ padding: 0; overflow: hidden; }}
    .side-head {{
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
      background: #f2efe6;
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: center;
    }}
    .side-head h2 {{ margin: 0; font-size: 15px; }}
    .model {{
      display: grid;
      grid-template-columns: 148px minmax(0, 1fr);
      gap: 0;
      font-size: 13px;
    }}
    .model div {{
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      min-width: 0;
      overflow-wrap: anywhere;
    }}
    .model div:nth-child(odd) {{
      color: var(--muted);
      background: #faf8f1;
      font-weight: 680;
    }}
    .diag {{
      padding: 14px 16px 16px;
      border-top: 1px solid var(--line);
    }}
    .diag h2 {{ margin: 0 0 10px; font-size: 15px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
    th, td {{
      text-align: left;
      border-bottom: 1px solid var(--line);
      padding: 7px 6px;
      vertical-align: top;
    }}
    th {{ color: var(--muted); font-weight: 720; background: #faf8f1; }}
    .empty {{ color: var(--muted); font-size: 13px; }}
    @media (max-width: 860px) {{
      header, .grid {{ grid-template-columns: 1fr; }}
      .chipbar {{ justify-content: flex-start; }}
      .model {{ grid-template-columns: 112px minmax(0,1fr); }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>{title}</h1>
        <p class="sub">{subtitle}</p>
      </div>
      <div class="chipbar" aria-label="model summary">
        <span class="chip">{model["student_parameters"]:,} params</span>
        <span class="chip">{model["sample_rate"]:,} Hz</span>
        <span class="chip">{model["device"]}</span>
      </div>
    </header>
    <div class="grid">
      <section class="composer">
        <label for="text">Text</label>
        <textarea id="text" spellcheck="false">{default_text}</textarea>
        <div class="actions">
          <button id="synth">Synthesize</button>
          <div class="control">
            <label for="durationSource">Durations</label>
            <select id="durationSource">
              <option value="student"{student_duration_selected}>Learned</option>
              <option value="oracle"{oracle_duration_selected}>Piper oracle</option>
            </select>
          </div>
          <div class="control">
            <label for="durationScale">Scale</label>
            <input id="durationScale" type="number" min="0.5" max="2.0" step="0.01" value="{duration_length_scale:.3f}">
          </div>
          <span id="status" class="status"></span>
        </div>
        <div class="samples" id="samples">
          <div class="sample">
            <h2><span>Student</span><span id="studentMeta"></span></h2>
            <audio id="studentAudio" controls preload="none"></audio>
            <div class="meter" id="studentMeter"></div>
          </div>
          <div class="sample">
            <h2><span>Enhanced student</span><span id="enhancedMeta"></span></h2>
            <audio id="enhancedAudio" controls preload="none"></audio>
            <div class="meter" id="enhancedMeter"></div>
          </div>
          <div class="sample">
            <h2><span>{oracle_decoder_label}</span><span id="oracleMeta"></span></h2>
            <audio id="oracleAudio" controls preload="none"></audio>
            <div class="meter" id="oracleMeter"></div>
          </div>
          <div class="sample">
            <h2><span>Teacher</span><span id="teacherMeta"></span></h2>
            <audio id="teacherAudio" controls preload="none"></audio>
            <div class="meter" id="teacherMeter"></div>
          </div>
        </div>
      </section>
      <section class="side">
        <div class="side-head">
          <h2>Current path</h2>
          <span class="chip">live</span>
        </div>
        <div class="model">
          <div>Acoustic</div><div>{Path(str(model["acoustic_checkpoint"])).name}</div>
          <div>Duration</div><div>{Path(str(model["duration_checkpoint"])).name if model.get("duration_checkpoint") else "Piper oracle"}</div>
          <div>Decoder</div><div>{Path(str(model["decoder"])).name}</div>
          <div>Enhancer</div><div>{Path(str(model["audio_enhancer"])).name if model.get("audio_enhancer") else "none"}</div>
          <div>Post DSP</div><div>{html.escape(str(model.get("postprocess_filter", "none")))} @ gain {float(model.get("postprocess_gain", 1.0)):.3f}</div>
          <div>Teacher</div><div>{Path(str(model["piper_model"])).name}</div>
          <div>Architecture</div><div>{str(acoustic_config.get("architecture", "unknown"))}</div>
          <div>Hidden</div><div>{str(acoustic_config.get("hidden", "unknown"))}</div>
          <div>Frame depth</div><div>{str(acoustic_config.get("depth", "unknown"))}</div>
          <div>Token depth</div><div>{str(acoustic_config.get("token_depth", "unknown"))}</div>
          <div>Durations</div><div>{model["durations"]}</div>
        </div>
        <div class="diag">
          <h2>Last render</h2>
          <div id="diag" class="empty">No render yet.</div>
        </div>
      </section>
    </div>
  </main>
  <script>
    const button = document.getElementById('synth');
    const defaultDurationScale = {duration_length_scale:.3f};
    const text = document.getElementById('text');
    const durationSource = document.getElementById('durationSource');
    const durationScale = document.getElementById('durationScale');
    const statusEl = document.getElementById('status');
    const studentAudio = document.getElementById('studentAudio');
    const enhancedAudio = document.getElementById('enhancedAudio');
    const oracleAudio = document.getElementById('oracleAudio');
    const teacherAudio = document.getElementById('teacherAudio');
    const studentMeta = document.getElementById('studentMeta');
    const enhancedMeta = document.getElementById('enhancedMeta');
    const oracleMeta = document.getElementById('oracleMeta');
    const teacherMeta = document.getElementById('teacherMeta');
    const studentMeter = document.getElementById('studentMeter');
    const enhancedMeter = document.getElementById('enhancedMeter');
    const oracleMeter = document.getElementById('oracleMeter');
    const teacherMeter = document.getElementById('teacherMeter');
    const diag = document.getElementById('diag');

    function seconds(value) {{
      return `${{Number(value).toFixed(2)}}s`;
    }}

    function setStatus(message, isError = false) {{
      statusEl.textContent = message;
      statusEl.className = isError ? 'status error' : 'status';
    }}

    function renderDiagnostics(data) {{
      const rows = data.chunks.map((chunk) => `
        <tr>
          <td>${{chunk.chunk_index}}</td>
          <td>${{chunk.phoneme_id_count}}</td>
          <td>${{chunk.frames}}</td>
          <td>${{chunk.oracle_frames}}</td>
          <td>${{chunk.duration_source}}</td>
          <td>${{chunk.student_samples}}</td>
          <td>${{chunk.oracle_samples}}</td>
          <td>${{chunk.teacher_samples}}</td>
        </tr>
      `).join('');
      diag.innerHTML = `
        <table>
          <thead><tr><th>Chunk</th><th>IDs</th><th>Frames</th><th>Oracle frames</th><th>Mode</th><th>Student samples</th><th>Oracle-decoder samples</th><th>Teacher samples</th></tr></thead>
          <tbody>${{rows}}</tbody>
        </table>
      `;
    }}

    async function synthesize() {{
      const value = text.value.trim();
      if (!value) {{
        setStatus('Enter text first.', true);
        return;
      }}
      button.disabled = true;
      setStatus('Synthesizing...');
      studentMeter.style.setProperty('--meter-offset', '-100%');
      enhancedMeter.style.setProperty('--meter-offset', '-100%');
      oracleMeter.style.setProperty('--meter-offset', '-100%');
      teacherMeter.style.setProperty('--meter-offset', '-100%');
      try {{
        const response = await fetch('/api/synthesize', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{
            text: value,
            duration_source: durationSource.value,
            duration_length_scale: Number(durationScale.value || defaultDurationScale),
          }}),
        }});
        const data = await response.json();
        if (!response.ok || !data.ok) {{
          throw new Error(data.error || `HTTP ${{response.status}}`);
        }}
        const cacheBust = `?t=${{Date.now()}}`;
        studentAudio.src = data.student_audio + cacheBust;
        enhancedAudio.src = data.enhanced_audio + cacheBust;
        oracleAudio.src = data.oracle_decoder_audio + cacheBust;
        teacherAudio.src = data.teacher_audio + cacheBust;
        studentMeta.textContent = `${{seconds(data.student_duration_s)}} · rms ${{Number(data.student_rms).toFixed(4)}}`;
        enhancedMeta.textContent = `${{seconds(data.enhanced_duration_s)}} · rms ${{Number(data.enhanced_rms).toFixed(4)}}`;
        oracleMeta.textContent = `${{seconds(data.oracle_decoder_duration_s)}} · rms ${{Number(data.oracle_decoder_rms).toFixed(4)}}`;
        teacherMeta.textContent = `${{seconds(data.teacher_duration_s)}} · rms ${{Number(data.teacher_rms).toFixed(4)}}`;
        setStatus(`Rendered ${{data.id}} in ${{data.elapsed_ms}} ms · ${{data.duration_source}} durations.`);
        studentMeter.style.setProperty('--meter-offset', '100%');
        enhancedMeter.style.setProperty('--meter-offset', '100%');
        oracleMeter.style.setProperty('--meter-offset', '100%');
        teacherMeter.style.setProperty('--meter-offset', '100%');
        renderDiagnostics(data);
      }} catch (error) {{
        setStatus(error.message, true);
      }} finally {{
        button.disabled = false;
      }}
    }}

    button.addEventListener('click', synthesize);
    text.addEventListener('keydown', (event) => {{
      if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') {{
        synthesize();
      }}
    }});
  </script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    state: DashboardState

    def log_message(self, format: str, *args: object) -> None:
        sys.stderr.write(f"[dashboard] {self.address_string()} - {format % args}\n")

    def do_HEAD(self) -> None:
        try:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                response_html_head(self, index_html(self.state.model_card))
                return
            if parsed.path == "/favicon.ico":
                response_no_content(self)
                return
            self.send_response(HTTPStatus.NOT_FOUND.value)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
        except Exception:
            traceback.print_exc()
            self.send_response(HTTPStatus.INTERNAL_SERVER_ERROR.value)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()

    def do_GET(self) -> None:
        try:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                response_html(self, index_html(self.state.model_card))
                return
            if parsed.path == "/favicon.ico":
                response_no_content(self)
                return
            if parsed.path.startswith("/audio/"):
                response_file(self, safe_audio_path(self.state.audio_dir, self.path), "audio/wav")
                return
            response_json(self, HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})
        except FileNotFoundError:
            response_json(self, HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})
        except Exception as exc:
            traceback.print_exc()
            response_json(self, HTTPStatus.INTERNAL_SERVER_ERROR, {"ok": False, "error": str(exc)})

    def do_POST(self) -> None:
        try:
            parsed = urlparse(self.path)
            if parsed.path != "/api/synthesize":
                response_json(self, HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})
                return
            length_raw = self.headers.get("Content-Length")
            if length_raw is None:
                raise ValueError("missing Content-Length")
            length = int(length_raw)
            if length <= 0 or length > 128 * 1024:
                raise ValueError("invalid request size")
            raw = self.rfile.read(length)
            try:
                payload = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError("request body must be JSON") from exc
            if not isinstance(payload, dict):
                raise ValueError("request JSON must be an object")
            text = payload.get("text")
            if not isinstance(text, str):
                raise ValueError("text must be a string")
            duration_source = payload.get("duration_source")
            if duration_source is not None and not isinstance(duration_source, str):
                raise ValueError("duration_source must be a string")
            duration_length_scale_raw = payload.get("duration_length_scale")
            duration_length_scale = None
            if duration_length_scale_raw is not None:
                try:
                    duration_length_scale = float(duration_length_scale_raw)
                except (TypeError, ValueError) as exc:
                    raise ValueError("duration_length_scale must be numeric") from exc
            result = self.state.synthesize(
                text,
                duration_source=duration_source,
                duration_length_scale=duration_length_scale,
            )
            response_json(self, HTTPStatus.OK, result)
        except ValueError as exc:
            response_json(self, HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
        except Exception as exc:
            traceback.print_exc()
            response_json(self, HTTPStatus.INTERNAL_SERVER_ERROR, {"ok": False, "error": str(exc)})


def main() -> None:
    args = parse_args()
    state = DashboardState(args)
    Handler.state = state
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(
        json.dumps(
            {
                "url": f"http://{args.host}:{args.port}/",
                "out_dir": str(args.out_dir),
                "model": state.model_card,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
