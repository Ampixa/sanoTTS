#!/usr/bin/env python3
"""Extract and validate a Piper/VITS decoder-only ONNX cut.

Root A needs a render primitive before training a latent student. This script
cuts a Piper graph at a selected decoder-side tensor and proves that saved
oracle tensors from a probe pack round-trip through the extracted decoder.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import tempfile
import wave
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, utils


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = (
    ROOT.parent
    / "g2p"
    / "data"
    / "external"
    / "piper_voices"
    / "ne_NP"
    / "chitwan-medium.onnx"
)
DEFAULT_PACK_DIR = (
    ROOT
    / "artifacts"
    / "sub10m-search"
    / "root-a-piper-vits"
    / "chitwan-medium-a1-32row-piper-native-pack-20260625"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "artifacts"
    / "sub10m-search"
    / "root-a-piper-vits"
    / "chitwan-medium-a2-decoder-cut-smoke-20260625"
)


def load_local_module(name: str, path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"module file not found: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--pack-dir", type=Path, default=DEFAULT_PACK_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--latent-name",
        default=None,
        help="Internal ONNX latent value to cut from. Defaults to resolved Root A generator_input.",
    )
    parser.add_argument("--output-name", default="output")
    parser.add_argument("--latent-channels", type=int, default=192)
    parser.add_argument(
        "--pack-tensor-key",
        default="generator_input",
        help="Tensor key inside each pack NPZ to feed during decoder cut validation.",
    )
    parser.add_argument("--max-row-mean-abs", type=float, default=5e-4)
    parser.add_argument("--min-row-cosine", type=float, default=0.9999)
    parser.add_argument("--max-rms-abs-error", type=float, default=5e-4)
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


def read_wav_float32(path: Path) -> tuple[np.ndarray, int]:
    require_file(path, "WAV")
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.getnframes()
        payload = wav.readframes(frames)
    if channels != 1:
        raise RuntimeError(f"expected mono WAV, got {channels} channels: {path}")
    if sample_width != 2:
        raise RuntimeError(f"expected 16-bit PCM WAV, got sample width {sample_width}: {path}")
    audio = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32767.0
    if audio.size <= 0:
        raise RuntimeError(f"empty WAV: {path}")
    return audio, int(sample_rate)


def rms(array: np.ndarray) -> float:
    value = np.asarray(array, dtype=np.float64).reshape(-1)
    if value.size <= 0:
        raise RuntimeError("cannot compute RMS of an empty array")
    return float(math.sqrt(float(np.mean(np.square(value)))))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a_f = np.asarray(a, dtype=np.float64).reshape(-1)
    b_f = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(a_f) * np.linalg.norm(b_f))
    if denom <= 0:
        raise RuntimeError("cannot compute cosine for zero-norm arrays")
    return float(np.dot(a_f, b_f) / denom)


def graph_value_names(model: onnx.ModelProto) -> set[str]:
    names: set[str] = set()
    for value in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        names.add(value.name)
    for node in model.graph.node:
        names.update(output for output in node.output if output)
    return names


def ensure_value_info(
    model: onnx.ModelProto,
    *,
    name: str,
    channels: int,
) -> onnx.ModelProto:
    value_names = graph_value_names(model)
    if name not in value_names:
        raise RuntimeError(f"latent cut tensor is not present in graph: {name}")

    typed_names = {
        value.name
        for value in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info)
    }
    if name not in typed_names:
        model.graph.value_info.append(
            helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, int(channels), "latent_frames"])
        )
    return model


def extract_decoder(args: argparse.Namespace) -> Path:
    require_file(args.model, "Piper ONNX model")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    decoder_path = args.out_dir / f"{args.model.stem}-decoder-from-generator-input.onnx"

    model = onnx.load(str(args.model))
    if args.latent_name is None:
        pack_mod = load_local_module(
            "roota_pack_builder_for_decoder_cut",
            ROOT / "tools" / "build_piper_vits_roota_probe_pack.py",
        )
        args.latent_name = pack_mod.resolve_tensor_outputs(model, ("generator_input",))["generator_input"]
    model = ensure_value_info(model, name=args.latent_name, channels=args.latent_channels)

    with tempfile.TemporaryDirectory(prefix="piper_decoder_cut_") as tmp_dir:
        tmp_model = Path(tmp_dir) / "model-with-cut-value-info.onnx"
        onnx.save(model, str(tmp_model))
        # ONNX's extractor preserves enough graph to run under ORT, but the
        # resulting dynamic input has incomplete checker metadata. Execution
        # validation below is the proof for this cut.
        utils.extract_model(
            str(tmp_model),
            str(decoder_path),
            [args.latent_name],
            [args.output_name],
            check_model=False,
        )
    require_file(decoder_path, "decoder ONNX")
    return decoder_path


def decode_chunk(
    session: ort.InferenceSession,
    latent_input_name: str,
    tensor_path: Path,
    *,
    pack_tensor_key: str,
) -> tuple[np.ndarray, float]:
    require_file(tensor_path, "tensor NPZ")
    with np.load(tensor_path) as tensors:
        if pack_tensor_key not in tensors.files:
            raise RuntimeError(f"{tensor_path}: missing {pack_tensor_key}; available={list(tensors.files)}")
        latent = np.asarray(tensors[pack_tensor_key], dtype=np.float32)
    if latent.ndim != 3:
        raise RuntimeError(f"{tensor_path}: expected 3D latent, got shape {latent.shape}")
    if not np.isfinite(latent).all():
        raise RuntimeError(f"{tensor_path}: non-finite latent")
    audio = np.asarray(session.run(None, {latent_input_name: latent})[0], dtype=np.float32).reshape(-1)
    if audio.size <= 0:
        raise RuntimeError(f"{tensor_path}: decoder returned empty audio")
    if not np.isfinite(audio).all():
        raise RuntimeError(f"{tensor_path}: decoder returned non-finite audio")
    return audio, rms(audio)


def validate_decoder(args: argparse.Namespace, decoder_path: Path) -> dict[str, Any]:
    require_dir(args.pack_dir, "A1 pack directory")
    rows = read_json(args.pack_dir / "rows.json")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"{args.pack_dir / 'rows.json'} must contain a non-empty list")

    session = ort.InferenceSession(str(decoder_path), providers=["CPUExecutionProvider"])
    input_names = [item.name for item in session.get_inputs()]
    output_names = [item.name for item in session.get_outputs()]
    if len(input_names) != 1:
        raise RuntimeError(f"decoder input mismatch: expected exactly one latent input, got {input_names}")
    if output_names != [args.output_name]:
        raise RuntimeError(f"decoder output mismatch: expected {[args.output_name]}, got {output_names}")
    latent_input_name = input_names[0]

    errors: list[str] = []
    chunk_rms_errors: list[float] = []
    chunk_sample_errors: list[int] = []
    row_comparisons: list[dict[str, Any]] = []
    chunks_decoded = 0

    for row in rows:
        row_id = str(row.get("row_id") or "")
        if not row_id:
            errors.append("row missing row_id")
            continue
        chunks = row.get("chunks")
        if not isinstance(chunks, list) or not chunks:
            errors.append(f"{row_id}: missing chunks")
            continue

        decoded_chunks: list[np.ndarray] = []
        for chunk in chunks:
            tensor_path = Path(str(chunk.get("tensor_npz") or ""))
            try:
                audio, audio_rms = decode_chunk(
                    session,
                    latent_input_name,
                    tensor_path,
                    pack_tensor_key=args.pack_tensor_key,
                )
            except Exception as exc:  # noqa: BLE001 - this is validation reporting
                errors.append(f"{row_id} chunk {chunk.get('chunk_index')}: {exc}")
                continue
            chunks_decoded += 1
            decoded_chunks.append(audio)

            expected_samples = int(chunk.get("audio_samples") or -1)
            sample_error = int(audio.size - expected_samples)
            chunk_sample_errors.append(abs(sample_error))
            if sample_error != 0:
                errors.append(
                    f"{row_id} chunk {chunk.get('chunk_index')}: sample count mismatch "
                    f"{audio.size} != {expected_samples}"
                )

            expected_rms = chunk.get("audio_rms")
            if expected_rms is not None:
                rms_error = abs(audio_rms - float(expected_rms))
                chunk_rms_errors.append(float(rms_error))
                if rms_error > args.max_rms_abs_error:
                    errors.append(
                        f"{row_id} chunk {chunk.get('chunk_index')}: RMS mismatch "
                        f"{audio_rms:.8f} vs {float(expected_rms):.8f}"
                    )

        if len(chunks) != 1 or len(decoded_chunks) != 1:
            continue

        wav_path = Path(str(row.get("audio") or ""))
        try:
            wav_audio, sample_rate = read_wav_float32(wav_path)
        except Exception as exc:  # noqa: BLE001 - this is validation reporting
            errors.append(f"{row_id}: {exc}")
            continue
        if sample_rate != int(row.get("sample_rate")):
            errors.append(f"{row_id}: sample rate mismatch {sample_rate} != {row.get('sample_rate')}")
            continue

        decoded = decoded_chunks[0]
        if decoded.size != wav_audio.size:
            errors.append(f"{row_id}: decoded/WAV sample mismatch {decoded.size} != {wav_audio.size}")
            continue

        diff = decoded - wav_audio
        mean_abs = float(np.mean(np.abs(diff)))
        max_abs = float(np.max(np.abs(diff)))
        rms_abs = rms(diff)
        cos = cosine(decoded, wav_audio)
        row_comparisons.append(
            {
                "row_id": row_id,
                "samples": int(decoded.size),
                "mean_abs": mean_abs,
                "max_abs": max_abs,
                "rms_abs": rms_abs,
                "cosine": cos,
                "decoder_rms": rms(decoded),
                "wav_rms": rms(wav_audio),
            }
        )
        if mean_abs > args.max_row_mean_abs:
            errors.append(f"{row_id}: mean abs too high {mean_abs:.8f}")
        if cos < args.min_row_cosine:
            errors.append(f"{row_id}: cosine too low {cos:.8f}")

    if not row_comparisons:
        errors.append("no single-chunk rows were available for WAV round-trip comparison")

    validation = {
        "passed": not errors,
        "model": str(args.model),
        "decoder_model": str(decoder_path),
        "pack_dir": str(args.pack_dir),
        "latent_name": args.latent_name,
        "pack_tensor_key": args.pack_tensor_key,
        "decoder_input_name": latent_input_name,
        "output_name": args.output_name,
        "rows": int(len(rows)),
        "chunks_decoded": int(chunks_decoded),
        "single_chunk_rows_compared": int(len(row_comparisons)),
        "max_chunk_sample_abs_error": int(max(chunk_sample_errors) if chunk_sample_errors else 0),
        "max_chunk_rms_abs_error": float(max(chunk_rms_errors) if chunk_rms_errors else 0.0),
        "mean_chunk_rms_abs_error": float(np.mean(chunk_rms_errors) if chunk_rms_errors else 0.0),
        "max_row_mean_abs": float(max((item["mean_abs"] for item in row_comparisons), default=0.0)),
        "max_row_rms_abs": float(max((item["rms_abs"] for item in row_comparisons), default=0.0)),
        "min_row_cosine": float(min((item["cosine"] for item in row_comparisons), default=1.0)),
        "row_comparisons": row_comparisons,
        "error_count": int(len(errors)),
        "errors": errors,
    }
    return validation


def main() -> None:
    args = parse_args()
    decoder_path = extract_decoder(args)
    validation = validate_decoder(args, decoder_path)
    write_json(args.out_dir / "decoder-cut-validation.json", validation)
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    if not validation["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
