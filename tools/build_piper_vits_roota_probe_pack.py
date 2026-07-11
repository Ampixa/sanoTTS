#!/usr/bin/env python3
"""Build a coherent Root A Piper/VITS probe pack from Piper/VITS inference.

The pack is intentionally Piper-native:

- Piper/eSpeak phoneme IDs are the frontend units.
- Piper/VITS `w_ceil` and path tensors are the timing/alignment evidence.
- Piper/VITS latent tensors and waveform are generated in the same inference
  pass.

It does not use external phone IDs or MFA durations as training labels.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import wave
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper
from piper.voice import PiperVoice


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
DEFAULT_CONFIG = DEFAULT_MODEL.with_suffix(".onnx.json")
DEFAULT_SOURCE_ROWS = (
    ROOT
    / "artifacts"
    / "nepali-mfa-male25-policy-clean"
    / "acoustic-duration-sat3-single-top-speaker.jsonl"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "artifacts"
    / "sub10m-search"
    / "root-a-piper-vits"
    / "chitwan-medium-a1-32row-piper-native-pack-20260625"
)
LEGACY_TENSOR_OUTPUTS = {
    "w_ceil": "w_ceil",
    "path": "path",
    "attn_mask": "attn_mask.3",
    "y_mask": "onnx::Unsqueeze_7230",
    "pre_final_flow": "tensor.31",
    "pre_decoder_latent": "onnx::Mul_7968",
    "generator_input": "onnx::Conv_7969",
    "generator_conv_pre": "x.191",
    "generator_first_upsample": "x.195",
}
ACOUSTIC_LOGICAL_OUTPUTS = (
    "w_ceil",
    "generator_input",
)
DECODER_LOGICAL_OUTPUTS = (
    "w_ceil",
    "generator_input",
    "generator_conv_pre",
    "generator_first_upsample",
)
FULL_LOGICAL_OUTPUTS = (
    "w_ceil",
    "path",
    "attn_mask",
    "y_mask",
    "pre_final_flow",
    "pre_decoder_latent",
    "generator_input",
    "generator_conv_pre",
    "generator_first_upsample",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-jsonl", type=Path, default=DEFAULT_SOURCE_ROWS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-rows", type=int, default=32)
    parser.add_argument(
        "--skip-rows",
        type=int,
        default=0,
        help="Skip this many eligible source rows before collecting --max-rows.",
    )
    parser.add_argument(
        "--eligible-index-modulo",
        type=int,
        default=0,
        help="If positive, select rows by 1-based eligible_index %% modulo.",
    )
    parser.add_argument(
        "--eligible-index-remainder",
        type=int,
        default=0,
        help="Remainder used with --eligible-index-modulo.",
    )
    parser.add_argument(
        "--invert-eligible-index-selection",
        action="store_true",
        help="Invert the modulo row selection, useful for train/eval splits.",
    )
    parser.add_argument("--min-source-seconds", type=float, default=0.5)
    parser.add_argument("--max-source-seconds", type=float, default=6.0)
    parser.add_argument(
        "--allow-text-only-source",
        action="store_true",
        help=(
            "Allow JSONL rows with text/target_text but no duration metadata. "
            "Useful when the Piper teacher itself generates the training targets."
        ),
    )
    parser.add_argument(
        "--default-text-only-seconds",
        type=float,
        default=2.0,
        help="Synthetic source duration recorded for text-only rows when --allow-text-only-source is set.",
    )
    parser.add_argument("--noise-scale", type=float, default=0.0)
    parser.add_argument("--length-scale", type=float, default=1.0)
    parser.add_argument("--noise-w", type=float, default=0.0)
    parser.add_argument("--sentence-silence", type=float, default=0.12)
    parser.add_argument(
        "--tensor-mode",
        choices=("full", "decoder", "acoustic"),
        default="full",
        help=(
            "full stores legacy debug tensors when available; decoder stores tensors needed for decoder "
            "training; acoustic stores only phoneme_ids, w_ceil, and generator_input."
        ),
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=1,
        help="Print progress every N rows; always prints the first and final row.",
    )
    return parser.parse_args()


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def finite_float(value: float) -> float:
    if not math.isfinite(value):
        raise RuntimeError(f"non-finite value: {value!r}")
    return value


def source_seconds(row: dict[str, Any]) -> float:
    for key in ("target_duration_s", "alignment_duration_s", "duration_sec"):
        value = row.get(key)
        if value is None:
            continue
        try:
            seconds = float(value)
        except (TypeError, ValueError):
            continue
        if seconds > 0:
            return finite_float(seconds)
    durations = row.get("hifigan_durations")
    if not isinstance(durations, list):
        raise RuntimeError("row missing usable duration field")
    return finite_float(float(sum(int(item) for item in durations) * 256.0 / 24000.0))


def read_source_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.max_rows <= 0:
        raise ValueError(f"--max-rows must be positive, got {args.max_rows}")
    if args.skip_rows < 0:
        raise ValueError(f"--skip-rows must be non-negative, got {args.skip_rows}")
    if args.eligible_index_modulo < 0:
        raise ValueError(f"--eligible-index-modulo must be non-negative, got {args.eligible_index_modulo}")
    if args.eligible_index_modulo == 0 and args.eligible_index_remainder != 0:
        raise ValueError("--eligible-index-remainder requires --eligible-index-modulo")
    if args.eligible_index_modulo > 0 and not (0 <= args.eligible_index_remainder < args.eligible_index_modulo):
        raise ValueError(
            f"--eligible-index-remainder must be in [0, {args.eligible_index_modulo}), "
            f"got {args.eligible_index_remainder}"
        )
    if args.default_text_only_seconds <= 0:
        raise ValueError(f"--default-text-only-seconds must be positive, got {args.default_text_only_seconds}")
    require_file(args.source_jsonl, "source JSONL")
    rows: list[dict[str, Any]] = []
    eligible_count = 0
    with args.source_jsonl.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{args.source_jsonl}:{line_no}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise RuntimeError(f"{args.source_jsonl}:{line_no}: expected JSON object")
            text = str(row.get("target_text") or row.get("text") or "").strip()
            if not text:
                raise RuntimeError(f"{args.source_jsonl}:{line_no}: missing target_text/text")
            try:
                seconds = source_seconds(row)
            except RuntimeError:
                if not args.allow_text_only_source:
                    raise
                seconds = finite_float(float(args.default_text_only_seconds))
            if not (args.min_source_seconds <= seconds <= args.max_source_seconds):
                continue
            eligible_count += 1
            if eligible_count <= args.skip_rows:
                continue
            if args.eligible_index_modulo > 0:
                selected = eligible_count % args.eligible_index_modulo == args.eligible_index_remainder
                if args.invert_eligible_index_selection:
                    selected = not selected
                if not selected:
                    continue
            row["_line_no"] = line_no
            row["_eligible_index"] = eligible_count
            row["_source_duration_s"] = seconds
            rows.append(row)
            if len(rows) >= args.max_rows:
                break
    if len(rows) < args.max_rows:
        raise RuntimeError(
            f"Only found {len(rows)} matching rows in {args.source_jsonl} after skipping "
            f"{args.skip_rows} eligible rows; expected {args.max_rows}"
        )
    return rows


def collect_value_names(model: onnx.ModelProto) -> set[str]:
    names: set[str] = set()
    for item in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        names.add(item.name)
    for node in model.graph.node:
        names.update(name for name in node.output if name)
    return names


def node_with_initializer(model: onnx.ModelProto, *, op_type: str, initializer_name: str) -> onnx.NodeProto:
    matches = [
        node
        for node in model.graph.node
        if node.op_type == op_type and initializer_name in set(node.input)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one {op_type} node consuming {initializer_name!r}, found {len(matches)}"
        )
    return matches[0]


def resolve_tensor_outputs(model: onnx.ModelProto, logical_outputs: tuple[str, ...]) -> dict[str, str]:
    value_names = collect_value_names(model)
    resolved: dict[str, str] = {}

    conv_pre_node: onnx.NodeProto | None = None
    first_upsample_node: onnx.NodeProto | None = None
    ceil_node: onnx.NodeProto | None = None

    for logical_name in logical_outputs:
        legacy_name = LEGACY_TENSOR_OUTPUTS.get(logical_name)
        if legacy_name in value_names:
            resolved[logical_name] = str(legacy_name)
            continue

        if logical_name == "w_ceil":
            if ceil_node is None:
                ceil_nodes = [node for node in model.graph.node if node.op_type == "Ceil"]
                if len(ceil_nodes) != 1:
                    raise RuntimeError(f"expected exactly one Ceil node for w_ceil, found {len(ceil_nodes)}")
                ceil_node = ceil_nodes[0]
            resolved[logical_name] = str(ceil_node.output[0])
        elif logical_name == "generator_input":
            if conv_pre_node is None:
                conv_pre_node = node_with_initializer(
                    model, op_type="Conv", initializer_name="dec.conv_pre.weight"
                )
            resolved[logical_name] = str(conv_pre_node.input[0])
        elif logical_name == "generator_conv_pre":
            if conv_pre_node is None:
                conv_pre_node = node_with_initializer(
                    model, op_type="Conv", initializer_name="dec.conv_pre.weight"
                )
            resolved[logical_name] = str(conv_pre_node.output[0])
        elif logical_name == "generator_first_upsample":
            if first_upsample_node is None:
                first_upsample_node = node_with_initializer(
                    model, op_type="ConvTranspose", initializer_name="dec.ups.0.weight"
                )
            resolved[logical_name] = str(first_upsample_node.output[0])
        else:
            raise RuntimeError(
                f"could not resolve logical tensor {logical_name!r}; legacy name {legacy_name!r} "
                "is not present in this ONNX graph"
            )

    missing = [name for name in resolved.values() if name not in value_names]
    if missing:
        raise RuntimeError(f"resolved ONNX outputs are not present in graph: {missing}")
    return resolved


def make_debug_model(model_path: Path, output_names: list[str]) -> onnx.ModelProto:
    require_file(model_path, "Piper ONNX")
    model = onnx.load(str(model_path))
    value_names = collect_value_names(model)
    missing = [name for name in output_names if name not in value_names]
    if missing:
        raise RuntimeError(f"Model is missing expected debug outputs: {missing}")
    existing = {output.name for output in model.graph.output}
    for name in output_names:
        if name not in existing:
            model.graph.output.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, None))
    return model


def array_stats(array: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(array)
    result: dict[str, Any] = {
        "shape": [int(value) for value in arr.shape],
        "dtype": str(arr.dtype),
        "numel": int(arr.size),
    }
    if not arr.size or not np.issubdtype(arr.dtype, np.number):
        return result
    arr_f = arr.astype(np.float64, copy=False)
    finite = np.isfinite(arr_f)
    result["finite_count"] = int(finite.sum())
    result["finite_ratio"] = float(finite.mean())
    if finite.any():
        values = arr_f[finite]
        result.update(
            {
                "min": finite_float(float(values.min())),
                "max": finite_float(float(values.max())),
                "sum": finite_float(float(values.sum())),
                "mean": finite_float(float(values.mean())),
                "std": finite_float(float(values.std())),
                "rms": finite_float(float(math.sqrt(float(np.mean(np.square(values)))))),
            }
        )
    return result


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        raise RuntimeError(f"refusing to write empty audio: {path}")
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())


def safe_row_id(index: int, line_no: int) -> str:
    return f"{index:05d}_line{line_no:06d}"


def run_piper_chunk(
    *,
    session: ort.InferenceSession,
    voice: PiperVoice,
    phonemes: list[str],
    scales: list[float],
    tensor_dir: Path,
    row_id: str,
    chunk_index: int,
    tensor_names: dict[str, str],
) -> tuple[np.ndarray, dict[str, Any]]:
    phoneme_ids = voice.phonemes_to_ids(phonemes)
    if not phoneme_ids:
        raise RuntimeError(f"{row_id} chunk {chunk_index}: empty Piper phoneme IDs")

    feeds = {
        "input": np.expand_dims(np.asarray(phoneme_ids, dtype=np.int64), 0),
        "input_lengths": np.asarray([len(phoneme_ids)], dtype=np.int64),
        "scales": np.asarray(scales, dtype=np.float32),
    }
    output_names = [output.name for output in session.get_outputs()]
    outputs = dict(zip(output_names, session.run(output_names, feeds), strict=True))
    audio = np.asarray(outputs["output"], dtype=np.float32).reshape(-1)

    chunk_slug = f"{row_id}_chunk{chunk_index:02d}"
    tensor_path = tensor_dir / f"{chunk_slug}.npz"
    w_ceil_raw = np.asarray(outputs[tensor_names["w_ceil"]], dtype=np.float32)
    generator_input_raw = np.asarray(outputs[tensor_names["generator_input"]], dtype=np.float32)
    saved_tensors: dict[str, np.ndarray] = {
        "phoneme_ids": np.asarray(phoneme_ids, dtype=np.int64),
        "w_ceil": w_ceil_raw,
        "generator_input": generator_input_raw,
    }
    for logical_name in (
        "path",
        "attn_mask",
        "y_mask",
        "pre_final_flow",
        "pre_decoder_latent",
        "generator_conv_pre",
        "generator_first_upsample",
    ):
        graph_name = tensor_names.get(logical_name)
        if graph_name is not None:
            saved_tensors[logical_name] = np.asarray(outputs[graph_name], dtype=np.float32)
    np.savez_compressed(tensor_path, **saved_tensors)

    w_ceil = w_ceil_raw.reshape(-1)
    generator_input = generator_input_raw
    path_name = tensor_names.get("path")
    path = np.asarray(outputs[path_name], dtype=np.float32) if path_name in outputs else None
    latent_frames = int(generator_input.shape[-1])
    w_ceil_sum = int(np.rint(w_ceil).sum())
    if latent_frames != w_ceil_sum:
        raise RuntimeError(
            f"{row_id} chunk {chunk_index}: latent frames {latent_frames} != w_ceil sum {w_ceil_sum}"
        )

    result = {
        "chunk_index": int(chunk_index),
        "phonemes": phonemes,
        "phoneme_ids": phoneme_ids,
        "phoneme_count": int(len(phonemes)),
        "phoneme_id_count": int(len(phoneme_ids)),
        "tensor_npz": str(tensor_path),
        "audio_samples": int(audio.size),
        "audio_rms": array_stats(audio).get("rms"),
        "w_ceil_sum": int(w_ceil_sum),
        "path_sum": finite_float(float(np.asarray(path, dtype=np.float64).sum())) if path is not None else None,
        "tensor_stats": {
            "w_ceil": array_stats(w_ceil_raw),
            "generator_input": array_stats(generator_input_raw),
        },
    }
    for logical_name in (
        "path",
        "attn_mask",
        "y_mask",
        "pre_final_flow",
        "pre_decoder_latent",
        "generator_conv_pre",
        "generator_first_upsample",
    ):
        graph_name = tensor_names.get(logical_name)
        if graph_name is not None:
            result["tensor_stats"][logical_name] = array_stats(outputs[graph_name])
    return audio, result


def html_page(rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    teacher_name = Path(str(summary["model"])).stem
    title = f"Root A {teacher_name} Piper-native Pack ({summary['rows']} rows, skip {summary['skip_rows']})"
    lines = [
        "<!doctype html>",
        '<meta charset="utf-8">',
        f"<title>{title}</title>",
        "<style>",
        "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:24px;line-height:1.35;color:#161616;background:#fafafa}",
        "table{border-collapse:collapse;width:100%;background:#fff}",
        "th,td{border:1px solid #ddd;padding:8px;vertical-align:top;font-size:13px}",
        "th{background:#f0f0f0;text-align:left}",
        "audio{width:220px}",
        ".text{font-size:16px;max-width:520px}",
        "</style>",
        f"<h1>{title}</h1>",
        f"<p>Rows: {summary['rows']} | total generated seconds: {summary['total_generated_seconds']:.3f} | "
        f"mean Piper IDs: {summary['mean_piper_id_count']:.2f}</p>",
        "<table>",
        "<thead><tr><th>#</th><th>Text</th><th>Audio</th><th>IDs</th><th>w_ceil</th><th>Latent</th></tr></thead>",
        "<tbody>",
    ]
    for row in rows:
        rel_audio = Path(row["audio"]).name
        chunk = row["chunks"][0]
        latent_shape = chunk["tensor_stats"]["generator_input"]["shape"]
        lines.append(
            "<tr>"
            f"<td>{row['index']}</td>"
            f"<td class='text'>{row['text']}</td>"
            f"<td><audio controls src='audio/{rel_audio}'></audio></td>"
            f"<td>{row['piper_id_count']}</td>"
            f"<td>{row['w_ceil_total']}</td>"
            f"<td>{latent_shape}</td>"
            "</tr>"
        )
    lines.extend(["</tbody>", "</table>"])
    return "\n".join(lines) + "\n"


def validate_pack(out_dir: Path, rows: list[dict[str, Any]], tensor_mode: str) -> dict[str, Any]:
    required_keys = {
        "phoneme_ids",
        "w_ceil",
        "generator_input",
    }
    if tensor_mode in {"full", "decoder"}:
        required_keys.update(
            {
                "generator_conv_pre",
                "generator_first_upsample",
            }
        )
    errors: list[str] = []
    for row in rows:
        audio_path = Path(row["audio"])
        if not audio_path.is_file():
            errors.append(f"{row['row_id']}: missing audio {audio_path}")
        else:
            with wave.open(str(audio_path), "rb") as wav:
                if wav.getframerate() != int(row["sample_rate"]):
                    errors.append(f"{row['row_id']}: sample rate mismatch")
                if wav.getnframes() <= 0:
                    errors.append(f"{row['row_id']}: empty audio")
        for chunk in row["chunks"]:
            tensor_path = Path(chunk["tensor_npz"])
            if not tensor_path.is_file():
                errors.append(f"{row['row_id']} chunk {chunk['chunk_index']}: missing {tensor_path}")
                continue
            with np.load(tensor_path) as tensors:
                missing = sorted(required_keys - set(tensors.files))
                if missing:
                    errors.append(f"{row['row_id']} chunk {chunk['chunk_index']}: missing keys {missing}")
                    continue
                phoneme_ids = tensors["phoneme_ids"]
                w_ceil = tensors["w_ceil"]
                generator_input = tensors["generator_input"]
                if int(phoneme_ids.shape[0]) != int(chunk["phoneme_id_count"]):
                    errors.append(f"{row['row_id']} chunk {chunk['chunk_index']}: phoneme ID count mismatch")
                w_ceil_sum = int(round(float(w_ceil.sum())))
                if w_ceil_sum != int(chunk["w_ceil_sum"]):
                    errors.append(f"{row['row_id']} chunk {chunk['chunk_index']}: w_ceil sum mismatch")
                if int(generator_input.shape[-1]) != int(chunk["w_ceil_sum"]):
                    errors.append(f"{row['row_id']} chunk {chunk['chunk_index']}: latent/w_ceil mismatch")
                if "y_mask" in tensors and int(tensors["y_mask"].shape[-1]) != int(chunk["w_ceil_sum"]):
                    errors.append(f"{row['row_id']} chunk {chunk['chunk_index']}: y_mask/w_ceil mismatch")
                for key in required_keys:
                    value = tensors[key]
                    if value.size == 0:
                        errors.append(f"{row['row_id']} chunk {chunk['chunk_index']}: empty {key}")
                    if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
                        errors.append(f"{row['row_id']} chunk {chunk['chunk_index']}: non-finite {key}")

    validation = {
        "passed": not errors,
        "rows": int(len(rows)),
        "tensor_mode": str(tensor_mode),
        "chunks": int(sum(len(row["chunks"]) for row in rows)),
        "audio_files": int(len(list((out_dir / "audio").glob("*.wav")))),
        "tensor_npz_files": int(len(list((out_dir / "tensors").glob("*.npz")))),
        "total_generated_seconds": finite_float(float(sum(row["generated_duration_s"] for row in rows))),
        "min_generated_seconds": finite_float(float(min(row["generated_duration_s"] for row in rows))),
        "max_generated_seconds": finite_float(float(max(row["generated_duration_s"] for row in rows))),
        "min_piper_id_count": int(min(row["piper_id_count"] for row in rows)),
        "max_piper_id_count": int(max(row["piper_id_count"] for row in rows)),
        "min_w_ceil_total": int(min(row["w_ceil_total"] for row in rows)),
        "max_w_ceil_total": int(max(row["w_ceil_total"] for row in rows)),
        "error_count": int(len(errors)),
        "errors": errors,
    }
    (out_dir / "validation-summary.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if errors:
        raise RuntimeError(f"Root A pack validation failed with {len(errors)} errors")
    return validation


def build_pack(args: argparse.Namespace) -> dict[str, Any]:
    require_file(args.model, "Piper ONNX")
    require_file(args.config, "Piper config")
    rows = read_source_rows(args)

    source_model = onnx.load(str(args.model))
    if args.tensor_mode == "acoustic":
        logical_outputs = ACOUSTIC_LOGICAL_OUTPUTS
    elif args.tensor_mode == "decoder":
        logical_outputs = DECODER_LOGICAL_OUTPUTS
    else:
        logical_outputs = FULL_LOGICAL_OUTPUTS
    tensor_names = resolve_tensor_outputs(source_model, logical_outputs)
    debug_outputs = list(dict.fromkeys(tensor_names.values()))
    debug_model = make_debug_model(args.model, debug_outputs)
    session = ort.InferenceSession(debug_model.SerializeToString(), providers=["CPUExecutionProvider"])
    voice = PiperVoice.load(args.model, args.config)
    sample_rate = int(voice.config.sample_rate)
    scales = [float(args.noise_scale), float(args.length_scale), float(args.noise_w)]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = args.out_dir / "audio"
    tensor_dir = args.out_dir / "tensors"
    audio_dir.mkdir(parents=True, exist_ok=True)
    tensor_dir.mkdir(parents=True, exist_ok=True)

    split_suffix = f"skip{args.skip_rows}-n{args.max_rows}"
    if args.eligible_index_modulo > 0:
        split_suffix += f"-mod{args.eligible_index_modulo}rem{args.eligible_index_remainder}"
        if args.invert_eligible_index_selection:
            split_suffix += "-invert"
    manifest_path = args.out_dir / f"root-a-piper-native-{split_suffix}.jsonl"
    row_summaries: list[dict[str, Any]] = []
    generated_seconds: list[float] = []
    piper_id_counts: list[int] = []
    w_ceil_totals: list[int] = []

    with manifest_path.open("w", encoding="utf-8") as out:
        for index, row in enumerate(rows, start=1):
            row_id = safe_row_id(index, int(row["_line_no"]))
            text = str(row.get("target_text") or row.get("text") or "").strip()
            sentence_phonemes = voice.phonemize(text)
            if not sentence_phonemes:
                raise RuntimeError(f"{row_id}: Piper produced no sentence phonemes")

            audio_parts: list[np.ndarray] = []
            chunks: list[dict[str, Any]] = []
            silence = np.zeros(int(round(args.sentence_silence * sample_rate)), dtype=np.float32)
            for chunk_index, phonemes in enumerate(sentence_phonemes):
                if not phonemes:
                    raise RuntimeError(f"{row_id} chunk {chunk_index}: empty phoneme list")
                if chunk_index > 0 and silence.size:
                    audio_parts.append(silence)
                audio, chunk_info = run_piper_chunk(
                    session=session,
                    voice=voice,
                    phonemes=phonemes,
                    scales=scales,
                    tensor_dir=tensor_dir,
                    row_id=row_id,
                    chunk_index=chunk_index,
                    tensor_names=tensor_names,
                )
                audio_parts.append(audio)
                chunks.append(chunk_info)

            full_audio = np.concatenate(audio_parts).astype(np.float32)
            audio_path = audio_dir / f"{row_id}.wav"
            write_wav(audio_path, full_audio, sample_rate)
            duration_s = finite_float(float(full_audio.size / sample_rate))
            piper_id_count = int(sum(chunk["phoneme_id_count"] for chunk in chunks))
            w_ceil_total = int(sum(chunk["w_ceil_sum"] for chunk in chunks))
            generated_seconds.append(duration_s)
            piper_id_counts.append(piper_id_count)
            w_ceil_totals.append(w_ceil_total)

            out_row = {
                "row_id": row_id,
                "index": int(index),
                "source_line_no": int(row["_line_no"]),
                "source_eligible_index": int(row["_eligible_index"]),
                "source_jsonl": str(args.source_jsonl),
                "source_duration_s": finite_float(float(row["_source_duration_s"])),
                "source_audio": str(row.get("target_audio") or row.get("audio") or row.get("audio_path") or ""),
                "text": text,
                "root": "root_a_piper_vits_native",
                "teacher_model": str(args.model),
                "teacher_config": str(args.config),
                "sample_rate": sample_rate,
                "tensor_mode": str(args.tensor_mode),
                "scales": {
                    "noise_scale": float(args.noise_scale),
                    "length_scale": float(args.length_scale),
                    "noise_w": float(args.noise_w),
                },
                "audio": str(audio_path),
                "generated_duration_s": duration_s,
                "sentence_count": int(len(chunks)),
                "piper_id_count": piper_id_count,
                "w_ceil_total": w_ceil_total,
                "chunks": chunks,
            }
            out.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            row_summaries.append(out_row)

            if index == 1 or index == len(rows) or index % max(int(args.progress_interval), 1) == 0:
                print(
                    json.dumps(
                        {
                            "progress": index,
                            "row_id": row_id,
                            "piper_id_count": piper_id_count,
                            "w_ceil_total": w_ceil_total,
                            "generated_duration_s": duration_s,
                        },
                        ensure_ascii=False,
                    ),
                    file=sys.stderr,
                    flush=True,
                )

    summary = {
        "passed": True,
        "rows": int(len(row_summaries)),
        "model": str(args.model),
        "config": str(args.config),
        "source_jsonl": str(args.source_jsonl),
        "max_rows": int(args.max_rows),
        "skip_rows": int(args.skip_rows),
        "eligible_index_modulo": int(args.eligible_index_modulo),
        "eligible_index_remainder": int(args.eligible_index_remainder),
        "invert_eligible_index_selection": bool(args.invert_eligible_index_selection),
        "manifest": str(manifest_path),
        "out_dir": str(args.out_dir),
        "sample_rate": sample_rate,
        "tensor_mode": str(args.tensor_mode),
        "scales": {
            "noise_scale": float(args.noise_scale),
            "length_scale": float(args.length_scale),
            "noise_w": float(args.noise_w),
        },
        "logical_outputs": dict(tensor_names),
        "debug_outputs": list(debug_outputs),
        "total_generated_seconds": finite_float(float(sum(generated_seconds))),
        "mean_generated_seconds": finite_float(float(statistics.fmean(generated_seconds))),
        "mean_piper_id_count": finite_float(float(statistics.fmean(piper_id_counts))),
        "mean_w_ceil_total": finite_float(float(statistics.fmean(w_ceil_totals))),
        "min_w_ceil_total": int(min(w_ceil_totals)),
        "max_w_ceil_total": int(max(w_ceil_totals)),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.out_dir / "rows.json").write_text(
        json.dumps(row_summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.out_dir / "index.html").write_text(html_page(row_summaries, summary), encoding="utf-8")
    validation = validate_pack(args.out_dir, row_summaries, str(args.tensor_mode))
    summary["validation"] = validation
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    args = parse_args()
    summary = build_pack(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
