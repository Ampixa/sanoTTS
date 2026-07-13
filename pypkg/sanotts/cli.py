"""sanotts command-line interface.

    sanotts say "Hello world" --voice amy -o out.wav
    sanotts say "Hello world" --voice-dir ./amy-en-1p46m -o out.wav
"""

from __future__ import annotations

import argparse
import logging
import sys
import wave
from pathlib import Path

import numpy as np

from .engine import Synthesizer
from .frontend import FrontendError
from .voicepack import VoicePackError


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    audio = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    pcm16 = (audio * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm16.tobytes())


def _add_say_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("text", help="Text to synthesize.")
    voice_group = parser.add_mutually_exclusive_group(required=True)
    voice_group.add_argument("--voice", help="Named voice to download/use from the cache, e.g. 'amy'.")
    voice_group.add_argument("--voice-dir", type=Path, help="Local voice package directory (has manifest.json).")
    parser.add_argument("-o", "--out", type=Path, default=Path("out.wav"), help="Output WAV path (default: out.wav).")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Override the voice download cache directory.")
    parser.add_argument(
        "--duration-length-scale",
        type=float,
        default=None,
        help="Override the voice's default speaking-rate scale (>0; larger = slower).",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable info-level logging.")


def _run_say(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(name)s: %(message)s")
    try:
        synth = Synthesizer(args.voice, voice_dir=args.voice_dir, cache_dir=args.cache_dir)
        result = synth.synthesize(args.text, duration_length_scale=args.duration_length_scale)
    except (FrontendError, VoicePackError, ValueError, RuntimeError, NotImplementedError) as exc:
        print(f"sanotts: error: {exc}", file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_wav(args.out, result.audio, result.sample_rate)
    duration_s = result.audio.shape[-1] / float(result.sample_rate)
    print(f"sanotts: wrote {args.out} ({duration_s:.2f}s at {result.sample_rate} Hz)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sanotts", description="Self-contained saanoTTS inference CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    say_parser = subparsers.add_parser("say", help="Synthesize text to a WAV file.")
    _add_say_args(say_parser)
    say_parser.set_defaults(func=_run_say)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
