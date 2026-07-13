"""Text -> Piper phoneme-id frontend.

Mirrors the exact algorithm used by ``piper.phonemize_espeak`` /
``piper.phoneme_ids`` (see the upstream piper1-gpl project) without
depending on the ``piper-tts`` package itself (that package pulls in
onnxruntime as a hard dependency and expects a full ONNX voice to be
loaded before it will phonemize anything).

Instead we drive the same underlying espeak-ng shared library through
``phonemizer-fork`` + ``espeakng-loader``, then apply the exact
NFD-decompose-to-codepoints and ``phonemes_to_ids`` framing rules that
piper uses. This has been verified byte-for-byte against
``piper.voice.PiperVoice.phonemize`` / ``.phonemes_to_ids`` for
representative English sentences (see the package's gate test) -- the
key ingredients are ``with_stress=True``, ``tie=False``,
``preserve_punctuation=True`` (with piper's own punctuation set) and
``language_switch="remove-flags"``, plus a final ``rstrip()`` before
NFD decomposition (phonemizer emits a trailing space that piper does
not).

Known espeakng-loader pitfall (see project memory): some published
``espeakng-loader`` wheels report a data path via
``get_data_path()`` that looks valid on disk but the bundled dylib
silently ignores it and falls back to a baked-in build-time path from
the wheel's CI runner (e.g.
``/Users/runner/work/espeakng-loader/.../espeak-ng-data``), which does
not exist on the end user's machine. We defend against this by
actually trying to construct a working backend for each data-path
candidate (loader path, then common system install locations) instead
of trusting ``os.path.exists`` alone.
"""

from __future__ import annotations

import glob
import logging
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("sanotts.frontend")

# Framing ids fixed by the Piper phoneme_id_map convention; every voice's
# piper-phoneme-config.json must agree (checked in load_phoneme_table).
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2

# Piper's own punctuation set (phoneme_id_map keys that are ASCII
# punctuation rather than IPA symbols). Passed to phonemizer so
# preserve_punctuation keeps exactly these and nothing extra.
PUNCTUATION_MARKS = "!'(),-.:;?\""

_DATA_PATH_CANDIDATES: list[str] = [
    # Homebrew (macOS arm64/x86_64) -- versioned Cellar path, resolved via glob.
    "/opt/homebrew/Cellar/espeak-ng/*/share/espeak-ng-data",
    "/opt/homebrew/share/espeak-ng-data",
    "/usr/local/share/espeak-ng-data",
    "/usr/share/espeak-ng-data",
    "/usr/lib/x86_64-linux-gnu/espeak-ng-data",
]


class FrontendError(RuntimeError):
    """Raised when the espeak-ng backend cannot be initialized or used."""


@dataclass(frozen=True)
class PhonemeTable:
    """A single voice's codepoint -> Piper phoneme-id map."""

    espeak_voice: str
    id_map: dict[str, int]


def load_phoneme_table(piper_config_path: Path) -> PhonemeTable:
    """Parse a Piper ``*.onnx.json`` / ``piper-phoneme-config.json`` file.

    Raises FrontendError if the file does not have the exact shape we
    depend on (single-codepoint keys, single-id values, and the
    pad/bos/eos framing ids piper hardcodes).
    """
    import json

    if not piper_config_path.is_file():
        raise FrontendError(f"phoneme config not found: {piper_config_path}")
    with piper_config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    phoneme_type = config.get("phoneme_type", "espeak")
    if phoneme_type not in (None, "espeak"):
        raise FrontendError(f"{piper_config_path}: unsupported phoneme_type={phoneme_type!r}")

    espeak_cfg = config.get("espeak") or {}
    espeak_voice = espeak_cfg.get("voice")
    if not espeak_voice:
        raise FrontendError(f"{piper_config_path}: missing espeak.voice")

    raw_map = config.get("phoneme_id_map")
    if not isinstance(raw_map, dict) or not raw_map:
        raise FrontendError(f"{piper_config_path}: missing/empty phoneme_id_map")

    id_map: dict[str, int] = {}
    for key, ids in raw_map.items():
        if len(key) != 1:
            raise FrontendError(f"{piper_config_path}: multi-codepoint map key {key!r}")
        if not isinstance(ids, list) or len(ids) != 1:
            raise FrontendError(f"{piper_config_path}: multi-id map value {key!r} -> {ids!r}")
        id_map[key] = int(ids[0])

    for sym, want in (("_", PAD_ID), ("^", BOS_ID), ("$", EOS_ID)):
        got = id_map.get(sym)
        if got != want:
            raise FrontendError(
                f"{piper_config_path}: framing symbol {sym!r} maps to {got}, expected {want}"
            )

    return PhonemeTable(espeak_voice=str(espeak_voice), id_map=id_map)


class EspeakEngine:
    """Lazily-initialized espeak-ng backend, shared across voices.

    One process-wide instance is enough: phonemizer's EspeakBackend
    takes the target espeak voice as a constructor argument, so we
    cache one backend per requested espeak voice string.
    """

    def __init__(self) -> None:
        self._configured = False
        self._backends: dict[str, Any] = {}

    def _configure_once(self) -> None:
        if self._configured:
            return
        try:
            import espeakng_loader
        except ImportError as exc:  # pragma: no cover - dependency missing
            raise FrontendError(
                "the 'espeakng-loader' package is required for phonemization; "
                "install it with `pip install espeakng-loader`"
            ) from exc
        try:
            from phonemizer.backend import EspeakBackend
            from phonemizer.backend.espeak.wrapper import EspeakWrapper
        except ImportError as exc:  # pragma: no cover - dependency missing
            raise FrontendError(
                "the 'phonemizer-fork' package is required for phonemization; "
                "install it with `pip install phonemizer-fork`"
            ) from exc

        library_path = espeakng_loader.get_library_path()
        data_candidates = [espeakng_loader.get_data_path()]
        for pattern in _DATA_PATH_CANDIDATES:
            data_candidates.extend(sorted(glob.glob(pattern)))

        last_error: Exception | None = None
        for data_path in data_candidates:
            if not data_path or not Path(data_path).is_dir():
                continue
            try:
                EspeakWrapper.set_library(library_path)
                EspeakWrapper.set_data_path(data_path)
                # Constructing a throwaway backend exercises espeak_Initialize
                # end to end; a bad data path raises here rather than later
                # inside phonemize(), matching the documented loader pitfall.
                probe = EspeakBackend(
                    "en-us",
                    preserve_punctuation=True,
                    with_stress=True,
                    tie=False,
                    language_switch="remove-flags",
                )
                probe.phonemize(["a"], strip=False, separator=None)
            except Exception as exc:  # noqa: BLE001 - probing many candidates on purpose
                last_error = exc
                logger.debug("espeak data path candidate failed: %s (%s)", data_path, exc)
                continue
            logger.info("sanotts: using espeak-ng data path %s", data_path)
            self._configured = True
            return

        raise FrontendError(
            "could not initialize espeak-ng: no working espeak-ng-data directory found. "
            "Tried the espeakng-loader bundled path plus common system locations "
            f"({', '.join(_DATA_PATH_CANDIDATES)}). This is a known espeakng-loader "
            "packaging issue where the wheel's compiled-in default path does not match "
            "the data it ships. Fix: `brew install espeak-ng` (macOS) or "
            "`apt-get install espeak-ng` (Debian/Ubuntu), then retry."
        ) from last_error

    def backend(self, espeak_voice: str) -> Any:
        self._configure_once()
        cached = self._backends.get(espeak_voice)
        if cached is not None:
            return cached
        from phonemizer.backend import EspeakBackend

        # Some older voice packages (e.g. kristin) were trained against an
        # espeak-ng version where a bare language code like "en" was a
        # directly selectable voice. Newer espeak-ng only exposes regional
        # variants (en-us, en-gb, ...) as primary voices -- "en" now only
        # appears as a secondary/"other language" tag, so EspeakBackend("en")
        # raises. Retry with a regional variant rather than failing outright;
        # this is a voice-*selection* compatibility shim only (it changes
        # which accent espeak uses), not a phoneme-table substitution.
        candidates = [espeak_voice]
        if "-" not in espeak_voice:
            candidates += [f"{espeak_voice}-us", f"{espeak_voice}-gb"]

        last_error: Exception | None = None
        for candidate in candidates:
            try:
                backend = EspeakBackend(
                    candidate,
                    preserve_punctuation=True,
                    punctuation_marks=PUNCTUATION_MARKS,
                    with_stress=True,
                    tie=False,
                    language_switch="remove-flags",
                )
            except Exception as exc:  # noqa: BLE001 - trying multiple voice-name spellings
                last_error = exc
                continue
            if candidate != espeak_voice:
                logger.warning(
                    "sanotts: espeak voice %r unavailable in this espeak-ng build, "
                    "using %r instead (phoneme/accent may differ slightly from training)",
                    espeak_voice, candidate,
                )
            self._backends[espeak_voice] = backend
            return backend
        raise FrontendError(
            f"espeak-ng has no voice matching {espeak_voice!r} (tried {candidates})"
        ) from last_error

    def phonemize(self, text: str, espeak_voice: str) -> str:
        backend = self.backend(espeak_voice)
        out = backend.phonemize([text], strip=False, separator=None)
        if not out or not out[0]:
            raise FrontendError(f"espeak-ng produced no phonemes for text: {text!r}")
        # phonemizer appends a trailing separator space that piper's own
        # clause-based espeakbridge does not emit at true end-of-input.
        return out[0].rstrip()


_ENGINE = EspeakEngine()


def phonemes_to_ids(phonemes: list[str], table: PhonemeTable) -> list[int]:
    """Reproduce piper.phoneme_ids.phonemes_to_ids exactly:

    bos, pad, then (id, pad) per phoneme, then eos. Phonemes missing from
    the table are skipped with a warning, matching piper's behavior.
    """
    ids: list[int] = [BOS_ID, PAD_ID]
    missing = 0
    for phoneme in phonemes:
        pid = table.id_map.get(phoneme)
        if pid is None:
            missing += 1
            continue
        ids.append(pid)
        ids.append(PAD_ID)
    if missing:
        logger.warning("sanotts: %d phoneme(s) missing from the voice's phoneme_id_map, skipped", missing)
    ids.append(EOS_ID)
    return ids


def text_to_phoneme_ids(text: str, table: PhonemeTable) -> np.ndarray:
    """Full text -> Piper phoneme-id array, matching PiperVoice exactly."""
    clean_text = text.strip()
    if not clean_text:
        raise FrontendError("text is empty")
    phonemized = _ENGINE.phonemize(clean_text, table.espeak_voice)
    phonemes = list(unicodedata.normalize("NFD", phonemized))
    ids = phonemes_to_ids(phonemes, table)
    if len(ids) <= 3:
        raise FrontendError(f"phonemization produced no usable phonemes for: {text!r}")
    return np.asarray(ids, dtype=np.int64)
