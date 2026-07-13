"""Voice package resolution: local directories or cached downloads.

A voice package is a directory containing:
  - manifest.json          (format "roota.raw-fp16.v1"; see
                             tools/export_roota_self_contained_package.py
                             in the saanoTTS research repo for the exporter)
  - weights.fp16.bin        (flat fp16 blob, tensors addressed by
                             manifest offset_bytes/nbytes)
  - piper-phoneme-config.json (codepoint -> phoneme-id table + espeak voice)
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("sanotts.voicepack")

# Placeholder release location -- change this constant to point at wherever
# voice packages are actually published; everything else in this module is
# indifferent to the exact hosting scheme as long as it serves a
# `<name>.tar.gz` containing manifest.json + weights.fp16.bin + piper-phoneme-config.json
# at its root.
VOICE_RELEASE_BASE_URL = "https://github.com/Ampixa/sanoTTS/releases/download/voices-v1"

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "sanotts"

# name -> package archive/dir basename, for the voices published alongside
# this package (see releases/multivoice-20260713 in the research repo).
KNOWN_VOICES_TABLE = Path(__file__).parent / "tables" / "voices.json"


class VoicePackError(RuntimeError):
    pass


@dataclass(frozen=True)
class VoicePack:
    name: str
    directory: Path
    manifest: dict[str, Any]
    weights: bytes

    @property
    def sample_rate(self) -> int:
        return int(self.manifest["sample_rate"])

    @property
    def duration_length_scale(self) -> float:
        return float(self.manifest.get("inference", {}).get("duration_length_scale", 1.0))

    def component_tensors(self, component: str) -> dict[str, np.ndarray]:
        """Materialize one component's tensors as float32 numpy arrays."""
        comp = self.manifest["components"].get(component)
        if comp is None:
            raise VoicePackError(f"manifest has no component {component!r}")
        out: dict[str, np.ndarray] = {}
        for tensor in comp["tensors"]:
            name = tensor["name"]
            shape = tuple(tensor["shape"])
            dtype = tensor["dtype"]
            offset = int(tensor["offset_bytes"])
            nbytes = int(tensor["nbytes"])
            raw = self.weights[offset:offset + nbytes]
            if len(raw) != nbytes:
                raise VoicePackError(
                    f"{component}.{name}: truncated weights blob "
                    f"(wanted {nbytes} bytes at {offset}, got {len(raw)})"
                )
            if dtype == "float16":
                array = np.frombuffer(raw, dtype="<f2").astype(np.float32)
            elif dtype == "int64":
                array = np.frombuffer(raw, dtype="<i8").astype(np.int64)
            elif dtype == "int32":
                array = np.frombuffer(raw, dtype="<i4").astype(np.int32)
            else:
                raise VoicePackError(f"{component}.{name}: unsupported dtype {dtype!r}")
            out[name] = array.reshape(shape)
        return out

    def component_config(self, component: str) -> dict[str, Any]:
        comp = self.manifest["components"].get(component)
        if comp is None:
            raise VoicePackError(f"manifest has no component {component!r}")
        return comp["config"]

    @property
    def phoneme_config_path(self) -> Path:
        included = self.manifest.get("frontend", {}).get("included_config") or "piper-phoneme-config.json"
        path = self.directory / included
        if not path.is_file():
            raise VoicePackError(f"voice pack {self.name!r} is missing its phoneme config: {path}")
        return path


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_from_directory(name: str, directory: Path) -> VoicePack:
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise VoicePackError(f"{directory}: missing manifest.json")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("format") != "roota.raw-fp16.v1":
        raise VoicePackError(
            f"{directory}: unsupported manifest format {manifest.get('format')!r}, "
            "expected 'roota.raw-fp16.v1'"
        )
    weights_path = directory / manifest["weights_file"]
    if not weights_path.is_file():
        raise VoicePackError(f"{directory}: missing weights file {weights_path}")
    weights = weights_path.read_bytes()
    expected_size = int(manifest.get("weights_size_bytes", -1))
    if expected_size >= 0 and len(weights) != expected_size:
        raise VoicePackError(
            f"{weights_path}: size {len(weights)} != manifest weights_size_bytes {expected_size}"
        )
    expected_sha = manifest.get("weights_sha256")
    if expected_sha:
        actual_sha = _sha256_hex(weights)
        if actual_sha != expected_sha:
            raise VoicePackError(
                f"{weights_path}: sha256 mismatch (manifest={expected_sha}, actual={actual_sha}); "
                "the voice package is corrupt or was tampered with"
            )
    return VoicePack(name=name, directory=directory, manifest=manifest, weights=weights)


def _known_voice_archive_name(voice: str) -> str:
    if not KNOWN_VOICES_TABLE.is_file():
        raise VoicePackError(f"missing bundled voice registry: {KNOWN_VOICES_TABLE}")
    with KNOWN_VOICES_TABLE.open("r", encoding="utf-8") as handle:
        registry = json.load(handle)
    entry = registry.get("voices", {}).get(voice)
    if entry is None:
        available = ", ".join(sorted(registry.get("voices", {})))
        raise VoicePackError(f"unknown voice {voice!r}; known voices: {available}")
    return str(entry["package"])


def _download_and_extract(voice: str, cache_dir: Path) -> Path:
    package_name = _known_voice_archive_name(voice)
    dest_dir = cache_dir / package_name
    if (dest_dir / "manifest.json").is_file():
        return dest_dir

    url = f"{VOICE_RELEASE_BASE_URL}/{package_name}.tar.gz"
    logger.info("sanotts: downloading voice %r from %s", voice, url)
    try:
        with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 - fixed https host
            archive_bytes = response.read()
    except OSError as exc:
        raise VoicePackError(
            f"could not download voice {voice!r} from {url}: {exc}. "
            f"Use --voice-dir to point at a local voice package instead, e.g. a directory "
            "produced by tools/export_roota_self_contained_package.py."
        ) from exc

    cache_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
        safe_root = dest_dir.resolve()
        for member in archive.getmembers():
            member_path = (dest_dir / member.name).resolve()
            if safe_root not in member_path.parents and member_path != safe_root:
                raise VoicePackError(f"refusing to extract unsafe archive member: {member.name}")
        archive.extractall(dest_dir)  # noqa: S202 - paths validated above
    return dest_dir


def load_voice(
    voice: str | None = None,
    *,
    voice_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
) -> VoicePack:
    """Resolve a voice pack from an explicit directory, or by name (downloading
    into `cache_dir` if it is not already cached there)."""
    if voice_dir is not None:
        directory = Path(voice_dir).expanduser().resolve()
        if not directory.is_dir():
            raise VoicePackError(f"--voice-dir {directory} is not a directory")
        name = voice or directory.name
        return _load_from_directory(name, directory)

    if not voice:
        raise VoicePackError("either voice or voice_dir must be given")
    resolved_cache_dir = Path(cache_dir).expanduser() if cache_dir else DEFAULT_CACHE_DIR
    directory = _download_and_extract(voice, resolved_cache_dir)
    return _load_from_directory(voice, directory)
