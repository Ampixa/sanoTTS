"""Voice package resolution: local directories or cached downloads.

Two package layouts exist, because the project has two graphs.

**piperlite** (voices-v1) -- run by `models.py`:
  - manifest.json          (format "roota.raw-fp16.v1"; see
                             tools/export_roota_self_contained_package.py
                             in the saanoTTS research repo for the exporter)
  - weights.fp16.bin        (flat fp16 blob, tensors addressed by
                             manifest offset_bytes/nbytes)
  - piper-phoneme-config.json (codepoint -> phoneme-id table + espeak voice)

**nano** (voices-v2) -- run by `nano.py`:
  - meta.json               (lineage, per-file sha256, sample rate, vocab)
  - front_q8.bin / model_q8.bin  (or *_f32.bin when meta says weights=f32)

The two are not interchangeable: the nano graph is mel-100 -> ConvNeXt1D ->
iSTFT with a noise-fed decoder, which `models.py` rejects outright. Which one
a voice needs is recorded in tables/voices.json, and `load_voice` returns the
matching object -- `VoicePack` or `NanoVoicePack`.

Both layouts are mirrored on two hosts. Hugging Face (`ampixa/sanoTTS`) is
tried first and holds each package as a directory of loose files; GitHub
releases are the fallback and hold one .tar.gz per package. Set
SANOTTS_VOICE_SOURCE=hf or =github to pin one of them.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import shutil
import tarfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("sanotts.voicepack")

# Where voice packages are published. Each release serves `<package>.tar.gz`
# with its files at the ARCHIVE ROOT -- not nested under a directory, which
# would extract one level too deep and break resolution.
#
# Per voice, because the two graphs shipped in different releases: the
# piperlite voices in voices-v1 and the nano voices in voices-v2. A voice's
# `release` key in tables/voices.json selects one; anything without that key
# keeps the historical default, so existing installs are unaffected.
VOICE_RELEASE_ROOT = "https://github.com/Ampixa/sanoTTS/releases/download"
DEFAULT_VOICE_RELEASE = "voices-v1"
VOICE_RELEASE_BASE_URL = f"{VOICE_RELEASE_ROOT}/{DEFAULT_VOICE_RELEASE}"

# Hugging Face is tried FIRST and GitHub is the fallback. Two reasons, in
# order: HF is where people look for models, and a download only counts
# towards a model's visibility there if it actually goes through HF. GitHub
# releases stay as the fallback so an HF outage, a rate limit or a network
# that blocks it cannot break `pip install sanotts` for anyone.
#
# The two hosts hold DIFFERENT layouts and that is deliberate. HF stores each
# package as a directory of loose files, which is idiomatic there and lets the
# weights be browsed and previewed on the web. GitHub stores one .tar.gz per
# package. Hence two fetchers rather than one with a swapped base URL.
HF_REPO = "ampixa/sanoTTS"
HF_API_URL = f"https://huggingface.co/api/models/{HF_REPO}"
HF_RESOLVE_BASE = f"https://huggingface.co/{HF_REPO}/resolve/main"

# Set SANOTTS_VOICE_SOURCE to "github" or "hf" to pin one host -- for
# debugging, for an air-gapped mirror, or to reproduce a download exactly.
VOICE_SOURCE_ENV = "SANOTTS_VOICE_SOURCE"

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


@dataclass(frozen=True)
class NanoVoicePack:
    """A voices-v2 package: two weight blobs plus meta.json.

    Unlike VoicePack this holds no tensor table -- the nano runtime addresses
    weights by the byte offsets in the lineage's generated header, so the
    blobs travel as opaque bytes and `nano.py` slices them.
    """

    name: str
    directory: Path
    meta: dict[str, Any]
    front: bytes
    dec: bytes
    offsets: dict[str, int]

    @property
    def sample_rate(self) -> int:
        return int(self.meta["sample_rate"])

    @property
    def lineage(self) -> str:
        return str(self.meta["lineage"])

    @property
    def parameters(self) -> int:
        return int(self.meta["params"])

    @property
    def vocab_size(self) -> int:
        return int(self.meta["vocab_size"])


def _load_nano_directory(name: str, directory: Path) -> NanoVoicePack:
    meta_path = directory / "meta.json"
    if not meta_path.is_file():
        raise VoicePackError(f"{directory}: missing meta.json")
    with meta_path.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)
    if meta.get("runtime") != "snt_nano":
        raise VoicePackError(
            f"{directory}: meta.json runtime is {meta.get('runtime')!r}, expected 'snt_nano'"
        )
    blobs: dict[str, bytes] = {}
    for key in ("front", "dec"):
        filename = meta.get(key)
        if not filename:
            raise VoicePackError(f"{directory}: meta.json has no {key!r} entry")
        path = directory / filename
        if not path.is_file():
            raise VoicePackError(f"{directory}: missing {key} blob {filename}")
        data = path.read_bytes()
        expected_bytes = meta.get(f"{key}_bytes")
        if expected_bytes is not None and len(data) != int(expected_bytes):
            raise VoicePackError(
                f"{path}: size {len(data)} != meta {key}_bytes {expected_bytes}"
            )
        # meta.json ships a sha256 per blob; a silently corrupt download would
        # otherwise surface as noise rather than as an error.
        expected_sha = meta.get(f"{key}_sha256")
        if expected_sha:
            actual = _sha256_hex(data)
            if actual != expected_sha:
                raise VoicePackError(
                    f"{path}: sha256 mismatch (meta={expected_sha}, actual={actual}); "
                    "the voice package is corrupt or was tampered with"
                )
        blobs[key] = data

    # The runtime addresses weights by byte offset, and those offsets belong
    # to the lineage, not to the package format -- a wider decoder moves every
    # one of them. The generated header travels with the weights so a new
    # lineage needs no library release.
    header_name = meta.get("offsets_header", "nano_q8_meta.h")
    header_path = directory / header_name
    if not header_path.is_file():
        raise VoicePackError(
            f"{directory}: missing {header_name}. Voice packages published before "
            "the numpy nano runtime did not carry it; re-download the voice, or "
            "pass --voice-dir at a package that has one."
        )
    expected_sha = meta.get("offsets_header_sha256")
    if expected_sha:
        actual = _sha256_hex(header_path.read_bytes())
        if actual != expected_sha:
            raise VoicePackError(
                f"{header_path}: sha256 mismatch (meta={expected_sha}, actual={actual})"
            )
    from .nano import parse_meta_header  # noqa: PLC0415 - avoids a cycle at import time

    offsets = parse_meta_header(header_path)
    return NanoVoicePack(name=name, directory=directory, meta=meta,
                         front=blobs["front"], dec=blobs["dec"], offsets=offsets)


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


def _registry_entry(voice: str) -> dict[str, Any]:
    if not KNOWN_VOICES_TABLE.is_file():
        raise VoicePackError(f"missing bundled voice registry: {KNOWN_VOICES_TABLE}")
    with KNOWN_VOICES_TABLE.open("r", encoding="utf-8") as handle:
        registry = json.load(handle)
    entry = registry.get("voices", {}).get(voice)
    if entry is None:
        available = ", ".join(sorted(registry.get("voices", {})))
        raise VoicePackError(f"unknown voice {voice!r}; known voices: {available}")
    return dict(entry)


def _known_voice_archive_name(voice: str) -> str:
    return str(_registry_entry(voice)["package"])


def voice_runtime(voice: str) -> str:
    """'nano' or 'piperlite'. Which graph this voice needs."""
    return str(_registry_entry(voice).get("runtime", "piperlite"))


def _http_get(url: str, timeout: int = 60) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310 - fixed https hosts
        return response.read()


def _hf_package_files(package_name: str) -> list[str]:
    """Repo-relative paths HF holds under `package_name/`.

    Listed rather than assumed: the two layouts differ per package (piperlite
    ships weights.fp16.bin, nano ships two blobs plus a generated header), and
    a hardcoded file list would silently rot the next time a lineage changes.
    """
    payload = json.loads(_http_get(HF_API_URL, timeout=30).decode("utf-8"))
    prefix = f"{package_name}/"
    return [
        name
        for sibling in payload.get("siblings", ())
        if (name := sibling.get("rfilename", "")).startswith(prefix)
    ]


def _fetch_from_hf(package_name: str, dest_dir: Path) -> None:
    files = _hf_package_files(package_name)
    if not files:
        raise VoicePackError(f"{HF_REPO} has no package directory {package_name!r}")

    # Download into a sibling and rename at the end, so an interrupted fetch
    # can never leave a half-written directory that the cache check would then
    # treat as complete.
    staging = dest_dir.with_name(dest_dir.name + ".partial")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    try:
        for rfilename in files:
            relative = rfilename[len(package_name) + 1 :]
            if not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
                raise VoicePackError(f"refusing unsafe path from {HF_REPO}: {rfilename}")
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(_http_get(f"{HF_RESOLVE_BASE}/{rfilename}"))
        shutil.rmtree(dest_dir, ignore_errors=True)
        staging.replace(dest_dir)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _fetch_from_github(entry: dict[str, Any], package_name: str, dest_dir: Path) -> None:
    release = str(entry.get("release", DEFAULT_VOICE_RELEASE))
    # The archive name is decoupled from the package/directory name because a
    # release asset URL is CDN-cached by path: replacing a file in place can
    # keep serving the old bytes for an unbounded time. A corrected package
    # therefore ships under a new, lineage-tagged filename rather than
    # silently depending on a cache expiring.
    archive_name = str(entry.get("archive", package_name))
    url = f"{VOICE_RELEASE_ROOT}/{release}/{archive_name}.tar.gz"
    archive_bytes = _http_get(url)
    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
        safe_root = dest_dir.resolve()
        for member in archive.getmembers():
            member_path = (dest_dir / member.name).resolve()
            if safe_root not in member_path.parents and member_path != safe_root:
                raise VoicePackError(f"refusing to extract unsafe archive member: {member.name}")
        archive.extractall(dest_dir)  # noqa: S202 - paths validated above


def _download_and_extract(voice: str, cache_dir: Path) -> Path:
    entry = _registry_entry(voice)
    package_name = str(entry["package"])
    dest_dir = cache_dir / package_name
    # Either layout counts as already cached; checking only manifest.json
    # would re-download a nano voice on every call.
    if (dest_dir / "manifest.json").is_file() or (dest_dir / "meta.json").is_file():
        return dest_dir

    cache_dir.mkdir(parents=True, exist_ok=True)
    pinned = os.environ.get(VOICE_SOURCE_ENV, "").strip().lower()
    if pinned not in ("", "hf", "huggingface", "github"):
        raise VoicePackError(
            f"{VOICE_SOURCE_ENV}={pinned!r} is not recognised; use 'hf' or 'github'"
        )

    sources: list[tuple[str, Any]] = []
    if pinned in ("", "hf", "huggingface"):
        sources.append((f"Hugging Face ({HF_REPO})", lambda: _fetch_from_hf(package_name, dest_dir)))
    if pinned in ("", "github"):
        sources.append(
            ("GitHub releases", lambda: _fetch_from_github(entry, package_name, dest_dir))
        )

    failures: list[str] = []
    for label, fetch in sources:
        logger.info("sanotts: downloading voice %r from %s", voice, label)
        try:
            fetch()
        except (OSError, urllib.error.URLError, VoicePackError, ValueError) as exc:
            failures.append(f"{label}: {exc}")
            logger.info("sanotts: %s failed for %r (%s)", label, voice, exc)
            continue
        return dest_dir

    tried = "; ".join(failures)
    raise VoicePackError(
        f"could not download voice {voice!r} from any source ({tried}). "
        f"Use --voice-dir to point at a local voice package instead, e.g. a directory "
        "produced by tools/export_roota_self_contained_package.py."
    )


def load_voice(
    voice: str | None = None,
    *,
    voice_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
) -> VoicePack | NanoVoicePack:
    """Resolve a voice pack from an explicit directory, or by name (downloading
    into `cache_dir` if it is not already cached there).

    Returns a `NanoVoicePack` for the nano voices and a `VoicePack` for the
    piperlite ones. A local directory is classified by which of meta.json /
    manifest.json it actually contains, so `--voice-dir` needs no extra flag.
    """
    if voice_dir is not None:
        directory = Path(voice_dir).expanduser().resolve()
        if not directory.is_dir():
            raise VoicePackError(f"--voice-dir {directory} is not a directory")
        name = voice or directory.name
        if (directory / "meta.json").is_file() and not (directory / "manifest.json").is_file():
            return _load_nano_directory(name, directory)
        return _load_from_directory(name, directory)

    if not voice:
        raise VoicePackError("either voice or voice_dir must be given")
    resolved_cache_dir = Path(cache_dir).expanduser() if cache_dir else DEFAULT_CACHE_DIR
    directory = _download_and_extract(voice, resolved_cache_dir)
    if voice_runtime(voice) == "nano":
        return _load_nano_directory(voice, directory)
    return _load_from_directory(voice, directory)
