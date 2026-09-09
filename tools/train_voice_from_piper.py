#!/usr/bin/env python3
"""Distil a complete sanoTTS voice from a Rhasspy/Piper teacher, in one command.

    python3 tools/train_voice_from_piper.py --voice vi_VN-vais1000-medium --auto-text

Everything from "a Piper voice exists on Hugging Face" to "a shipped package
directory" runs from here: teacher download, preflight, the four teacher packs,
the decoder cut, three students, the two decoder adaptation stages, the joint
finetune, the quality gate and the export.

This is the id/vi path that actually shipped (releases/multivoice-20260713),
reproduced stage for stage from the campaign script that produced it. It is not
a new recipe and the hyperparameters below are not tuned here -- they are copied
from the runs whose checkpoints are in that release. See
docs/distillation-recipe.md for why the pipeline has this shape.

THREE THINGS THIS ADDS over running the stages by hand:

1. Signature nodes are read off the teacher graph instead of being typed in.
   tools/build_piper_vits_decoder_signature_pack.py falls back to value names
   from a 2026-06 Nepali export when --node is not given. Those names do not
   exist in a modern Piper export, so the fallback silently produces a
   signature pack the decoder trainer then cannot use. This resolves the names
   from the teacher's own graph and refuses to continue if the five keys the
   trainer needs are not all present.

2. The teacher's front end is exercised before any compute. Whatever it is --
   eSpeak, pinyin/g2pW, Hebrew niqqud -- if it cannot phonemize the first row
   of the corpus there is no voice to build, and finding that out after the
   packs have rendered is a wasted hour. The students are also sized against
   the teacher's phoneme_id_map rather than against the ids the corpus happens
   to contain, so the embedding covers every id the front end may later emit.

3. The export stage exists. The campaign script stopped at the joint finetune
   and the package was built by hand afterwards.

RESUMABLE. Every stage declares the file it writes last, and is skipped when
that file exists. A crash mid-stage leaves no done-file, so a re-run repeats
exactly the stage that failed. Never point a done-file at a directory a tool
creates early -- a crash would leave the directory behind and fake completion.

COST. The teacher renders twice (once for the acoustic pack, once for the
decoder pack), which is most of the wall time. Reported runs are several hours
on an M-series box. Use --dry-run first: it prints every command without
running anything.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import resource
import shutil
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

ROOT = Path(__file__).resolve().parents[1]

# Rhasspy publishes every Piper voice in one repo, keyed by a voices.json index
# that carries the per-voice file list. Resolving through the index rather than
# guessing the directory layout means a renamed or restructured voice fails
# loudly here instead of 404-ing mid-download.
PIPER_REPO = "rhasspy/piper-voices"
PIPER_BASE = f"https://huggingface.co/{PIPER_REPO}/resolve/main"
PIPER_INDEX = f"{PIPER_BASE}/voices.json"

# The five activation signatures the decoder trainer consumes. The signature
# pack may carry more; these are the ones whose absence makes the run pointless.
REQUIRED_SIGNATURE_KEYS = ("stage0_mix", "stage1_mix", "stage2_mix", "pre_tanh", "audio")

# Languages tools/make_multilang_text_distill_corpus.py can generate text for.
AUTO_TEXT_LANGUAGES = ("en_US", "hi_IN", "zh_CN", "id_ID", "vi_VN", "te_IN")

# Rows held out of training, taken from the front of the corpus: the first 12
# render the smoke pack that the decoder cut is validated against, the next 128
# are the evaluation pack. Everything after row 140 is training text.
SMOKE_ROWS = 12
EVAL_ROWS = 128
HELDOUT_ROWS = SMOKE_ROWS + EVAL_ROWS
DECODER_TRAIN_ROWS = 512


@dataclass
class Profile:
    """Step counts per stage. `shipped` is what produced the id/vi release."""

    name: str
    duration_steps: int
    acoustic_steps: int
    decoder_steps: int
    zmix_steps: int
    joint_steps: int


@dataclass
class Size:
    """Model widths. Parameters scale with the square of these, so halving a
    width quarters that stage."""

    name: str
    duration_hidden: int
    acoustic_hidden: int
    acoustic_depth: int
    decoder_channels: str


SIZES = {
    # What the id/vi release shipped: ~1.56M total.
    "standard": Size("standard", 64, 96, 4, "160,80,40,20"),
    # Under 500k for microcontroller-class deployment. Duration drops to the
    # width the English recipe uses; the acoustic and decoder halve, which
    # quarters their parameter counts.
    "tiny": Size("tiny", 32, 48, 3, "96,48,24,12"),
}


PROFILES = {
    # Verified against the train_args serialized into the shipped id/vi
    # checkpoints, not just the campaign script's text.
    "shipped": Profile("shipped", 4000, 5000, 2800, 2500, 4000),
    # For proving the plumbing on a new voice before committing a night to it.
    # A smoke voice will sound bad; that is not a regression.
    "smoke": Profile("smoke", 200, 200, 100, 100, 100),
}


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def die(message: str) -> NoReturn:
    print(f"\nERROR: {message}\n", file=sys.stderr)
    raise SystemExit(1)


# --------------------------------------------------------------------------
# teacher resolution
# --------------------------------------------------------------------------

def fetch_json(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 - fixed https host
        return json.load(response)


def download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        log(f"  have {dest.name} ({dest.stat().st_size:,} bytes)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    log(f"  downloading {dest.name} …")
    with urllib.request.urlopen(url, timeout=600) as response:  # noqa: S310
        with tmp.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    tmp.replace(dest)
    log(f"  got {dest.name} ({dest.stat().st_size:,} bytes)")


def resolve_teacher(voice: str, teacher_dir: Path) -> tuple[Path, Path]:
    """Return (onnx, onnx.json) for a Rhasspy voice key, downloading if needed."""
    index = fetch_json(PIPER_INDEX)
    entry = index.get(voice)
    if entry is None:
        matches = [k for k in index if voice.lower() in k.lower()]
        hint = ("\n  did you mean: " + ", ".join(sorted(matches)[:8])) if matches else ""
        die(f"{voice!r} is not a voice in {PIPER_REPO}.{hint}")
    files = list(entry.get("files") or {})
    onnx_rel = next((f for f in files if f.endswith(".onnx")), None)
    config_rel = next((f for f in files if f.endswith(".onnx.json")), None)
    if onnx_rel is None or config_rel is None:
        die(f"{voice}: the index lists no .onnx/.onnx.json pair (files: {files})")
    onnx = teacher_dir / voice / Path(onnx_rel).name
    config = teacher_dir / voice / Path(config_rel).name
    download(f"{PIPER_BASE}/{onnx_rel}", onnx)
    download(f"{PIPER_BASE}/{config_rel}", config)
    return onnx, config


# --------------------------------------------------------------------------
# preflight
# --------------------------------------------------------------------------

def espeak_can_speak(voice: str) -> tuple[bool, str]:
    """Can the eSpeak the PACK BUILDER will use actually speak this voice?

    Piper ships its own compiled espeakbridge and espeak-ng-data, and that --
    not any system install -- is what renders the packs. Checking `espeak-ng
    --voices` on PATH asks the wrong question: k2 has no espeak binary and
    trains fine. So this drives piper's own phonemizer and sees whether real
    phonemes come back.

    Falls back to the system binary only when piper's bridge cannot be
    imported, so the check still says something useful outside a piper env.
    """
    try:
        from piper.phonemize_espeak import EspeakPhonemizer  # noqa: PLC0415

        phonemizer = EspeakPhonemizer()
        for candidate in ([voice] if "-" in voice else [voice, f"{voice}-us", f"{voice}-gb"]):
            try:
                out = phonemizer.phonemize(candidate, "test")
            except Exception:  # noqa: BLE001 - an unknown voice raises from the C bridge
                continue
            if out and any(out):
                return True, f"piper's bundled eSpeak ({candidate})"
        return False, "piper's bundled eSpeak returned no phonemes"
    except ImportError:
        pass
    binary = shutil.which("espeak-ng") or shutil.which("espeak")
    if binary is None:
        return False, "no piper espeak bridge and no espeak-ng on PATH"
    out = subprocess.run([binary, "--voices"], capture_output=True, text=True, check=True).stdout
    codes = {p[1].lower() for p in (l.split() for l in out.splitlines()[1:]) if len(p) >= 2}
    ok = any(c in codes for c in ([voice.lower()] if "-" in voice
                                  else [voice.lower(), f"{voice.lower()}-us", f"{voice.lower()}-gb"]))
    return ok, f"system espeak-ng at {binary}"


def teacher_vocab_size(config: dict[str, Any], config_path: Path) -> int:
    """Embedding rows the students need to cover every id this teacher can emit.

    Both students size their phoneme embedding from the ids they OBSERVE in the
    packs unless told otherwise, and a corpus never exercises the whole
    inventory. The ten voices shipped on 2026-09-08 are all built this way and
    all of them are short: de_DE has 142 rows against a 152-symbol teacher,
    it_IT 143 against 157. Nothing in training notices -- the missing ids are
    the rare ones -- but at inference the front end is free to emit any id in
    the map, and pypkg's embedding lookup then indexes past the end of the
    table and raises. The voice is one unusual sentence away from crashing.

    The teacher's phoneme_id_map is the authority on that range, so read it
    here and force both students to it. Sizing from the map instead of the data
    costs (max_id + 1 - observed) rows: about 10 embedding rows, ~1.6k
    parameters at the standard width. It is not a tuning choice.

    This is also the assumption that a non-eSpeak teacher breaks loudly rather
    than quietly: zh_CN-xiao_ya-medium's pinyin inventory is 85 symbols over 73
    distinct ids, half of eSpeak's 152, so anything that assumed the eSpeak
    range would size the embedding at twice what the teacher can address.
    """
    id_map = config.get("phoneme_id_map") or {}
    ids = [int(i) for seq in id_map.values() for i in seq]
    if not ids:
        die(f"{config_path} has no phoneme_id_map; the students cannot be sized "
            "against the teacher's phoneme inventory.")
    if min(ids) < 0:
        die(f"{config_path}: phoneme_id_map contains a negative id ({min(ids)}).")
    return max(ids) + 1


def probe_front_end(teacher: Path, config_path: Path, sample_text: str,
                    vocab_size: int) -> dict[str, Any]:
    """Drive the teacher's OWN front end on real corpus text, before any compute.

    espeak_can_speak() answers this question for eSpeak teachers only, and the
    driver had nothing at all for the other front ends -- a pinyin teacher with
    g2pW missing, or a Hebrew one without its niqqud model, got a log line
    saying "training works" and then failed an hour later in the first pack.
    This runs the exact call the pack builder makes (PiperVoice.phonemize ->
    phonemes_to_ids), so whatever the front end needs is either present now or
    the run stops here.

    It also confirms the ids land inside the embedding the students are about
    to be built with, which is the contract teacher_vocab_size() is asserting.
    """
    from piper import PiperVoice  # noqa: PLC0415

    sys.path.insert(0, str(ROOT / "tools"))
    from piper_frontend_workers import disable_g2p_dataloader_workers  # noqa: PLC0415

    voice_obj = PiperVoice.load(str(teacher), config_path=str(config_path))
    disable_g2p_dataloader_workers(voice_obj)
    chunks = voice_obj.phonemize(sample_text)
    phonemes = [p for chunk in chunks for p in chunk]
    if not phonemes:
        die(f"the teacher's front end produced no phonemes for the first corpus "
            f"row ({sample_text[:40]!r}). There is no voice to distil.")
    ids = voice_obj.phonemes_to_ids(phonemes)
    if not ids:
        die(f"the teacher's front end produced {len(phonemes)} phonemes but no "
            "phoneme ids; the phoneme_id_map does not match its own front end.")
    if max(ids) >= vocab_size:
        die(f"the front end emitted phoneme id {max(ids)} but the teacher's "
            f"phoneme_id_map only reaches {vocab_size - 1}. The students would "
            "be built with an embedding that cannot address it.")
    return {"phonemes": len(phonemes), "ids": len(ids), "max_id": max(ids),
            "sample": phonemes[:12]}


def preflight(config_path: Path) -> dict[str, Any]:
    """Read the teacher config and refuse anything this pipeline cannot distil."""
    config = json.loads(config_path.read_text())

    # Which front ends exist is a property of the INSTALLED piper, not of this
    # script: 1.4.2 had espeak/text/pinyin, 1.8.0 added hebrew (Nakdimon niqqud
    # + a rule-based IPA G2P), japanese and thai. Asking the library means a
    # teacher becomes trainable the moment piper is upgraded, with no edit here.
    phoneme_type = str(config.get("phoneme_type") or "espeak")
    try:
        from piper.config import PhonemeType  # noqa: PLC0415
        supported = {e.value for e in PhonemeType}
    except ImportError:
        die("piper-tts is not installed in this interpreter "
            "(pip install piper-tts); the pack builder needs it.")
    if phoneme_type not in supported:
        die(f"the installed piper cannot phonemize {phoneme_type!r} "
            f"(it supports {sorted(supported)}).\n"
            f"  piper-tts 1.8.0 added hebrew, japanese and thai, and bundles the "
            f"models they need.\n  Try:  pip install -U piper-tts")

    espeak_voice = ""
    if phoneme_type == "espeak":
        espeak_voice = ((config.get("espeak") or {}).get("voice") or "").strip()
        if not espeak_voice:
            die(f"{config_path} has no espeak.voice; cannot pick a front end.")
        ok, how = espeak_can_speak(espeak_voice)
        if not ok:
            die(f"eSpeak cannot speak {espeak_voice!r}: {how}. "
                f"Install piper-tts (it bundles espeak-ng-data) or a fuller "
                f"espeak-ng build, or pick another teacher.")
        log(f"  espeak via   {how}")
    else:
        # Training only needs piper to produce the ids. Inference is the catch:
        # the sanoTTS runtime phonemizes with espeak (or a bundled per-language
        # G2P), so a voice on any other front end trains fine and then has no
        # way to turn arbitrary text into ids at serve time until that G2P is
        # ported. Say it now rather than at packaging.
        log(f"  NOTE: front end is {phoneme_type!r}, not espeak. Training works -- "
            "piper produces the ids -- but the")
        log("        shipped voice will need that same G2P at inference; the "
            "sanoTTS runtime does not have it yet.")

    # The compact decoder is initialised by channel-slicing the teacher's
    # generator, and that path assumes a 256-channel conv_pre. Piper's "high"
    # voices are 512 and blow up five stages in with "channel score length 256
    # does not match tensor shape (512, ...)". Catch it before the packs render.
    onnx_path = config_path.with_suffix("")
    if onnx_path.suffix != ".onnx":
        onnx_path = Path(str(config_path)[: -len(".json")])
    if onnx_path.is_file():
        try:
            import onnx  # noqa: PLC0415

            graph = onnx.load(str(onnx_path), load_external_data=False).graph
            width = next((int(t.dims[0]) for t in graph.initializer
                          if t.name.endswith("dec.conv_pre.weight") and len(t.dims) == 3),
                         None)
            if width is not None and width != 256:
                die(f"teacher decoder is {width} channels wide at conv_pre; the "
                    "teacher-init channel slicing only handles 256.\n"
                    "  Piper's 'high' voices are 512 -- pick the 'medium' voice for "
                    "this language instead.\n  (This fails five stages in, after the "
                    "packs, if it is not caught here.)")
        except ImportError:
            pass

    vocab_size = teacher_vocab_size(config, config_path)

    speakers = int(config.get("num_speakers") or 1)
    if speakers != 1:
        die(f"teacher has {speakers} speakers. The students have no speaker "
            "conditioning, so a multi-speaker teacher would be distilled into a "
            "blend of all of them. Use a single-speaker voice.")

    sample_rate = int((config.get("audio") or {}).get("sample_rate") or 0)
    if sample_rate <= 0:
        die(f"{config_path} has no audio.sample_rate.")

    inference = config.get("inference") or {}
    return {
        "phoneme_type": phoneme_type,
        "espeak_voice": espeak_voice,
        "sample_rate": sample_rate,
        "language": ((config.get("language") or {}).get("code")
                     or config.get("dataset") or "und"),
        "phoneme_id_map_size": len(config.get("phoneme_id_map") or {}),
        "vocab_size": vocab_size,
        "length_scale": float(inference.get("length_scale") or 1.0),
    }


def resolve_signature_nodes(decoder_onnx: Path) -> list[str]:
    """Signature value names read off the decoder-cut graph.

    build_piper_vits_decoder_signature_pack.py maps ONNX value names to
    semantic labels through its own NODE_LABELS table, and falls back to a
    2026-06 Nepali export's names when none are given. Those do not exist in a
    modern export, so the fallback yields a pack missing every key the decoder
    trainer asks for -- and the failure surfaces two stages later. Intersecting
    the table with the graph's real node outputs makes the choice explicit, and
    a missing key stops the run here.
    """
    try:
        import onnx  # noqa: PLC0415
    except ImportError:
        die("the onnx package is required to resolve signature nodes "
            "(pip install onnx)")
        raise  # unreachable; keeps `onnx` bound for static analysis
    sys.path.insert(0, str(ROOT / "tools"))
    import importlib.util  # noqa: PLC0415

    spec = importlib.util.spec_from_file_location(
        "_sigpack", ROOT / "tools" / "build_piper_vits_decoder_signature_pack.py")
    if spec is None or spec.loader is None:
        die("could not load build_piper_vits_decoder_signature_pack.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    labels: dict[str, str] = dict(module.NODE_LABELS)

    graph = onnx.load(str(decoder_onnx), load_external_data=False).graph
    present = {out for node in graph.node for out in node.output}
    present |= {o.name for o in graph.output}
    # The latent is the cut graph's INPUT, not any node's output -- cutting the
    # decoder out of the teacher turns that value into the new graph's entry
    # point. Leaving inputs out of the search finds every signature except the
    # one the tool insists on ("exactly one latent node labelled 'latent'").
    present |= {i.name for i in graph.input}

    chosen = [name for name in labels if name in present]
    covered = {labels[name] for name in chosen}
    missing = [k for k in (*REQUIRED_SIGNATURE_KEYS, "latent") if k not in covered]
    if missing:
        die(f"{decoder_onnx.name}: no known value name maps to {missing}. "
            f"The graph exposes {len(present)} node outputs and matched "
            f"{len(chosen)} known names ({sorted(covered)}). This teacher was "
            "exported by a Piper version whose decoder value names are not in "
            "build_piper_vits_decoder_signature_pack.py's NODE_LABELS -- add "
            "them there rather than letting the tool fall back to its defaults.")
    log(f"  signature nodes resolved: {len(chosen)} nodes -> {sorted(covered)}")
    return chosen


# --------------------------------------------------------------------------
# corpus
# --------------------------------------------------------------------------

def ensure_corpus(text_path: Path | None, language: str, rows: int,
                  auto_text: bool, python_bin: str, dry_run: bool) -> Path:
    if text_path is not None:
        if not text_path.is_file():
            die(f"--text {text_path} does not exist")
        return text_path
    if not auto_text:
        die("no --text given. Pass a JSONL of {\"id\": ..., \"text\": ...} rows, "
            "or --auto-text to generate one for a supported language.")
    if language not in AUTO_TEXT_LANGUAGES:
        die(f"--auto-text cannot generate {language!r} "
            f"(tools/make_multilang_text_distill_corpus.py knows "
            f"{', '.join(AUTO_TEXT_LANGUAGES)}). Supply --text instead.")
    out_dir = ROOT / "data" / "textsets" / "multilang-distill-v1"
    generated = out_dir / f"{language}.train.jsonl"
    if not generated.is_file():
        run(["python3", "tools/make_multilang_text_distill_corpus.py",
             "--languages", language, "--train-rows", str(rows),
             "--out-dir", str(out_dir)], dry_run=dry_run, python_bin=python_bin)
    # The generated corpus is a template cross-product. It is fine for proving
    # the pipeline and it is what id/vi shipped on, but templated text inflates
    # every naturalness score (docs/distillation-recipe.md: ~1.3 SCOREQ), so a
    # score measured on it is not a score on real text.
    log("  NOTE: generated text is templated. Scores on it are optimistic; "
        "gate on varied real text before believing a number.")
    return generated


def count_rows(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


# --------------------------------------------------------------------------
# stage runner
# --------------------------------------------------------------------------

def raise_open_file_limit(target: int = 16384) -> None:
    """Lift the descriptor limit the stages inherit from this process.

    macOS ships a 256 soft limit. That is fine for a trainer and not fine for a
    pack: the Chinese front end used to leak descriptors per sentence and died
    at row 200 of 1781 with "Too many open files". That leak is fixed at source
    in tools/piper_frontend_workers.py, but 256 is a low ceiling for any stage
    that opens a file per row, and subprocesses inherit whatever this process
    has, so raise it once here rather than in each tool.
    """
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft >= target:
        return
    ceiling = target if hard == resource.RLIM_INFINITY else min(target, hard)
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (ceiling, hard))
    except (OSError, ValueError) as exc:
        # kern.maxfilesperproc can be below the target; not fatal, so say so
        # and carry on with whatever the kernel allowed.
        log(f"  WARNING: could not raise the open-file limit from {soft} "
            f"to {ceiling}: {exc}")
        return
    log(f"  file limit   {soft} -> {ceiling} descriptors (inherited by stages)")


def free_gb() -> float:
    """Best-effort free memory, so a long run can wait rather than swap."""
    try:
        if platform.system() == "Darwin":
            out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout
            # Page size is 16384 on Apple Silicon and 4096 on Intel; vm_stat
            # prints it, so read it rather than assuming (assuming 4096 on an
            # M-series box under-reports free memory by 4x and the gate then
            # waits forever on a machine that is 96% free).
            match = re.search(r"page size of (\d+) bytes", out)
            page = int(match.group(1)) if match else 4096
            counts = {}
            for line in out.splitlines():
                if ":" in line and line.startswith("Pages "):
                    name, _, value = line.partition(":")
                    try:
                        counts[name.strip()] = int(value.strip().rstrip("."))
                    except ValueError:
                        continue
            # Inactive pages are file-backed and reclaimed on demand, so macOS
            # counts them as available; leaving them out is what makes a healthy
            # machine look starved.
            available = (counts.get("Pages free", 0)
                         + counts.get("Pages speculative", 0)
                         + counts.get("Pages inactive", 0)
                         + counts.get("Pages purgeable", 0))
            return available * page / 1e9
        meminfo = Path("/proc/meminfo")
        if meminfo.is_file():
            for line in meminfo.read_text().splitlines():
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1e6
    except (OSError, ValueError, subprocess.CalledProcessError):
        return float("inf")
    return float("inf")


def wait_for_memory(min_gb: float) -> None:
    if min_gb <= 0:
        return
    while free_gb() < min_gb:
        log(f"  waiting for memory (free {free_gb():.1f} GB < {min_gb} GB)")
        time.sleep(120)


def run(command: list[str], *, dry_run: bool, python_bin: str,
        log_path: Path | None = None) -> None:
    if command and command[0] == "python3":
        command = [python_bin, *command[1:]]
    printable = " ".join(str(c) for c in command)
    if dry_run:
        print(f"  $ {printable}")
        return
    env = dict(os.environ)
    # Several trainers hit ops with no MPS kernel; without this they abort on
    # an Apple box instead of falling back to CPU for that op.
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    if log_path is None:
        subprocess.run(command, cwd=ROOT, env=env, check=True)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n$ {printable}\n")
        handle.flush()
        result = subprocess.run(command, cwd=ROOT, env=env,
                                stdout=handle, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0:
        die(f"stage failed (exit {result.returncode}). Log: {log_path}\n"
            f"  command: {printable}")


class Runner:
    def __init__(self, run_dir: Path, log_dir: Path, python_bin: str,
                 dry_run: bool, min_free_gb: float) -> None:
        self.run_dir = run_dir
        self.log_dir = log_dir
        self.python_bin = python_bin
        self.dry_run = dry_run
        self.min_free_gb = min_free_gb

    def stage(self, name: str, done: Path, command: list[str]) -> None:
        """Run one stage unless `done` already exists.

        `done` must be the file the tool writes LAST. Pointing it at a
        directory the tool creates early would let a crashed stage look
        finished on the next run.
        """
        if done.exists() and not self.dry_run:
            log(f"SKIP  {name} ({done.relative_to(ROOT) if done.is_relative_to(ROOT) else done} exists)")
            return
        log(f"START {name}")
        if not self.dry_run:
            wait_for_memory(self.min_free_gb)
        started = time.time()
        run(command, dry_run=self.dry_run, python_bin=self.python_bin,
            log_path=None if self.dry_run else self.log_dir / f"{name}.log")
        if self.dry_run:
            return
        if not done.exists():
            die(f"{name} reported success but did not write {done}")
        log(f"DONE  {name} ({time.time() - started:.0f}s)")


# --------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_argument_group("teacher")
    source.add_argument("--voice", help="Rhasspy voice key, e.g. vi_VN-vais1000-medium")
    source.add_argument("--teacher-onnx", type=Path,
                        help="Use a local teacher instead of downloading one")
    source.add_argument("--teacher-config", type=Path,
                        help="Local teacher .onnx.json (defaults to <onnx>.json)")
    source.add_argument("--teacher-dir", type=Path, default=ROOT / "models" / "teachers",
                        help="Where downloaded teachers are cached")

    text = parser.add_argument_group("text")
    text.add_argument("--text", type=Path, help="JSONL corpus with a 'text' field per row")
    text.add_argument("--auto-text", action="store_true",
                      help="Generate a templated corpus for a supported language")
    text.add_argument("--auto-text-rows", type=int, default=2048)

    parser.add_argument("--out-dir", type=Path,
                        help="Run directory (default artifacts/voices/<voice>)")
    parser.add_argument("--package-name", help="Exported package name (default <voice>)")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="shipped")
    parser.add_argument("--size", choices=sorted(SIZES), default="standard",
                        help="standard = the ~1.56M id/vi shape; tiny = under 500k")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--python", dest="python_bin", default=sys.executable,
                        help="Interpreter for the stage tools (needs torch/onnxruntime)")
    parser.add_argument("--duration-length-scale", type=float, default=1.0,
                        help="Inference-time duration scale baked into the package. "
                             "id/vi shipped at 1.16; tune by ear, no retraining needed.")
    parser.add_argument("--min-free-gb", type=float, default=0.0,
                        help="Wait for this much free RAM before each stage (0 = never)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print every command and exit without running them")
    parser.add_argument("--stop-after", help="Stop after this stage name")
    args = parser.parse_args()

    if not args.voice and not args.teacher_onnx:
        die("pass --voice <rhasspy key> or --teacher-onnx <path>")

    profile = PROFILES[args.profile]
    size = SIZES[args.size]
    if not args.dry_run:
        raise_open_file_limit()

    # ---- teacher -------------------------------------------------------
    log("preflight")
    if args.teacher_onnx:
        teacher = args.teacher_onnx
        config_path = args.teacher_config or Path(str(teacher) + ".json")
        if not teacher.is_file():
            die(f"--teacher-onnx {teacher} does not exist")
        if not config_path.is_file():
            die(f"teacher config {config_path} does not exist (pass --teacher-config)")
        voice = args.voice or teacher.stem
    else:
        voice = args.voice
        teacher, config_path = resolve_teacher(voice, args.teacher_dir)

    facts = preflight(config_path)
    log(f"  voice        {voice}")
    log(f"  front end    {facts['phoneme_type']}"
        + (f" ({facts['espeak_voice']}, supported)" if facts["espeak_voice"] else ""))
    log(f"  sample rate  {facts['sample_rate']} Hz")
    log(f"  phoneme ids  {facts['phoneme_id_map_size']} symbols -> "
        f"{facts['vocab_size']} embedding rows")

    run_dir = args.out_dir or (ROOT / "artifacts" / "voices" / voice)
    log_dir = run_dir / "logs"
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---- text ----------------------------------------------------------
    # The teacher config's language.code is authoritative; deriving it from the
    # voice key instead put the filename stem ("test") into the manifest.
    language = str(facts["language"])
    corpus = ensure_corpus(args.text, language, args.auto_text_rows,
                           args.auto_text, args.python_bin, args.dry_run)
    rows = count_rows(corpus) if corpus.is_file() else args.auto_text_rows
    acoustic_rows = rows - HELDOUT_ROWS
    if acoustic_rows < 256:
        die(f"{corpus} has {rows} rows; after holding out {HELDOUT_ROWS} for the "
            "smoke and eval packs that leaves too few to train on.")
    if acoustic_rows < 1900:
        log(f"  WARNING: {acoustic_rows} training rows. id/vi shipped on 1908 and "
            "the recipe calls the acoustic data-limited; expect a weaker voice.")
    log(f"  corpus       {corpus} ({rows} rows -> {acoustic_rows} training)")

    # The front end is checked against a row of the corpus it will actually be
    # asked to phonemize, not a hardcoded probe string, so a corpus in the
    # wrong script fails here too.
    if not args.dry_run:
        with corpus.open(encoding="utf-8") as handle:
            first = next((json.loads(line) for line in handle if line.strip()), {})
        sample_text = str(first.get("text") or first.get("target_text") or "").strip()
        if not sample_text:
            die(f"{corpus}: the first row has no 'text' field.")
        probe = probe_front_end(teacher, config_path, sample_text, facts["vocab_size"])
        log(f"  front end ok {probe['phonemes']} phonemes -> {probe['ids']} ids "
            f"(max id {probe['max_id']} < {facts['vocab_size']}): {probe['sample']}")

    runner = Runner(run_dir, log_dir, args.python_bin, args.dry_run, args.min_free_gb)

    smoke = run_dir / "smoke12"
    evalp = run_dir / "eval128"
    train_ac = run_dir / "train-acoustic"
    train_dec = run_dir / "train512-decoder"

    pack_common = ["--allow-text-only-source", "--noise-scale", "0",
                   "--length-scale", "1", "--noise-w", "0"]

    def pack(out: Path, max_rows: int, skip: int, mode: str) -> list[str]:
        return ["python3", "tools/build_piper_vits_roota_probe_pack.py",
                "--model", str(teacher), "--config", str(config_path),
                "--source-jsonl", str(corpus), "--out-dir", str(out),
                "--max-rows", str(max_rows), "--skip-rows", str(skip),
                "--tensor-mode", mode, *pack_common]

    # ---- S1: the four teacher packs ------------------------------------
    # The teacher is the data source: each pack stores its phoneme ids, the
    # durations it chose, its 192-dim latent and its audio. Sampling is turned
    # off (noise 0, length 1) so the targets are deterministic.
    runner.stage("s1a_smoke", smoke / "rows.json", pack(smoke, SMOKE_ROWS, 0, "decoder"))
    runner.stage("s1b_eval", evalp / "rows.json", pack(evalp, EVAL_ROWS, SMOKE_ROWS, "decoder"))
    runner.stage("s1c_acoustic", train_ac / "rows.json",
                 pack(train_ac, acoustic_rows, HELDOUT_ROWS, "acoustic"))
    runner.stage("s1d_decoder", train_dec / "rows.json",
                 pack(train_dec, min(DECODER_TRAIN_ROWS, acoustic_rows), HELDOUT_ROWS, "decoder"))
    if args.stop_after == "s1":
        return 0

    # ---- S2: cut the teacher's decoder out as an ONNX oracle ------------
    cut_dir = run_dir / "decoder-cut"
    cut_onnx = cut_dir / f"{teacher.stem}-decoder-from-generator-input.onnx"
    runner.stage("s2_decoder_cut", cut_onnx,
                 ["python3", "tools/extract_piper_vits_decoder_cut.py",
                  "--model", str(teacher), "--pack-dir", str(smoke),
                  "--out-dir", str(cut_dir)])

    # ---- S3: duration student ------------------------------------------
    duration_pt = run_dir / "duration" / "duration-student.pt"
    runner.stage("s3_duration", duration_pt,
                 ["python3", "tools/train_roota_piper_duration_student.py",
                  "--pack-dir", str(train_ac), "--eval-pack-dir", str(evalp),
                  "--out-dir", str(run_dir / "duration"),
                  "--hidden", str(size.duration_hidden), "--depth", "3",
                  "--kernel-size", "5",
                  # Sized from the teacher's phoneme_id_map, not from the ids the
                  # corpus happens to contain. See teacher_vocab_size().
                  "--duration-vocab-size", str(facts["vocab_size"]),
                  # Unreachable while the vocabulary covers the whole map; id 0
                  # is Piper's pad in every inventory, so it is a valid fallback
                  # for any teacher, which the default (59) is not.
                  "--duration-oov-id", "0",
                  "--steps", str(profile.duration_steps), "--device", args.device])

    # ---- S4: acoustic student, de-smoothed ------------------------------
    # A plain regressor over-smooths the latent; the latent-adversarial term is
    # training-only and never ships, and is worth roughly +1.3 SCOREQ.
    acoustic_pt = run_dir / "acoustic" / "latent-student.pt"
    runner.stage("s4_acoustic", acoustic_pt,
                 ["python3", "tools/train_roota_piper_latent_student.py",
                  "--pack-dir", str(train_ac), "--eval-pack-dir", str(evalp),
                  "--out-dir", str(run_dir / "acoustic"),
                  "--architecture", "token_context",
                  "--hidden", str(size.acoustic_hidden), "--token-depth", "3",
                  "--depth", str(size.acoustic_depth), "--kernel-size", "5",
                  # Same contract as the duration student above.
                  "--vocab-size", str(facts["vocab_size"]),
                  "--norm-l1-weight", "0.25", "--delta-l1-weight", "0.10",
                  "--channel-stat-weight", "0.05",
                  "--latent-adv-weight", "0.1", "--latent-adv-start-step", "1500",
                  "--decoder", str(cut_onnx),
                  "--steps", str(profile.acoustic_steps), "--device", args.device])

    # ---- S5: teacher parity, which exports the decoder init -------------
    # This is where the compact decoder's initial weights come from: the
    # teacher's own generator, channel-sliced. From-scratch decoders at this
    # size are a dead class.
    parity_pt = run_dir / "parity" / "piper-decoder-teacher.pt"
    runner.stage("s5_parity", parity_pt,
                 ["python3", "tools/verify_piper_decoder_torch_parity.py",
                  "--decoder", str(cut_onnx), "--pack-dir", str(evalp),
                  "--out-dir", str(run_dir / "parity"), "--rows", "32",
                  "--export-checkpoint"])

    # ---- S6: activation signatures, nodes read off the graph ------------
    sig_dir = run_dir / "signatures"
    if not (sig_dir / "summary.json").exists():
        if args.dry_run and not cut_onnx.exists():
            nodes = ["<resolved from the decoder-cut graph at run time>"]
        else:
            nodes = resolve_signature_nodes(cut_onnx)
    else:
        nodes = []
    node_args: list[str] = []
    for name in nodes:
        node_args += ["--node", name]
    runner.stage("s6_signatures", sig_dir / "summary.json",
                 ["python3", "tools/build_piper_vits_decoder_signature_pack.py",
                  "--model", str(cut_onnx), "--pack-dir", str(train_dec),
                  "--out-dir", str(sig_dir), "--feed-latent-from-pack",
                  "--dtype", "float16", *node_args])

    # ---- S7: compact decoder, initialised from the teacher --------------
    decoder_common = [
        "--variant", "piperlite", "--channels", size.decoder_channels,
        "--activation", "leaky_relu",
        "--signature-hint-weight", "0.05", "--signature-temporal-weight", "0.4",
        "--feature-hint-weight", "0.05",
        "--quiet-ceiling-weight", "0.010", "--quiet-ceiling-margin-db", "0.5",
        "--adv-weight", "0.075", "--adv-feature-weight", "0.75",
        "--render-rows", "0", "--device", args.device,
    ]
    a91 = run_dir / "decoder" / "decoder-student.pt"
    runner.stage("s7_decoder", a91,
                 ["python3", "tools/train_roota_piper_decoder_student.py",
                  "--pack-dir", str(train_dec), "--eval-pack-dir", str(evalp),
                  "--teacher-decoder", str(cut_onnx),
                  "--teacher-init-checkpoint", str(parity_pt),
                  "--teacher-init-method", "importance",
                  "--acoustic-checkpoint", str(acoustic_pt),
                  "--signature-pack-dir", str(sig_dir),
                  "--out-dir", str(run_dir / "decoder"),
                  "--adv-start-step", "1500",
                  "--steps", str(profile.decoder_steps), "--lr", "2e-4",
                  *decoder_common])

    # ---- S8: z-mix, so the decoder survives the acoustic's error ---------
    # Stage 7 only ever saw the teacher's latents. Mixing in the acoustic
    # student's predicted latents is what closes the interface; skipping it is
    # the classic failure where every loss looks fine and the voice is brittle.
    a92 = run_dir / "decoder-zmix" / "decoder-student.pt"
    runner.stage("s8_zmix", a92,
                 ["python3", "tools/train_roota_piper_decoder_student.py",
                  "--pack-dir", str(train_dec), "--eval-pack-dir", str(evalp),
                  "--teacher-decoder", str(cut_onnx),
                  "--init-decoder-checkpoint", str(a91),
                  "--acoustic-checkpoint", str(acoustic_pt),
                  "--acoustic-latent-mix-prob", "0.5",
                  "--signature-pack-dir", str(sig_dir),
                  "--out-dir", str(run_dir / "decoder-zmix"),
                  "--adv-start-step", "1000",
                  "--steps", str(profile.zmix_steps), "--lr", "1e-4",
                  *decoder_common])

    # ---- S9: joint finetune; emits the two shipped checkpoints ----------
    joint_dir = run_dir / "joint"
    joint_decoder = joint_dir / "decoder-student.pt"
    joint_acoustic = joint_dir / "latent-student.pt"
    runner.stage("s9_joint", joint_decoder,
                 ["python3", "tools/train_roota_joint_z_finetune.py",
                  "--pack-dir", str(train_dec), "--teacher-decoder", str(cut_onnx),
                  "--acoustic-checkpoint", str(acoustic_pt),
                  "--decoder-checkpoint", str(a92),
                  "--out-dir", str(joint_dir),
                  "--z-anchor-weight", "0.5",
                  "--steps", str(profile.joint_steps), "--device", args.device])
    if args.stop_after == "s9":
        return 0

    # ---- S10: five-lane gate -------------------------------------------
    # Best-effort: a failed gate is a result to read, not a reason to discard
    # the checkpoints, so this does not abort the run.
    feedback = run_dir / "feedback"
    if not (feedback / "scoreq-summary.json").exists() and not args.dry_run:
        log("START s10_gate")
        try:
            run(["python3", "tools/run_roota_feedback_loop.py",
                 "--eval-pack", str(evalp),
                 "--acoustic-checkpoint", str(joint_acoustic),
                 "--duration-checkpoint", str(duration_pt),
                 "--piper-model", str(teacher), "--piper-config", str(config_path),
                 "--decoder-report", str(run_dir / "decoder" / "train-report.json"),
                 "--candidate-label", f"{voice}-joint",
                 "--out-dir", str(feedback), "--rows", "64",
                 "--param-budget", "1500000",
                 "--force-render", "--force-score", "--device", "cpu"],
                dry_run=False, python_bin=args.python_bin,
                log_path=log_dir / "s10_gate.log")
            log("DONE  s10_gate")
        except SystemExit:
            log("WARN  s10_gate failed; run it by hand. Checkpoints are unaffected.")
    elif args.dry_run:
        print("  $ (s10 gate: tools/run_roota_feedback_loop.py …)")

    # ---- S11: export the package ----------------------------------------
    package_name = args.package_name or voice
    package_dir = run_dir / "package"
    runner.stage("s11_export", package_dir / "manifest.json",
                 ["python3", "tools/export_roota_self_contained_package.py",
                  "--package-name", package_name,
                  "--language", str(facts["language"]),
                  "--voice", voice,
                  "--acoustic-checkpoint", str(joint_acoustic),
                  "--duration-checkpoint", str(duration_pt),
                  "--decoder-checkpoint", str(joint_decoder),
                  "--piper-config", str(config_path),
                  "--sample-rate", str(facts["sample_rate"]),
                  "--duration-length-scale", str(args.duration_length_scale),
                  "--out-dir", str(package_dir)])

    if args.dry_run:
        print("\n(dry run — nothing was executed)")
        return 0

    log(f"COMPLETE  package at {package_dir}")
    manifest_path = package_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        log(f"  parameters {manifest.get('total_parameters'):,}")
        log(f"  weights    {manifest.get('weights_size_bytes'):,} bytes")
    log("  Listen before believing any number: "
        "tools/serve_roota_arbitrary_tts_dashboard.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
