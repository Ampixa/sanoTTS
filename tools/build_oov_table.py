"""Build g2p_data/oov_table_en_us.npz.xz: the OOV model's own output for common words.

The English nano front end sends every word its dictionary lacks to a small
BART (sanotts/nano_g2p_oov.py). Decoding one word costs about a millisecond
on one CPU thread, so a sentence with a few names spends most of its
time-to-first-audio there. This tool runs that same model, single-word path
(`OOVPhonemizer.__call__`), over a fixed general word list and stores the
results, so the runtime can look them up instead of decoding them. A table
entry is, by construction, what the model returns for that word; the table
only changes how fast the answer arrives.

Word list (fixed rule, decided before measuring coverage on any eval set):
wordfreq 3.1.1, English, wordlist="large" (321,180 entries; data under
CC-BY-SA 4.0, see g2p_data/NOTICE.md). Kept: entries made only of letters with
inner apostrophes, hyphens or periods that fit the model's position table.
Each kept word is stored in lowercase and capitalised form, because the model
is case-sensitive and names reach it capitalised. Evaluation texts are never
an input.

Dictionary words are kept too. A first build left out words whose lowercase
form is a us_gold/us_silver key, on the theory that the fallback never sees
them; profiling showed it does (the back-off hands it "by", "Sea" or "C" in
some contexts), so the filter was dropped. Whatever reaches the fallback, a
table entry is still the model's own answer.

Usage (from the repo root, with wordfreq==3.1.1 installed):
    python tools/build_oov_table.py build entries.json.xz     # decode (~15 min, one thread)
    python tools/build_oov_table.py pack entries.json.xz pypkg/sanotts/g2p_data/oov_table_en_us.npz.xz

`build` writes the sorted entries with their provenance as xz-compressed JSON
(kept as the reviewable record); `pack` turns that into the arrays the
runtime reads (see nano_g2p_oov._WordTable).
"""

from __future__ import annotations

import hashlib
import json
import lzma
import platform
import re
import sys
import time
from importlib import metadata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pypkg"))

from sanotts import nano_g2p_oov  # noqa: E402
from sanotts.nano_g2p import _fold_to_model_alphabet  # noqa: E402

WORDFREQ_VERSION = "3.1.1"
WORD_RE = re.compile(r"[a-z]+(?:['\-.][a-z]+)*")


def build(out_path: str) -> None:
    import wordfreq  # noqa: PLC0415  (build-time dependency only)

    got = metadata.version("wordfreq")
    if got != WORDFREQ_VERSION:
        raise SystemExit(f"need wordfreq=={WORDFREQ_VERSION}, found {got}")
    words = wordfreq.get_frequency_dict("en", wordlist="large")

    from threadpoolctl import threadpool_info, threadpool_limits  # noqa: PLC0415

    # One BLAS thread, as at runtime: a multi-threaded gemm may round
    # differently, and the table must be what a one-thread decode returns.
    threadpool_limits(limits=1)
    if any(p.get("num_threads") != 1 for p in threadpool_info()):
        raise SystemExit(f"could not cap BLAS at one thread: {threadpool_info()}")
    model = nano_g2p_oov.OOVPhonemizer(nano_g2p_oov.TABLE_MODEL_PATH, table_path=None)
    model.warm()
    limit = model.max_positions - 2  # bos + eos
    keys: list[str] = []
    for w in words:
        if not WORD_RE.fullmatch(w) or len(w) > limit:
            continue
        for form in (w, w.capitalize()):
            folded = _fold_to_model_alphabet(form)
            if folded == form:
                keys.append(form)
    keys = list(dict.fromkeys(keys))

    entries: dict[str, str] = {}
    t0 = time.perf_counter()
    for i, key in enumerate(keys):
        # A fresh memo per word is not needed: __call__ is a pure function of
        # the word and the memo only returns earlier results of that function.
        entries[key] = model(key)
        if i % 20000 == 0:
            print(f"{i}/{len(keys)} {time.perf_counter() - t0:.0f}s", flush=True)

    model_sha = hashlib.sha256(Path(nano_g2p_oov.TABLE_MODEL_PATH).read_bytes()).hexdigest()
    payload = {
        "format": 1,
        "model_sha256": model_sha,
        "source": f"wordfreq {WORDFREQ_VERSION} en large; tools/build_oov_table.py",
        "built_on": f"{platform.machine()} {platform.system()}, numpy {metadata.version('numpy')}",
        "entries": dict(sorted(entries.items())),
    }
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    Path(out_path).write_bytes(lzma.compress(raw, preset=9 | lzma.PRESET_EXTREME))
    print(f"wrote {out_path}: {len(entries)} entries, {len(raw)} bytes raw, "
          f"{Path(out_path).stat().st_size} compressed, {time.perf_counter() - t0:.0f}s")


def pack(entries_path: str, out_path: str) -> None:
    payload = json.loads(lzma.decompress(Path(entries_path).read_bytes()))
    model_sha = hashlib.sha256(Path(nano_g2p_oov.TABLE_MODEL_PATH).read_bytes()).hexdigest()
    if payload["model_sha256"] != model_sha:
        raise SystemExit(f"{entries_path} was built for model {payload['model_sha256']}, not {model_sha}")
    meta = {k: payload[k] for k in ("source", "built_on")}
    meta["entries_sha256"] = hashlib.sha256(Path(entries_path).read_bytes()).hexdigest()
    nano_g2p_oov.write_table(out_path, payload["entries"], model_sha, meta)
    print(f"wrote {out_path}: {len(payload['entries'])} rows, {Path(out_path).stat().st_size} bytes")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "build":
        build(sys.argv[2])
    elif len(sys.argv) == 4 and sys.argv[1] == "pack":
        pack(sys.argv[2], sys.argv[3])
    else:
        raise SystemExit(__doc__)
