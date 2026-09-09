#!/usr/bin/env python3
"""Measure nano_lex_g2p.c against pypkg/sanotts/nano_g2p.py.

    python3 esphome/g2p/lex/run_parity.py [--json out.json]

Three numbers, and they answer different questions:

  port correctness   C vs `nano_g2p.py` with its neural fallback DISABLED.
                     That is the same algorithm the C implements, so anything
                     below 100% is a bug in the C, not a property of English.
  fallback cost      the same Python with the neural OOV model ENABLED, vs the
                     C. The gap is exactly what dropping the 2.8 MB model costs.
  phonetic fidelity  C vs the recorded real-misaki ids in
                     pypkg/tests/data/nano-g2p-misaki-reference.jsonl -- the
                     ids every nano training and eval pack was built from.
                     Bounded above by what the Python front end itself reaches
                     (82.1% exact / 0.30% TER, docs/nano-espeak-free-frontend.md).

The oracle is `nano_g2p.py` itself, imported and called; nothing about it is
reimplemented here. numpy is imported (nano_g2p imports the OOV module), torch
is not, and no model runs unless --with-fallback is on, which it is by default
only for the fallback-cost row.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "pypkg"))

from sanotts import nano_g2p as NG          # noqa: E402
from sanotts.nano_frontend import FrontendError  # noqa: E402

BINARY = HERE / "test_nano_lex_g2p"
HA_CORPUS = HERE / "ha_corpus.txt"
EDGE_CORPUS = HERE / "edge_corpus.txt"
MISAKI_REFERENCE = ROOT / "pypkg" / "tests" / "data" / "nano-g2p-misaki-reference.jsonl"

# The Python cap is a policy, not a buffer size; raise it so the comparison is
# about phonemes rather than about who truncates first. The C is given a cap
# far above anything these corpora reach, for the same reason.
NO_CAP = 1 << 20


class ParityError(RuntimeError):
    """Anything that makes a measurement meaningless. Never swallowed."""


# --------------------------------------------------------------------------
# Corpora
# --------------------------------------------------------------------------


def load_plain_corpus(path: Path, prefix: str) -> list[dict]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ParityError(f"cannot read {path}: {exc}") from exc
    rows = []
    for lineno, line in enumerate(raw.splitlines(), start=1):
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        rows.append({"row_id": f"{prefix}-{lineno:03d}", "text": text})
    if not rows:
        raise ParityError(f"{path} has no sentences")
    return rows


def load_misaki_reference() -> list[dict]:
    try:
        raw = MISAKI_REFERENCE.read_text(encoding="utf-8")
    except OSError as exc:
        raise ParityError(f"cannot read {MISAKI_REFERENCE}: {exc}") from exc
    rows = []
    for lineno, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ParityError(f"{MISAKI_REFERENCE}:{lineno} is not valid JSON: {exc}") from exc
        for field in ("row_id", "text", "ids"):
            if field not in row:
                raise ParityError(f"{MISAKI_REFERENCE}:{lineno} has no {field!r}")
        rows.append(row)
    if not rows:
        raise ParityError(f"{MISAKI_REFERENCE} has no rows")
    return rows


# --------------------------------------------------------------------------
# Oracles
# --------------------------------------------------------------------------


def build_oracle(*, with_fallback: bool) -> NG.NanoG2P:
    """`nano_g2p.NanoG2P`, with the neural fallback on or off.

    Off is not `NanoG2P(fallback=None)`: that constructor argument means
    "use the default", so the attribute is cleared afterwards instead. The C
    implements exactly the fallback-is-None behaviour.
    """
    engine = NG.NanoG2P()
    if not with_fallback:
        engine.fallback = None
    return engine


def oracle_ids(engine: NG.NanoG2P, text: str) -> dict:
    try:
        ids, dropped = NG.phonemize(text, max_tokens=NO_CAP, g2p=engine)
    except FrontendError as exc:
        return {"ok": False, "error": f"FrontendError({exc.kind}): {exc}"}
    except NG.NumberError as exc:
        return {"ok": False, "error": f"NumberError: {exc}"}
    try:
        phonemes = NG.phonemize_to_string(text, g2p=engine)
    except (FrontendError, NG.NumberError) as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"ok": True, "ids": list(ids), "phonemes": phonemes, "dropped": dropped}


# --------------------------------------------------------------------------
# The C under test
# --------------------------------------------------------------------------


def run_candidate(texts: list[str]) -> list[dict]:
    if not BINARY.is_file():
        raise ParityError(f"{BINARY} is not built; run `make` in {HERE}")
    for text in texts:
        if "\n" in text or "\r" in text:
            raise ParityError(f"input has a newline, which the line protocol cannot carry: "
                              f"{text!r}")
    payload = ("\n".join(texts) + "\n").encode("utf-8")
    try:
        proc = subprocess.run([str(BINARY)], input=payload, capture_output=True, check=False)
    except OSError as exc:
        raise ParityError(f"could not run {BINARY}: {exc}") from exc
    if proc.returncode != 0:
        raise ParityError(f"{BINARY} exited {proc.returncode}: "
                          f"{proc.stderr.decode('utf-8', 'replace')}")
    lines = proc.stdout.decode("utf-8").splitlines()
    if len(lines) != len(texts):
        raise ParityError(f"{BINARY} returned {len(lines)} lines for {len(texts)} inputs")

    out = []
    for text, line in zip(texts, lines):
        parts = line.split("\t")
        if parts and parts[0] == "!ERROR":
            out.append({"ok": False, "error": "\t".join(parts[1:])})
            continue
        if len(parts) != 3:
            raise ParityError(f"malformed harness line for {text!r}: {line!r}")
        try:
            ids = [int(v) for v in parts[0].split(",") if v]
            stats = [int(v) for v in parts[2].split(",")]
        except ValueError as exc:
            raise ParityError(f"malformed numbers for {text!r}: {line!r} ({exc})") from exc
        if len(stats) != 5:
            raise ParityError(f"expected 5 stat fields for {text!r}, got {len(stats)}")
        out.append({
            "ok": True,
            "ids": ids,
            "phonemes": parts[1],
            "lexicon_words": stats[0],
            "oov_words": stats[1],
            "chunks": stats[2],
            "dropped": stats[3],
            "arena_peak": stats[4],
        })
    return out


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def levenshtein(ref: list[int], cand: list[int]) -> int:
    if not ref:
        return len(cand)
    previous = list(range(len(cand) + 1))
    for i, r in enumerate(ref, start=1):
        current = [i]
        for j, c in enumerate(cand, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (0 if r == c else 1)))
        previous = current
    return previous[-1]


def score(reference: list[dict], candidate: list[dict], rows: list[dict],
          label: str) -> dict:
    if len(reference) != len(candidate) or len(reference) != len(rows):
        raise ParityError("reference, candidate and rows disagree on length")
    exact = 0
    compared = 0
    distance = 0
    ref_tokens = 0
    ref_failed = 0
    cand_failed = 0
    both_failed = 0
    mismatches = []
    for row, ref, cand in zip(rows, reference, candidate):
        if not ref["ok"]:
            if not cand["ok"]:
                # Both refuse the row. That is agreement, not a miss: the C is
                # meant to fail exactly where the Python does.
                both_failed += 1
                continue
            ref_failed += 1
            mismatches.append({"row_id": row["row_id"], "text": row["text"],
                               "why": "oracle failed but C succeeded",
                               "oracle_error": ref["error"]})
            continue
        if not cand["ok"]:
            cand_failed += 1
            mismatches.append({"row_id": row["row_id"], "text": row["text"],
                               "why": "C failed but oracle succeeded",
                               "c_error": cand["error"]})
            continue
        compared += 1
        ref_tokens += len(ref["ids"])
        if ref["ids"] == cand["ids"]:
            exact += 1
        else:
            distance += levenshtein(ref["ids"], cand["ids"])
            if len(mismatches) < 25:
                mismatches.append({
                    "row_id": row["row_id"],
                    "text": row["text"],
                    "oracle_phonemes": ref.get("phonemes"),
                    "c_phonemes": cand.get("phonemes"),
                })
    return {
        "label": label,
        "rows": len(rows),
        "compared": compared,
        "oracle_only_errors": ref_failed,
        "c_only_errors": cand_failed,
        "both_refused": both_failed,
        "exact_match": exact,
        "exact_match_rate": (exact / compared) if compared else 0.0,
        "reference_tokens": ref_tokens,
        "edit_distance": distance,
        "token_error_rate": (distance / ref_tokens) if ref_tokens else 0.0,
        "mismatches": mismatches,
    }


def print_report(report: dict) -> None:
    print(f"\n{report['label']}")
    print(f"  rows              {report['rows']}")
    print(f"  compared          {report['compared']}"
          f"   (both refused {report['both_refused']},"
          f" oracle-only errors {report['oracle_only_errors']},"
          f" C-only errors {report['c_only_errors']})")
    print(f"  exact sequence    {report['exact_match']}/{report['compared']}"
          f"   = {100.0 * report['exact_match_rate']:.2f}%")
    print(f"  token error rate  {report['edit_distance']}/{report['reference_tokens']}"
          f"   = {100.0 * report['token_error_rate']:.3f}%")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", type=Path, help="write the full report here")
    parser.add_argument("--limit", type=int, default=0,
                        help="use only the first N rows of each corpus")
    args = parser.parse_args(argv)

    try:
        corpora = {
            "gutenberg-330 (repo eval set)": load_misaki_reference(),
            "home-assistant (written for this port)": load_plain_corpus(HA_CORPUS, "ha"),
            "edge cases (written for this port)": load_plain_corpus(EDGE_CORPUS, "edge"),
        }
    except ParityError as exc:
        print(f"run_parity: {exc}", file=sys.stderr)
        return 2

    if args.limit:
        corpora = {k: v[:args.limit] for k, v in corpora.items()}

    no_fallback = build_oracle(with_fallback=False)
    with_fallback = build_oracle(with_fallback=True)

    out: dict = {"corpora": {}}
    try:
        for name, rows in corpora.items():
            texts = [r["text"] for r in rows]
            cand = run_candidate(texts)
            plain = [oracle_ids(no_fallback, t) for t in texts]
            neural = [oracle_ids(with_fallback, t) for t in texts]

            reports = [
                score(plain, cand, rows,
                      f"[{name}] C vs nano_g2p.py, fallback DISABLED  (port correctness)"),
                score(neural, cand, rows,
                      f"[{name}] C vs nano_g2p.py, fallback ENABLED   (cost of dropping the "
                      f"OOV model)"),
            ]
            if all("ids" in r for r in rows):
                misaki = [{"ok": True, "ids": list(r["ids"])} for r in rows]
                reports.append(score(misaki, cand, rows,
                                     f"[{name}] C vs recorded real misaki  (phonetic fidelity)"))

            oov_words = sum(c.get("oov_words", 0) for c in cand if c["ok"])
            lex_words = sum(c.get("lexicon_words", 0) for c in cand if c["ok"])
            arena_peak = max((c.get("arena_peak", 0) for c in cand if c["ok"]), default=0)
            chunks = sum(c.get("chunks", 0) for c in cand if c["ok"])

            print(f"\n=== {name}: {len(rows)} sentences ===")
            for report in reports:
                print_report(report)
            print(f"\n  OOV word groups   {oov_words}/{lex_words}"
                  f"   = {100.0 * oov_words / lex_words if lex_words else 0.0:.2f}%")
            print(f"  chunks emitted    {chunks}")
            print(f"  arena high water  {arena_peak} bytes of {6144}")

            out["corpora"][name] = {
                "reports": reports,
                "oov_words": oov_words,
                "lexicon_words": lex_words,
                "oov_rate": (oov_words / lex_words) if lex_words else 0.0,
                "chunks": chunks,
                "arena_peak_bytes": arena_peak,
            }
    except ParityError as exc:
        print(f"run_parity: {exc}", file=sys.stderr)
        return 2

    if args.json:
        try:
            args.json.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                                 encoding="utf-8")
        except OSError as exc:
            print(f"run_parity: cannot write {args.json}: {exc}", file=sys.stderr)
            return 2
        print(f"\nwrote {args.json}")

    worst = min((r["exact_match_rate"]
                 for c in out["corpora"].values()
                 for r in c["reports"] if "fallback DISABLED" in r["label"]),
                default=0.0)
    return 0 if worst >= 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
