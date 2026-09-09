#!/usr/bin/env python3
"""Measure the C front end against the repo's recorded reference id sequences.

Runs `nano_ids_host` (the ESP32 component's own nano_g2p.c, compiled for the
host) over an evaluation set and reports, per system:

  * exact id-sequence match rate,
  * token error rate = Levenshtein(candidate, reference) / len(reference),
  * the substitutions/insertions/deletions that dominate the residual.

Two references, because they answer different questions:

  shipped-espeak.jsonl   what pypkg/sanotts/nano_frontend.py produces. Agreement
                         here is a PORT-CORRECTNESS number: anything below 100%
                         is a bug in the C, not a property of espeak.
  misaki-reference.jsonl what real misaki/Kokoro produces, i.e. the ids every
                         nano training and eval pack was built from. Agreement
                         here is the PHONETIC-FIDELITY number, and it is bounded
                         above by what the Python front end itself achieves.

Standard library only; no model is loaded and no audio is rendered.
"""

from __future__ import annotations

import argparse
import collections
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


class ParityError(RuntimeError):
    pass


def levenshtein_ops(ref: list[int], cand: list[int]) -> tuple[int, list[tuple[str, int, int]]]:
    """Edit distance plus a backtrace of the operations, ref -> cand."""
    n, m = len(ref), len(cand)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == cand[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    ops: list[tuple[str, int, int]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (0 if ref[i - 1] == cand[j - 1] else 1):
            if ref[i - 1] != cand[j - 1]:
                ops.append(("sub", ref[i - 1], cand[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("del", ref[i - 1], -1))
            i -= 1
        else:
            ops.append(("ins", -1, cand[j - 1]))
            j -= 1
    return dp[n][m], ops


def load_jsonl(path: Path) -> list[dict]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ParityError(f"cannot read {path}: {exc}") from exc
    rows = []
    for lineno, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ParityError(f"{path}:{lineno} is not valid JSON: {exc}") from exc
    if not rows:
        raise ParityError(f"{path} has no rows")
    return rows


def run_candidate(binary: Path, data_dir: Path, texts: list[str]) -> list[dict]:
    """Drive the C harness. Any row it rejects is reported, never dropped."""
    if not binary.is_file():
        raise ParityError(f"{binary} not built; run esphome/g2p/host/build_host.sh first")
    for text in texts:
        if "\n" in text or "\r" in text:
            raise ParityError(f"input contains a newline, which the TSV protocol cannot carry: {text!r}")
    payload = "\n".join(texts) + "\n"
    try:
        proc = subprocess.run(
            [str(binary), str(data_dir)],
            input=payload.encode("utf-8"),
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        raise ParityError(f"could not run {binary}: {exc}") from exc
    if proc.returncode != 0:
        raise ParityError(
            f"{binary} exited {proc.returncode}: {proc.stderr.decode('utf-8', 'replace')}"
        )
    out_lines = proc.stdout.decode("utf-8").splitlines()
    if len(out_lines) != len(texts):
        raise ParityError(f"harness returned {len(out_lines)} lines for {len(texts)} inputs")

    results = []
    for text, line in zip(texts, out_lines):
        parts = line.split("\t")
        if parts[0] == "!ERROR":
            results.append({"text": text, "ok": False, "error": "\t".join(parts[1:])})
            continue
        if len(parts) != 2:
            raise ParityError(f"malformed harness line for {text!r}: {line!r}")
        ps, ids_csv = parts
        try:
            ids = [int(x) for x in ids_csv.split(",") if x != ""]
        except ValueError as exc:
            raise ParityError(f"malformed id list for {text!r}: {ids_csv!r}") from exc
        results.append({"text": text, "ok": True, "ps": ps, "ids": ids})
    return results


def score(name: str, reference: list[dict], candidate: list[dict], vocab_inv: dict[int, str]) -> dict:
    exact = 0
    ps_exact = 0
    total_ref = 0
    total_edits = 0
    confusions: collections.Counter = collections.Counter()
    failures = []
    per_row = []

    for ref, cand in zip(reference, candidate):
        if not cand["ok"]:
            failures.append((ref.get("row_id", ref["text"][:40]), cand["error"]))
            continue
        ref_ids = list(ref["ids"])
        cand_ids = cand["ids"]
        dist, ops = levenshtein_ops(ref_ids, cand_ids)
        total_ref += len(ref_ids)
        total_edits += dist
        if dist == 0:
            exact += 1
        if "ps" in ref and ref["ps"] == cand["ps"]:
            ps_exact += 1
        for kind, a, b in ops:
            sym_a = vocab_inv.get(a, "<ins>") if kind != "ins" else "<ins>"
            sym_b = vocab_inv.get(b, "<del>") if kind != "del" else "<del>"
            confusions[(sym_a, sym_b)] += 1
        per_row.append(
            {
                "row_id": ref.get("row_id", ""),
                "ter": round(dist / max(1, len(ref_ids)), 4),
                "ref_ids": len(ref_ids),
                "cand_ids": len(cand_ids),
            }
        )

    scored = len(per_row)
    return {
        "system": name,
        "rows": len(reference),
        "rows_scored": scored,
        "rows_failed": len(failures),
        "failures": failures[:10],
        "exact_sequence_match": round(exact / scored, 6) if scored else 0.0,
        "phoneme_string_exact": round(ps_exact / scored, 6) if scored else None,
        "token_error_rate": round(total_edits / total_ref, 6) if total_ref else None,
        "reference_tokens": total_ref,
        "edit_distance_total": total_edits,
        "top_confusions": [
            {"reference": a, "candidate": b, "count": c}
            for (a, b), c in confusions.most_common(20)
        ],
        "worst_rows": sorted(per_row, key=lambda r: -r["ter"])[:10],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--binary", type=Path, default=HERE / "nano_ids_host")
    parser.add_argument(
        "--espeak-data", type=Path, default=Path("/opt/homebrew/share/espeak-ng-data")
    )
    parser.add_argument(
        "--reference",
        type=Path,
        action="append",
        default=None,
        help="jsonl with text+ids; repeatable. Defaults to the two 20260904 sets.",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    refs = args.reference or [
        ROOT / "artifacts/nano-g2p-espeak-free-20260904/shipped-espeak.jsonl",
        ROOT / "artifacts/nano-g2p-espeak-free-20260904/misaki-reference.jsonl",
    ]

    try:
        vocab = json.loads(
            (ROOT / "artifacts/kokoro-corpus-af_heart-20260713/kokoro_vocab.json").read_text(
                encoding="utf-8"
            )
        )
    except OSError as exc:
        raise SystemExit(f"error: cannot read the vocabulary: {exc}")
    vocab_inv = {int(v): k for k, v in vocab.items()}

    reports = []
    try:
        for ref_path in refs:
            rows = load_jsonl(ref_path)
            texts = [r["text"] for r in rows]
            cand = run_candidate(args.binary, args.espeak_data, texts)
            reports.append(score(ref_path.name, rows, cand, vocab_inv))
    except ParityError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    for rep in reports:
        print(f"\n=== C front end vs {rep['system']} ===")
        print(f"  rows                 {rep['rows_scored']} scored, {rep['rows_failed']} failed")
        print(f"  exact id sequences   {rep['exact_sequence_match']:.4%}")
        if rep["phoneme_string_exact"] is not None:
            print(f"  exact phoneme string {rep['phoneme_string_exact']:.4%}")
        print(
            f"  token error rate     {rep['token_error_rate']:.4%} "
            f"({rep['edit_distance_total']}/{rep['reference_tokens']} id edits)"
        )
        for fail in rep["failures"]:
            print(f"  FAILED {fail[0]}: {fail[1]}")
        if rep["top_confusions"]:
            print("  top confusions (reference -> candidate):")
            for c in rep["top_confusions"][:12]:
                print(f"    {c['reference']!r:>8} -> {c['candidate']!r:<8} {c['count']}")

    if args.json_out:
        try:
            args.json_out.write_text(json.dumps(reports, indent=1, ensure_ascii=False) + "\n",
                                     encoding="utf-8")
        except OSError as exc:
            print(f"error: cannot write {args.json_out}: {exc}", file=sys.stderr)
            return 1
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
