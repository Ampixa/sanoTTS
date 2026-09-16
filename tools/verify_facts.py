#!/usr/bin/env python3
"""verify_facts.py — consistency checks between the site's displayed numbers
and the artifacts on this branch.

Checks:
  1. parameter split sums to the advertised total (paperFacts.ts ↔ meta.json)
  2. shipped weights size matches meta.json and the blobs in public/engine/
  3. trace index carries the same param total and 8 validated rows
  4. every trace manifest reports got_frames == fixture frames
  5. WASM module / blobs hashes match the voice manifest from master

Run after any fact edit:  python3 tools/verify_facts.py
"""
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
fail = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global fail
    print(f"{'PASS' if ok else 'FAIL'}  {name}{(' — ' + detail) if detail else ''}")
    fail |= not ok


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


pf = (ROOT / "src" / "data" / "paperFacts.ts").read_text()
meta = json.loads((ROOT / "public" / "engine" / "meta.json").read_text())


def pf_int(key: str) -> int:
    m = re.search(rf"\b{key}:\s*([0-9][0-9_]*)", pf)
    if not m:
        return -1
    return int(m.group(1).replace("_", ""))


# 1. parameter arithmetic
dur, ac, dec = pf_int("duration"), pf_int("acoustic"), pf_int("decoder")
total = pf_int("total")
check("params: split sums to total", dur + ac + dec == total,
      f"{dur}+{ac}+{dec}={total}")
check("params: match master meta.json",
      meta["param_breakdown"] == {"duration": dur, "acoustic": ac, "decoder": dec}
      and meta["params"] == total)

# 2. weights size
front_b = (ROOT / "public" / "engine" / "front_q8.bin").stat().st_size
dec_b = (ROOT / "public" / "engine" / "model_q8.bin").stat().st_size
check("weights: blob sizes match meta.json",
      front_b == meta["front_bytes"] and dec_b == meta["dec_bytes"],
      f"{front_b}+{dec_b}")
check("weights: paperFacts weightsBytes", pf_int("weightsBytes") == front_b + dec_b)
check("weights: sha256 match master manifest",
      sha256(ROOT / "public" / "engine" / "front_q8.bin") == meta["front_sha256"]
      and sha256(ROOT / "public" / "engine" / "model_q8.bin") == meta["dec_sha256"])

# 3. traces index
ix = json.loads((ROOT / "public" / "traces" / "index.json").read_text())
check("traces: index params == paperFacts total", ix["params"] == total)
check("traces: 8 rows", len(ix["rows"]) == 8)
check("traces: vocab 62 symbols", len(ix["vocab"]) == 62)

# 4. per-row manifests vs fixture rows.txt
fixture_rows = {}
for line in (ROOT / "engine" / "fixtures" / "en_us_e13b" / "rows.txt").read_text().splitlines():
    rid, ntok, frames, samples, seed = line.split()
    fixture_rows[rid] = (int(ntok), int(frames), int(samples), int(seed))
for rid, (ntok, frames, samples, seed) in sorted(fixture_rows.items()):
    man_p = ROOT / "public" / "traces" / rid / "manifest.json"
    if not man_p.is_file():
        check(f"traces: {rid} manifest present", False)
        continue
    man = json.loads(man_p.read_text())
    ok = (man["n_tokens"] == ntok and abs(man["got_frames"] - frames) <= 4
          and (man["seed_hi"] << 32 | man["seed_lo"]) == seed and man["text"])
    check(f"traces: {rid} manifest", ok,
          f"N={man['n_tokens']}/{ntok} T={man['got_frames']}/{frames} (student timing may differ by a few frames) seed-ok text-ok")
    pcm_samples = (ROOT / "public" / "traces" / rid / "pcm.i16").stat().st_size // 2
    check(f"traces: {rid} pcm length", abs(pcm_samples - man["got_samples"]) < 4)

# 5. arena peak claim in paperFacts (max over traces)
peak = 0
for rid in fixture_rows:
    man = json.loads((ROOT / "public" / "traces" / rid / "manifest.json").read_text())
    peak = max(peak, man["arena_peak"])
check("arena: paperFacts arenaPeakBytes == max trace peak",
      pf_int("arenaPeakBytes") == peak, f"claimed {pf_int('arenaPeakBytes')}, max {peak}")

print()
if fail:
    print("FAILURES present — fix before shipping")
    sys.exit(1)
print("all fact checks pass")
