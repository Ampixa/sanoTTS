#!/usr/bin/env python3
"""Measure the parity ceiling of a per-word espeak dictionary vs in-context espeak.

Reference = whole-sentence espeak -> Kristin ids (the 15.3%-WER path).
Candidate = each word phonemized in ISOLATION, then assembled in piper's id layout
            (BOS,pad, w1 phonemes w/pads, space, w2..., '.', EOS).
Reports per-sentence phoneme-id error rate (Levenshtein/ref-len) = the loss a
per-word dictionary incurs from ignoring sentence context.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from piper.phonemize_espeak import EspeakPhonemizer
from piper.phoneme_ids import phonemes_to_ids

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "models" / "teachers" / "kristin-medium" / "en_US-kristin-medium.onnx.json"
ID_MAP = json.load(open(CFG, encoding="utf-8"))["phoneme_id_map"]
PH = EspeakPhonemizer()


def lev(a, b):
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return dp[n]


def sent_phonemes(text: str) -> list[str]:
    return [p for s in PH.phonemize("en-us", text) for p in s]


def word_phonemes(word: str) -> list[str]:
    out = []
    for s in PH.phonemize("en-us", word):
        out.extend(s)
    return [p for p in out if p not in (" ", ".", ",", "!", "?", ";", ":")]


import re


def assemble_perword(text: str) -> list[str]:
    words = re.findall(r"[A-Za-z']+", text)
    phon: list[str] = []
    for i, w in enumerate(words):
        if i:
            phon.append(" ")
        phon.extend(word_phonemes(w))
    phon.append(".")
    return phon


def main():
    textset = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "data/textsets/multilang-distill-v1/en_US.diverse-heldout24.jsonl")
    texts = [json.loads(l)["text"] for l in Path(textset).read_text().splitlines() if l.strip()]
    tot_err = tot_len = 0
    for t in texts:
        ref = phonemes_to_ids(sent_phonemes(t), ID_MAP)
        cand = phonemes_to_ids(assemble_perword(t), ID_MAP)
        e = lev(ref, cand)
        tot_err += e
        tot_len += len(ref)
        print(f"PER {e/max(1,len(ref)):5.1%}  ref={len(ref):3d} cand={len(cand):3d}  {t[:44]!r}")
    print(f"\nper-word dictionary parity ceiling: PER {tot_err/max(1,tot_len):.2%} "
          f"({tot_err}/{tot_len} id edits)")


if __name__ == "__main__":
    main()
