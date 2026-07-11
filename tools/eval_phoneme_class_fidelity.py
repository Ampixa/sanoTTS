#!/usr/bin/env python3
"""Per-phoneme-class spectral-fidelity eval for distilled TTS stacks.

WHY THIS EXISTS
---------------
Whole-utterance MOS predictors (SCOREQ, UTMOS, DNSMOS) collapse a sentence to one
number and routinely MISS localized artifacts. Concrete case (2026-07-08): our
Kristin student scored SCOREQ 4.09 / UTMOS 3.98 (beats TinyTTS) yet had an audible
whistly/metallic sibilant. Global metrics were blind to it because it is confined
to /s ʃ z ʒ/, a small fraction of each utterance.

This tool segments the rendered audio by phoneme (using the stack's own duration
alignment), buckets phonemes into articulatory classes, and measures the 2-8 kHz
spectral flatness (noise-like-ness) per class against the teacher. A sibilant is
broadband noise (high flatness); if the student's flatness collapses, it is
rendering that fricative as a tone -> the metallic/whistly artifact.

TWO OUTPUTS THAT MATTER
-----------------------
* per-class flatness gap (student vs teacher): WHICH phoneme classes degrade.
* the oracle-decoder lane (teacher's own latents -> our decoder): attributes each
  degradation to the ACOUSTIC vs the DECODER. If oracle-decoder matches the teacher
  but the full student doesn't, the acoustic is the culprit (it smooths that
  class's latent); if oracle-decoder already degrades, the decoder is.

Flatness is a NOISE-fidelity metric; it catches fricative/sibilant/aspiration
smoothing. It does NOT catch tonal ringing on vowels (a phase artifact) -- pair it
with listening (use --emit-audio for per-class concatenated wavs).

USAGE
-----
  python3 tools/eval_phoneme_class_fidelity.py \
    --acoustic  <latent-student.pt> \
    --decoder-student <decoder-student.pt> \
    --duration  <duration-student.pt> \
    --piper-model <teacher.onnx> --piper-config <teacher.onnx.json> \
    --textset   data/textsets/multilang-distill-v1/en_US.diverse-heldout24.jsonl \
    --out-dir   artifacts/phoneme-fidelity/<run> --emit-audio

Emits report.json (+ per-class {class}_{teacher,oracle,student}.wav if --emit-audio).
Reuses tools/serve_roota_arbitrary_tts_dashboard.py (DashboardState) as the renderer.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import stft

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# Articulatory classes. A phoneme may belong to several (e.g. ʃ is both sibilant
# and postalveolar). espeak-ng / Piper IPA symbols; phonemize() returns one symbol
# per list element, so classification is per-element.
CLASSES: dict[str, set[str]] = {
    "sibilant (s z ʃ ʒ)": set("szʃʒ"),
    "postalv/palatal (ʃ ʒ tʃ dʒ j)": {"ʃ", "ʒ", "tʃ", "dʒ", "j"},
    "affricate (tʃ dʒ)": {"tʃ", "dʒ"},
    "fricative-nonsib (f v θ ð h)": {"f", "v", "θ", "ð", "h"},
    "alveolar (t d n l s z ɹ)": {"t", "d", "n", "l", "s", "z", "ɹ", "ɾ"},
    "nasal (m n ŋ)": {"m", "n", "ŋ"},
    "rhotic/retroflex (ɹ ɻ ɚ)": {"ɹ", "ɻ", "ɚ", "ɝ", "r"},
    "lateral (l)": {"l"},
    "plosive (p b t d k g)": {"p", "b", "t", "d", "k", "ɡ", "g"},
    "vowel": {"i", "iː", "ɪ", "ɛ", "e", "æ", "ə", "ʌ", "ɑ", "ɑː", "ɒ", "ɔ", "ɔː", "o", "oː",
              "ʊ", "u", "uː", "ɜ", "ɜː", "aɪ", "aʊ", "ɔɪ", "eɪ", "oʊ", "əʊ", "ɪə", "eə", "ʊə", "ɚ", "ɝ"},
}


def classify(p: str) -> list[str]:
    return [c for c, s in CLASSES.items() if p in s]


def flatness_2_8k(a: np.ndarray, sr: int) -> float | None:
    """Mean spectral flatness (geo/arith mean) in 2-8 kHz over the signal. Higher = noisier."""
    if len(a) < 512:
        return None
    f, _, Z = stft(a, sr, nperseg=min(1024, len(a)))
    S = np.abs(Z) + 1e-9
    band = (f >= 2000) & (f <= 8000)
    Sb = S[band]
    if Sb.shape[1] == 0:
        return None
    return float((np.exp(np.log(Sb).mean(0)) / Sb.mean(0)).mean())


def load_serve():
    spec = importlib.util.spec_from_file_location(
        "serve_mod", ROOT / "tools" / "serve_roota_arbitrary_tts_dashboard.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load serve_roota_arbitrary_tts_dashboard.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["serve_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


def build_state(serve, a: argparse.Namespace, work: Path):
    args = argparse.Namespace(
        host="127.0.0.1", port=0,
        acoustic_checkpoint=Path(a.acoustic), decoder_backend="student", decoder=ROOT / "unused.onnx",
        decoder_student_checkpoint=Path(a.decoder_student),
        duration_source="student", duration_checkpoint=Path(a.duration), duration_length_scale=a.length_scale,
        audio_enhancer_checkpoint=None, postprocess_gain=1.0, postprocess_filter="none",
        piper_model=Path(a.piper_model), piper_config=Path(a.piper_config),
        out_dir=work, device=a.device, noise_scale=0.0, length_scale=1.0, noise_w=0.0,
        sentence_silence=0.12, text_chunking="none", dashboard_title="x", dashboard_subtitle="", default_text="x")
    return serve.DashboardState(args)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--acoustic", required=True)
    ap.add_argument("--decoder-student", required=True)
    ap.add_argument("--duration", required=True)
    ap.add_argument("--piper-model", required=True)
    ap.add_argument("--piper-config", required=True)
    ap.add_argument("--textset", required=True, help="jsonl with a 'text' field per line")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--length-scale", type=float, default=1.08)
    ap.add_argument("--device", default="cpu", help="cpu recommended: MPS iSTFT hits a >65536-channel limit on full sentences")
    ap.add_argument("--emit-audio", action="store_true", help="write per-class concatenated wavs for listening")
    ap.add_argument("--sibilant-calib", default=None, help="calib npz (tea_std, sib_ids) to enable sibilant injection on the student lane")
    ap.add_argument("--sibilant-beta", type=float, default=0.0, help="injection strength (0=off)")
    a = ap.parse_args()

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    serve = load_serve()
    st = build_state(serve, a, out / "_work")
    if a.sibilant_beta > 0.0 and a.sibilant_calib:
        cal = np.load(a.sibilant_calib)
        st.sibilant_beta = float(a.sibilant_beta)
        st.sibilant_tea_std = cal["tea_std"].astype(np.float32)[:, None]
        st.sibilant_ids = set(int(x) for x in cal["sib_ids"].tolist())
        print(f"[inject] beta={a.sibilant_beta} sib_ids={sorted(st.sibilant_ids)} tea_std[{st.sibilant_tea_std.shape[0]}]")
    sr = st.sample_rate
    sil = st.silence.size

    # capture phoneme symbols + durations that synthesize() uses, aligned to the audio it renders
    cap_ph: list[tuple[list, list]] = []
    cap_du: list[np.ndarray] = []
    orig_ids = st.voice.phonemes_to_ids
    st.voice.phonemes_to_ids = lambda ph: (cap_ph.append((list(ph), list(orig_ids(ph)))) or orig_ids(ph))  # type: ignore
    orig_du = st.predict_student_durations

    def wrap_du(ids, oracle, src, ls):
        d, s = orig_du(ids, oracle, src, ls)
        cap_du.append(np.asarray(d).reshape(-1))
        return d, s
    st.predict_student_durations = wrap_du  # type: ignore

    texts = [json.loads(l)["text"] for l in Path(a.textset).read_text().splitlines() if l.strip()]
    # lane -> class -> list of segments
    aud: dict[str, dict[str, list]] = {"teacher": defaultdict(list), "oracle": defaultdict(list), "student": defaultdict(list)}
    lane_key = {"teacher": "teacher_audio", "oracle": "oracle_decoder_audio", "student": "student_audio"}
    n_ok = 0
    for text in texts:
        cap_ph.clear(); cap_du.clear()
        try:
            meta = st.synthesize(text, duration_source="student", duration_length_scale=a.length_scale)
        except Exception as exc:  # keep going; report at the end
            print(f"[warn] synth failed: {exc}", file=sys.stderr)
            continue
        if len(cap_ph) != len(cap_du) or not cap_du:
            continue
        wavs = {}
        for lane, key in lane_key.items():
            w, _ = sf.read(str(st.audio_dir / Path(meta[key]).name))
            wavs[lane] = w.mean(1) if w.ndim > 1 else w
        totf = sum(int(d.sum()) for d in cap_du)
        nq = len(cap_du)
        hop = {lane: (len(wavs[lane]) - (nq - 1) * sil) / max(1, totf) for lane in wavs}
        off = {lane: 0.0 for lane in wavs}
        for (phonemes, _ids), durs in zip(cap_ph, cap_du):
            P, L = len(phonemes), len(durs)
            # Piper id layout [BOS, pad, p0, pad, p1, pad, ...] -> phoneme i at index 2+2i
            if L >= 2 * P + 1:
                pos = [2 + 2 * i for i in range(P)]
            elif L == P:
                pos = list(range(P))
            else:
                pos = [min(L - 1, int(round((i + 0.5) * L / P))) for i in range(P)]
            cum = np.concatenate([[0], np.cumsum(durs)])
            for i, p in enumerate(phonemes):
                cs = classify(p)
                if not cs:
                    continue
                fa, fb = cum[pos[i]], cum[pos[i] + 1]
                for lane in wavs:
                    seg = wavs[lane][int(off[lane] + fa * hop[lane]):int(off[lane] + fb * hop[lane])]
                    if len(seg) > 0:
                        for c in cs:
                            aud[lane][c].append(seg)
            for lane in wavs:
                off[lane] += int(durs.sum()) * hop[lane] + sil
        n_ok += 1

    def flatcat(segs):
        return flatness_2_8k(np.concatenate(segs), sr) if segs else None

    report = {"n_sentences": n_ok, "length_scale": a.length_scale, "classes": []}
    print(f"\nRendered {n_ok}/{len(texts)} sentences. Flatness 2-8kHz (higher = crisper noise; lower = smoothed/tonal).")
    print(f"{'class':32s} {'teacher':>8}{'oracle-dec':>11}{'student':>8}{'stu-tea':>9}{'attribute':>11}{'n':>6}")
    for c in CLASSES:
        if not aud["student"][c]:
            continue
        t = flatcat(aud["teacher"][c]); o = flatcat(aud["oracle"][c]); s = flatcat(aud["student"][c])
        if s is None:
            continue
        t = t if t is not None else float("nan")
        o = o if o is not None else float("nan")
        # attribution: gap the oracle lane already has = decoder; extra gap in student = acoustic
        dec_gap = (o - t) if (o == o and t == t) else float("nan")
        aco_gap = (s - o) if (s == s and o == o) else float("nan")
        attr = "acoustic" if (aco_gap == aco_gap and aco_gap < -0.03) else ("decoder" if (dec_gap == dec_gap and dec_gap < -0.03) else "-")
        n = len(aud["student"][c])
        print(f"{c:32s} {t:8.3f}{o:11.3f}{s:8.3f}{s - t:+9.3f}{attr:>11}{n:6d}")
        row = {"cls": c, "teacher": t, "oracle_decoder": o, "student": s,
               "student_minus_teacher": s - t, "decoder_gap": dec_gap, "acoustic_gap": aco_gap,
               "attribution": attr, "n": n}
        report["classes"].append(row)
        if a.emit_audio:
            def cat(segs):
                g = np.zeros(int(0.05 * sr), dtype=np.float32)
                o2, tot = [], 0
                for x in segs:
                    if tot > 7 * sr:
                        break
                    o2 += [x.astype(np.float32), g]; tot += len(x) + len(g)
                return np.concatenate(o2) if o2 else np.zeros(1, dtype=np.float32)
            slug = c.split()[0].replace("/", "_")
            for lane in ("teacher", "oracle", "student"):
                sf.write(str(out / f"{slug}_{lane}.wav"), np.clip(cat(aud[lane][c]), -1, 1), sr)
    (out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"\nwrote {out / 'report.json'}")
    print("Attribution: 'acoustic' = student smooths this class's latent (decoder is fine on teacher latents); "
          "'decoder' = the decoder itself degrades it. Pair with --emit-audio + ears for tonal artifacts flatness misses.")


if __name__ == "__main__":
    main()
