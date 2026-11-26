"""The harmonic-plus-noise decoder (sanotts/nano_dsp.py) and its voice-package wiring.

Checked here, with random weights (every property is structural):

* render() returns (T - 1) * 256 samples and DspStream, fed in any chunk schedule, concatenates to it.
* The per-frame noise does not depend on the range it is drawn in.
* The exact min-phase matrix equals the FFT computation when the envelope clamp is inactive.
* DspWeights survives save_npz -> from_bytes.
* nano_dsp.stream_local (first words first) concatenates to the whole-utterance render, on the heart-nano
  (en_us_e13b) fixture front end with the local flag set, as test_nano_local.py does for the neural decoder.
* voicepack refuses a dsp_decoder with a bad sha256, one without a sha256, and one on a non-local voice.

Run:  python3 -m pytest pypkg/tests/test_nano_dsp.py -v
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pypkg"))

from sanotts import nano, nano_dsp, voicepack  # noqa: E402

SCHEDULES = ((1, 1), (3, 7), (4, 16), (8, 128), (10_000, 10_000))
FIXTURE = ROOT / "mcu/test/fixtures/en_us_e13b"
HEADER = ROOT / "mcu/models/en_us_e13b/nano_q8_meta.h"


def _weights(seed=0, minphase=True, dim=16, blocks=2, hb=12, nb=8, noise_fft=1024, mp_noclamp=False):
    r = np.random.default_rng(seed)
    n = lambda *s: r.standard_normal(s) * 0.1  # noqa: E731
    blk = [{"dw_w": n(dim, 1, 5), "dw_b": n(dim), "n_w": 1 + n(dim), "n_b": n(dim), "p1_w": n(2 * dim, dim),
            "p1_b": n(2 * dim), "p2_w": n(dim, 2 * dim), "p2_b": n(dim), "gamma": 0.1 + n(dim), "left": 3, "right": 1}
           for _ in range(blocks)]
    head_b = np.concatenate([[0.0, 2.0], np.full(hb, -3.0), np.full(nb, -5.0)])
    return nano_dsp.DspWeights(n(dim, 100, 3), n(dim), 1, 1, blk, 1 + n(dim), n(dim), n(2 + hb + nb, dim), head_b,
                               n(nano_dsp.N_HARM), minphase, hb, nb, noise_fft, mp_noclamp)


def _mel(t, seed=1):
    return (np.random.default_rng(seed).standard_normal((100, t)) - 4.0).astype(np.float32)


@pytest.mark.parametrize("minphase,noise_fft,mp_noclamp", [(False, 1024, False), (True, 1024, False), (True, 512, True)])
def test_stream_matches_render(minphase, noise_fft, mp_noclamp):
    w = _weights(minphase=minphase, noise_fft=noise_fft, mp_noclamp=mp_noclamp)
    for t in (2, 9, 57, 300):
        mel = _mel(t)
        ref = nano_dsp.render(w, mel)
        assert ref.shape == ((t - 1) * nano_dsp.HOP,)
        for first, chunk in SCHEDULES:
            s, k, out = nano_dsp.DspStream(w), 0, []
            size = first
            while k < t:
                b = min(t, k + size)
                out.append(s.push(mel[:, k:b], final=b == t))
                k, size = b, chunk
            got = np.concatenate(out)
            assert got.shape == ref.shape, (t, first, chunk)
            assert np.abs(got - ref).max() <= 1e-6, (t, first, chunk)


def test_noise_is_range_independent():
    syn = nano_dsp._Synth(_weights(), seed=5)
    whole = syn.white(0, 300)
    for lo, hi in ((0, 1), (127, 131), (200, 300)):
        assert np.array_equal(syn.white(lo, hi), whole[:, lo:hi]), (lo, hi)
    x = np.concatenate([whole.real.ravel(), whole.imag.ravel()])
    assert abs(x.mean()) < 0.02 and abs(x.std() - 1.0) < 0.02


def test_minphase_matrix_is_exact():
    hb = 12
    bands = np.random.default_rng(2).standard_normal((20, hb)) - 3.0
    fast = bands @ nano_dsp._minphase_matrix(hb)
    slow = nano_dsp._minimum_phase(np.minimum(bands @ nano_dsp._band_interp_matrix(hb), 4.0))
    assert np.abs(fast - slow).max() < 1e-9


def test_npz_round_trip():
    w = _weights()
    buf = io.BytesIO()
    w.save_npz(buf)
    w2 = nano_dsp.DspWeights.from_bytes(buf.getvalue())
    mel = _mel(40)
    # weights are stored as float32: the round trip is exact to float32 weight rounding
    ref = nano_dsp.render(w, mel)
    assert np.abs(nano_dsp.render(w2, mel) - ref).max() <= 1e-4 * max(1.0, float(np.abs(ref).max()))
    assert (w2.minphase, w2.harm_bands, w2.noise_bands) == (True, 12, 8)


def _local_front():
    meta = dict(nano.parse_meta_header(HEADER))
    meta["NANO_LOCAL_FEATURES"] = 1
    return nano.NanoWeights((FIXTURE / "front_q8.bin").read_bytes(), (FIXTURE / "model_q8.bin").read_bytes(), meta)


def test_stream_local_matches_whole_render():
    fw, w = _local_front(), _weights()
    for row in range(3):
        ids = np.fromfile(FIXTURE / f"r{row:02d}_ids.bin", dtype=np.int32).astype(np.int64)
        d = nano.duration_forward(fw, ids).astype(np.int64)
        plan = nano.acoustic_plan(fw, ids, d)
        ref = nano_dsp.render(w, nano.acoustic_frames(fw, plan, 0, plan.frames))
        for k in (None, 20, 40):
            prefix = None if k is None else ids[:k].tolist()
            pieces = list(nano_dsp.stream_local(fw, w, prefix, lambda: ids))
            got = np.concatenate(pieces)
            assert pieces and got.shape == ref.shape, (row, k)
            # phase 1 (prefix) and phase 2 (whole text) acoustic frames differ by float rounding only
            assert np.abs(got - ref).max() <= 5e-4 * max(1.0, float(np.abs(ref).max())), (row, k)


def test_stream_local_rejects_global_voice():
    meta = nano.parse_meta_header(HEADER)
    fw = nano.NanoWeights((FIXTURE / "front_q8.bin").read_bytes(), (FIXTURE / "model_q8.bin").read_bytes(), meta)
    with pytest.raises(ValueError):
        list(nano_dsp.stream_local(fw, _weights(), None, lambda: [1, 2, 3]))


def _voice_dir(tmp_path, local, dsp_bytes, sha):
    d = tmp_path / "voice"
    d.mkdir()
    shutil.copy(FIXTURE / "front_q8.bin", d / "front_q8.bin")
    shutil.copy(FIXTURE / "model_q8.bin", d / "model_q8.bin")
    hdr = HEADER.read_text(encoding="utf-8")
    if local:
        hdr += "\n#define NANO_LOCAL_FEATURES 1\n"
    (d / "nano_q8_meta.h").write_text(hdr, encoding="utf-8")
    (d / "dsp_decoder.npz").write_bytes(dsp_bytes)
    meta = {"key": "t", "runtime": "snt_nano", "lineage": "test", "front": "front_q8.bin", "dec": "model_q8.bin",
            "sample_rate": 24000, "params": 1, "vocab_size": 62, "offsets_header": "nano_q8_meta.h",
            "dsp_decoder": "dsp_decoder.npz"}
    if sha is not None:
        meta["dsp_decoder_sha256"] = sha
    (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return d


def _dsp_bytes():
    buf = io.BytesIO()
    _weights().save_npz(buf)
    return buf.getvalue()


def test_voicepack_loads_dsp_decoder(tmp_path):
    data = _dsp_bytes()
    pack = voicepack.load_voice("t", voice_dir=_voice_dir(tmp_path, True, data, hashlib.sha256(data).hexdigest()))
    assert pack.dsp == data


@pytest.mark.parametrize("case", ["bad_sha", "no_sha", "not_local"])
def test_voicepack_rejects_bad_dsp_decoder(tmp_path, case):
    data = _dsp_bytes()
    sha = {"bad_sha": "0" * 64, "no_sha": None, "not_local": hashlib.sha256(data).hexdigest()}[case]
    with pytest.raises(voicepack.VoicePackError):
        voicepack.load_voice("t", voice_dir=_voice_dir(tmp_path, case != "not_local", data, sha))


@pytest.mark.parametrize("n_fft", [512, 1024])
def test_ola_matches_nano_istft(n_fft):
    r = np.random.default_rng(3)
    for t in (2, 7, 100):
        spec = r.standard_normal((n_fft // 2 + 1, t)) + 1j * r.standard_normal((n_fft // 2 + 1, t))
        ref = nano.istft(spec, n_fft, nano_dsp.HOP).astype(np.float64)
        o = nano_dsp._OLA(n_fft); o.set_total(t)
        assert np.abs(o.push(spec) - ref).max() < 1e-6
        s, parts, k = nano_dsp._OLA(n_fft), [], 0
        while k < t:
            b = min(t, k + 3)
            if b == t:
                s.set_total(t)
            parts.append(s.push(spec[:, k:b])); k = b
        assert np.abs(np.concatenate(parts) - ref).max() < 1e-6


def _drop_dec(d: Path, keep_dsp: bool):
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    for k in ("dec", "dec_bytes", "dec_sha256"):
        meta.pop(k, None)
    if not keep_dsp:
        meta.pop("dsp_decoder", None)
        meta.pop("dsp_decoder_sha256", None)
    (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (d / "model_q8.bin").unlink()


def test_dsp_voice_needs_no_neural_decoder_blob(tmp_path):
    data = _dsp_bytes()
    d = _voice_dir(tmp_path, True, data, hashlib.sha256(data).hexdigest())
    _drop_dec(d, keep_dsp=True)
    pack = voicepack.load_voice("t", voice_dir=d)
    assert pack.dec is None and pack.dsp == data
    w = nano.NanoWeights(pack.front, pack.dec, pack.offsets)
    with pytest.raises(ValueError):
        w.dec_f32("HEAD_B", 1) if "DOFF_HEAD_B_F32" in pack.offsets else w._need_dec()


def test_voice_without_any_decoder_is_refused(tmp_path):
    data = _dsp_bytes()
    d = _voice_dir(tmp_path, True, data, hashlib.sha256(data).hexdigest())
    _drop_dec(d, keep_dsp=False)
    with pytest.raises(voicepack.VoicePackError):
        voicepack.load_voice("t", voice_dir=d)
