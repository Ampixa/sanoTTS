"""Streaming synthesis, and the vectorised erf and noise generator it relies on.

Every claim the 0.5.1 docstrings make is checked here:

* `_erf` agrees with math.erf to float64 rounding.
* The vectorised MT19937 / Gaussian noise is bit-identical to the scalar
  implementation it replaced (kept below as the reference).
* `nano.stream_ids` concatenates to `nano.synthesize_ids` for the three
  shipped nano lineages and many chunk schedules.
* The piperlite decoder's `decoder_halo_frames` bounds its measured receptive
  field, and `decoder_stream` concatenates to `decoder_forward`.
* 0.5.2's faster silu, conv1d, layer_norm, erf, frame features and seeding
  are bit-identical to the 0.5.1 forms (kept below as references).

Run:  python3 -m pytest pypkg/tests/test_nano_stream.py -v
  or: python3 pypkg/tests/test_nano_stream.py
"""

from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pypkg"))

from sanotts import models, nano, nano_rng  # noqa: E402

# Chunked decoding differs from the whole pass only by float rounding (BLAS
# association over a shorter window, FFT size in the DC blocker). Measured
# worst cases are ~1e-12 for nano and 0 for piperlite; this is the bound
# promised in the docstrings.
STREAM_TOL = 1e-6

LINEAGES = (("en_us_e12nano", "q8"), ("en_us_e13b", "q8"), ("en_us_r227f32", "f32"))
SCHEDULES = ((1, 1), (1, 2), (3, 5), (8, 64), (2, 7), (10_000, 10_000))


# ---- reference: the scalar generator shipped up to 0.5.0 -----------------

class _ScalarMT19937:
    def __init__(self, seed: int) -> None:
        state = [seed & 0xFFFFFFFF]
        for i in range(1, 624):
            prev = state[-1]
            state.append((1812433253 * (prev ^ (prev >> 30)) + i) & 0xFFFFFFFF)
        self.s, self.left, self.next = state, 1, 0

    def _twist(self) -> None:
        s = self.s
        for i in range(624):
            v = s[(i + 1) % 624]
            mixed = (s[i] & 0x80000000) | (v & 0x7FFFFFFF)
            s[i] = s[(i + 397) % 624] ^ (mixed >> 1) ^ (0x9908B0DF if v & 1 else 0)
        self.left, self.next = 624, 0

    def next_uint32(self) -> int:
        self.left -= 1
        if self.left <= 0:
            self._twist()
        y = self.s[self.next]
        self.next += 1
        y ^= y >> 11
        y = (y ^ ((y << 7) & 0x9D2C5680)) & 0xFFFFFFFF
        y = (y ^ ((y << 15) & 0xEFC60000)) & 0xFFFFFFFF
        return (y ^ (y >> 18)) & 0xFFFFFFFF

    def uniform(self, n: int) -> np.ndarray:
        return np.array([np.float32((self.next_uint32() & 0xFFFFFF) * (1.0 / 16777216.0))
                         for _ in range(n)], dtype=np.float32)


def _scalar_noise(seed: int, channels: int, frames: int) -> np.ndarray:
    size = channels * frames
    gen = _ScalarMT19937(seed)
    flat = gen.uniform(size)
    for i in range(0, (size // 16) * 16, 16):
        nano_rng._normal_fill_16(flat[i:i + 16])
    if size % 16:
        tail = gen.uniform(16)
        nano_rng._normal_fill_16(tail)
        flat[size - 16:] = tail
    return flat.reshape(channels, frames)


SEEDS = (0, 1, 2236265385529901705, 2**32 - 1, 2**32, 12345678901234)


def test_uniform_matches_scalar_reference():
    for seed in SEEDS:
        for n in (1, 16, 623, 624, 625, 1249, 3000):
            got = nano_rng.uniform_stream(seed, n)
            ref = _ScalarMT19937(seed).uniform(n)
            assert np.array_equal(got.view(np.uint32), ref.view(np.uint32)), (seed, n)


def test_uniform_draws_in_pieces_continue_the_stream():
    gen, ref = nano_rng.ATenMT19937(5), _ScalarMT19937(5)
    got = np.concatenate([gen.next_uint32_array(k) for k in (3, 700, 0, 1, 2000)])
    assert np.array_equal(got, np.array([ref.next_uint32() for _ in range(got.size)], dtype=np.uint32))


def test_noise_matches_scalar_reference():
    for seed in SEEDS:
        for channels, frames in ((4, 4), (4, 5), (4, 659), (4, 1003), (3, 7)):
            got = nano_rng.seeded_noise(seed, channels, frames)
            ref = _scalar_noise(seed, channels, frames)
            assert np.array_equal(got.view(np.uint32), ref.view(np.uint32)), (seed, channels, frames)


# ---- erf ----------------------------------------------------------------

def test_erf_matches_math_erf():
    rng = np.random.default_rng(1)
    xs = np.concatenate([rng.uniform(-7.0, 7.0, 200_000), rng.normal(0.0, 1.0, 100_000),
                         np.arange(-6 * 1024, 6 * 1024 + 1) / 1024 + 2.0**-12,
                         [0.0, -0.0, 1e-300, 5.999999, 6.0, 6.5, 30.0, -30.0]])
    ref = np.array([math.erf(v) for v in xs.tolist()])
    assert np.abs(nano._erf(xs) - ref).max() <= 1e-15
    special = nano._erf(np.array([np.nan, np.inf, -np.inf]))
    assert np.isnan(special[0]) and special[1] == 1.0 and special[2] == -1.0


# ---- nano streaming ------------------------------------------------------

def _lineage(lineage: str, ext: str):
    fixture = ROOT / "mcu/test/fixtures" / lineage
    meta = nano.parse_meta_header(ROOT / "mcu/models" / lineage / "nano_q8_meta.h")
    return (fixture, meta, (fixture / f"front_{ext}.bin").read_bytes(),
            (fixture / f"model_{ext}.bin").read_bytes())


def test_nano_stream_matches_full_pass():
    for lineage, ext in LINEAGES:
        fixture, meta, front, dec = _lineage(lineage, ext)
        weights = nano.NanoWeights(front, dec, meta)
        for row in range(3):
            ids = np.fromfile(fixture / f"r{row:02d}_ids.bin", dtype=np.int32)
            durs = np.fromfile(fixture / f"r{row:02d}_durs.bin", dtype=np.int32)
            full = nano.synthesize_ids(front, dec, meta, ids, seed=7, durations=durs, weights=weights)
            for first, chunk in SCHEDULES:
                pieces = list(nano.stream_ids(front, dec, meta, ids, seed=7, durations=durs, weights=weights,
                                              first_chunk_frames=first, chunk_frames=chunk))
                assert all(p.size for p in pieces), (lineage, row, first, chunk)
                got = np.concatenate(pieces)
                assert got.shape == full.shape, (lineage, row, first, chunk, got.shape, full.shape)
                assert np.abs(got - full).max() <= STREAM_TOL, (lineage, row, first, chunk)


def test_nano_cached_weights_change_nothing():
    fixture, meta, front, dec = _lineage("en_us_e13b", "q8")
    ids = np.fromfile(fixture / "r00_ids.bin", dtype=np.int32)
    fresh = nano.synthesize_ids(front, dec, meta, ids, seed=3)
    weights = nano.NanoWeights(front, dec, meta)
    nano.synthesize_ids(front, dec, meta, ids, seed=3, weights=weights)    # fills the cache
    assert np.array_equal(nano.synthesize_ids(front, dec, meta, ids, seed=3, weights=weights), fresh)


def test_nano_stream_rejects_bad_schedule():
    fixture, meta, front, dec = _lineage("en_us_e13b", "q8")
    ids = np.fromfile(fixture / "r00_ids.bin", dtype=np.int32)
    for first, chunk in ((0, 8), (8, 4)):
        try:
            next(nano.stream_ids(front, dec, meta, ids, seed=1, first_chunk_frames=first, chunk_frames=chunk))
        except ValueError:
            continue
        raise AssertionError(f"accepted first={first} chunk={chunk}")


# ---- piperlite streaming -------------------------------------------------

def _random_piperlite(rng: np.random.Generator, branches=(0, 1, 2)):
    """A small randomly initialised piperlite decoder with the shipped layer layout."""
    c = (16, 8, 8, 4)
    t = {"pre.weight": rng.normal(0, 0.2, (c[0], 12, 7)), "pre.bias": rng.normal(0, 0.1, c[0]),
         "post.weight": rng.normal(0, 0.2, (1, c[3], 7)), "post.bias": rng.normal(0, 0.1, 1)}
    for stage, (cin, cout, k) in enumerate(((c[0], c[1], 16), (c[1], c[2], 16), (c[2], c[3], 8))):
        t[f"up{stage}.weight"] = rng.normal(0, 0.2, (cin, cout, k))
        t[f"up{stage}.bias"] = rng.normal(0, 0.1, cout)
        for b in branches:
            kb = models._BANK_KERNELS[b]
            for conv in ("conv1", "conv2"):
                t[f"res{stage}.0.blocks.{b}.{conv}.weight"] = rng.normal(0, 0.2, (cout, cout, kb))
                t[f"res{stage}.0.blocks.{b}.{conv}.bias"] = rng.normal(0, 0.1, cout)
    config = {"variant": "piperlite", "channels": list(c), "res_layers": 1,
              **{f"stage{i}_branches": list(branches) for i in range(3)}}
    return {k: v.astype(np.float32) for k, v in t.items()}, config


def test_piperlite_halo_bounds_measured_receptive_field():
    rng = np.random.default_rng(0)
    for branches in ((0, 1, 2), (0,), (2,)):
        tensors, config = _random_piperlite(rng, branches)
        frames, f0 = 80, 40
        latent = rng.normal(0, 1, (12, frames)).astype(np.float32)
        base = models.decoder_forward(tensors, config, latent)
        poked = latent.copy()
        poked[:, f0] += 5.0
        changed = np.nonzero(models.decoder_forward(tensors, config, poked) != base)[0]
        hop = models.decoder_hop(config)
        reach = max(f0 - changed.min() / hop, (changed.max() + 1) / hop - (f0 + 1))
        assert 0 < reach <= models.decoder_halo_frames(tensors, config), (branches, reach)


def test_piperlite_stream_matches_full_pass():
    rng = np.random.default_rng(1)
    tensors, config = _random_piperlite(rng)
    for frames in (1, 5, 37, 150):
        latent = rng.normal(0, 1, (12, frames)).astype(np.float32)
        full = models.decoder_forward(tensors, config, latent)
        for first, chunk in SCHEDULES:
            pieces = list(models.decoder_stream(tensors, config, latent,
                                                first_chunk_frames=first, chunk_frames=chunk))
            assert all(p.size for p in pieces)
            got = np.concatenate(pieces)
            assert got.shape == full.shape and np.abs(got - full).max() <= STREAM_TOL, (frames, first, chunk)


# ---- 0.5.2: the faster primitives are bit-identical to the 0.5.1 forms ----

def _silu_ref(x):
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = x[pos] / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[~pos])
    out[~pos] = x[~pos] * e / (1.0 + e)
    return out


def _conv1d_ref(x, w, b, kernel):
    c_in, t = x.shape
    pad = kernel // 2
    xp = np.pad(x, ((0, 0), (pad, pad)))
    gathered = np.empty((c_in * kernel, t), dtype=np.float32)
    for k in range(kernel):
        gathered[k::kernel] = xp[:, k : k + t]
    return (w @ gathered + b[:, None]).astype(np.float32)


def _layer_norm_ref(x, w, b, eps=nano.LAYER_NORM_EPS):
    mu = x.mean(axis=0, keepdims=True)
    var = x.var(axis=0, keepdims=True)
    return ((x - mu) / np.sqrt(var + eps) * w[:, None] + b[:, None]).astype(np.float32)


def _erf_ref(x):
    x = np.asarray(x, dtype=np.float64)
    a = np.abs(x)
    nan = np.isnan(a)
    a = np.minimum(np.where(nan, 0.0, a), nano._ERF_KNOTS * nano._ERF_STEP)
    k = np.rint(a * (1.0 / nano._ERF_STEP)).astype(np.intp)
    x0 = nano._ERF_X0[k]
    t = a - x0
    x2 = x0 * x0
    c2 = -x0
    c3 = (4.0 * x2 - 2.0) / 6.0
    c4 = (12.0 * x0 - 8.0 * x0 * x2) / 24.0
    c5 = (16.0 * x2 * x2 - 48.0 * x2 + 12.0) / 120.0
    poly = t * (1.0 + t * (c2 + t * (c3 + t * (c4 + t * c5))))
    out = np.copysign(nano._ERF_TABLE[k] + nano._ERF_DERIV[k] * poly, x)
    if nan.any():
        out[nan] = np.nan
    return out


def _bits_equal(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def test_primitives_bit_identical_to_previous_forms():
    rng = np.random.default_rng(3)
    specials = np.array([0.0, -0.0, np.inf, -np.inf, np.nan, 1e-30, -1e-30, 40.0, -40.0])
    for scale in (1e-3, 1.0, 6.0, 80.0):
        x = rng.standard_normal((37, 29)) * scale
        x.flat[: specials.size] = specials
        assert _bits_equal(nano._erf(x), _erf_ref(x)), scale
        xf = x.astype(np.float32)
        with np.errstate(over="ignore", invalid="ignore"):
            assert _bits_equal(nano.silu(xf), _silu_ref(xf)), scale
            assert _bits_equal(nano.gelu(xf), (0.5 * xf * (1.0 + _erf_ref(xf.astype(np.float64) / math.sqrt(2.0)))).astype(np.float32))
        finite = rng.standard_normal((37, 29)).astype(np.float32) * np.float32(scale)
        w = rng.standard_normal(37).astype(np.float32)
        b = rng.standard_normal(37).astype(np.float32)
        assert _bits_equal(nano.layer_norm(finite, w, b), _layer_norm_ref(finite, w, b)), scale


def test_conv1d_bit_identical_including_sequences_shorter_than_the_kernel():
    rng = np.random.default_rng(4)
    for kernel in (1, 3, 5, 7, 9):
        for t in (1, 2, 3, 4, 5, 11, 64):
            x = rng.standard_normal((6, t)).astype(np.float32)
            w = rng.standard_normal((5, 6 * kernel)).astype(np.float32)
            b = rng.standard_normal(5).astype(np.float32)
            assert _bits_equal(nano.conv1d(x, w, b, kernel), _conv1d_ref(x, w, b, kernel)), (kernel, t)


def test_frame_features_match_the_per_token_loop():
    _fixture, meta, front, dec = _lineage(*LINEAGES[0])
    weights = nano.NanoWeights(front, dec, meta)
    rng = np.random.default_rng(5)
    for n in (1, 2, 9, 60):
        ids = rng.integers(1, meta["NANO_VOCAB"], n)
        durs = rng.integers(1, 9, n)
        durs[rng.random(n) < 0.3] = 1
        plan = nano.acoustic_plan(weights, ids, durs)
        token_count = max(n - 1, 1)
        frames = int(durs.sum())
        token_pos = np.empty(frames, dtype=np.float32)
        duration_pos = np.empty(frames, dtype=np.float32)
        at = 0
        for ti, d in enumerate(durs.tolist()):
            token_pos[at : at + d] = np.float32(ti) / np.float32(token_count)
            duration_pos[at : at + d] = 0.0 if d == 1 else np.arange(d, dtype=np.float32) / np.float32(d - 1)
            at += d
        assert _bits_equal(plan.frame_feats[1], token_pos), n
        assert _bits_equal(plan.frame_feats[2], duration_pos), n


def test_cached_seed_state_is_not_shared_between_generators():
    a, b = nano_rng.ATenMT19937(99), nano_rng.ATenMT19937(99)
    first = a.uniform(2000)
    assert np.array_equal(b.uniform(2000), first)
    assert np.array_equal(nano_rng.ATenMT19937(99).uniform(2000), first)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")


# ---- 0.5.4: cached noise, buffered iSTFT, cropped decoder window ---------

def test_cached_noise_is_bit_identical_to_a_fresh_generator():
    for seed in (2236265385529901705, 7, 0):
        nano_rng._PREFIXES.clear()
        # Growing, shrinking and repeated sizes, so the prefix is reused,
        # regrown and sliced; sizes with and without a redrawn tail block.
        for frames in (4, 5, 400, 37, 4001, 5000, 400, 16560):
            got = nano_rng.seeded_noise(seed, 4, frames)
            want = nano_rng.seeded_noise_uncached(seed, 4, frames)
            assert _bits_equal(got, want), (seed, frames)
    # The cached arrays are read-only and callers get a private copy.
    first = nano_rng.seeded_noise(5, 4, 100)
    first[:] = 0.0
    assert _bits_equal(nano_rng.seeded_noise(5, 4, 100), nano_rng.seeded_noise_uncached(5, 4, 100))
    assert not nano_rng._PREFIXES[5].normal.flags.writeable
    for bad in (1, 3):  # fewer than 16 values: torch's scalar path, not implemented
        try:
            nano_rng.seeded_noise(5, 4, bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"seeded_noise accepted {4 * bad} values")


def _depthwise_ref(x, dw_w):
    dk = dw_w.shape[1]
    pad = dk // 2
    xp = np.zeros((x.shape[0], x.shape[1] + 2 * pad), dtype=x.dtype)
    xp[:, pad : pad + x.shape[1]] = x
    h = np.zeros_like(x)
    for k in range(dk):
        h += dw_w[:, k : k + 1] * xp[:, k : k + x.shape[1]]
    return h


def test_depthwise_bit_identical_with_shipped_kernels():
    rng = np.random.default_rng(11)
    for lineage, ext in LINEAGES:
        _, meta, front, dec = _lineage(lineage, ext)
        w = nano.NanoWeights(front, dec, meta)
        dim, dk = meta["NANO_DIM"], meta["NANO_DW_KERNEL"]
        for b in range(meta["NANO_BLOCKS"]):
            dw_w = w.dec_f32(f"B{b}_DW_W", dim * dk).reshape(dim, dk)
            dw_b = w.dec_f32(f"B{b}_DW_B", dim)
            for t in (1, 2, 3, 7, 20, 200):
                x = (rng.standard_normal((dim, t)) * 4).astype(np.float32)
                x[rng.random(x.shape) < 0.1] = 0.0
                assert _bits_equal(nano._depthwise(x, dw_w) + dw_b[:, None],
                                   _depthwise_ref(x, dw_w) + dw_b[:, None]), (lineage, b, t)


def test_unit_ramp_is_linspace():
    for n in (1, 2, 3, 73, 207, 1000, 16560):
        want = np.linspace(0.0, 1.0, n, dtype=np.float32) if n > 1 else np.zeros(1, np.float32)
        assert _bits_equal(nano._unit_ramp(n), want), n
        assert not nano._unit_ramp(n).flags.writeable


def test_cropped_decoder_window_matches_uncropped():
    for lineage, ext in LINEAGES:
        fixture, meta, front, dec = _lineage(lineage, ext)
        w = nano.NanoWeights(front, dec, meta)
        ids = np.fromfile(fixture / "r00_ids.bin", dtype=np.int32).astype(np.int64)
        durs = np.fromfile(fixture / "r00_durs.bin", dtype=np.int32).astype(np.int64)
        plan, noise = nano._front_end(w, ids, seed=7, durations=durs, length_scale=1.0)
        total, halo = plan.frames, nano.decoder_halo_frames(meta)
        mel = nano.acoustic_frames(w, plan, 0, total)
        for a, b in ((0, 1), (0, 8), (5, 13), (total // 2, total // 2 + 16), (total - 3, total)):
            lo, hi = max(0, a - halo), min(total, b + halo)
            full = nano.decoder_forward(w, mel[:, lo:hi], noise[:, lo:hi])[:, a - lo : b - lo]
            kept = nano.decoder_forward(w, mel[:, lo:hi], noise[:, lo:hi], keep=(a - lo, b - lo))
            assert kept.shape == full.shape, (lineage, a, b)
            # Measured: at most 1.05e-5 relative over every start of the three
            # fixture rows (float32 rounding through four blocks, amplified by
            # the exp of the magnitude head); the bound is 3x that.
            rel = np.abs(kept - full) / np.maximum(np.abs(full), 1e-3)
            assert rel.max() <= 3e-5, (lineage, a, b, float(rel.max()))
        try:
            nano.decoder_forward(w, mel[:, :10], noise[:, :10], keep=(4, 11))
        except ValueError:
            pass
        else:
            raise AssertionError("keep outside the window was accepted")


def test_streaming_istft_buffers_stay_small_and_match_istft():
    rng = np.random.default_rng(5)
    n_fft, hop, total = 1024, 256, 300
    spec = (rng.standard_normal((513, total)) + 1j * rng.standard_normal((513, total))).astype(np.complex64)
    whole = nano.istft(spec, n_fft, hop)
    ola = nano._StreamingISTFT(n_fft, hop, total)
    pieces, a, size, widest = [], 0, 1, 0
    while a < total:
        b = min(total, a + size)
        pieces.append(ola.push(spec[:, a:b]))
        widest = max(widest, ola.out.size)
        a, size = b, min(64, size * 2)
    assert _bits_equal(np.concatenate(pieces), whole)
    assert widest <= 64 * hop + n_fft, widest
