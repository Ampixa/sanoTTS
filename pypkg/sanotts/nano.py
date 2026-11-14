"""Numpy runtime for the nano voices (heart, heart-nano).

The piperlite voices in `models.py` are a different graph: `duration_conv` /
`token_context` / `piperlite`. The nano lineage is duration student ->
contextual acoustic student -> mel-100 -> a ConvNeXt1D decoder -> iSTFT, with
a noise-fed decoder. `models.py` raises NotImplementedError on it, which is
why heart could not be played from Python.

WEIGHTS
    Read straight from the shipped `front_q8.bin` / `model_q8.bin` using the
    byte offsets in `nano_q8_meta.h`, so a voice package needs no re-export.
    Rows are dequantised to float32 (`w = q * scale`) and the arithmetic runs
    in float.

    That is deliberately NOT what the C runtime does. It also quantises
    activations, one dynamic scale per output column, because an MCU has int8
    kernels and no spare cycles. Reproducing that here would buy nothing: a
    Python caller has floats, and dequantised-weight float math lands closer
    to the PyTorch reference than the device does, not further away. So this
    is gated against the same PyTorch reference the C is gated against, at the
    same 0.98 threshold, rather than against the C's output.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re

import numpy as np

from .nano_rng import seeded_noise

LENGTH_SCALE = 1.0
DC_BLOCK_R = 0.9973
DC_BLOCK_TAPS = 4096


def parse_meta_header(path: Path) -> dict[str, int]:
    """`#define NAME value` -> dict. The header is generated, never hand-edited."""
    out: dict[str, int] = {}
    pattern = re.compile(r"^#define\s+([A-Z0-9_]+)\s+(-?\d+)\s*$")
    for line in path.read_text(encoding="utf-8").splitlines():
        m = pattern.match(line.strip())
        if m:
            out[m.group(1)] = int(m.group(2))
    if "NANO_FRONT_BYTES" not in out:
        raise ValueError(f"{path} does not look like a nano meta header")
    return out


@dataclass
class Q8Layer:
    """One quantised layer: int8 rows, a per-row scale, and a bias.

    `weight` is already dequantised and trimmed to `in_flat`; the exporter pads
    each row out to a 16-byte multiple so the device SIMD kernels can read
    whole groups, and those trailing zeros are shipped bytes, not parameters.
    """
    weight: np.ndarray   # [out, in_flat] float32
    bias: np.ndarray     # [out] float32


class NanoWeights:
    """Dequantised layers, read from the shipped blobs on first use and kept.

    Dequantising is deterministic, so caching changes no output; it only stops
    every utterance -- and every streamed chunk -- from paying for it again.
    Callers must treat the returned arrays as read-only.
    """

    def __init__(self, front: bytes, dec: bytes, meta: dict[str, int]):
        self._front, self._dec, self.m = front, dec, meta
        self._cache: dict[tuple, object] = {}

    def _cached(self, key: tuple, build):
        hit = self._cache.get(key)
        if hit is None:
            hit = build()
            self._cache[key] = hit
        return hit

    def _q8(self, blob: bytes, off_w: int, off_s: int, off_b: int,
            out_ch: int, n16: int, in_flat: int) -> Q8Layer:
        """Read one weight region.

        NANO_WEIGHT_FORMAT selects the row dtype: 0 is int8 with a
        per-output-channel scale, 1 is float32 rows with unit scales (the
        browser build, and the 2.27M `heart` voice, which does not survive
        int8). The region layout is otherwise identical, so both go through
        the same offsets; only the element type and stride differ.
        """
        if self.m.get("NANO_WEIGHT_FORMAT", 0) == 1:
            rows = np.frombuffer(blob, dtype=np.float32, count=out_ch * n16,
                                 offset=off_w).reshape(out_ch, n16)
            w = rows[:, :in_flat].astype(np.float32)
        else:
            q = np.frombuffer(blob, dtype=np.int8, count=out_ch * n16,
                              offset=off_w).reshape(out_ch, n16)
            scale = np.frombuffer(blob, dtype=np.float32, count=out_ch, offset=off_s)
            w = q[:, :in_flat].astype(np.float32) * scale[:, None]
        bias = np.frombuffer(blob, dtype=np.float32, count=out_ch, offset=off_b)
        return Q8Layer(np.ascontiguousarray(w), bias.astype(np.float32).copy())

    def _f32(self, blob: bytes, off: int, count: int) -> np.ndarray:
        return np.frombuffer(blob, dtype=np.float32, count=count, offset=off).copy()

    def _front_q8_uncached(self, name: str, out_ch: int, in_flat: int) -> Q8Layer:
        m = self.m
        return self._q8(self._front, m[f"NOFF_{name}_W8"], m[f"NOFF_{name}_SCALE"],
                        m[f"NOFF_{name}_BIAS"], out_ch, m[f"NANO_{name}_N16"], in_flat)

    def front_q8(self, name, out_ch, in_flat):
        return self._cached(("front_q8", name, out_ch, in_flat), lambda: self._front_q8_uncached(name, out_ch, in_flat))

    def _front_f32_uncached(self, name: str, count: int) -> np.ndarray:
        return self._f32(self._front, self.m[f"NOFF_{name}_F32"], count)

    def front_f32(self, name, count):
        return self._cached(("front_f32", name, count), lambda: self._front_f32_uncached(name, count))

    def _dec_q8_uncached(self, name: str, out_ch: int, in_flat: int) -> Q8Layer:
        m = self.m
        return self._q8(self._dec, m[f"DOFF_{name}_W8"], m[f"DOFF_{name}_SCALE"],
                        m[f"DOFF_{name}_BIAS"], out_ch, m[f"NANO_{name}_N16"], in_flat)

    def dec_q8(self, name, out_ch, in_flat):
        return self._cached(("dec_q8", name, out_ch, in_flat), lambda: self._dec_q8_uncached(name, out_ch, in_flat))

    def _dec_f32_uncached(self, name: str, count: int) -> np.ndarray:
        return self._f32(self._dec, self.m[f"DOFF_{name}_F32"], count)

    def dec_f32(self, name, count):
        return self._cached(("dec_f32", name, count), lambda: self._dec_f32_uncached(name, count))


# ---- primitives ---------------------------------------------------------

def silu(x: np.ndarray) -> np.ndarray:
    """x * sigmoid(x), computed without overflowing.

    The naive 1/(1+exp(-x)) overflows for large negative x. The result is
    still right -- x/inf is 0 -- but it raises a RuntimeWarning on every
    utterance, and a library that cries wolf teaches its callers to ignore
    warnings. Branching on the sign keeps both exponentials bounded.
    """
    # One exp over exp(-|x|) is the same per-element arithmetic as the two
    # masked branches (x / (1 + exp(-x)) for x >= 0, x e / (1 + e) with
    # e = exp(x) below), without the boolean gathers and scatters.
    e = np.exp(-np.abs(x))
    denom = e + 1.0
    return np.where(x >= 0, x / denom, x * e / denom)


def gelu(x: np.ndarray) -> np.ndarray:
    """Exact erf GELU -- what nn.GELU() computes by default, NOT the tanh
    approximation. numpy has no erf, and pulling in scipy for one function
    would add a heavy dependency to a package that advertises numpy only.

    The erf is `_erf` below, which agrees with math.erf to float64 rounding.
    It replaced a np.vectorize(math.erf) that called back into Python once per
    element and was ~90% of the decoder's time."""
    return (0.5 * x * (1.0 + _erf(x.astype(np.float64) / math.sqrt(2.0)))).astype(np.float32)


# erf through a table and a short Taylor series, in float64.
#
# The table holds erf and its derivative at every multiple of 1/1024 on
# [0, 6]; math.erf(6.0) is already exactly 1.0, so larger |x| clamp there. A
# point is taken from its nearest knot x0, so |t| = |x - x0| <= 2**-11, and
# expanded through the fifth-order Taylor term. Every derivative of erf is a
# polynomial in x0 times D = 2/sqrt(pi) * exp(-x0**2), and the first omitted
# term is below 1e-20, far under float64 rounding. The measured agreement with
# math.erf is checked in tests/test_nano_stream.py.
_ERF_STEP = 1.0 / 1024.0
_ERF_KNOTS = 6 * 1024
_ERF_X0 = np.arange(_ERF_KNOTS + 1, dtype=np.float64) * _ERF_STEP
_ERF_TABLE = np.array([math.erf(v) for v in _ERF_X0.tolist()], dtype=np.float64)
_ERF_DERIV = (2.0 / math.sqrt(math.pi)) * np.exp(-_ERF_X0 * _ERF_X0)


def _erf(x: np.ndarray) -> np.ndarray:
    """float64 erf, elementwise. NaN in, NaN out; +-inf go to +-1."""
    x = np.asarray(x, dtype=np.float64)
    a = np.abs(x)
    nan = np.isnan(a)
    has_nan = bool(nan.any())
    if has_nan:
        a[nan] = 0.0
    np.minimum(a, _ERF_KNOTS * _ERF_STEP, out=a)
    k = np.rint(a * (1.0 / _ERF_STEP)).astype(np.intp)
    x0 = _ERF_X0[k]
    t = a - x0
    x2 = x0 * x0
    # erf(x0 + t) = erf(x0) + D * (t + c2 t^2 + c3 t^3 + c4 t^4 + c5 t^5).
    # The 2nd..5th derivatives are D times -2x, 4x^2 - 2, 12x - 8x^3 and
    # 16x^4 - 48x^2 + 12; each c_n is that polynomial over n!. The in-place
    # steps below are these same operations in the same order.
    c3 = 4.0 * x2
    c3 -= 2.0
    c3 /= 6.0
    c4 = 8.0 * x0
    c4 *= x2
    np.subtract(12.0 * x0, c4, out=c4)
    c4 /= 24.0
    c5 = 16.0 * x2
    c5 *= x2
    c5 -= 48.0 * x2
    c5 += 12.0
    c5 /= 120.0
    poly = t * c5          # t * c5
    poly += c4             # c4 + t*c5
    poly *= t
    poly += c3
    poly *= t
    np.subtract(poly, x0, out=poly)   # c2 + (...) with c2 = -x0
    poly *= t
    poly += 1.0
    poly *= t
    poly *= _ERF_DERIV[k]
    poly += _ERF_TABLE[k]
    out = np.copysign(poly, x, out=poly)
    if has_nan:
        out[nan] = np.nan
    return out


def conv1d(x: np.ndarray, w: np.ndarray, b: np.ndarray, kernel: int) -> np.ndarray:
    """[C_in, T] -> [C_out, T], 'same' padding, weight rows [C_out, C_in*K].

    Rows are laid out exactly as the device kernel consumes them, which is
    torch's [C_out, C_in, K] flattened, so the gather below must walk the
    kernel axis fastest.
    """
    c_in, t = x.shape
    pad = kernel // 2
    # Row k of each input channel is x shifted by k - pad, zero outside it:
    # the same matrix np.pad + slicing built, written directly.
    gathered = np.zeros((c_in * kernel, t), dtype=np.float32)
    for k in range(kernel):
        shift = k - pad
        if abs(shift) >= t:
            continue                     # this tap only ever sees padding
        if shift < 0:
            gathered[k::kernel, -shift:] = x[:, : t + shift]
        elif shift > 0:
            gathered[k::kernel, : t - shift] = x[:, shift:]
        else:
            gathered[k::kernel] = x
    return (w @ gathered + b[:, None]).astype(np.float32)


def residual_block(x: np.ndarray, c0: Q8Layer, c1: Q8Layer, scale: float,
                   kernel: int) -> np.ndarray:
    """x + scale * conv1(silu(conv0(x))) -- the front end's ResidualConvBlock.

    The training code multiplies by a padding mask on both sides; inference is
    a single unpadded sequence, so the mask is all ones and drops out.
    """
    h = conv1d(x, c0.weight, c0.bias, kernel)
    h = silu(h)
    h = conv1d(h, c1.weight, c1.bias, kernel)
    return (x + scale * h).astype(np.float32)


# The decoder's norms are built by make_norm(), which asks for eps=1e-6 --
# NOT torch's 1e-5 default. The difference looks negligible and is not: it
# compounds through four blocks and is then amplified by the exp() in the
# magnitude head, which cost 0.06 of correlation and a third of the output
# amplitude before it was found.
LAYER_NORM_EPS = 1e-6


def layer_norm(x: np.ndarray, w: np.ndarray, b: np.ndarray,
               eps: float = LAYER_NORM_EPS) -> np.ndarray:
    """Over the channel axis of [C, T], as torch does on channel-last."""
    # x.mean / x.var as numpy's _mean/_var compute them (add.reduce, then
    # true_divide by an intp count with unsafe casting), minus their wrapper
    # overhead; bit-identical, checked in tests/test_nano_stream.py.
    count = np.intp(x.shape[0])
    mu = np.add.reduce(x, axis=0, keepdims=True)
    np.true_divide(mu, count, out=mu, casting="unsafe")
    dev = x - mu
    var = np.add.reduce(np.multiply(dev, dev), axis=0, keepdims=True)
    np.true_divide(var, count, out=var, casting="unsafe")
    return (dev / np.sqrt(var + eps) * w[:, None] + b[:, None]).astype(np.float32)


# ---- stages -------------------------------------------------------------

def duration_forward(w: NanoWeights, ids: np.ndarray,
                     length_scale: float = 1.0) -> np.ndarray:
    """Phoneme ids -> integer frame counts, one per token.

    `length_scale` > 1 slows speech down, < 1 speeds it up, applied before
    rounding and the per-token clamp exactly as the C runtime does.
    """
    m = w.m
    H = m["NANO_DUR_HIDDEN"]
    K = m["NANO_DUR_KERNEL"]
    n = int(ids.size)

    emb = w.front_f32("DUR_EMB", m["NANO_VOCAB"] * H).reshape(m["NANO_VOCAB"], H)
    x = emb[ids].T.astype(np.float32)                       # [H, n]

    # positions, length hint and valid hint, exactly as DurationStudent builds
    # them. Inference is one unpadded sequence, so valid_hint is all ones and
    # lengths == n.
    positions = np.linspace(0.0, 1.0, n, dtype=np.float32) if n > 1 else np.zeros(1, np.float32)
    length_hint = np.float32(math.log1p(max(n, 1)) / math.log1p(float(m["NANO_DUR_MAX_TOKENS"])))
    feats = np.stack([positions,
                      np.full(n, length_hint, dtype=np.float32),
                      np.ones(n, dtype=np.float32)])
    x = np.concatenate([x, feats], axis=0)                  # [H+3, n]

    proj = w.front_q8("DUR_PROJ", H, H + 3)
    x = conv1d(x, proj.weight, proj.bias, 1)
    for b in range(m["NANO_DUR_DEPTH"]):
        c0 = w.front_q8(f"DUR_B{b}_C0", H, H * K)
        c1 = w.front_q8(f"DUR_B{b}_C1", H, H * K)
        scale = float(w.front_f32(f"DUR_B{b}_SCALE", 1)[0])
        x = residual_block(x, c0, c1, scale, K)

    out = w.front_q8("DUR_OUT", 1, H)
    log_d = conv1d(x, out.weight, out.bias, 1)[0]
    d = np.round(np.maximum(np.exp(log_d), 1.0) * float(length_scale))
    return np.clip(d, 1, m["NANO_DUR_MAX_DURATION"]).astype(np.int32)


@dataclass
class AcousticPlan:
    """The utterance-global half of the acoustic model, computed once.

    `token_x` is the token stage's output [H, n]; `frame_token` maps each of
    the T frames to its token; `frame_feats` holds the three per-frame
    features [3, T] (frame_pos, token_pos, duration_pos). Everything after
    this is local in time, so any frame range can be produced from it.
    """
    token_x: np.ndarray
    frame_token: np.ndarray
    frame_feats: np.ndarray

    @property
    def frames(self) -> int:
        return int(self.frame_token.size)


def acoustic_plan(w: NanoWeights, ids: np.ndarray, durations: np.ndarray) -> AcousticPlan:
    """Token stage and frame features of ContextualLatentStudent."""
    m = w.m
    H = m["NANO_AC_HIDDEN"]
    K = m["NANO_AC_KERNEL"]
    n = int(ids.size)

    emb = w.front_f32("AC_EMB", m["NANO_VOCAB"] * H).reshape(m["NANO_VOCAB"], H)
    token_x = emb[ids].T.astype(np.float32)                 # [H, n]

    token_count = max(n - 1, 1)
    token_pos_tok = (np.linspace(0.0, 1.0, n, dtype=np.float32) if n > 1
                     else np.zeros(1, np.float32))
    dur_f = durations.astype(np.float32)
    max_dur = max(float(dur_f.max()), 1.0)
    duration_hint = (np.log1p(dur_f) / np.log1p(max_dur)).astype(np.float32)

    token_x = np.concatenate([token_x, np.stack([token_pos_tok, duration_hint])], axis=0)
    tproj = w.front_q8("AC_TPROJ", H, H + 2)
    token_x = conv1d(token_x, tproj.weight, tproj.bias, 1)
    for b in range(m["NANO_AC_TOKEN_DEPTH"]):
        c0 = w.front_q8(f"AC_TB{b}_C0", H, H * K)
        c1 = w.front_q8(f"AC_TB{b}_C1", H, H * K)
        scale = float(w.front_f32(f"AC_TB{b}_SCALE", 1)[0])
        token_x = residual_block(token_x, c0, c1, scale, K)

    # Each contextual token state is repeated across its own frames, and the
    # three frame features are built the same way expand_features does --
    # note token_pos divides by (n_tokens - 1) while duration_pos divides by
    # (duration - 1), and a duration of 1 contributes a single 0.0 rather
    # than a division.
    frame_token = np.repeat(np.arange(n), durations)
    frames = int(frame_token.size)
    frame_pos = (np.linspace(0.0, 1.0, frames, dtype=np.float32) if frames > 1
                 else np.zeros(1, np.float32))
    # Per token ti with d frames: token_pos = ti / token_count on each of
    # them, and duration_pos = j / (d - 1) for j in 0..d-1 (0.0 when d == 1),
    # each a single float32 division as before, done for all frames at once.
    token_pos = (np.arange(n, dtype=np.float32) / np.float32(token_count))[frame_token]
    starts = np.cumsum(durations) - durations
    within = (np.arange(frames) - starts[frame_token]).astype(np.float32)
    span = (durations - 1).astype(np.float32)[frame_token]
    duration_pos = np.zeros(frames, dtype=np.float32)
    multi = span > 0
    duration_pos[multi] = within[multi] / span[multi]
    return AcousticPlan(token_x, frame_token, np.stack([frame_pos, token_pos, duration_pos]))


def acoustic_halo_frames(meta: dict[str, int]) -> int:
    """Frames of context each side that make `acoustic_frames` exact: the
    frame stage is NANO_AC_DEPTH residual blocks of two convolutions each."""
    return meta["NANO_AC_DEPTH"] * 2 * (meta["NANO_AC_KERNEL"] // 2)


def acoustic_frames(w: NanoWeights, plan: AcousticPlan, lo: int, hi: int) -> np.ndarray:
    """The frame stage over frames [lo, hi) only -> mel-100 [100, hi - lo].

    Exact for the frames at least `acoustic_halo_frames` inside the range
    (or at the utterance's own ends); callers widen the range by that much
    and crop. [0, T) is the whole-utterance pass.
    """
    m = w.m
    H = m["NANO_AC_HIDDEN"]
    K = m["NANO_AC_KERNEL"]
    x = np.concatenate([plan.token_x[:, plan.frame_token[lo:hi]], plan.frame_feats[:, lo:hi]], axis=0)
    fproj = w.front_q8("AC_FPROJ", H, H + 3)
    x = conv1d(x, fproj.weight, fproj.bias, 1)
    for b in range(m["NANO_AC_DEPTH"]):
        c0 = w.front_q8(f"AC_FB{b}_C0", H, H * K)
        c1 = w.front_q8(f"AC_FB{b}_C1", H, H * K)
        scale = float(w.front_f32(f"AC_FB{b}_SCALE", 1)[0])
        x = residual_block(x, c0, c1, scale, K)

    out = w.front_q8("AC_OUT", m["NANO_MELS"], H)
    return conv1d(x, out.weight, out.bias, 1)


def acoustic_forward(w: NanoWeights, ids: np.ndarray, durations: np.ndarray) -> np.ndarray:
    """ids + durations -> mel-100 [100, T]. ContextualLatentStudent."""
    plan = acoustic_plan(w, ids, durations)
    return acoustic_frames(w, plan, 0, plan.frames)


def decoder_forward(w: NanoWeights, mel: np.ndarray, noise: np.ndarray) -> np.ndarray:
    """mel-100 [100, T] + noise [4, T] -> complex spectrum [513, T]."""
    m = w.m
    dim, ek, dk = m["NANO_DIM"], m["NANO_EMBED_KERNEL"], m["NANO_DW_KERNEL"]

    embed = w.dec_q8("EMBED", dim, m["NANO_MELS"] * ek)
    x = conv1d(mel, embed.weight, embed.bias, ek)
    adapter = w.dec_q8("NOISE", dim, m["NANO_NOISE_CH"] * ek)
    x = x + conv1d(noise, adapter.weight, adapter.bias, ek)

    x = layer_norm(x, w.dec_f32("NORM_W", dim), w.dec_f32("NORM_B", dim))

    for b in range(m["NANO_BLOCKS"]):
        residual = x
        dw_w = w.dec_f32(f"B{b}_DW_W", dim * dk).reshape(dim, dk)
        dw_b = w.dec_f32(f"B{b}_DW_B", dim)
        # depthwise: one kernel per channel, groups == dim
        pad = dk // 2
        xp = np.zeros((x.shape[0], x.shape[1] + 2 * pad), dtype=x.dtype)
        xp[:, pad : pad + x.shape[1]] = x
        h = np.zeros_like(x)
        for k in range(dk):
            h += dw_w[:, k : k + 1] * xp[:, k : k + x.shape[1]]
        h = h + dw_b[:, None]
        h = layer_norm(h, w.dec_f32(f"B{b}_NORM_W", dim), w.dec_f32(f"B{b}_NORM_B", dim))
        pw0 = w.dec_q8(f"B{b}_PW0", m["NANO_PW_HIDDEN"], dim)
        h = pw0.weight @ h + pw0.bias[:, None]
        h = gelu(h)
        pw1 = w.dec_q8(f"B{b}_PW1", dim, m["NANO_PW_HIDDEN"])
        h = pw1.weight @ h + pw1.bias[:, None]
        h = h * w.dec_f32(f"B{b}_GAMMA", dim)[:, None]
        x = (residual + h).astype(np.float32)

    x = layer_norm(x, w.dec_f32("FNORM_W", dim), w.dec_f32("FNORM_B", dim))
    head = w.dec_q8("HEAD", m["NANO_HEAD_OUT"], dim)
    out = head.weight @ x + head.bias[:, None]              # [1026, T]

    bins = m["NANO_BINS"]
    mag, phase = out[:bins], out[bins:]
    mag = np.exp(mag)
    # Bin 0 and Nyquist stay zeroed: the mag*(cos,sin) parametrisation
    # phase-collapses at bin 0, which is where the frame-DC artefact came from.
    mag[0] = 0.0
    mag[-1] = 0.0
    mag = np.clip(mag, None, 1e2)
    return mag * (np.cos(phase) + 1j * np.sin(phase))


def istft(spec: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    """Inverse STFT with a Hann window and centre padding, matching torch.

    torch.istft divides by the summed squared window rather than assuming the
    COLA condition, so a frame the window sums to zero over is not silently
    scaled wrong; the same normalisation is applied here.
    """
    bins, frames = spec.shape
    window = np.hanning(n_fft + 1)[:n_fft].astype(np.float64)  # periodic, as torch
    length = (frames - 1) * hop + n_fft
    out = np.zeros(length, dtype=np.float64)
    norm = np.zeros(length, dtype=np.float64)
    frames_td = np.fft.irfft(spec, n=n_fft, axis=0)            # [n_fft, frames]
    wsq = window * window
    for i in range(frames):
        start = i * hop
        out[start : start + n_fft] += frames_td[:, i] * window
        norm[start : start + n_fft] += wsq
    pad = n_fft // 2
    out = out[pad : length - pad]
    norm = norm[pad : length - pad]
    nonzero = norm > 1e-11
    out[nonzero] /= norm[nonzero]
    return out.astype(np.float32)


def dc_block(wav: np.ndarray) -> np.ndarray:
    """H(z) = (1 - z^-1)/(1 - R z^-1) as a truncated FIR, via FFT convolution."""
    impulse = np.zeros(DC_BLOCK_TAPS, dtype=np.float64)
    impulse[0] = 1.0
    n = np.arange(1, DC_BLOCK_TAPS, dtype=np.float64)
    impulse[1:] = (DC_BLOCK_R - 1.0) * DC_BLOCK_R ** (n - 1.0)
    samples = wav.size
    n_lin = samples + DC_BLOCK_TAPS - 1
    n_fft = 1 << (n_lin - 1).bit_length()
    filtered = np.fft.irfft(np.fft.rfft(wav.astype(np.float64), n_fft)
                            * np.fft.rfft(impulse, n_fft), n_fft)
    return filtered[:samples].astype(np.float32)


def _front_end(w: NanoWeights, ids: np.ndarray, *, seed: int,
               durations: np.ndarray | None, length_scale: float):
    """ids -> (AcousticPlan, noise [4, T]): the utterance-global stages.

    Durations, the acoustic token stage and the frame features depend on the
    whole utterance (frame positions are normalised by the total frame
    count), and so does the noise (frame t of channel c is draw c*T + t), so
    these always run over all of it. They are cheap; the acoustic frame
    stage and the decoder, which are not, are local in time.
    """
    ids = np.asarray(ids, dtype=np.int64)
    if ids.size == 0:
        raise ValueError("nano: empty phoneme id sequence")
    if durations is None:
        durations = duration_forward(w, ids, length_scale=length_scale)
    durations = np.asarray(durations, dtype=np.int64)
    if durations.shape != ids.shape:
        raise ValueError(f"nano: {durations.size} durations for {ids.size} ids")
    plan = acoustic_plan(w, ids, durations)
    noise = seeded_noise(seed, w.m["NANO_NOISE_CH"], plan.frames)
    return plan, noise


def synthesize_ids(front: bytes, dec: bytes, meta: dict[str, int],
                   ids: np.ndarray, *, seed: int,
                   durations: np.ndarray | None = None,
                   length_scale: float = 1.0,
                   weights: NanoWeights | None = None) -> np.ndarray:
    """Phoneme ids -> waveform. `durations` overrides the duration model.

    `weights` is a NanoWeights already built over the same blobs; passing one
    reuses its dequantised layers instead of rebuilding them.
    """
    w = weights if weights is not None else NanoWeights(front, dec, meta)
    plan, noise = _front_end(w, ids, seed=seed, durations=durations, length_scale=length_scale)
    spec = decoder_forward(w, acoustic_frames(w, plan, 0, plan.frames), noise)
    return dc_block(istft(spec, meta["NANO_N_FFT"], meta["NANO_HOP"]))


# ---- streaming ----------------------------------------------------------
#
# The acoustic frame stage is local in time (NANO_AC_DEPTH blocks of two
# NANO_AC_KERNEL convs over per-frame inputs), and so is the decoder: an embed conv of NANO_EMBED_KERNEL, then
# NANO_BLOCKS depthwise convs of NANO_DW_KERNEL, with every norm and pointwise
# layer acting on one frame. A spectrum column therefore depends only on the
# mel/noise frames within `decoder_halo_frames` of it, so decoding a window
# that reaches that far past each side of a chunk gives the chunk's columns
# exactly as the whole-utterance pass does. The iSTFT is an overlap-add, so a
# sample is final once every frame overlapping it has been added, and the DC
# blocker is a causal FIR, so it only needs the samples before the chunk.
#
# What is not identical is floating-point association: a matrix product over
# a 40-column window and over a 700-column one may round differently in the
# last bit. Streamed audio therefore matches synthesize_ids() to float32
# rounding, not bit for bit; tests/test_nano_stream.py measures it.

def decoder_halo_frames(meta: dict[str, int]) -> int:
    """Frames of context each side of a chunk that make its decoder output exact."""
    return meta["NANO_EMBED_KERNEL"] // 2 + meta["NANO_BLOCKS"] * (meta["NANO_DW_KERNEL"] // 2)


_DC_IMPULSE_FFT: dict[tuple[int, int], np.ndarray] = {}


def _dc_impulse_fft(taps: int, n_fft: int) -> np.ndarray:
    """rfft of the first `taps` DC-block taps at size n_fft, cached."""
    key = (taps, n_fft)
    spectrum = _DC_IMPULSE_FFT.get(key)
    if spectrum is None:
        impulse = np.zeros(taps, dtype=np.float64)
        impulse[0] = 1.0
        n = np.arange(1, taps, dtype=np.float64)
        impulse[1:] = (DC_BLOCK_R - 1.0) * DC_BLOCK_R ** (n - 1.0)
        if len(_DC_IMPULSE_FFT) >= 64:
            _DC_IMPULSE_FFT.clear()
        spectrum = _DC_IMPULSE_FFT[key] = np.fft.rfft(impulse, n_fft)
    return spectrum


class _StreamingDCBlock:
    """`dc_block` applied chunk by chunk.

    The filter is the same truncated FIR; each chunk is convolved together
    with the last DC_BLOCK_TAPS - 1 input samples before it, which is all of
    the past a causal FIR of that length can see.
    """

    def __init__(self) -> None:
        # Past input samples, at most DC_BLOCK_TAPS - 1; before the first
        # chunk there are none, and the filter sees zeros there anyway.
        self._history = np.zeros(0, dtype=np.float64)

    def __call__(self, chunk: np.ndarray) -> np.ndarray:
        if chunk.size == 0:
            return np.zeros(0, dtype=np.float32)
        segment = np.concatenate([self._history, chunk.astype(np.float64)])
        # Taps past the start of the signal only ever meet zeros, so a short
        # segment needs only its own length in taps.
        taps = min(DC_BLOCK_TAPS, segment.size)
        n_lin = segment.size + taps - 1
        n_fft = 1 << (n_lin - 1).bit_length()
        filtered = np.fft.irfft(np.fft.rfft(segment, n_fft) * _dc_impulse_fft(taps, n_fft), n_fft)
        out = filtered[self._history.size : self._history.size + chunk.size]
        self._history = segment[-(DC_BLOCK_TAPS - 1):]
        return out.astype(np.float32)


class _StreamingISTFT:
    """`istft` as an incremental overlap-add over consecutive spectrum columns.

    Frames are added in order, the same order `istft` adds them in, so every
    sample sees the same float64 sums. A sample is emitted once no later frame
    can reach it.
    """

    def __init__(self, n_fft: int, hop: int, total_frames: int) -> None:
        self.n_fft, self.hop, self.total = n_fft, hop, total_frames
        self.window = np.hanning(n_fft + 1)[:n_fft].astype(np.float64)
        self.wsq = self.window * self.window
        self.length = (total_frames - 1) * hop + n_fft      # padded length
        self.pad = n_fft // 2
        self.out = np.zeros(self.length, dtype=np.float64)
        self.norm = np.zeros(self.length, dtype=np.float64)
        self.added = 0          # spectrum frames added so far
        self.emitted = self.pad  # padded position up to which samples were returned

    def push(self, spec: np.ndarray) -> np.ndarray:
        """Add the next columns; return the samples that are now final."""
        frames_td = np.fft.irfft(spec, n=self.n_fft, axis=0)
        for j in range(frames_td.shape[1]):
            start = (self.added + j) * self.hop
            self.out[start : start + self.n_fft] += frames_td[:, j] * self.window
            self.norm[start : start + self.n_fft] += self.wsq
        self.added += frames_td.shape[1]
        if self.added > self.total:
            raise RuntimeError(f"istft: {self.added} frames pushed, {self.total} expected")
        # The next frame starts at added * hop, so everything before it is
        # final; once the last frame is in, everything up to the end is.
        ready = self.length - self.pad if self.added == self.total else self.added * self.hop
        lo = self.emitted
        hi = min(ready, self.length - self.pad)
        if hi <= lo:
            return np.zeros(0, dtype=np.float32)
        out = self.out[lo:hi].copy()
        norm = self.norm[lo:hi]
        nonzero = norm > 1e-11
        out[nonzero] /= norm[nonzero]
        self.emitted = hi
        return out.astype(np.float32)


def stream_ids(front: bytes, dec: bytes, meta: dict[str, int],
               ids: np.ndarray, *, seed: int,
               durations: np.ndarray | None = None,
               length_scale: float = 1.0,
               weights: NanoWeights | None = None,
               first_chunk_frames: int = 8,
               chunk_frames: int = 128):
    """Phoneme ids -> waveform, yielded in pieces as they are decoded.

    The concatenation of the pieces is `synthesize_ids(...)` for the same
    arguments, to float32 rounding. The first piece is decoded from
    `first_chunk_frames` spectrum frames, so it arrives after the
    utterance-global stages plus a small acoustic and decoder window; later
    pieces grow by doubling up to `chunk_frames`, which keeps the halo
    recomputation small.
    Every yielded array is non-empty float32 (not yet clipped).
    """
    if first_chunk_frames < 1 or chunk_frames < first_chunk_frames:
        raise ValueError(
            f"need 1 <= first_chunk_frames <= chunk_frames, got {first_chunk_frames}, {chunk_frames}")
    w = weights if weights is not None else NanoWeights(front, dec, meta)
    plan, noise = _front_end(w, ids, seed=seed, durations=durations, length_scale=length_scale)
    total = plan.frames
    halo = decoder_halo_frames(meta)
    ac_halo = acoustic_halo_frames(meta)
    ola = _StreamingISTFT(meta["NANO_N_FFT"], meta["NANO_HOP"], total)
    dc = _StreamingDCBlock()
    a, size = 0, first_chunk_frames
    while a < total:
        b = min(total, a + size)
        # decoder window [lo, hi), and the wider acoustic window that makes
        # every mel frame in it exact
        lo, hi = max(0, a - halo), min(total, b + halo)
        alo, ahi = max(0, lo - ac_halo), min(total, hi + ac_halo)
        mel = acoustic_frames(w, plan, alo, ahi)[:, lo - alo : hi - alo]
        spec = decoder_forward(w, mel, noise[:, lo:hi])[:, a - lo : b - lo]
        piece = dc(ola.push(spec))
        if piece.size:
            yield piece
        a, size = b, min(chunk_frames, size * 2)
    if ola.emitted != ola.length - ola.pad:
        raise RuntimeError(f"istft: emitted to {ola.emitted}, expected {ola.length - ola.pad}")
