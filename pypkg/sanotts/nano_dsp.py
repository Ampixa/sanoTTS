"""
Harmonic-plus-noise ("DSP") decoder for local nano voices: mel-100 -> audio.

A voice whose meta.json names a `dsp_decoder` file (an .npz of float weights) is decoded here instead of by
the neural decoder in nano.py; the neural decoder blob still ships in the package, so older sanotts versions
play that voice with it. The network predicts per-frame controls only (pitch, voicing, a harmonic and a noise
envelope); fixed DSP renders the waveform: up to 80 sinusoids locked to the pitch (running-sum phase,
per-frame minimum-phase harmonic phases) plus noise shaped by the noise envelope and overlap-added by the same
iSTFT nano.py uses.

Contract: mel frames [100, T] -> (T - 1) * 256 samples at 24 kHz, the package's length convention.
Streaming: `DspStream.push(mel_frames, final)` returns the samples that are final. A frame's controls need
LEFT_HALO frames before it and LOOKAHEAD after it; replicate padding is applied only at the true utterance
edges, as in the whole pass, so the chunks concatenate to `render()` to float rounding (tests/test_nano_dsp.py).
The noise is drawn per frame from a counter-based hash, so it does not depend on the chunking either.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

import io

from . import nano as _nano  # the iSTFT (matches torch.istft) and its streaming form

SR, HOP, N_FFT = 24000, 256, 1024
N_HARM, HARM_BANDS, NOISE_BANDS = 80, 40, 24
F0_REF, F0_SCALE, MEL_MEAN, MEL_STD = 200.0, 0.5, -4.0, 4.0
N_BINS = N_FFT // 2 + 1
TABLE_LINEAR = False      # 1024-point cubic lookup (a 2048-point linear table was slower: bigger FFT)
TABLE_DTYPE = np.float32  # tables and harmonic arithmetic in float32 (memory-bound); phase stays float64
TABLE = 1024  # wavetable points per period, read with cubic (Catmull-Rom) interpolation: max error 1.1e-4 of peak
              # against the exact harmonic sum on real speech (table_acc.py), the same as 4096 points read linearly


def _mel_centres(n, fmax=SR / 2):
    mmax = 2595.0 * math.log10(1.0 + fmax / 700.0)
    m = np.linspace(0.0, mmax, n)
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def _interp_bands(log_env, freqs, centres):
    """log_env [..., B], freqs [..., K] -> [..., K], linear in mel (dspdec.interp_bands)."""
    n = centres.size
    mel_c = 2595.0 * np.log10(1.0 + centres / 700.0)
    mel_f = 2595.0 * np.log10(1.0 + np.maximum(freqs, 0.0) / 700.0)
    pos = np.clip(mel_f / mel_c[-1] * (n - 1), 0.0, n - 1)
    lo = np.minimum(np.floor(pos).astype(np.int64), n - 2)
    a = pos - lo
    v_lo = np.take_along_axis(log_env, lo, axis=-1)
    v_hi = np.take_along_axis(log_env, lo + 1, axis=-1)
    return v_lo * (1.0 - a) + v_hi * a


_ERF_H = 1e-4
_ERF_VAL = np.array([math.erf(v) for v in np.arange(-60000, 60002) * _ERF_H])  # [-6, 6 + h]


def _gelu(x):
    """Exact-erf GELU (torch's default) via a uniform erf table with direct indexing (no search): linear
    interpolation error < 2e-9; erf is +-1 to double precision beyond |6|."""
    u = np.clip(x * (1.0 / math.sqrt(2.0)), -6.0, 6.0) * (1.0 / _ERF_H) + 60000.0
    i = u.astype(np.int64)
    f = u - i
    return 0.5 * x * (1.0 + _ERF_VAL[i] * (1.0 - f) + _ERF_VAL[i + 1] * f)


def _layernorm(x, w, b, eps=1e-5):  # x [T, C]
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


@dataclass
class DspWeights:
    inp_w: np.ndarray   # [C, 100, k]
    inp_b: np.ndarray
    in_left: int
    in_right: int
    blocks: list        # dicts: dw_w [C,1,k], dw_b, n_w, n_b, p1_w [2C, C], p1_b, p2_w [C, 2C], p2_b, gamma, left, right
    norm_w: np.ndarray
    norm_b: np.ndarray
    head_w: np.ndarray  # [66, C]
    head_b: np.ndarray
    theta: np.ndarray   # [80]
    minphase: bool = False  # harmonic phases = min phase of each frame's envelope + theta (dspdec --minphase)
    harm_bands: int = HARM_BANDS
    noise_bands: int = NOISE_BANDS
    noise_fft: int = N_FFT          # noise synthesis FFT size (hop 256); dspdec --noise-fft
    mp_noclamp: bool = False        # min-phase from the unclamped envelope (exact matrix path); --minphase-noclamp

    @property
    def left_halo(self):
        return self.in_left + sum(b["left"] for b in self.blocks)

    @property
    def lookahead(self):
        return self.in_right + sum(b["right"] for b in self.blocks)

    def save_npz(self, path):
        arrays = {"inp_w": self.inp_w, "inp_b": self.inp_b, "norm_w": self.norm_w, "norm_b": self.norm_b,
                  "head_w": self.head_w, "head_b": self.head_b, "theta": self.theta,
                  "minphase": np.array(int(self.minphase)), "bands": np.array([self.harm_bands, self.noise_bands]),
                  "noise_fft": np.array(self.noise_fft), "mp_noclamp": np.array(int(self.mp_noclamp)),
                  "pads": np.array([self.in_left, self.in_right] + [v for b in self.blocks for v in (b["left"], b["right"])])}
        for i, b in enumerate(self.blocks):
            for k in ("dw_w", "dw_b", "n_w", "n_b", "p1_w", "p1_b", "p2_w", "p2_b", "gamma"):
                arrays[f"b{i}_{k}"] = b[k]
        # float32 on disk (half the size); everything is computed in float64 after loading
        np.savez(path, **{k: (v.astype(np.float32) if v.dtype.kind == "f" else v) for k, v in arrays.items()})

    @classmethod
    def from_bytes(cls, data: bytes):
        return cls.load_npz(io.BytesIO(data))

    @classmethod
    def load_npz(cls, path):
        z = np.load(path, allow_pickle=False)
        z = {k: (z[k].astype(np.float64) if z[k].dtype.kind == "f" else z[k]) for k in z.files}
        pads = z["pads"].tolist()
        n = (len(pads) - 2) // 2
        blocks = [{k: z[f"b{i}_{k}"] for k in ("dw_w", "dw_b", "n_w", "n_b", "p1_w", "p1_b", "p2_w", "p2_b", "gamma")}
                  | {"left": pads[2 + 2 * i], "right": pads[3 + 2 * i]} for i in range(n)]
        return cls(z["inp_w"], z["inp_b"], pads[0], pads[1], blocks, z["norm_w"], z["norm_b"], z["head_w"],
                   z["head_b"], z["theta"], bool(int(z["minphase"])) if "minphase" in z else False,
                   *(int(v) for v in z["bands"]) if "bands" in z else (HARM_BANDS, NOISE_BANDS),
                   int(z["noise_fft"]) if "noise_fft" in z else N_FFT,
                   bool(int(z["mp_noclamp"])) if "mp_noclamp" in z else False)



def _splitmix64(x):
    """splitmix64 finaliser on uint64 arrays (wrapping arithmetic)."""
    with np.errstate(over="ignore"):
        z = x + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def _conv_valid(x, w, b, groups=1):
    """x [T, Cin], w [Cout, Cin/groups, k] -> [T - k + 1, Cout] (cross-correlation, as torch)."""
    k = w.shape[-1]
    t = x.shape[0] - k + 1
    if groups == 1:
        cols = np.concatenate([x[j:j + t] for j in range(k)], axis=1)  # [t, k * Cin], tap-major
        wm = w.transpose(2, 1, 0).reshape(k * w.shape[1], w.shape[0])   # [k * Cin, Cout]
        return cols @ wm + b
    out = np.zeros((t, w.shape[0]))
    for j in range(k):
        out += x[j:j + t] * w[:, 0, j]
    return out + b


def _pad_edges(x, left, right, at_start, at_end):
    """Replicate-pad only at true utterance edges; elsewhere the window already carries real context."""
    if at_start and left:
        x = np.concatenate([np.repeat(x[:1], left, 0), x], 0)
    if at_end and right:
        x = np.concatenate([x, np.repeat(x[-1:], right, 0)], 0)
    return x


def controls(w: DspWeights, mel_win: np.ndarray, at_start: bool, at_end: bool) -> np.ndarray:
    """mel_win [100, n] -> head outputs [n - (halo not covered by edges), 66].

    The window starts at frame lo and ends at hi. Each layer drops its left/right padding from the window
    unless that side is a true utterance edge, where it replicate-pads like the whole pass.
    """
    x = ((mel_win.T.astype(np.float64)) - MEL_MEAN) / MEL_STD  # [n, 100]
    x = _conv_valid(_pad_edges(x, w.in_left, w.in_right, at_start, at_end), w.inp_w, w.inp_b)
    for b in w.blocks:
        k = b["dw_w"].shape[-1]
        xp = _pad_edges(x, b["left"], b["right"], at_start, at_end)
        y = _conv_valid(xp, b["dw_w"], b["dw_b"], groups=x.shape[1])
        # the residual input is the centre part of x aligned with y
        lo = 0 if at_start else b["left"]
        hi = x.shape[0] if at_end else x.shape[0] - b["right"]
        res = x[lo:hi]
        y = _layernorm(y, b["n_w"], b["n_b"])
        y = _gelu(y @ b["p1_w"].T + b["p1_b"]) @ b["p2_w"].T + b["p2_b"]
        x = res + b["gamma"] * y
    return _layernorm(x, w.norm_w, w.norm_b) @ w.head_w.T + w.head_b


_CACHE: dict = {}
WHITE_CACHE_FRAMES = 4096  # noise of the first 4096 frames (~44 s) is precomputed per seed (complex64, ~17 MB): it
                           # does not depend on the text; later frames are drawn on demand


def _band_interp_matrix(bands, n_fft=N_FFT):
    """[bands, n_fft/2+1]: linear map from a band log-envelope to its value at the bin frequencies."""
    key = ("bim", bands, n_fft)
    if key not in _CACHE:
        nb = n_fft // 2 + 1
        eye = np.eye(bands)
        _CACHE[key] = _interp_bands(eye[:, None, :].repeat(nb, 1),
                                    np.broadcast_to(np.arange(nb) * SR / n_fft, (bands, nb))[..., None],
                                    _mel_centres(bands))[..., 0]
    return _CACHE[key]


def _minimum_phase(log_amp):
    """[t, 513] log magnitude -> [t, 513] minimum-phase phase response (dspdec.minimum_phase); keeps the input dtype."""
    c = np.fft.irfft(log_amp, n=N_FFT, axis=-1)
    w = np.zeros(N_FFT, dtype=c.dtype)
    w[0] = 1.0
    w[1:N_FFT // 2] = 2.0
    w[N_FFT // 2] = 1.0
    return np.fft.rfft(c * w, n=N_FFT, axis=-1).imag


def _minphase_matrix(bands=HARM_BANDS):
    """[40, 513]: minimum phase is linear in the log magnitude (a Hilbert transform), and the log magnitude at the
    bins is linear in the 40 band values, so phase = bands @ this matrix exactly (no per-frame FFTs), as long as
    the training-side clamp of the log envelope at 4 is inactive (checked per frame by the caller)."""
    if ("mpm", bands) not in _CACHE:
        _CACHE[("mpm", bands)] = _minimum_phase(_band_interp_matrix(bands))
    return _CACHE[("mpm", bands)]


def _noise_interp_matrix(bands=NOISE_BANDS, n_fft=N_FFT):
    """The noise envelope is always read at the same bin frequencies: precompute the linear map [bands, bins]."""
    return _band_interp_matrix(bands, n_fft)


class _Synth:
    def __init__(self, w: DspWeights, seed: int):
        self.w, self.seed = w, seed
        self.hb, self.nb = w.harm_bands, w.noise_bands
        self.harm_c, self.noise_c = _mel_centres(self.hb), _mel_centres(self.nb)
        self.bin_f = np.arange(N_BINS) * SR / N_FFT
        self.orders = np.arange(1, N_HARM + 1, dtype=np.float64)
        self.nfft = w.noise_fft
        self.n_nb = self.nfft // 2 + 1
        self.nim = _noise_interp_matrix(self.nb, self.nfft)
        self.him = _band_interp_matrix(self.hb) if w.minphase else None
        self.mpm = _minphase_matrix(self.hb) if w.minphase else None
        self.mpm32 = self.mpm.astype(np.float32) if w.minphase else None
        self.nim32 = self.nim.astype(np.float32)
        self.him32 = self.him.astype(np.float32) if w.minphase else None
        self.theta32 = w.theta.astype(np.float32)
        # band position of a frequency: log1p(f / 700) * scale == mel(f) / mel(fmax) * (bands - 1)
        self._mel_scale = (2595.0 / math.log(10.0)) / (2595.0 * math.log10(1.0 + (SR / 2) / 700.0)) * (self.hb - 1)
        key = ("white", seed, self.n_nb)
        if key not in _CACHE:
            _CACHE[key] = self._white(0, WHITE_CACHE_FRAMES).astype(np.complex64)
        self.white_cache = _CACHE[key]

    def frame_params(self, h):
        """head outputs [t, 66] -> f0 [t] (float64), harmonic amplitudes [t, 80], noise spectra [513, t] (before white
        noise), harmonic phases [t, 80]. Envelope arithmetic in float32 (contiguous copies of the head slices);
        f0, which drives the running phase, stays float64."""
        f0 = F0_REF * np.exp(F0_SCALE * h[:, 0])
        voiced = (1.0 / (1.0 + np.exp(-h[:, 1]))).astype(np.float32)
        kf = f0[:, None] * self.orders  # float64 [t, 80]
        bands = np.ascontiguousarray(h[:, 2:2 + self.hb], dtype=np.float32)
        nbands = np.ascontiguousarray(h[:, 2 + self.hb:], dtype=np.float32)
        # harmonic envelope read at k * f0, linear in mel (as _interp_bands / dspdec.interp_bands)
        pos = np.clip(np.log1p(kf * (1.0 / 700.0)) * self._mel_scale, 0.0, self.hb - 1).astype(np.float32)
        lo = np.minimum(pos.astype(np.int64), self.hb - 2)
        a = pos - lo
        v_lo = np.take_along_axis(bands, lo, -1)
        la = v_lo + (np.take_along_axis(bands, lo + 1, -1) - v_lo) * a
        amp = np.exp(np.minimum(la, 4.0)) * (self._fade(kf) * voiced[:, None])
        env = np.exp(np.minimum(nbands @ self.nim32, 4.0))
        if self.w.minphase:
            if self.w.mp_noclamp or bands.max() < 4.0:  # no clamp: the phase is linear in the bands
                ph = bands @ self.mpm32  # [t, 513], exact up to float32
            else:
                # the clamp at 4 is active (this model's bands reach ~15 in 25-40 % of frames): exact FFT path, float32
                ph = _minimum_phase(np.minimum(bands @ self.him32, 4.0))
            p2 = np.clip(kf * (N_FFT / SR), 0.0, N_BINS - 1).astype(np.float32)
            l2 = np.minimum(p2.astype(np.int64), N_BINS - 2)
            a2 = p2 - l2
            g = np.take_along_axis(ph, l2, -1)
            th = g + (np.take_along_axis(ph, l2 + 1, -1) - g) * a2 + self.theta32
        else:
            th = np.broadcast_to(self.theta32, amp.shape)
        return f0, amp, env.T, th

    def _fade(self, kf):
        return np.clip((0.95 * SR / 2 - kf) * (1.0 / (0.05 * SR / 2)), 0.0, 1.0).astype(np.float32)

    def white(self, lo, hi):
        if hi <= WHITE_CACHE_FRAMES:
            return self.white_cache[:, lo:hi].astype(np.complex128)
        if lo < WHITE_CACHE_FRAMES:
            return np.concatenate([self.white_cache[:, lo:].astype(np.complex128), self._white(WHITE_CACHE_FRAMES, hi)], 1)
        return self._white(lo, hi)

    def _white(self, lo, hi):
        """Standard-normal complex spectrum per frame [513, hi - lo], counter-based: value (frame f, slot j) is
        Box-Muller of two splitmix64 hashes of (seed, f, j), so a frame's noise does not depend on the chunking
        and is cheap to vectorise (and to port to C)."""
        f = np.arange(lo, hi, dtype=np.uint64)[None, :]
        j = np.arange(self.n_nb, dtype=np.uint64)[:, None]
        with np.errstate(over="ignore"):  # uint64 multiply wraps by design
            key = (np.uint64(self.seed) * np.uint64(0x9E3779B97F4A7C15)) ^ (f << np.uint64(20)) ^ j
        u1 = (_splitmix64(key) >> np.uint64(11)).astype(np.float64) * (1.0 / 9007199254740992.0)
        u2 = (_splitmix64(key ^ np.uint64(0xD1B54A32D192ED03)) >> np.uint64(11)).astype(np.float64) * (1.0 / 9007199254740992.0)
        r = np.sqrt(-2.0 * np.log1p(-u1))  # u1 in [0, 1): 1 - u1 in (0, 1]
        return r * np.cos(2.0 * math.pi * u2) + 1j * (r * np.sin(2.0 * math.pi * u2))

    def tables(self, amp, th):
        """One period of each frame's harmonic waveform, sum_k amp[t, k] sin(k phi + theta_k), sampled at
        TABLE points (+1 wrap sample): [t, TABLE + 1]. Exact identity: with amplitudes linearly interpolated
        between frames, the harmonic sum is the same crossfade of the two frames' tables."""
        spec = np.zeros((amp.shape[0], TABLE // 2 + 1), dtype=np.complex128)
        spec[:, 1:N_HARM + 1] = (TABLE / 2) * amp * np.exp(1j * (th - math.pi / 2))
        tab = np.fft.irfft(spec, n=TABLE, axis=1)
        return np.concatenate([tab[:, -1:], tab, tab[:, :2]], axis=1).astype(TABLE_DTYPE)  # wrap: 1 before, 2 after





class _OLA:
    """Streaming iSTFT (centre padding, Hann, normalised by the summed squared window: nano.istft's contract) with
    the overlap-add vectorised over frames: frame j adds its quarter q to output block j + q, so a push is four
    strided adds instead of a Python loop per frame. Positions are in the padded signal (n_fft / 2 on each side)."""

    def __init__(self, n_fft=N_FFT):
        if n_fft % HOP:
            raise ValueError(f"n_fft {n_fft} must be a multiple of the hop {HOP}")
        self.n_fft, self.m = n_fft, n_fft // HOP
        self.win = np.hanning(n_fft + 1)[:n_fft]  # periodic, as torch / nano.istft
        self.wsq = (self.win * self.win).reshape(self.m, HOP)
        self.out = np.zeros((0, HOP))     # blocks [base, base + len) of the padded signal, accumulated
        self.norm = np.zeros((0, HOP))
        self.base = 0                     # first block kept
        self.added = 0                    # frames added
        self.emitted = n_fft // 2         # padded sample position up to which samples were returned
        self.total = None

    def set_total(self, frames):
        self.total = frames

    def push(self, spec):
        td = np.fft.irfft(spec, n=self.n_fft, axis=0).T * self.win  # [count, n_fft]
        count = td.shape[0]
        need = self.added + count + self.m - 1 - self.base  # blocks touched: frame j covers blocks j .. j+m-1
        if need > self.out.shape[0]:
            grow = need - self.out.shape[0]
            self.out = np.concatenate([self.out, np.zeros((grow, HOP))])
            self.norm = np.concatenate([self.norm, np.zeros((grow, HOP))])
        qm = td.reshape(count, self.m, HOP)
        j0 = self.added - self.base
        for q in range(self.m):
            self.out[j0 + q:j0 + q + count] += qm[:, q]
            self.norm[j0 + q:j0 + q + count] += self.wsq[q]
        self.added += count
        pad = self.n_fft // 2
        if self.total is not None and self.added == self.total:
            hi = (self.total - 1) * HOP + self.n_fft - pad  # end of the unpadded signal, in padded positions
        else:
            hi = self.added * HOP                       # no later frame reaches before its own start
        lo = self.emitted
        if hi <= lo:
            return np.zeros(0)
        o = self.out.reshape(-1)[lo - self.base * HOP:hi - self.base * HOP].copy()
        nm = self.norm.reshape(-1)[lo - self.base * HOP:hi - self.base * HOP]
        nz = nm > 1e-11
        o[nz] /= nm[nz]
        self.emitted = hi
        keep = hi // HOP  # drop whole blocks that are fully emitted
        drop = keep - self.base
        if drop > 0:
            self.out, self.norm, self.base = self.out[drop:], self.norm[drop:], keep
        return o


def _harm_from_tables(tab, lo, hi, a, phase):
    """tab [t, TABLE + 1]; per-sample frame indices lo/hi, crossfade a, wrapped phase -> harmonic samples."""
    x = phase * (TABLE / (2.0 * math.pi))
    i = np.minimum(x.astype(np.int64), TABLE - 1)
    f = (x - i).astype(TABLE_DTYPE)
    a = a.astype(TABLE_DTYPE)
    if TABLE_LINEAR:
        flat_lo, flat_hi = lo * (TABLE + 3) + i + 1, hi * (TABLE + 3) + i + 1
        t = tab.reshape(-1)
        v_lo = t[flat_lo] + (t[flat_lo + 1] - t[flat_lo]) * f
        v_hi = t[flat_hi] + (t[flat_hi + 1] - t[flat_hi]) * f
        return (v_lo + (v_hi - v_lo) * a).astype(np.float64)

    def cubic(rows):
        p0, p1, p2, p3 = tab[rows, i], tab[rows, i + 1], tab[rows, i + 2], tab[rows, i + 3]
        return p1 + 0.5 * f * (p2 - p0 + f * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3 + f * (3.0 * (p1 - p2) + p3 - p0)))
    c_lo = cubic(lo)
    return (c_lo + (cubic(hi) - c_lo) * a).astype(np.float64)


def render(w: DspWeights, mel: np.ndarray, seed: int = 7) -> np.ndarray:
    """Whole-utterance render: mel [100, T] -> float32 audio, (T - 1) * 256 samples."""
    t = mel.shape[1]
    h = controls(w, mel, True, True)
    syn = _Synth(w, seed)
    f0, amp, env, th = syn.frame_params(h)
    n = (t - 1) * HOP
    pos = np.arange(n) / HOP
    lo = np.minimum(np.floor(pos).astype(np.int64), t - 1)
    hi = np.minimum(lo + 1, t - 1)
    a = np.clip(pos - lo, 0.0, 1.0)
    f0_s = f0[lo] * (1 - a) + f0[hi] * a
    phase = np.remainder(np.cumsum(2.0 * math.pi * f0_s / SR), 2.0 * math.pi)
    harm = _harm_from_tables(syn.tables(amp, th), lo, hi, a, phase)
    ola = _OLA(w.noise_fft)
    ola.set_total(t)
    noise = ola.push(env * syn.white(0, t))
    return (harm + noise[:n]).astype(np.float32)


class DspStream:
    """Incremental render. push(mel_frames [100, k], final) -> newly final samples (float32)."""

    def __init__(self, w: DspWeights, seed: int = 7):
        self.w, self.syn = w, _Synth(w, seed)
        self.mel = np.zeros((100, 0))
        self.final = False
        self.ctl = np.zeros((0, 2 + w.harm_bands + w.noise_bands))  # controls computed for frames [0, len)
        self.f0 = np.zeros(0)
        self.tab = np.zeros((0, TABLE + 3), dtype=TABLE_DTYPE)
        self.istft = _OLA(w.noise_fft)
        self.noise_buf = np.zeros(0)
        self.noise_pushed = 0
        self.harm_done = 0      # samples of the harmonic part emitted so far
        self.out_done = 0       # samples returned so far
        self.phase = 0.0        # running phase (float64) after harm_done samples
        self.harm_buf = np.zeros(0)

    def _compute_controls(self):
        t_have = self.mel.shape[1]
        done = self.ctl.shape[0]
        end = t_have if self.final else t_have - self.w.lookahead
        if end <= done:
            return
        lo = max(0, done - self.w.left_halo)
        hi = t_have if self.final else end + self.w.lookahead
        h = controls(self.w, self.mel[:, lo:hi], lo == 0, self.final)
        # h covers frames [lo if lo == 0 else lo + left_halo, ...); keep the new ones
        first = 0 if lo == 0 else lo + self.w.left_halo
        h = h[done - first:end - first]
        self.ctl = np.concatenate([self.ctl, h], 0)
        f0, amp, env, th = self.syn.frame_params(h)
        self.f0 = np.concatenate([self.f0, f0])
        self.tab = np.concatenate([self.tab, self.syn.tables(amp, th)], 0)
        if self.final:
            self.istft.set_total(end)
        self.noise_buf = np.concatenate([self.noise_buf, self.istft.push(env * self.syn.white(done, end))])

    def _harmonic(self):
        t = self.f0.size
        total = (self.mel.shape[1] - 1) * HOP if self.final else None
        # a sample n needs frames floor(n / HOP) and +1 (unless +1 is past the last frame)
        ready = (t - 1) * HOP if not self.final or t < self.mel.shape[1] else total
        if ready is None or ready <= self.harm_done:
            return
        n = np.arange(self.harm_done, ready)
        pos = n / HOP
        lo = np.minimum(np.floor(pos).astype(np.int64), t - 1)
        hi = np.minimum(lo + 1, t - 1)
        a = np.clip(pos - lo, 0.0, 1.0)
        f0_s = self.f0[lo] * (1 - a) + self.f0[hi] * a
        cum = self.phase + np.cumsum(2.0 * math.pi * f0_s / SR)
        self.phase = float(cum[-1])
        phase = np.remainder(cum, 2.0 * math.pi)
        h = _harm_from_tables(self.tab, lo, hi, a, phase)
        self.harm_buf = np.concatenate([self.harm_buf, h])
        self.harm_done = ready

    def push(self, mel_frames: np.ndarray, final: bool = False) -> np.ndarray:
        if self.final:
            raise RuntimeError("push after final")
        self.mel = np.concatenate([self.mel, mel_frames.astype(np.float64)], 1)
        self.final = final
        self._compute_controls()
        self._harmonic()
        k = min(self.harm_buf.size, self.noise_buf.size)
        if final:
            k = min(k, (self.mel.shape[1] - 1) * HOP - self.out_done)
        out = self.harm_buf[:k] + self.noise_buf[:k]
        self.harm_buf, self.noise_buf = self.harm_buf[k:], self.noise_buf[k:]
        self.out_done += k
        return out.astype(np.float32)


def stream_local(front_w, dspw: DspWeights, prefix_ids, full_ids, *, seed: int = 7, length_scale: float = 1.0,
                 first_chunk_frames: int = 4, chunk_frames: int = 128):
    """nano.stream_local with the DSP decoder: audio from the first words, then the rest.

    `front_w` is the voice's nano.NanoWeights (its duration and acoustic models are used; its neural decoder is
    not). Phase 1 runs durations and the acoustic token stage on `prefix_ids` and pushes the mel frames whose
    acoustic receptive field lies inside the trusted prefix; phase 2 converts `full_ids()`, keeps phase 1's
    durations for the tokens it used, and pushes the rest. Frames from both phases are the same up to float
    rounding (local features), so the stream concatenates to the whole-utterance render.
    """
    m = front_w.m
    if not _nano.is_local(m):
        raise ValueError("nano_dsp.stream_local needs a voice with NANO_LOCAL_FEATURES")
    if first_chunk_frames < 1 or chunk_frames < first_chunk_frames:
        raise ValueError(f"need 1 <= first_chunk_frames <= chunk_frames, got {first_chunk_frames}, {chunk_frames}")
    ac_halo = _nano.acoustic_halo_frames(m)
    ds = DspStream(dspw, seed=seed)
    # the first push carries the decoder's look-ahead so it already returns audio
    pushed, size, kept = 0, first_chunk_frames + dspw.lookahead, None

    def frames(plan, total, a, b):
        alo = max(0, a - ac_halo)
        return _nano.acoustic_frames(front_w, plan, alo, min(total, b + ac_halo))[:, a - alo: b - alo]

    if prefix_ids is not None:
        ids_p = np.asarray(prefix_ids, dtype=np.int64)
        trusted_dur = ids_p.size - _nano.duration_halo_tokens(m)
        trusted_tok = trusted_dur - _nano.token_halo_tokens(m)
        if trusted_tok > 0:
            dur_p = _nano.duration_forward(front_w, ids_p, length_scale=length_scale)
            plan_p = _nano.acoustic_plan(front_w, ids_p, dur_p)
            limit = int(dur_p[:trusted_tok].sum()) - ac_halo
            if limit >= size:
                kept = dur_p[:trusted_dur]
                while pushed + size <= limit:
                    out = ds.push(frames(plan_p, plan_p.frames, pushed, pushed + size))
                    if out.size:
                        yield out
                    pushed, size = pushed + size, min(chunk_frames, size * 2)
    ids = np.asarray(full_ids(), dtype=np.int64)
    if ids.size == 0:
        raise ValueError("nano: empty phoneme id sequence")
    durations = _nano.duration_forward(front_w, ids, length_scale=length_scale).astype(np.int64)
    if kept is not None:
        durations[: kept.size] = kept
    plan = _nano.acoustic_plan(front_w, ids, durations)
    total = plan.frames
    if total < 2:
        raise ValueError("nano_dsp: need at least 2 frames")
    while pushed < total:
        b = min(total, pushed + size)
        out = ds.push(frames(plan, total, pushed, b), final=b == total)
        if out.size:
            yield out
        pushed, size = b, min(chunk_frames, size * 2)
