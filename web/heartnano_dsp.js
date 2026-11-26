/* heartnano_dsp.js -- browser runtime for the sanotts 0.5.7 heart-nano voice
 * (104,215 parameters): the int8 local-feature front end of sanotts/nano.py
 * (durations + acoustic mel-100 frames) and the harmonic-plus-noise decoder of
 * sanotts/nano_dsp.py (mel-100 -> 24 kHz audio). Plain JavaScript, no build
 * step, no dependencies.
 *
 * The snt_nano C runtime (snt_nano_heartnano.js) cannot run this voice: it has
 * no local-feature path and no DSP decoder. This file is a port of the Python
 * package, not of the C, and is checked against the package's own output by
 * tools/verify_heartnano_dsp_node.mjs.
 *
 * Numerics follow the package step by step: the front end runs on the folded
 * weights of nano._front_plan with every float32 rounding point the numpy code
 * has (Math.fround), dot products accumulated in float64; the decoder's
 * control network runs in float64 as nano_dsp.controls does; the envelope,
 * minimum-phase and wavetable arithmetic keep the package's float32 casts; the
 * running phase stays float64. The noise is the package's counter-based
 * splitmix64 + Box-Muller draw, rounded to float32 as the package's noise cache
 * stores it. The seed is meta.json "seed" (sanotts engine.DEFAULT_NANO_SEED).
 *
 * Bundle (web/voices/heartnano-dsp/, written by tools/convert_heartnano_dsp_web.py):
 *   meta.json            offsets (from nano_q8_meta.h), dsp_tensors, dsp_config, seed
 *   front_q8.bin         the package's front-end blob, unchanged
 *   dsp_decoder_f32.bin  dsp_decoder.npz's float tensors, little-endian float32
 *
 * API (globalThis.SaanoNanoDsp):
 *   const v = SaanoNanoDsp.createVoice({meta, front, dsp});   // Uint8Arrays
 *   const r = v.synthesize(ids);   // {pcm: Float32Array, durations, frames, sampleRate}
 *   v.durations(ids), v.mel(ids, durations), v.render(mel, frames)  (stages, for tests)
 */
(function (root) {
  "use strict";

  var fr = Math.fround;
  var SR = 24000, HOP = 256, N_FFT = 1024, N_BINS = 513, N_HARM = 80, TABLE = 1024;
  var F0_REF = 200.0, F0_SCALE = 0.5, MEL_MEAN = -4.0, MEL_STD = 4.0;
  var TWO_PI = 2.0 * Math.PI;

  function fail(msg) { throw new Error("heart-nano (dsp): " + msg); }

  /* ------------------------------------------------------------ erf / GELU */

  // nano_dsp._gelu: exact-erf GELU through a uniform erf table (step 1e-4 on
  // [-6, 6 + 1e-4]) read with linear interpolation. The table values come
  // from the all-positive series erf(x) = 2/sqrt(pi) e^{-x^2} sum_n 2^n
  // x^{2n+1} / (1*3*...*(2n+1)) (Abramowitz & Stegun 7.1.6), which has no
  // cancellation; it agrees with Python's math.erf to a few float64 ulp,
  // far below the table's own 2e-9 interpolation error.
  var ERF_H = 1e-4, ERF_INV_H = 1.0 / ERF_H, INV_SQRT2 = 1.0 / Math.sqrt(2.0);
  var ERF_VAL = null;
  function erfSeries(x) {
    if (x === 0) return 0;
    var x2 = x * x, term = x, sum = x;
    for (var n = 1; n < 1000; n++) {
      term *= 2.0 * x2 / (2 * n + 1);
      sum += term;
      if (Math.abs(term) < 1e-17 * Math.abs(sum)) break;
    }
    var v = 2.0 / Math.sqrt(Math.PI) * Math.exp(-x2) * sum;
    return v > 1 ? 1 : v < -1 ? -1 : v;
  }
  function erfTable() {
    if (!ERF_VAL) {
      ERF_VAL = new Float64Array(120002);
      for (var i = 0; i < 120002; i++) ERF_VAL[i] = erfSeries((i - 60000) * ERF_H);
    }
    return ERF_VAL;
  }
  function gelu(x) {
    var c = x * INV_SQRT2;
    if (c < -6.0) c = -6.0; else if (c > 6.0) c = 6.0;
    var u = c * ERF_INV_H + 60000.0;
    var i = Math.trunc(u), f = u - i;
    return 0.5 * x * (1.0 + ERF_VAL[i] * (1.0 - f) + ERF_VAL[i + 1] * f);
  }

  /* ------------------------------------------------------------------ FFT */

  // In-place iterative radix-2 complex FFT; sign -1 forward, +1 inverse
  // (unnormalised). Used for irfft/rfft of sizes 512 and 1024.
  var FFT_PLANS = {};
  function fftPlan(n) {
    if (FFT_PLANS[n]) return FFT_PLANS[n];
    if (n & (n - 1)) fail("FFT size " + n + " is not a power of two");
    var bits = Math.round(Math.log2(n)), rev = new Uint32Array(n);
    for (var i = 0; i < n; i++) {
      var r = 0;
      for (var b = 0; b < bits; b++) r |= ((i >>> b) & 1) << (bits - 1 - b);
      rev[i] = r;
    }
    var cos = new Float64Array(n / 2), sin = new Float64Array(n / 2);
    for (var k = 0; k < n / 2; k++) { cos[k] = Math.cos(TWO_PI * k / n); sin[k] = Math.sin(TWO_PI * k / n); }
    FFT_PLANS[n] = {n: n, rev: rev, cos: cos, sin: sin, re: new Float64Array(n), im: new Float64Array(n)};
    return FFT_PLANS[n];
  }
  function fftInPlace(p, re, im, sign) {
    var n = p.n, rev = p.rev, i, j, t;
    for (i = 0; i < n; i++) {
      j = rev[i];
      if (j > i) { t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
    }
    for (var size = 2; size <= n; size <<= 1) {
      var half = size >>> 1, step = n / size;
      for (var start = 0; start < n; start += size) {
        for (var k = 0; k < half; k++) {
          var wr = p.cos[k * step], wi = sign * p.sin[k * step];
          var a = start + k, b = a + half;
          var xr = re[b] * wr - im[b] * wi, xi = re[b] * wi + im[b] * wr;
          re[b] = re[a] - xr; im[b] = im[a] - xi;
          re[a] += xr; im[a] += xi;
        }
      }
    }
  }
  // numpy.fft.irfft(X, n) for X of n/2+1 bins: the imaginary parts of DC and
  // Nyquist are ignored, as pocketfft's C2R does. Writes n samples to out.
  function irfft(p, xr, xi, out) {
    var n = p.n, h = n >>> 1, re = p.re, im = p.im, k;
    re[0] = xr[0]; im[0] = 0;
    re[h] = xr[h]; im[h] = 0;
    for (k = 1; k < h; k++) { re[k] = xr[k]; im[k] = xi[k]; re[n - k] = xr[k]; im[n - k] = -xi[k]; }
    fftInPlace(p, re, im, +1);
    for (k = 0; k < n; k++) out[k] = re[k] / n;
  }

  /* ------------------------------------------------------- band matrices */

  function melCentres(n) {   // nano_dsp._mel_centres(n, SR / 2)
    var mmax = 2595.0 * Math.log10(1.0 + (SR / 2) / 700.0), out = new Float64Array(n);
    var step = mmax / (n - 1);
    for (var i = 0; i < n; i++) {
      var m = i === n - 1 ? mmax : i * step;   // np.linspace sets the endpoint exactly
      out[i] = 700.0 * (Math.pow(10.0, m / 2595.0) - 1.0);
    }
    return out;
  }
  // nano_dsp._band_interp_matrix: [bands, nfft/2+1], float64.
  function bandInterpMatrix(bands, nfft) {
    var nb = nfft / 2 + 1, c = melCentres(bands), M = new Float64Array(bands * nb);
    var melTop = 2595.0 * Math.log10(1.0 + c[bands - 1] / 700.0);
    for (var k = 0; k < nb; k++) {
      var f = k * SR / nfft;
      var melF = 2595.0 * Math.log10(1.0 + Math.max(f, 0.0) / 700.0);
      var pos = melF / melTop * (bands - 1);
      if (pos < 0) pos = 0; else if (pos > bands - 1) pos = bands - 1;
      var lo = Math.min(Math.floor(pos), bands - 2), a = pos - lo;
      M[lo * nb + k] += 1.0 - a;   // the one-hot rows: v_lo * (1 - a) + v_hi * a
      M[(lo + 1) * nb + k] += a;
    }
    return M;
  }
  // nano_dsp._minimum_phase applied to each row of the band matrix: the phase
  // response is linear in the band log-envelope (mp_noclamp voices).
  function minphaseMatrix(bands) {
    var him = bandInterpMatrix(bands, N_FFT), p = fftPlan(N_FFT), out = new Float64Array(bands * N_BINS);
    var xr = new Float64Array(N_BINS), xi = new Float64Array(N_BINS), c = new Float64Array(N_FFT);
    var re = new Float64Array(N_FFT), im = new Float64Array(N_FFT);
    for (var b = 0; b < bands; b++) {
      for (var k = 0; k < N_BINS; k++) { xr[k] = him[b * N_BINS + k]; xi[k] = 0; }
      irfft(p, xr, xi, c);
      for (var n = 0; n < N_FFT; n++) {
        var w = n === 0 || n === N_FFT / 2 ? 1.0 : n < N_FFT / 2 ? 2.0 : 0.0;
        re[n] = c[n] * w; im[n] = 0;
      }
      fftInPlace(p, re, im, -1);
      for (k = 0; k < N_BINS; k++) out[b * N_BINS + k] = im[k];
    }
    return out;
  }
  function toF32(a) { var o = new Float32Array(a.length); for (var i = 0; i < a.length; i++) o[i] = a[i]; return o; }

  /* -------------------------------------------------------- noise (splitmix64) */

  // 64-bit words as (hi, lo) uint32 pairs; results in H, L.
  var H = 0, L = 0;
  function mulhi32(a, b) {
    var a0 = a & 0xffff, a1 = a >>> 16, b0 = b & 0xffff, b1 = b >>> 16;
    var p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
    var mid = (p00 >>> 16) + (p01 & 0xffff) + (p10 & 0xffff);
    return (p11 + Math.floor(p01 / 65536) + Math.floor(p10 / 65536) + Math.floor(mid / 65536)) >>> 0;
  }
  function mul64(ah, al, bh, bl) {   // low 64 bits of a * b
    L = Math.imul(al, bl) >>> 0;
    H = (mulhi32(al, bl) + Math.imul(ah, bl) + Math.imul(al, bh)) >>> 0;
  }
  function splitmix64(h, l) {        // nano_dsp._splitmix64
    var lo = l + 0x7F4A7C15, carry = lo > 0xFFFFFFFF ? 1 : 0;
    lo = lo >>> 0;
    var hi = (h + 0x9E3779B9 + carry) >>> 0;
    lo = (lo ^ ((lo >>> 30) | (hi << 2))) >>> 0; hi = (hi ^ (hi >>> 30)) >>> 0;
    mul64(hi, lo, 0xBF58476D, 0x1CE4E5B9); hi = H; lo = L;
    lo = (lo ^ ((lo >>> 27) | (hi << 5))) >>> 0; hi = (hi ^ (hi >>> 27)) >>> 0;
    mul64(hi, lo, 0x94D049BB, 0x133111EB); hi = H; lo = L;
    L = (lo ^ ((lo >>> 31) | (hi << 1))) >>> 0; H = (hi ^ (hi >>> 31)) >>> 0;
  }
  function seedWords(seedStr) {
    if (typeof seedStr !== "string" || !/^\d+$/.test(seedStr)) fail("seed must be a decimal string, got " + seedStr);
    if (typeof BigInt !== "function") fail("this browser has no BigInt");
    var s = BigInt(seedStr), M64 = (BigInt(1) << BigInt(64)) - BigInt(1);
    if (s > M64) fail("seed " + seedStr + " does not fit in 64 bits");
    var k = (s * BigInt("0x9E3779B97F4A7C15")) & M64;
    return {hi: Number(k >> BigInt(32)) >>> 0, lo: Number(k & BigInt(0xFFFFFFFF)) >>> 0};
  }
  // nano_dsp._Synth._white for one frame: bins standard-normal complex values,
  // rounded to float32 (the package reads them from its complex64 cache).
  function whiteFrame(sw, f, bins, outRe, outIm) {
    var fh = Math.floor(f / 4096) >>> 0, fl = ((f & 0xfff) << 20) >>> 0;
    for (var j = 0; j < bins; j++) {
      var kh = (sw.hi ^ fh) >>> 0, kl = (sw.lo ^ fl ^ j) >>> 0;
      splitmix64(kh, kl);
      var u1 = (H * 2097152 + (L >>> 11)) * (1.0 / 9007199254740992.0);
      splitmix64((kh ^ 0xD1B54A32) >>> 0, (kl ^ 0xD192ED03) >>> 0);
      var u2 = (H * 2097152 + (L >>> 11)) * (1.0 / 9007199254740992.0);
      var r = Math.sqrt(-2.0 * Math.log1p(-u1));
      outRe[j] = fr(r * Math.cos(TWO_PI * u2));
      outIm[j] = fr(r * Math.sin(TWO_PI * u2));
    }
  }

  /* ----------------------------------------------------------- front end */

  function readF32(buf, off, count) {
    if (off < 0 || off + 4 * count > buf.byteLength) fail("float32 read [" + off + ", +" + 4 * count + ") is outside the blob");
    var dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength), out = new Float32Array(count);
    for (var i = 0; i < count; i++) out[i] = dv.getFloat32(off + 4 * i, true);
    return out;
  }
  function need(m, k) { if (!(k in m) || typeof m[k] !== "number") fail("meta.json offsets lack " + k); return m[k]; }
  // nano.NanoWeights._q8 (NANO_WEIGHT_FORMAT 0): int8 rows padded to n16,
  // dequantised in float32 (w = q * scale), trimmed to inFlat.
  function q8(front, m, name, out, inFlat) {
    var offW = need(m, "NOFF_" + name + "_W8"), n16 = need(m, "NANO_" + name + "_N16");
    if (inFlat > n16) fail(name + ": " + inFlat + " inputs but rows of " + n16);
    if (offW + out * n16 > front.byteLength) fail(name + ": int8 rows run past the front blob");
    var scale = readF32(front, need(m, "NOFF_" + name + "_SCALE"), out);
    var bias = readF32(front, need(m, "NOFF_" + name + "_BIAS"), out);
    var w = new Float32Array(out * inFlat), q = new Int8Array(front.buffer, front.byteOffset + offW, out * n16);
    for (var o = 0; o < out; o++) for (var i = 0; i < inFlat; i++) w[o * inFlat + i] = fr(q[o * n16 + i] * scale[o]);
    return {w: w, b: bias};
  }
  // nano._front_plan: embeddings folded through the input projections, each
  // residual scale folded into its block's second conv (float64, then float32).
  function frontPlan(front, m) {
    var V = need(m, "NANO_VOCAB"), HD = need(m, "NANO_DUR_HIDDEN"), HA = need(m, "NANO_AC_HIDDEN");
    function blocks(prefix, depth, hid, K) {
      var out = [];
      for (var b = 0; b < depth; b++) {
        var c0 = q8(front, m, prefix + b + "_C0", hid, hid * K), c1 = q8(front, m, prefix + b + "_C1", hid, hid * K);
        var s = readF32(front, need(m, "NOFF_" + prefix + b + "_SCALE_F32"), 1)[0];
        var w1 = new Float32Array(c1.w.length), b1 = new Float32Array(hid);
        for (var i = 0; i < w1.length; i++) w1[i] = c1.w[i] * s;
        for (i = 0; i < hid; i++) b1[i] = c1.b[i] * s;
        out.push({w0: c0.w, b0: c0.b, w1: w1, b1: b1});
      }
      return out;
    }
    function fold(projW, inW, emb, hid) {   // (W[:, :hid] @ E^T) -> [hid, V], float64 then float32
      var out = new Float32Array(hid * V);
      for (var h = 0; h < hid; h++) for (var v = 0; v < V; v++) {
        var acc = 0;
        for (var j = 0; j < hid; j++) acc += projW[h * inW + j] * emb[v * hid + j];
        out[h * V + v] = acc;
      }
      return out;
    }
    function cols(W, inW, rows, from, n) {   // W[:, from:from+n] -> [rows, n]
      var out = new Float32Array(rows * n);
      for (var r = 0; r < rows; r++) for (var c = 0; c < n; c++) out[r * n + c] = W[r * inW + from + c];
      return out;
    }
    var durE = readF32(front, need(m, "NOFF_DUR_EMB_F32"), V * HD), acE = readF32(front, need(m, "NOFF_AC_EMB_F32"), V * HA);
    var proj = q8(front, m, "DUR_PROJ", HD, HD + 3), tproj = q8(front, m, "AC_TPROJ", HA, HA + 2);
    var fproj = q8(front, m, "AC_FPROJ", HA, HA + 3), dout = q8(front, m, "DUR_OUT", 1, HD);
    var mels = need(m, "NANO_MELS"), aout = q8(front, m, "AC_OUT", mels, HA);
    return {
      V: V, HD: HD, HA: HA, mels: mels,
      durK: need(m, "NANO_DUR_KERNEL"), acK: need(m, "NANO_AC_KERNEL"),
      durEmb: fold(proj.w, HD + 3, durE, HD), durFeat: cols(proj.w, HD + 3, HD, HD, 3), durB: proj.b,
      durBlocks: blocks("DUR_B", need(m, "NANO_DUR_DEPTH"), HD, need(m, "NANO_DUR_KERNEL")),
      durOutW: dout.w, durOutB: dout.b[0],
      tokEmb: fold(tproj.w, HA + 2, acE, HA), tokFeat: cols(tproj.w, HA + 2, HA, HA, 2), tokB: tproj.b,
      tokBlocks: blocks("AC_TB", need(m, "NANO_AC_TOKEN_DEPTH"), HA, need(m, "NANO_AC_KERNEL")),
      frTok: cols(fproj.w, HA + 3, HA, 0, HA), frFeat: cols(fproj.w, HA + 3, HA, HA, 3), frB: fproj.b,
      frBlocks: blocks("AC_FB", need(m, "NANO_AC_DEPTH"), HA, need(m, "NANO_AC_KERNEL")),
      outW: aout.w, outB: aout.b,
      maxDur: need(m, "NANO_DUR_MAX_DURATION"),
    };
  }
  // nano._conv_k: 'same' conv over x [C, T] (zero padding), weight rows
  // [Cout, C*K] with the kernel axis fastest; (w @ gathered) rounded to
  // float32, then the bias added in float32.
  function convSame(x, C, T, w, b, Cout, K, out) {
    var pad = K >> 1, CK = C * K;
    for (var o = 0; o < Cout; o++) {
      var row = o * CK, bo = b[o];
      for (var t = 0; t < T; t++) {
        var acc = 0;
        for (var c = 0; c < C; c++) {
          var xb = c * T, wb = row + c * K;
          for (var k = 0; k < K; k++) {
            var ti = t + k - pad;
            if (ti >= 0 && ti < T) acc += w[wb + k] * x[xb + ti];
          }
        }
        out[o * T + t] = fr(fr(acc) + bo);
      }
    }
  }
  // nano._residual_stack with nano._silu_tanh (x * (1 + tanh(x/2)) / 2, float32 steps).
  function residualStack(x, C, T, blocks, K) {
    var h = new Float32Array(C * T), y = new Float32Array(C * T), i;
    for (var bi = 0; bi < blocks.length; bi++) {
      var bl = blocks[bi];
      convSame(x, C, T, bl.w0, bl.b0, C, K, h);
      for (i = 0; i < h.length; i++) {
        var v = h[i];
        h[i] = fr(fr(fr(fr(1.0 + fr(Math.tanh(fr(v * 0.5)))) * v)) * 0.5);
      }
      convSame(h, C, T, bl.w1, bl.b1, C, K, y);
      for (i = 0; i < y.length; i++) y[i] = fr(y[i] + x[i]);
      var t = x; x = y; y = t;
    }
    return x;
  }
  function rintHalfEven(x) {   // np.round
    var f = Math.floor(x), d = x - f;
    if (d > 0.5) return f + 1;
    if (d < 0.5) return f;
    return f % 2 === 0 ? f : f + 1;
  }
  // nano.duration_forward, local-feature path.
  function durationForward(p, ids, lengthScale) {
    var n = ids.length, HD = p.HD, V = p.V, x = new Float32Array(HD * n), h, i;
    for (h = 0; h < HD; h++) {
      var fb = fr(p.durFeat[h * 3 + 2] + p.durB[h]);
      for (i = 0; i < n; i++) x[h * n + i] = fr(p.durEmb[h * V + ids[i]] + fb);
    }
    x = residualStack(x, HD, n, p.durBlocks, p.durK);
    var d = new Int32Array(n);
    for (i = 0; i < n; i++) {
      var acc = 0;
      for (h = 0; h < HD; h++) acc += p.durOutW[h] * x[h * n + i];
      var logd = fr(fr(acc) + p.durOutB);
      var v = rintHalfEven(fr(Math.max(fr(Math.exp(logd)), 1.0) * lengthScale));
      d[i] = Math.min(Math.max(v, 1), p.maxDur);
    }
    return d;
  }
  // nano.acoustic_plan + nano.acoustic_frames over [0, T), local-feature path.
  function acousticForward(p, ids, durs) {
    var n = ids.length, HA = p.HA, V = p.V, x = new Float32Array(HA * n), h, i, t, j;
    var lmax = Math.log1p(p.maxDur);
    for (i = 0; i < n; i++) {
      var hint = fr(fr(Math.log1p(durs[i])) / lmax);
      for (h = 0; h < HA; h++)
        x[h * n + i] = fr(fr(p.tokEmb[h * V + ids[i]] + fr(p.tokFeat[h * 2 + 1] * hint)) + p.tokB[h]);
    }
    var tokX = residualStack(x, HA, n, p.tokBlocks, p.acK);
    var tokF = new Float32Array(HA * n);
    for (h = 0; h < HA; h++) for (i = 0; i < n; i++) {
      var acc = 0;
      for (j = 0; j < HA; j++) acc += p.frTok[h * HA + j] * tokX[j * n + i];
      tokF[h * n + i] = fr(fr(acc) + p.frB[h]);
    }
    var T = 0;
    for (i = 0; i < n; i++) T += durs[i];
    var tokOf = new Int32Array(T), dpos = new Float32Array(T);
    for (i = 0, t = 0; i < n; i++) for (j = 0; j < durs[i]; j++, t++) {
      tokOf[t] = i;
      dpos[t] = durs[i] > 1 ? fr(j / (durs[i] - 1)) : 0;
    }
    var fx = new Float32Array(HA * T);
    for (h = 0; h < HA; h++) {
      var wd = p.frFeat[h * 3 + 2];
      for (t = 0; t < T; t++) fx[h * T + t] = fr(tokF[h * n + tokOf[t]] + fr(wd * dpos[t]));
    }
    fx = residualStack(fx, HA, T, p.frBlocks, p.acK);
    var mel = new Float32Array(p.mels * T);
    convSame(fx, HA, T, p.outW, p.outB, p.mels, 1, mel);
    return {mel: mel, frames: T};
  }

  /* ---------------------------------------------------------- DSP decoder */

  function dspWeights(dsp, meta) {
    var cfg = meta.dsp_config, tensors = meta.dsp_tensors;
    if (!cfg || !Array.isArray(tensors)) fail("meta.json lacks dsp_config / dsp_tensors");
    if (dsp.byteLength % 4) fail("dsp blob is not a whole number of float32s");
    var all = readF32(dsp, 0, dsp.byteLength / 4), byName = {};
    for (var i = 0; i < tensors.length; i++) {
      var t = tensors[i], n = 1;
      for (var s = 0; s < t.shape.length; s++) n *= t.shape[s];
      if (t.offset + n > all.length) fail("tensor " + t.name + " runs past the dsp blob");
      byName[t.name] = {shape: t.shape, data: all.subarray(t.offset, t.offset + n)};
    }
    function get(name, shape) {
      var t = byName[name];
      if (!t) fail("dsp tensor " + name + " is missing");
      if (t.shape.join(",") !== shape.join(",")) fail("dsp tensor " + name + " has shape [" + t.shape + "], expected [" + shape + "]");
      return t.data;
    }
    var pads = cfg.pads;
    if (!Array.isArray(pads) || pads.length < 2 || pads.length % 2) fail("bad dsp_config.pads");
    var C = byName.inp_w ? byName.inp_w.shape[0] : fail("dsp tensor inp_w is missing");
    var kIn = byName.inp_w.shape[2], hb = cfg.harm_bands, nb = cfg.noise_bands;
    // Only the configuration this voice ships is implemented (and verified).
    if (cfg.minphase !== true || cfg.mp_noclamp !== true)
      fail("only minphase + mp_noclamp decoders are supported (got minphase=" + cfg.minphase + ", mp_noclamp=" + cfg.mp_noclamp + ")");
    if (cfg.noise_fft % HOP || cfg.noise_fft & (cfg.noise_fft - 1)) fail("noise_fft " + cfg.noise_fft + " is not a power-of-two multiple of " + HOP);
    if (kIn !== pads[0] + pads[1] + 1) fail("inp_w kernel " + kIn + " does not match pads " + pads[0] + "+" + pads[1]);
    var w = {
      C: C, kIn: kIn, inL: pads[0], inR: pads[1], hb: hb, nb: nb, noiseFft: cfg.noise_fft,
      inpW: get("inp_w", [C, 100, kIn]), inpB: get("inp_b", [C]),
      normW: get("norm_w", [C]), normB: get("norm_b", [C]),
      headW: get("head_w", [2 + hb + nb, C]), headB: get("head_b", [2 + hb + nb]),
      theta: get("theta", [N_HARM]), blocks: [],
    };
    for (var b = 0; 2 + 2 * b < pads.length; b++) {
      var k = byName["b" + b + "_dw_w"] ? byName["b" + b + "_dw_w"].shape[2] : fail("dsp tensor b" + b + "_dw_w is missing");
      var hid = byName["b" + b + "_p1_w"] ? byName["b" + b + "_p1_w"].shape[0] : fail("dsp tensor b" + b + "_p1_w is missing");
      var L0 = pads[2 + 2 * b], R0 = pads[3 + 2 * b];
      if (k !== L0 + R0 + 1) fail("block " + b + " kernel " + k + " does not match pads " + L0 + "+" + R0);
      w.blocks.push({
        k: k, left: L0, right: R0, hid: hid,
        dwW: get("b" + b + "_dw_w", [C, 1, k]), dwB: get("b" + b + "_dw_b", [C]),
        nW: get("b" + b + "_n_w", [C]), nB: get("b" + b + "_n_b", [C]),
        p1W: get("b" + b + "_p1_w", [hid, C]), p1B: get("b" + b + "_p1_b", [hid]),
        p2W: get("b" + b + "_p2_w", [C, hid]), p2B: get("b" + b + "_p2_b", [C]),
        gamma: get("b" + b + "_gamma", [C]),
      });
    }
    return w;
  }
  function padRows(x, T, C, left, right) {   // replicate-pad [T, C] rows (utterance edges)
    var out = new Float64Array((T + left + right) * C);
    for (var t = 0; t < T + left + right; t++) {
      var src = Math.min(Math.max(t - left, 0), T - 1);
      for (var c = 0; c < C; c++) out[t * C + c] = x[src * C + c];
    }
    return out;
  }
  function layerNormRow(x, off, C, w, b, eps, out, oOff) {
    var mu = 0, c;
    for (c = 0; c < C; c++) mu += x[off + c];
    mu /= C;
    var v = 0;
    for (c = 0; c < C; c++) { var d = x[off + c] - mu; v += d * d; }
    v /= C;
    var s = Math.sqrt(v + eps);
    for (c = 0; c < C; c++) out[oOff + c] = (x[off + c] - mu) / s * w[c] + b[c];
  }
  // nano_dsp.controls over a whole utterance: mel [100, T] -> head outputs [T, 66], float64.
  function controls(w, mel, T) {
    erfTable();
    var C = w.C, x0 = new Float64Array(T * 100), t, c, j, o;
    for (c = 0; c < 100; c++) for (t = 0; t < T; t++) x0[t * 100 + c] = (mel[c * T + t] - MEL_MEAN) / MEL_STD;
    var xp = padRows(x0, T, 100, w.inL, w.inR), x = new Float64Array(T * C);
    for (t = 0; t < T; t++) for (o = 0; o < C; o++) {
      var acc = 0;
      for (j = 0; j < w.kIn; j++) {
        var xr = (t + j) * 100, wr = o * 100 * w.kIn + j;
        for (c = 0; c < 100; c++) acc += xp[xr + c] * w.inpW[wr + c * w.kIn];
      }
      x[t * C + o] = acc + w.inpB[o];
    }
    var y = new Float64Array(T * C), yn = new Float64Array(C), z = new Float64Array(256);
    for (var bi = 0; bi < w.blocks.length; bi++) {
      var bl = w.blocks[bi];
      var xq = padRows(x, T, C, bl.left, bl.right);
      for (t = 0; t < T; t++) for (c = 0; c < C; c++) {
        var a = 0;
        for (j = 0; j < bl.k; j++) a += xq[(t + j) * C + c] * bl.dwW[c * bl.k + j];
        y[t * C + c] = a + bl.dwB[c];
      }
      if (bl.hid > z.length) z = new Float64Array(bl.hid);
      for (t = 0; t < T; t++) {
        layerNormRow(y, t * C, C, bl.nW, bl.nB, 1e-5, yn, 0);
        for (o = 0; o < bl.hid; o++) {
          var s1 = 0;
          for (c = 0; c < C; c++) s1 += yn[c] * bl.p1W[o * C + c];
          z[o] = gelu(s1 + bl.p1B[o]);
        }
        for (c = 0; c < C; c++) {
          var s2 = 0;
          for (o = 0; o < bl.hid; o++) s2 += z[o] * bl.p2W[c * bl.hid + o];
          x[t * C + c] = x[t * C + c] + bl.gamma[c] * (s2 + bl.p2B[c]);
        }
      }
    }
    var R = 2 + w.hb + w.nb, head = new Float64Array(T * R);
    for (t = 0; t < T; t++) {
      layerNormRow(x, t * C, C, w.normW, w.normB, 1e-5, yn, 0);
      for (o = 0; o < R; o++) {
        var s3 = 0;
        for (c = 0; c < C; c++) s3 += yn[c] * w.headW[o * C + c];
        head[t * R + o] = s3 + w.headB[o];
      }
    }
    return head;
  }
  function synthTables(w) {
    if (!w.tables) {
      w.tables = {
        mpm32: toF32(minphaseMatrix(w.hb)),                  // [hb, 513]
        nim32: toF32(bandInterpMatrix(w.nb, w.noiseFft)),    // [nb, noise_fft/2+1]
        melScale: (2595.0 / Math.log(10.0)) / (2595.0 * Math.log10(1.0 + (SR / 2) / 700.0)) * (w.hb - 1),
      };
    }
    return w.tables;
  }
  // nano_dsp.render: mel [100, T] -> (T - 1) * 256 float64 samples (unclipped).
  function render(w, mel, T, seedStr) {
    if (T < 2) fail("need at least 2 frames, got " + T);
    var tb = synthTables(w), hb = w.hb, nb = w.nb, R = 2 + hb + nb;
    var h = controls(w, mel, T), sw = seedWords(seedStr);
    var nfft = w.noiseFft, nbin = nfft / 2 + 1, n = (T - 1) * HOP, t, k, i;
    var f0 = new Float64Array(T);
    var tabs = new Float32Array(T * (TABLE + 3));
    var bands = new Float32Array(hb), ph = new Float32Array(N_BINS);
    var sr = new Float64Array(TABLE / 2 + 1), si = new Float64Array(TABLE / 2 + 1), tab = new Float64Array(TABLE);
    var pT = fftPlan(TABLE), pN = fftPlan(nfft);
    var env = new Float32Array(nbin), wr = new Float32Array(nbin), wi = new Float32Array(nbin);
    var xr = new Float64Array(nbin), xi = new Float64Array(nbin), td = new Float64Array(nfft);
    var win = new Float64Array(nfft);
    for (i = 0; i < nfft; i++) win[i] = 0.5 + 0.5 * Math.cos(Math.PI * (2 * i - nfft) / nfft);  // np.hanning(nfft+1)[:nfft]
    var olaLen = (T - 1) * HOP + nfft, ola = new Float64Array(olaLen), norm = new Float64Array(olaLen);
    var fadeDen = 1.0 / (0.05 * SR / 2), fadeTop = 0.95 * SR / 2, binPerHz = N_FFT / SR;
    for (t = 0; t < T; t++) {
      var hr = t * R;
      f0[t] = F0_REF * Math.exp(F0_SCALE * h[hr]);
      var voiced = fr(1.0 / (1.0 + Math.exp(-h[hr + 1])));
      for (k = 0; k < hb; k++) bands[k] = h[hr + 2 + k];
      // phase response at the bins: bands @ mpm32 (float32)
      for (var q = 0; q < N_BINS; q++) {
        var acc = 0;
        for (k = 0; k < hb; k++) acc += bands[k] * tb.mpm32[k * N_BINS + q];
        ph[q] = acc;
      }
      sr[0] = 0; si[0] = 0;
      for (k = 1; k <= N_HARM; k++) {
        var kf = f0[t] * k;
        var pos = Math.log1p(kf * (1.0 / 700.0)) * tb.melScale;
        pos = fr(pos < 0 ? 0 : pos > hb - 1 ? hb - 1 : pos);
        var lo = Math.min(Math.trunc(pos), hb - 2), a = pos - lo;
        var la = bands[lo] + fr(bands[lo + 1] - bands[lo]) * a;
        var fade = (fadeTop - kf) * fadeDen;
        fade = fr(fade < 0 ? 0 : fade > 1 ? 1 : fade);
        var amp = Math.exp(Math.min(la, 4.0)) * fr(fade * voiced);
        var p2 = kf * binPerHz;
        p2 = fr(p2 < 0 ? 0 : p2 > N_BINS - 1 ? N_BINS - 1 : p2);
        var l2 = Math.min(Math.trunc(p2), N_BINS - 2), a2 = p2 - l2;
        var th = ph[l2] + fr(ph[l2 + 1] - ph[l2]) * a2 + w.theta[k - 1];
        var g = (TABLE / 2) * amp, ang = th - Math.PI / 2;
        sr[k] = g * Math.cos(ang); si[k] = g * Math.sin(ang);
      }
      for (k = N_HARM + 1; k <= TABLE / 2; k++) { sr[k] = 0; si[k] = 0; }
      irfft(pT, sr, si, tab);
      var base = t * (TABLE + 3);
      tabs[base] = tab[TABLE - 1];
      for (i = 0; i < TABLE; i++) tabs[base + 1 + i] = tab[i];
      tabs[base + TABLE + 1] = tab[0]; tabs[base + TABLE + 2] = tab[1];
      // noise: envelope (float32) times the frame's white spectrum, iSTFT frame
      for (q = 0; q < nbin; q++) {
        var e = 0;
        for (k = 0; k < nb; k++) e += fr(h[hr + 2 + hb + k]) * tb.nim32[k * nbin + q];
        env[q] = fr(Math.exp(Math.min(fr(e), 4.0)));
      }
      whiteFrame(sw, t, nbin, wr, wi);
      for (q = 0; q < nbin; q++) { xr[q] = env[q] * wr[q]; xi[q] = env[q] * wi[q]; }
      irfft(pN, xr, xi, td);
      var o0 = t * HOP;
      for (i = 0; i < nfft; i++) { ola[o0 + i] += td[i] * win[i]; norm[o0 + i] += win[i] * win[i]; }
    }
    var out = new Float64Array(n), cum = 0, pad = nfft / 2;
    for (var s = 0; s < n; s++) {
      var pp = s / HOP, lo2 = Math.min(Math.floor(pp), T - 1), hi2 = Math.min(lo2 + 1, T - 1);
      var al = pp - lo2;
      al = al < 0 ? 0 : al > 1 ? 1 : al;
      cum += TWO_PI * (f0[lo2] * (1 - al) + f0[hi2] * al) / SR;
      var x = (cum % TWO_PI) * (TABLE / TWO_PI);
      var ii = Math.min(Math.trunc(x), TABLE - 1), ff = fr(x - ii), a32 = fr(al);
      var cLo = cubic(tabs, lo2 * (TABLE + 3) + ii, ff), cHi = cubic(tabs, hi2 * (TABLE + 3) + ii, ff);
      var harm = fr(cLo + fr(fr(cHi - cLo) * a32));
      var nm = norm[s + pad], nz = ola[s + pad];
      out[s] = harm + (nm > 1e-11 ? nz / nm : nz);
    }
    return out;
  }
  // Catmull-Rom read of one table row at float32 arithmetic (nano_dsp._harm_from_tables).
  function cubic(tabs, at, f) {
    var p0 = tabs[at], p1 = tabs[at + 1], p2 = tabs[at + 2], p3 = tabs[at + 3];
    var inner = fr(fr(fr(3.0 * fr(p1 - p2)) + p3) - p0);
    var mid = fr(fr(fr(fr(fr(2.0 * p0) - fr(5.0 * p1)) + fr(4.0 * p2)) - p3) + fr(f * inner));
    var outer = fr(fr(p2 - p0) + fr(f * mid));
    return fr(p1 + fr(fr(0.5 * f) * outer));
  }

  /* ---------------------------------------------------------------- voice */

  function createVoice(opts) {
    var meta = opts && opts.meta, front = opts && opts.front, dsp = opts && opts.dsp;
    if (!meta || !(front instanceof Uint8Array) || !(dsp instanceof Uint8Array)) fail("createVoice needs {meta, front: Uint8Array, dsp: Uint8Array}");
    if (meta.runtime !== "nano_dsp_js") fail("meta.json runtime is " + meta.runtime + ", expected nano_dsp_js");
    var m = meta.offsets;
    if (!m) fail("meta.json has no offsets");
    if (m.NANO_LOCAL_FEATURES !== 1) fail("the voice does not declare NANO_LOCAL_FEATURES 1");
    if ((m.NANO_WEIGHT_FORMAT || 0) !== 0) fail("expected int8 front weights");
    if (front.byteLength !== meta.front_bytes || front.byteLength !== m.NANO_FRONT_BYTES)
      fail("front blob is " + front.byteLength + " bytes, meta.json says " + meta.front_bytes + " / header " + m.NANO_FRONT_BYTES);
    if (dsp.byteLength !== meta.dsp_bytes) fail("dsp blob is " + dsp.byteLength + " bytes, meta.json says " + meta.dsp_bytes);
    if (meta.sample_rate !== SR || meta.hop !== HOP) fail("expected 24 kHz / hop 256, meta.json says " + meta.sample_rate + " / " + meta.hop);
    if (m.NANO_MELS !== 100) fail("expected 100 mel channels, header says " + m.NANO_MELS);
    var plan = frontPlan(front, m), dw = dspWeights(dsp, meta), seed = meta.seed;
    seedWords(seed);
    function checkIds(ids) {
      if (!ids || !ids.length) fail("empty phoneme id sequence");
      var a = Int32Array.from(ids);
      for (var i = 0; i < a.length; i++) if (a[i] < 0 || a[i] >= plan.V || a[i] !== Number(ids[i])) fail("phoneme id " + ids[i] + " at " + i + " is outside the " + plan.V + "-symbol vocabulary");
      return a;
    }
    var voice = {
      sampleRate: SR,
      params: meta.params,
      durations: function (ids, lengthScale) {
        var ls = lengthScale === undefined ? 1.0 : Number(lengthScale);
        if (!(ls > 0)) fail("length scale must be positive, got " + lengthScale);
        return durationForward(plan, checkIds(ids), ls);
      },
      mel: function (ids, durs) { return acousticForward(plan, checkIds(ids), durs); },
      controls: function (mel, frames) { return controls(dw, mel, frames); },
      render: function (mel, frames, seedOverride) { return render(dw, mel, frames, seedOverride === undefined ? seed : seedOverride); },
      tables: function () { return synthTables(dw); },
      white: function (f) { var re = new Float32Array(dw.noiseFft / 2 + 1), im = new Float32Array(re.length); whiteFrame(seedWords(seed), f, re.length, re, im); return {re: re, im: im}; },
      // Synthesizer.synthesize_ids: durations, mel, decoder, clipped to [-1, 1].
      synthesize: function (ids, o) {
        o = o || {};
        var a = checkIds(ids), durs = voice.durations(a, o.lengthScale);
        var am = acousticForward(plan, a, durs);
        var y = render(dw, am.mel, am.frames, o.seed === undefined ? seed : String(o.seed));
        var pcm = new Float32Array(y.length);
        for (var i = 0; i < y.length; i++) { var v = fr(y[i]); pcm[i] = v > 1 ? 1 : v < -1 ? -1 : v; }
        return {pcm: pcm, durations: durs, frames: am.frames, sampleRate: SR};
      },
    };
    return voice;
  }

  root.SaanoNanoDsp = {createVoice: createVoice, erf: erfSeries};
})(typeof globalThis !== "undefined" ? globalThis : this);
