/* snt_front_f32.c -- portable fp32 reference of the saanoTTS front half
 * (duration student + token_context acoustic student + optional output
 * adapter). Plain C99, no malloc: all working memory comes from the caller's
 * arena. Semantics mirror tools/train_roota_piper_duration_student.py
 * (predict_durations, the dashboard duration_source="student" path) and
 * tools/train_roota_piper_latent_student.py (expand_features +
 * predict_latent_tensor) exactly; see snt_front_f32.h for the graph.
 *
 * Weight slot order is fixed by tools/export_front_golden.py; meta.bin
 * carries dims plus a per-slot (offset,size) table, and init() verifies each
 * slot size against the shape implied by the dims, so a stale or reordered
 * blob fails loudly instead of synthesizing garbage.
 *
 * Float-semantics notes (required for exact integer duration parity):
 *  - torch.linspace(0,1,n) computes step = 1/(n-1) in fp32, fills i < n/2 as
 *    step*i and i >= n/2 as fma(-step, n-1-i, 1) -- replicated below.
 *  - torch.round is round-half-to-even; rintf matches under the default
 *    FE_TONEAREST rounding mode.
 *  - expand_features builds token_pos/duration_pos in double then casts to
 *    fp32 -- replicated below.
 */
#include "snt_front_f32.h"

#include <math.h>
#include <string.h>

/* ---- meta.bin parsing ------------------------------------------------- */

static int32_t rd_i32(const unsigned char *p) {
    /* little-endian, alignment-safe */
    return (int32_t)((uint32_t)p[0] | ((uint32_t)p[1] << 8) |
                     ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24));
}

/* Slots used by a residual block group (scale, conv1 w/b, conv2 w/b). */
static long block_slot_floats(int r, int h, int k) {
    int part = r % 5;
    if (part == 0) return 1;                    /* learned scalar scale */
    if (part == 1 || part == 3) return (long)h * h * k;
    return h;
}

static int adapter_slot_count(int mode) {
    switch (mode) {
    case SNT_FRONT_ADAPTER_NONE: return 0;
    case SNT_FRONT_ADAPTER_AFFINE: return 2;
    case SNT_FRONT_ADAPTER_DEPTHWISE: return 3;
    case SNT_FRONT_ADAPTER_LOWRANK: return 6;
    case SNT_FRONT_ADAPTER_DEPTHWISE_LOWRANK: return 7;
    default: return -1;
    }
}

/* Expected float count of weight slot `idx` given the model dims;
 * returns -1 for an out-of-range slot. Order must match the exporter. */
static long slot_floats(const snt_front_model *m, int idx) {
    int base = 0;
    if (idx < 0) return -1;
    /* duration student */
    if (idx == 0) return (long)m->d_vocab * m->d_hidden;
    if (idx == 1) return (long)m->d_hidden * (m->d_hidden + 3);
    if (idx == 2) return m->d_hidden;
    base = 3;
    if (idx < base + 5 * m->d_depth)
        return block_slot_floats(idx - base, m->d_hidden, m->d_kernel);
    base += 5 * m->d_depth;
    if (idx == base) return m->d_hidden;        /* output.weight [1,h,1] */
    if (idx == base + 1) return 1;              /* output.bias */
    base += 2;
    /* acoustic student */
    if (idx == base) return (long)m->a_vocab * m->a_hidden;
    if (idx == base + 1) return (long)m->a_hidden * (m->a_hidden + 2);
    if (idx == base + 2) return m->a_hidden;
    base += 3;
    if (idx < base + 5 * m->a_token_depth)
        return block_slot_floats(idx - base, m->a_hidden, m->a_kernel);
    base += 5 * m->a_token_depth;
    if (idx == base) return (long)m->a_hidden * (m->a_hidden + 3);
    if (idx == base + 1) return m->a_hidden;
    base += 2;
    if (idx < base + 5 * m->a_depth)
        return block_slot_floats(idx - base, m->a_hidden, m->a_kernel);
    base += 5 * m->a_depth;
    if (idx == base) return (long)m->a_out * m->a_hidden;
    if (idx == base + 1) return m->a_out;
    base += 2;
    /* optional output adapter */
    if (m->adapter_mode == SNT_FRONT_ADAPTER_DEPTHWISE ||
        m->adapter_mode == SNT_FRONT_ADAPTER_DEPTHWISE_LOWRANK) {
        if (idx == base) return (long)m->a_out * m->adapter_kernel;
        base += 1;
    }
    if (m->adapter_mode == SNT_FRONT_ADAPTER_LOWRANK ||
        m->adapter_mode == SNT_FRONT_ADAPTER_DEPTHWISE_LOWRANK) {
        if (idx == base) return (long)m->adapter_rank * m->a_out;
        if (idx == base + 1) return m->adapter_rank;
        if (idx == base + 2) return (long)m->a_out * m->adapter_rank;
        if (idx == base + 3) return m->a_out;
        base += 4;
    }
    if (m->adapter_mode != SNT_FRONT_ADAPTER_NONE) {
        if (idx == base) return m->a_out;       /* adapter.scale */
        if (idx == base + 1) return m->a_out;   /* adapter.bias */
    }
    return -1;
}

int snt_front_init(snt_front_model *m,
                   const void *meta, size_t meta_bytes,
                   const float *weights, size_t weight_floats) {
    const unsigned char *p = (const unsigned char *)meta;
    int i, expected_tensors, adapter_slots;
    if (!m || !p || !weights) return -1;
    if (meta_bytes < 18 * 4) return -2;
    memset(m, 0, sizeof *m);
    if (rd_i32(p) != (int32_t)SNT_FRONT_MAGIC) return -3;
    if (rd_i32(p + 4) != 1) return -4; /* version */
    m->d_vocab = rd_i32(p + 8);
    m->d_hidden = rd_i32(p + 12);
    m->d_depth = rd_i32(p + 16);
    m->d_kernel = rd_i32(p + 20);
    m->d_max_tokens = rd_i32(p + 24);
    m->d_max_duration = rd_i32(p + 28);
    m->a_vocab = rd_i32(p + 32);
    m->a_hidden = rd_i32(p + 36);
    m->a_token_depth = rd_i32(p + 40);
    m->a_depth = rd_i32(p + 44);
    m->a_kernel = rd_i32(p + 48);
    m->a_out = rd_i32(p + 52);
    m->adapter_mode = rd_i32(p + 56);
    m->adapter_kernel = rd_i32(p + 60);
    m->adapter_rank = rd_i32(p + 64);
    m->n_tensors = rd_i32(p + 68);
    if (m->d_vocab <= 0 || m->d_hidden <= 0 || m->d_depth <= 0 ||
        m->d_kernel <= 0 || m->d_kernel % 2 == 0 ||
        m->d_max_tokens <= 0 || m->d_max_duration <= 0)
        return -5;
    if (m->a_vocab <= 0 || m->a_hidden <= 0 || m->a_token_depth <= 0 ||
        m->a_depth <= 0 || m->a_kernel <= 0 || m->a_kernel % 2 == 0 ||
        m->a_out <= 0)
        return -5;
    adapter_slots = adapter_slot_count(m->adapter_mode);
    if (adapter_slots < 0) return -5;
    if ((m->adapter_mode == SNT_FRONT_ADAPTER_DEPTHWISE ||
         m->adapter_mode == SNT_FRONT_ADAPTER_DEPTHWISE_LOWRANK) &&
        (m->adapter_kernel <= 0 || m->adapter_kernel % 2 == 0))
        return -5;
    if ((m->adapter_mode == SNT_FRONT_ADAPTER_LOWRANK ||
         m->adapter_mode == SNT_FRONT_ADAPTER_DEPTHWISE_LOWRANK) &&
        m->adapter_rank <= 0)
        return -5;
    expected_tensors = (5 + 5 * m->d_depth) +
                       (7 + 5 * m->a_token_depth + 5 * m->a_depth) +
                       adapter_slots;
    if (m->n_tensors != expected_tensors ||
        m->n_tensors > SNT_FRONT_MAX_TENSORS)
        return -6;
    if (meta_bytes < (size_t)(18 + 2 * m->n_tensors) * 4) return -2;
    for (i = 0; i < m->n_tensors; i++) {
        long off = rd_i32(p + 72 + 8 * i);
        long size = rd_i32(p + 76 + 8 * i);
        long want = slot_floats(m, i);
        if (off < 0 || size <= 0 || want < 0) return -7;
        if (size != want) return -8;
        if ((size_t)off + (size_t)size > weight_floats) return -9;
        m->w[i] = weights + off;
    }
    return 0;
}

/* ---- kernels ----------------------------------------------------------- */

/* Conv1d, PyTorch "same" semantics: pad = K/2 zeros each side.
 * w layout [out_ch, in_ch, K]; x,out channel-major [ch][T].
 * accumulate!=0: out += bias + conv; else out = bias + conv. */
static void fr_conv1d(float *out, const float *x, const float *w,
                      const float *b, int in_ch, int out_ch, int T,
                      int K, int accumulate) {
    int pad = K / 2;
    int oc, ic, k, t;
    for (oc = 0; oc < out_ch; oc++) {
        float *orow = out + (long)oc * T;
        float bias = b[oc];
        if (accumulate)
            for (t = 0; t < T; t++) orow[t] += bias;
        else
            for (t = 0; t < T; t++) orow[t] = bias;
        for (ic = 0; ic < in_ch; ic++) {
            const float *xrow = x + (long)ic * T;
            const float *wrow = w + ((long)oc * in_ch + ic) * K;
            for (k = 0; k < K; k++) {
                float wv = wrow[k];
                int off = k - pad;
                int lo = off < 0 ? -off : 0;
                int hi = off > 0 ? T - off : T;
                if (wv != 0.0f)
                    for (t = lo; t < hi; t++) orow[t] += wv * xrow[t + off];
            }
        }
    }
}

static void fr_silu(float *x, long n) {
    long i;
    for (i = 0; i < n; i++) {
        float v = x[i];
        x[i] = v / (1.0f + expf(-v)); /* aten silu: x / (1 + exp(-x)) */
    }
}

/* torch.linspace(0, 1, n) with exact CPU-kernel float semantics. */
static void fr_linspace01(float *dst, int n) {
    int half = n / 2, i;
    float step;
    if (n <= 0) return;
    if (n == 1) { dst[0] = 0.0f; return; }
    step = 1.0f / (float)(n - 1);
    for (i = 0; i < half; i++) dst[i] = step * (float)i;
    for (i = half; i < n; i++)
        dst[i] = fmaf(-step, (float)(n - 1 - i), 1.0f);
}

static void fr_tap(const snt_front_model *m, const char *name,
                   const float *data, int ch, long len) {
    if (m->stage_cb) m->stage_cb(name, data, ch, (int)len, m->stage_user);
}

/* Run `depth` residual blocks starting at slot base (5 slots per block). */
static void fr_run_blocks(const snt_front_model *m, int slot_base, int depth,
                          int h, int K, int T, float *x, float *t, float *u) {
    long n = (long)h * T;
    long i;
    int b;
    for (b = 0; b < depth; b++) {
        const float *const *ws = &m->w[slot_base + 5 * b];
        float scale = ws[0][0];
        fr_conv1d(t, x, ws[1], ws[2], h, h, T, K, 0);
        fr_silu(t, n);
        fr_conv1d(u, t, ws[3], ws[4], h, h, T, K, 0);
        for (i = 0; i < n; i++) x[i] += scale * u[i];
    }
}

/* ---- duration student --------------------------------------------------- */

size_t snt_front_duration_arena_floats(const snt_front_model *m,
                                       int n_tokens) {
    if (!m || n_tokens <= 0) return 0;
    /* feat (h+3)*N + x h*N + t h*N + u h*N + logits N */
    return (size_t)((long)(4 * m->d_hidden + 4) * n_tokens);
}

long snt_front_durations(const snt_front_model *m,
                         const int32_t *ids, int n_tokens,
                         float length_scale,
                         int32_t *dur_out,
                         float *arena, size_t arena_floats) {
    int h, N, i, c;
    float *feat, *x, *t, *u, *logits;
    float length_hint;
    long total;
    if (!m || !ids || n_tokens <= 0 || !dur_out || !arena) return -1;
    if (!(length_scale > 0.0f) || !isfinite(length_scale)) return -1;
    if (arena_floats < snt_front_duration_arena_floats(m, n_tokens)) return -2;
    h = m->d_hidden;
    N = n_tokens;
    feat = arena;
    x = feat + (long)(h + 3) * N;
    t = x + (long)h * N;
    u = t + (long)h * N;
    logits = u + (long)h * N;

    /* input features: embedded ids [h][N] + positions, length_hint, valid */
    for (i = 0; i < N; i++) {
        long id = ids[i];
        const float *erow;
        if (id < 0 || id >= m->d_vocab) return -3;
        erow = m->w[0] + id * h;
        for (c = 0; c < h; c++) feat[(long)c * N + i] = erow[c];
    }
    fr_linspace01(feat + (long)h * N, N);
    /* lengths = N (full mask); hint = log1p(N)/log1p(max_tokens), fp32 */
    length_hint = log1pf((float)N) / (float)log1p((double)m->d_max_tokens);
    for (i = 0; i < N; i++) {
        feat[(long)(h + 1) * N + i] = length_hint;
        feat[(long)(h + 2) * N + i] = 1.0f; /* valid_hint, mask all-true */
    }
    fr_tap(m, "dur_feats", feat + (long)h * N, 3, N);

    fr_conv1d(x, feat, m->w[1], m->w[2], h + 3, h, N, 1, 0);
    fr_run_blocks(m, 3, m->d_depth, h, m->d_kernel, N, x, t, u);
    fr_conv1d(logits, x, m->w[3 + 5 * m->d_depth],
              m->w[4 + 5 * m->d_depth], h, 1, N, 1, 0);
    fr_tap(m, "dur_log", logits, 1, N);

    /* predict_durations: exp -> clamp_min(1) -> *scale -> round -> clamp */
    total = 0;
    for (i = 0; i < N; i++) {
        float v = expf(logits[i]);
        if (v < 1.0f) v = 1.0f;
        v = rintf(v * length_scale); /* torch.round: ties to even */
        if (v < 1.0f) v = 1.0f;
        if (v > (float)m->d_max_duration) v = (float)m->d_max_duration;
        dur_out[i] = (int32_t)v;
        total += (long)dur_out[i];
    }
    return total;
}

/* ---- acoustic student ---------------------------------------------------- */

static long fr_big_rows(const snt_front_model *m) {
    return m->a_out > m->a_hidden + 3 ? m->a_out : m->a_hidden + 3;
}

static long fr_t_rows(const snt_front_model *m) {
    return m->adapter_rank > m->a_hidden ? m->adapter_rank : m->a_hidden;
}

size_t snt_front_latent_arena_floats(const snt_front_model *m,
                                     int n_tokens, long frames) {
    if (!m || n_tokens <= 0 || frames < n_tokens) return 0;
    return (size_t)((long)m->a_hidden * n_tokens +
                    (fr_big_rows(m) + 2 * m->a_hidden + fr_t_rows(m)) * frames);
}

int snt_front_latent(const snt_front_model *m,
                     const int32_t *ids, const int32_t *durations,
                     int n_tokens, long frames,
                     float *latent_out,
                     float *arena, size_t arena_floats) {
    int h, C, N, i, c;
    long T, sum, ti, j, pos;
    int A0, F0, O0, AD;
    float *tok, *big, *x, *t, *u;
    float maxd, log_maxd;
    if (!m || !ids || !durations || n_tokens <= 0 || frames <= 0 ||
        !latent_out || !arena)
        return -1;
    sum = 0;
    for (i = 0; i < n_tokens; i++) {
        if (durations[i] < 1) return -3;
        sum += durations[i];
    }
    if (sum != frames) return -3;
    if (frames > 0x7FFFFFFFL / (long)(m->a_out > m->a_hidden + 3
                                          ? m->a_out : m->a_hidden + 3))
        return -1;
    if (arena_floats < snt_front_latent_arena_floats(m, n_tokens, frames))
        return -2;
    h = m->a_hidden;
    C = m->a_out;
    N = n_tokens;
    T = frames;
    A0 = 5 + 5 * m->d_depth;                 /* acoustic slot base */
    F0 = A0 + 3 + 5 * m->a_token_depth;      /* frame_input_proj slot */
    O0 = F0 + 2 + 5 * m->a_depth;            /* output conv slot */
    AD = O0 + 2;                             /* adapter slot base */

    tok = arena;
    big = tok + (long)h * N;
    x = big + fr_big_rows(m) * T;
    t = x + (long)h * T;
    u = t + fr_t_rows(m) * T;

    /* -- token stage: embed + (token_pos, duration_hint) -> token blocks -- */
    for (i = 0; i < N; i++) {
        long id = ids[i];
        const float *erow;
        if (id < 0 || id >= m->a_vocab) return -3;
        erow = m->w[A0] + id * h;
        for (c = 0; c < h; c++) big[(long)c * N + i] = erow[c];
    }
    fr_linspace01(big + (long)h * N, N);
    maxd = 1.0f;
    for (i = 0; i < N; i++)
        if ((float)durations[i] > maxd) maxd = (float)durations[i];
    log_maxd = log1pf(maxd);
    for (i = 0; i < N; i++)
        big[(long)(h + 1) * N + i] = log1pf((float)durations[i]) / log_maxd;
    fr_conv1d(tok, big, m->w[A0 + 1], m->w[A0 + 2], h + 2, h, N, 1, 0);
    fr_run_blocks(m, A0 + 3, m->a_token_depth, h, m->a_kernel, N, tok, t, u);
    fr_tap(m, "tok_ctx", tok, h, N);

    /* -- expand token context + positional features to frames -- */
    for (c = 0; c < h; c++) {
        float *orow = big + (long)c * T;
        const float *trow = tok + (long)c * N;
        pos = 0;
        for (ti = 0; ti < N; ti++) {
            float v = trow[ti];
            for (j = 0; j < durations[ti]; j++) orow[pos++] = v;
        }
    }
    fr_linspace01(big + (long)h * T, (int)T); /* frame_pos */
    {
        /* expand_features: python doubles cast to fp32 */
        long tcount = N > 1 ? N - 1 : 1;
        float *tp = big + (long)(h + 1) * T;
        float *dp = big + (long)(h + 2) * T;
        pos = 0;
        for (ti = 0; ti < N; ti++) {
            long d = durations[ti];
            float tv = (float)((double)ti / (double)tcount);
            for (j = 0; j < d; j++) {
                tp[pos] = tv;
                dp[pos] = d == 1 ? 0.0f
                                 : (float)((double)j / (double)(d - 1));
                pos++;
            }
        }
    }
    fr_tap(m, "frame_feats", big + (long)h * T, 3, T);

    /* -- frame stage -> latent [C][T] -- */
    fr_conv1d(x, big, m->w[F0], m->w[F0 + 1], h + 3, h, (int)T, 1, 0);
    fr_run_blocks(m, F0 + 2, m->a_depth, h, m->a_kernel, (int)T, x, t, u);
    fr_conv1d(latent_out, x, m->w[O0], m->w[O0 + 1], h, C, (int)T, 1, 0);
    fr_tap(m, "latent_base", latent_out, C, T);

    /* -- optional output adapter (full-channel scope) -- */
    if (m->adapter_mode != SNT_FRONT_ADAPTER_NONE) {
        int slot = AD;
        const float *ad_scale, *ad_bias;
        long n = (long)C * T;
        if (m->adapter_mode == SNT_FRONT_ADAPTER_DEPTHWISE ||
            m->adapter_mode == SNT_FRONT_ADAPTER_DEPTHWISE_LOWRANK) {
            /* depthwise conv, weight [C,1,K], no bias */
            int K = m->adapter_kernel, pad = K / 2, k;
            const float *dw = m->w[slot++];
            for (c = 0; c < C; c++) {
                float *orow = big + (long)c * T;
                const float *xrow = latent_out + (long)c * T;
                const float *wrow = dw + (long)c * K;
                for (j = 0; j < T; j++) orow[j] = 0.0f;
                for (k = 0; k < K; k++) {
                    float wv = wrow[k];
                    int off = k - pad;
                    long lo = off < 0 ? -(long)off : 0;
                    long hi = off > 0 ? T - off : T;
                    if (wv != 0.0f)
                        for (j = lo; j < hi; j++)
                            orow[j] += wv * xrow[j + off];
                }
            }
        } else {
            memcpy(big, latent_out, (size_t)n * sizeof(float));
        }
        if (m->adapter_mode == SNT_FRONT_ADAPTER_LOWRANK ||
            m->adapter_mode == SNT_FRONT_ADAPTER_DEPTHWISE_LOWRANK) {
            /* big += up_bias + up(tanh(down(big))) */
            int rank = m->adapter_rank;
            long zn = (long)rank * T;
            fr_conv1d(t, big, m->w[slot], m->w[slot + 1], C, rank, (int)T,
                      1, 0);
            for (j = 0; j < zn; j++) t[j] = tanhf(t[j]);
            fr_conv1d(big, t, m->w[slot + 2], m->w[slot + 3], rank, C,
                      (int)T, 1, 1);
            slot += 4;
        }
        ad_scale = m->w[slot];
        ad_bias = m->w[slot + 1];
        for (c = 0; c < C; c++) {
            float s = ad_scale[c], b = ad_bias[c];
            float *orow = latent_out + (long)c * T;
            const float *irow = big + (long)c * T;
            for (j = 0; j < T; j++) orow[j] = irow[j] * s + b;
        }
    }
    fr_tap(m, "latent", latent_out, C, T);
    return 0;
}
