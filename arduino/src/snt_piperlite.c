/* snt_piperlite.c -- portable fp32 reference of the piperlite decoder.
 * Plain C99, no malloc: all working memory comes from the caller's arena.
 * Semantics mirror tools/train_roota_piper_decoder_student.py exactly
 * (variant "piperlite", leaky_relu): see snt_piperlite.h for the graph.
 *
 * Weight slot order is fixed by tools/export_piperlite_golden.py; meta.bin
 * carries dims plus a per-slot (offset,size) table, and init() verifies each
 * slot size against the shape implied by the dims, so a stale or reordered
 * blob fails loudly instead of decoding garbage.
 */
#include "snt_piperlite.h"

#include <math.h>
#include <string.h>

#define PL_HOP SNT_PIPERLITE_HOP

/* ---- meta.bin parsing ------------------------------------------------- */

static int32_t rd_i32(const unsigned char *p) {
    /* little-endian, alignment-safe */
    return (int32_t)((uint32_t)p[0] | ((uint32_t)p[1] << 8) |
                     ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24));
}

static float rd_f32(const unsigned char *p) {
    union { uint32_t u; float f; } v;
    v.u = (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
          ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
    return v.f;
}

/* Expected float count of weight slot `idx` given the model dims;
 * returns -1 for an out-of-range slot. Order must match the exporter. */
static long slot_floats(const snt_piperlite_model *m, int idx) {
    static const int bk[3] = {3, 5, 7}; /* residual-bank branch kernels */
    if (idx < 0) return -1;
    if (idx == 0) return (long)m->c0 * m->in_ch * 7;
    if (idx == 1) return m->c0;
    /* three upsample stages, each: up_w, up_b, then 3 branches x 4 tensors */
    {
        static const int up_k[3] = {16, 16, 8};
        int in_c[3], out_c[3];
        int s, base = 2;
        in_c[0] = m->c0; out_c[0] = m->c1;
        in_c[1] = m->c1; out_c[1] = m->c2;
        in_c[2] = m->c2; out_c[2] = m->c3;
        for (s = 0; s < 3; s++) {
            if (idx == base) return (long)in_c[s] * out_c[s] * up_k[s];
            if (idx == base + 1) return out_c[s];
            if (idx < base + 14) {
                int r = idx - (base + 2);      /* 0..11 within the bank */
                int branch = r / 4, part = r % 4;
                int ch = out_c[s];
                if (part == 0 || part == 2) return (long)ch * ch * bk[branch];
                return ch;
            }
            base += 14;
        }
        /* base == 44 here */
        if (idx == base) return (long)m->c3 * 7;      /* post.weight [1,c3,7] */
        if (idx == base + 1) return 1;                 /* post.bias */
        base += 2;
        if (m->pf_channels > 0) {
            int l;
            if (idx == base) return (long)m->pf_channels * m->pf_kernel;
            if (idx == base + 1) return m->pf_channels;
            base += 2;
            for (l = 0; l < m->pf_layers; l++) {
                if (idx == base) return 1;             /* unit scale */
                if (idx == base + 1 || idx == base + 3)
                    return (long)m->pf_channels * m->pf_channels * 3;
                if (idx == base + 2 || idx == base + 4) return m->pf_channels;
                base += 5;
            }
            if (idx == base) return (long)m->pf_channels * m->pf_kernel;
            if (idx == base + 1) return 1;
        }
    }
    return -1;
}

int snt_piperlite_init(snt_piperlite_model *m,
                       const void *meta, size_t meta_bytes,
                       const float *weights, size_t weight_floats) {
    const unsigned char *p = (const unsigned char *)meta;
    int i, expected_tensors;
    if (!m || !p || !weights) return -1;
    if (meta_bytes < 12 * 4) return -2;
    memset(m, 0, sizeof *m);
    if (rd_i32(p) != (int32_t)SNT_PIPERLITE_MAGIC) return -3;
    if (rd_i32(p + 4) != 1) return -4; /* version */
    m->in_ch = rd_i32(p + 8);
    m->c0 = rd_i32(p + 12);
    m->c1 = rd_i32(p + 16);
    m->c2 = rd_i32(p + 20);
    m->c3 = rd_i32(p + 24);
    m->pf_channels = rd_i32(p + 28);
    m->pf_layers = rd_i32(p + 32);
    m->pf_kernel = rd_i32(p + 36);
    m->pf_scale = rd_f32(p + 40);
    m->n_tensors = rd_i32(p + 44);
    if (m->in_ch <= 0 || m->c0 <= 0 || m->c1 <= 0 || m->c2 <= 0 || m->c3 <= 0)
        return -5;
    if (m->pf_channels < 0 || m->pf_layers < 0 ||
        (m->pf_channels == 0) != (m->pf_layers == 0))
        return -5;
    if (m->pf_channels > 0 && (m->pf_kernel <= 0 || m->pf_kernel % 2 == 0))
        return -5;
    expected_tensors =
        46 + (m->pf_channels > 0 ? 4 + 5 * m->pf_layers : 0);
    if (m->n_tensors != expected_tensors ||
        m->n_tensors > SNT_PIPERLITE_MAX_TENSORS)
        return -6;
    if (meta_bytes < (size_t)(12 + 2 * m->n_tensors) * 4) return -2;
    for (i = 0; i < m->n_tensors; i++) {
        long off = rd_i32(p + 48 + 8 * i);
        long size = rd_i32(p + 52 + 8 * i);
        long want = slot_floats(m, i);
        if (off < 0 || size <= 0 || want < 0) return -7;
        if (size != want) return -8;
        if ((size_t)off + (size_t)size > weight_floats) return -9;
        m->w[i] = weights + off;
    }
    return 0;
}

/* ---- kernels ----------------------------------------------------------- */

static void pl_leaky_copy(float *dst, const float *src, long n, float slope) {
    long i;
    for (i = 0; i < n; i++) {
        float v = src[i];
        dst[i] = v > 0.0f ? v : slope * v;
    }
}

/* Conv1d, PyTorch "same" semantics: pad = dil*(K/2) zeros each side.
 * w layout [out_ch, in_ch, K]; x,out channel-major [ch][T].
 * accumulate!=0: out += bias + conv; else out = bias + conv. */
static void pl_conv1d(float *out, const float *x, const float *w,
                      const float *b, int in_ch, int out_ch, int T,
                      int K, int dil, int accumulate) {
    int pad = dil * (K / 2);
    int oc, ic, k;
    for (oc = 0; oc < out_ch; oc++) {
        float *orow = out + (long)oc * T;
        float bias = b[oc];
        int t;
        if (accumulate)
            for (t = 0; t < T; t++) orow[t] += bias;
        else
            for (t = 0; t < T; t++) orow[t] = bias;
        for (ic = 0; ic < in_ch; ic++) {
            const float *xrow = x + (long)ic * T;
            const float *wrow = w + ((long)oc * in_ch + ic) * K;
            for (k = 0; k < K; k++) {
                float wv = wrow[k];
                int off = k * dil - pad;
                int lo = off < 0 ? -off : 0;
                int hi = off > 0 ? T - off : T;
                if (wv != 0.0f)
                    for (t = lo; t < hi; t++) orow[t] += wv * xrow[t + off];
            }
        }
    }
}

/* ConvTranspose1d, w layout [in_ch, out_ch, K]; out length
 * (T-1)*stride - 2*pad + K. Scatter form, exact PyTorch semantics. */
static void pl_convtr1d(float *out, const float *x, const float *w,
                        const float *b, int in_ch, int out_ch, int T,
                        int K, int stride, int pad) {
    int L = (T - 1) * stride - 2 * pad + K;
    int oc, ic, k;
    for (oc = 0; oc < out_ch; oc++) {
        float *orow = out + (long)oc * L;
        float bias = b[oc];
        int j;
        for (j = 0; j < L; j++) orow[j] = bias;
        for (ic = 0; ic < in_ch; ic++) {
            const float *xrow = x + (long)ic * T;
            const float *wrow = w + ((long)ic * out_ch + oc) * K;
            for (k = 0; k < K; k++) {
                float wv = wrow[k];
                int shift = k - pad;
                /* j = t*stride + shift must satisfy 0 <= j < L */
                int t_lo = 0, t_hi = T, t;
                long lim;
                if (shift >= L) continue;
                if (shift < 0)
                    t_lo = (-shift + stride - 1) / stride;
                lim = ((long)L - 1 - shift) / stride + 1;
                if (lim < t_hi) t_hi = (int)lim;
                for (t = t_lo; t < t_hi; t++)
                    orow[(long)t * stride + shift] += wv * xrow[t];
            }
        }
    }
}

/* PiperResidualBank: 3 branches (kernel, dil1, dil2) = (3,1,2), (5,2,6),
 * (7,3,12); each branch y1 = conv1(lrelu(x)) + x, y2 = conv2(lrelu(y1)) + y1;
 * bank out = (y2_0 + y2_1 + y2_2) / 3. `ws` points at the 12 bank slots.
 * Buffers sum/t/u are ch*T floats each; out may alias sum. */
static void pl_res_bank(float *out, const float *x, const float *const *ws,
                        int ch, int T, float *sum, float *t, float *u) {
    static const int bk[3] = {3, 5, 7};
    static const int bd1[3] = {1, 2, 3};
    static const int bd2[3] = {2, 6, 12};
    long n = (long)ch * T;
    long i;
    int br;
    memset(sum, 0, (size_t)n * sizeof(float));
    for (br = 0; br < 3; br++) {
        const float *w1 = ws[br * 4 + 0], *b1 = ws[br * 4 + 1];
        const float *w2 = ws[br * 4 + 2], *b2 = ws[br * 4 + 3];
        pl_leaky_copy(t, x, n, 0.1f);
        pl_conv1d(u, t, w1, b1, ch, ch, T, bk[br], bd1[br], 0);
        for (i = 0; i < n; i++) u[i] += x[i];      /* y1 */
        pl_leaky_copy(t, u, n, 0.1f);
        pl_conv1d(sum, t, w2, b2, ch, ch, T, bk[br], bd2[br], 1); /* += conv */
        for (i = 0; i < n; i++) sum[i] += u[i];    /* += y1 */
    }
    for (i = 0; i < n; i++) out[i] = sum[i] * (1.0f / 3.0f);
}

/* ---- arena sizing ------------------------------------------------------ */

static long pl_lane_floats(const snt_piperlite_model *m, int frames) {
    long s = (long)m->c0 * frames;
    long v = (long)m->c1 * 8 * frames;
    if (v > s) s = v;
    v = (long)m->c2 * 64 * frames;
    if (v > s) s = v;
    v = (long)m->c3 * PL_HOP * frames;
    if (v > s) s = v;
    v = (long)(m->pf_channels > 0 ? m->pf_channels : 1) * PL_HOP * frames;
    if (v > s) s = v;
    return s;
}

size_t snt_piperlite_arena_floats(const snt_piperlite_model *m, int frames) {
    if (!m || frames <= 0) return 0;
    return (size_t)(4 * pl_lane_floats(m, frames));
}

/* ---- forward ------------------------------------------------------------ */

static void pl_tap(const snt_piperlite_model *m, const char *name,
                   const float *data, int ch, long len) {
    if (m->stage_cb) m->stage_cb(name, data, ch, (int)len, m->stage_user);
}

int snt_piperlite_synthesize(const snt_piperlite_model *m,
                             const float *z, int frames,
                             float *audio_out,
                             float *arena, size_t arena_floats) {
    long lane, n_samples;
    float *ln[4];
    float *x;
    int cur, s, i;
    long j;
    /* per-stage geometry: up kernel/stride/pad, weight slot bases */
    static const int up_k[3] = {16, 16, 8};
    static const int up_s[3] = {8, 8, 4};
    static const int up_p[3] = {4, 4, 2};
    static const char *up_name[3] = {"up0", "up1", "up2"};
    static const char *mix_name[3] = {"stage0_mix", "stage1_mix", "stage2_mix"};

    if (!m || !z || frames <= 0 || !audio_out || !arena) return -1;
    if (arena_floats < snt_piperlite_arena_floats(m, frames)) return -2;
    lane = pl_lane_floats(m, frames);
    for (i = 0; i < 4; i++) ln[i] = arena + (long)i * lane;
    n_samples = (long)frames * PL_HOP;

    /* pre: Conv1d(in_ch, c0, 7, pad 3) */
    pl_conv1d(ln[0], z, m->w[0], m->w[1], m->in_ch, m->c0, frames, 7, 1, 0);
    pl_tap(m, "pre", ln[0], m->c0, frames);
    x = ln[0];
    cur = 0;

    {
        int T = frames;
        int in_c[3], out_c[3];
        in_c[0] = m->c0; out_c[0] = m->c1;
        in_c[1] = m->c1; out_c[1] = m->c2;
        in_c[2] = m->c2; out_c[2] = m->c3;
        for (s = 0; s < 3; s++) {
            int base = 2 + s * 14;
            int nxt = (cur + 1) % 4;
            int b_sum = (cur + 2) % 4, b_t = (cur + 3) % 4, b_u = cur;
            int L = (T - 1) * up_s[s] - 2 * up_p[s] + up_k[s];
            /* act (slope 0.1) then upsample */
            pl_leaky_copy(x, x, (long)in_c[s] * T, 0.1f);
            pl_convtr1d(ln[nxt], x, m->w[base], m->w[base + 1],
                        in_c[s], out_c[s], T, up_k[s], up_s[s], up_p[s]);
            pl_tap(m, up_name[s], ln[nxt], out_c[s], L);
            /* residual bank; result lands in the sum lane */
            pl_res_bank(ln[b_sum], ln[nxt], &m->w[base + 2], out_c[s], L,
                        ln[b_sum], ln[b_t], ln[b_u]);
            pl_tap(m, mix_name[s], ln[b_sum], out_c[s], L);
            x = ln[b_sum];
            cur = b_sum;
            T = L;
        }
        /* post: lrelu(0.01) -> Conv1d(c3, 1, 7, pad 3) -> tanh */
        pl_leaky_copy(x, x, (long)m->c3 * T, 0.01f);
        pl_conv1d(audio_out, x, m->w[44], m->w[45], m->c3, 1, T, 7, 1, 0);
        pl_tap(m, "pre_tanh", audio_out, 1, T);
        for (j = 0; j < n_samples; j++) audio_out[j] = tanhf(audio_out[j]);
        pl_tap(m, "audio_pre_filter", audio_out, 1, T);
    }

    /* optional WaveformPostFilter:
     * r = in_conv(audio); r = ResidualUnit^layers(r); r2 = out_conv(r);
     * audio = tanh(audio + pf_scale * r2). ResidualUnit(dil=1+l):
     * r += scale * conv2(lrelu(conv1(lrelu(r)))). */
    if (m->pf_channels > 0) {
        int pf = m->pf_channels, K = m->pf_kernel;
        long n = (long)pf * n_samples;
        float *r = ln[0], *t = ln[1], *u = ln[2], *v = ln[3];
        int base = 46, l;
        pl_conv1d(r, audio_out, m->w[base], m->w[base + 1], 1, pf,
                  (int)n_samples, K, 1, 0);
        base += 2;
        for (l = 0; l < m->pf_layers; l++) {
            float scale = m->w[base][0];
            pl_leaky_copy(t, r, n, 0.1f);
            pl_conv1d(u, t, m->w[base + 1], m->w[base + 2], pf, pf,
                      (int)n_samples, 3, 1 + l, 0);
            pl_leaky_copy(t, u, n, 0.1f);
            pl_conv1d(v, t, m->w[base + 3], m->w[base + 4], pf, pf,
                      (int)n_samples, 3, 1, 0);
            for (j = 0; j < n; j++) r[j] += scale * v[j];
            base += 5;
        }
        pl_conv1d(t, r, m->w[base], m->w[base + 1], pf, 1,
                  (int)n_samples, K, 1, 0);
        for (j = 0; j < n_samples; j++)
            audio_out[j] = tanhf(audio_out[j] + m->pf_scale * t[j]);
    }
    pl_tap(m, "audio", audio_out, 1, n_samples);
    return 0;
}
