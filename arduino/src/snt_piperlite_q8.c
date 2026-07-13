/* snt_piperlite_q8.c -- int8-weight piperlite decoder (see snt_piperlite_q8.h).
 * Plain C99, caller arena, no malloc.
 *
 * Numerics: weights are per-output-channel int8 (the distribution blob);
 * activations ride an int16 lane holding 12-bit values (sat to +/-2047,
 * scale = calibrated_clip/2047). Why not int8 activations: measured on the
 * amy golden, W8A8 tops out at 0.985 corr because the trunk tensors are
 * heavy-tailed (clip/rms 7..21) and the error is spread evenly across all
 * three stages (fp32-weight floors: stage0 0.9987, stage1 0.9961, stage2
 * 0.9953, all-stages 0.9904) -- no single-stage int16 lane can pass the
 * 0.99 gate. 12-bit (not 15) keeps every int32 accumulator overflow-safe:
 * worst case 2047*127*in_ch*K = 349M << 2^31 for the largest conv (pre,
 * 192ch k7). The int8 weight blob is unchanged either way.
 *
 * Convolutions accumulate int16 x int8 -> int32 row-wise per output channel
 * and requantize with fp32 multipliers; residual adds happen at the requant
 * point in the shared output scale; the residual-bank mean (/3) is folded
 * into the bank-output quant. The ap tensor (leaky 0.01 of stage2 mix) uses
 * the full 15-bit int16 range (post conv fan-in is only c3*7), and the final
 * post conv requantizes straight to fp32: pre_tanh is never quantized. The
 * optional waveform post filter runs fp32 with weights dequantized on the
 * fly.
 */
#include "snt_piperlite_q8.h"

#include <math.h>
#include <string.h>

#define PLQ_HOP SNT_PIPERLITE_Q8_HOP
#define PLQ_QMAX 2047 /* 12-bit activation lane, int32-safe accumulation */
/* meta act scales are calibrated_clip/127; rescale to the 12-bit grid */
#define PLQ_A12(s) ((s) * (127.0f / 2047.0f))

/* ---- meta parsing ------------------------------------------------------ */

static int32_t rd_i32(const unsigned char *p) {
    return (int32_t)((uint32_t)p[0] | ((uint32_t)p[1] << 8) |
                     ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24));
}

static float rd_f32(const unsigned char *p) {
    union { uint32_t u; float f; } v;
    v.u = (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
          ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
    return v.f;
}

/* Expected (kind, elems, out_ch) for slot idx; kind 0 = int8 weight,
 * kind 1 = fp32 (bias / pf unit scalar). Returns 0 ok, -1 bad idx. */
static int slot_spec(const snt_piperlite_q8_model *m, int idx,
                     int *kind, long *elems, int *out_ch) {
    static const int bk[3] = {3, 5, 7};
    static const int up_k[3] = {16, 16, 8};
    int in_c[3], out_c[3];
    int s, base;
    if (idx < 0) return -1;
    if (idx == 0) { *kind = 0; *elems = (long)m->c0 * m->in_ch * 7; *out_ch = m->c0; return 0; }
    if (idx == 1) { *kind = 1; *elems = m->c0; *out_ch = 0; return 0; }
    in_c[0] = m->c0; out_c[0] = m->c1;
    in_c[1] = m->c1; out_c[1] = m->c2;
    in_c[2] = m->c2; out_c[2] = m->c3;
    base = 2;
    for (s = 0; s < 3; s++) {
        if (idx == base) { *kind = 0; *elems = (long)in_c[s] * out_c[s] * up_k[s]; *out_ch = out_c[s]; return 0; }
        if (idx == base + 1) { *kind = 1; *elems = out_c[s]; *out_ch = 0; return 0; }
        if (idx < base + 14) {
            int r = idx - (base + 2), branch = r / 4, part = r % 4;
            int ch = out_c[s];
            if (part == 0 || part == 2) { *kind = 0; *elems = (long)ch * ch * bk[branch]; *out_ch = ch; }
            else { *kind = 1; *elems = ch; *out_ch = 0; }
            return 0;
        }
        base += 14;
    }
    if (idx == base) { *kind = 0; *elems = (long)m->c3 * 7; *out_ch = 1; return 0; }
    if (idx == base + 1) { *kind = 1; *elems = 1; *out_ch = 0; return 0; }
    base += 2;
    if (m->pf_channels > 0) {
        int l;
        if (idx == base) { *kind = 0; *elems = (long)m->pf_channels * m->pf_kernel; *out_ch = m->pf_channels; return 0; }
        if (idx == base + 1) { *kind = 1; *elems = m->pf_channels; *out_ch = 0; return 0; }
        base += 2;
        for (l = 0; l < m->pf_layers; l++) {
            if (idx == base) { *kind = 1; *elems = 1; *out_ch = 0; return 0; } /* unit scale */
            if (idx == base + 1 || idx == base + 3) {
                *kind = 0; *elems = (long)m->pf_channels * m->pf_channels * 3;
                *out_ch = m->pf_channels; return 0;
            }
            if (idx == base + 2 || idx == base + 4) { *kind = 1; *elems = m->pf_channels; *out_ch = 0; return 0; }
            base += 5;
        }
        if (idx == base) { *kind = 0; *elems = (long)m->pf_channels * m->pf_kernel; *out_ch = 1; return 0; }
        if (idx == base + 1) { *kind = 1; *elems = 1; *out_ch = 0; return 0; }
    }
    return -1;
}

int snt_piperlite_q8_init(snt_piperlite_q8_model *m,
                          const void *meta, size_t meta_bytes,
                          const int8_t *weights, size_t weight_bytes) {
    const unsigned char *p = (const unsigned char *)meta;
    size_t cur;
    const float *pool;
    long pool_n;
    int i, n_act, expected;
    if (!m || !p || !weights) return -1;
    if (meta_bytes < 13 * 4) return -2;
    memset(m, 0, sizeof *m);
    if (rd_i32(p) != (int32_t)SNT_PIPERLITE_Q8_MAGIC) return -3;
    if (rd_i32(p + 4) != 1) return -4;
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
    n_act = rd_i32(p + 48);
    if (m->in_ch <= 0 || m->c0 <= 0 || m->c1 <= 0 || m->c2 <= 0 || m->c3 <= 0)
        return -5;
    if (m->pf_channels < 0 || m->pf_layers < 0 ||
        (m->pf_channels == 0) != (m->pf_layers == 0))
        return -5;
    expected = 46 + (m->pf_channels > 0 ? 4 + 5 * m->pf_layers : 0);
    if (m->n_tensors != expected || m->n_tensors > SNT_PIPERLITE_Q8_MAX_TENSORS)
        return -6;
    if (n_act != SNT_PIPERLITE_Q8_ACTS) return -6;
    cur = 52;
    if (meta_bytes < cur + (size_t)n_act * 4) return -2;
    for (i = 0; i < n_act; i++) {
        m->act[i] = rd_f32(p + cur);
        if (!(m->act[i] > 0.0f)) return -7;
        cur += 4;
    }
    if (meta_bytes < cur + (size_t)m->n_tensors * 20 + 4) return -2;
    /* pool sits after the records */
    {
        size_t pool_hdr = cur + (size_t)m->n_tensors * 20;
        pool_n = rd_i32(p + pool_hdr);
        if (pool_n < 0 ||
            meta_bytes < pool_hdr + 4 + (size_t)pool_n * 4)
            return -2;
        pool = (const float *)(const void *)(p + pool_hdr + 4);
    }
    for (i = 0; i < m->n_tensors; i++) {
        int kind = rd_i32(p + cur);
        long off = rd_i32(p + cur + 4);
        long size = rd_i32(p + cur + 8);
        long aux_off = rd_i32(p + cur + 12);
        long aux_n = rd_i32(p + cur + 16);
        int want_kind, want_oc;
        long want_elems;
        cur += 20;
        if (slot_spec(m, i, &want_kind, &want_elems, &want_oc) != 0) return -8;
        if (kind != want_kind || size != want_elems || off < 0) return -8;
        if (kind == 0) {
            if ((size_t)off + (size_t)size > weight_bytes) return -9;
            if (aux_n != want_oc || aux_off < 0 || aux_off + aux_n > pool_n)
                return -9;
            m->wq[i] = weights + off;
            m->wscale[i] = pool + aux_off;
        } else {
            if (off + size > pool_n) return -9;
            m->f32[i] = pool + off;
        }
    }
    /* NOTE: pool points into the caller's meta buffer; it must outlive m.
     * (Alignment: the exporter keeps every section 4-aligned.) */
    return 0;
}

/* ---- kernels ------------------------------------------------------------ */

static int16_t sat12(float v) {
    long r = lrintf(v);
    if (r > PLQ_QMAX) return PLQ_QMAX;
    if (r < -PLQ_QMAX) return -PLQ_QMAX;
    return (int16_t)r;
}

static int16_t sat15(float v) {
    long r = lrintf(v);
    if (r > 32767) return 32767;
    if (r < -32767) return -32767;
    return (int16_t)r;
}

/* one output row of Conv1d "same": acc[t] = sum_ic sum_k w*x, w = [in_ch*K] */
static void q16_conv1d_row(int32_t *acc, const int16_t *x, const int8_t *w,
                           int in_ch, int T, int K, int dil) {
    int pad = dil * (K / 2);
    int ic, k, t;
    memset(acc, 0, (size_t)T * sizeof(int32_t));
    for (ic = 0; ic < in_ch; ic++) {
        const int16_t *xrow = x + (long)ic * T;
        const int8_t *wrow = w + (long)ic * K;
        for (k = 0; k < K; k++) {
            int32_t wv = wrow[k];
            int off = k * dil - pad;
            int lo = off < 0 ? -off : 0;
            int hi = off > 0 ? T - off : T;
            if (wv != 0)
                for (t = lo; t < hi; t++) acc[t] += wv * xrow[t + off];
        }
    }
}

/* one output row of ConvTranspose1d; w base layout [in_ch][out_ch][K] */
static void q16_convtr1d_row(int32_t *acc, const int16_t *x, const int8_t *w,
                             int oc, int in_ch, int out_ch, int T,
                             int K, int stride, int pad, int L) {
    int ic, k, t;
    memset(acc, 0, (size_t)L * sizeof(int32_t));
    for (ic = 0; ic < in_ch; ic++) {
        const int16_t *xrow = x + (long)ic * T;
        const int8_t *wrow = w + ((long)ic * out_ch + oc) * K;
        for (k = 0; k < K; k++) {
            int32_t wv = wrow[k];
            int shift = k - pad;
            int t_lo = 0, t_hi = T;
            long lim;
            if (wv == 0 || shift >= L) continue;
            if (shift < 0) t_lo = (-shift + stride - 1) / stride;
            lim = ((long)L - 1 - shift) / stride + 1;
            if (lim < t_hi) t_hi = (int)lim;
            for (t = t_lo; t < t_hi; t++)
                acc[(long)t * stride + shift] += wv * xrow[t];
        }
    }
}

/* leaky_relu between activation planes with static in/out scales */
static void q_leaky_requant(int16_t *dst, const int16_t *src, long n,
                            float slope, float s_in, float s_out, int use15) {
    float mp = s_in / s_out, mn = slope * s_in / s_out;
    long i;
    for (i = 0; i < n; i++) {
        int q = src[i];
        float v = (float)q * (q > 0 ? mp : mn);
        dst[i] = use15 ? sat15(v) : sat12(v);
    }
}

/* fp32 conv with int8 weights dequantized on the fly (post filter only) */
static void q8_conv1d_f32(float *out, const float *x, const int8_t *wq,
                          const float *ws, const float *b, int in_ch,
                          int out_ch, int T, int K, int dil) {
    int pad = dil * (K / 2);
    int oc, ic, k, t;
    for (oc = 0; oc < out_ch; oc++) {
        float *orow = out + (long)oc * T;
        float bias = b[oc], sw = ws[oc];
        for (t = 0; t < T; t++) orow[t] = bias;
        for (ic = 0; ic < in_ch; ic++) {
            const float *xrow = x + (long)ic * T;
            const int8_t *wrow = wq + ((long)oc * in_ch + ic) * K;
            for (k = 0; k < K; k++) {
                float wv = (float)wrow[k] * sw;
                int off = k * dil - pad;
                int lo = off < 0 ? -off : 0;
                int hi = off > 0 ? T - off : T;
                if (wv != 0.0f)
                    for (t = lo; t < hi; t++) orow[t] += wv * xrow[t + off];
            }
        }
    }
}

static void f32_leaky(float *dst, const float *src, long n, float slope) {
    long i;
    for (i = 0; i < n; i++) {
        float v = src[i];
        dst[i] = v > 0.0f ? v : slope * v;
    }
}

/* ---- arena sizing ------------------------------------------------------ */

static long plq_plane_elems(const snt_piperlite_q8_model *m, int frames) {
    long s = (long)m->c0 * frames;
    long v = (long)m->c1 * 8 * frames;
    if (v > s) s = v;
    v = (long)m->c2 * 64 * frames;
    if (v > s) s = v;
    v = (long)m->c3 * PLQ_HOP * frames;
    if (v > s) s = v;
    v = (long)m->in_ch * frames;
    if (v > s) s = v;
    return s;
}

size_t snt_piperlite_q8_arena_bytes(const snt_piperlite_q8_model *m,
                                    int frames) {
    long S, N, main_bytes, pf_bytes;
    if (!m || frames <= 0) return 0;
    S = plq_plane_elems(m, frames);
    N = (long)PLQ_HOP * frames;
    /* sumf (4S) + rowbuf (4N) + 4 int16 planes (2S each) + slack */
    main_bytes = 4 * S + 4 * N + 8 * S + 64;
    /* post filter reuses the arena as 4 fp32 planes of pf_ch*N */
    pf_bytes = m->pf_channels > 0 ? 16L * m->pf_channels * N + 64 : 0;
    return (size_t)(main_bytes > pf_bytes ? main_bytes : pf_bytes);
}

/* ---- forward ------------------------------------------------------------ */

static void plq_tap(const snt_piperlite_q8_model *m, const char *name,
                    const int16_t *q, float s, int ch, long len,
                    float *scratch) {
    long n = (long)ch * len, i;
    if (!m->stage_cb) return;
    for (i = 0; i < n; i++) scratch[i] = (float)q[i] * s;
    m->stage_cb(name, scratch, ch, (int)len, m->stage_user);
}

int snt_piperlite_q8_synthesize(const snt_piperlite_q8_model *m,
                                const float *z, int frames,
                                float *audio_out,
                                void *arena, size_t arena_bytes) {
    static const int up_k[3] = {16, 16, 8};
    static const int up_s[3] = {8, 8, 4};
    static const int up_p[3] = {4, 4, 2};
    static const int bk[3] = {3, 5, 7};
    static const int bd1[3] = {1, 2, 3};
    static const int bd2[3] = {2, 6, 12};
    static const char *up_name[3] = {"up0", "up1", "up2"};
    static const char *mix_name[3] = {"stage0_mix", "stage1_mix", "stage2_mix"};

    long S, N;
    float *sumf;
    int32_t *rowbuf;
    int16_t *pl[4];
    unsigned char *base;
    int cur, s, i;
    long j;
    float s_cur;

    if (!m || !z || frames <= 0 || !audio_out || !arena) return -1;
    if (arena_bytes < snt_piperlite_q8_arena_bytes(m, frames)) return -2;
    S = plq_plane_elems(m, frames);
    N = (long)PLQ_HOP * frames;
    base = (unsigned char *)arena;
    base += (16 - ((uintptr_t)base & 15)) & 15;
    sumf = (float *)(void *)base;
    rowbuf = (int32_t *)(void *)(base + 4 * S);
    for (i = 0; i < 4; i++)
        pl[i] = (int16_t *)(void *)(base + 4 * S + 4 * N + (long)i * 2 * S);

    /* quantize z into the 12-bit lane */
    {
        float inv = 1.0f / PLQ_A12(m->act[0]);
        long n = (long)m->in_ch * frames;
        for (j = 0; j < n; j++) pl[0][j] = sat12(z[j] * inv);
    }

    /* pre conv */
    {
        float s_z = PLQ_A12(m->act[0]), s_pre = PLQ_A12(m->act[1]);
        const float *b = m->f32[1];
        int oc, T = frames;
        for (oc = 0; oc < m->c0; oc++) {
            float mm = s_z * m->wscale[0][oc] / s_pre;
            float cc = b[oc] / s_pre;
            int16_t *orow = pl[1] + (long)oc * T;
            int t;
            q16_conv1d_row(rowbuf, pl[0], m->wq[0] + (long)oc * m->in_ch * 7,
                           m->in_ch, T, 7, 1);
            for (t = 0; t < T; t++)
                orow[t] = sat12((float)rowbuf[t] * mm + cc);
        }
        plq_tap(m, "pre", pl[1], s_pre, m->c0, T, sumf);
    }
    cur = 1;
    s_cur = PLQ_A12(m->act[1]);

    {
        int T = frames;
        int in_c[3], out_c[3];
        in_c[0] = m->c0; out_c[0] = m->c1;
        in_c[1] = m->c1; out_c[1] = m->c2;
        in_c[2] = m->c2; out_c[2] = m->c3;
        for (s = 0; s < 3; s++) {
            int wbase = 2 + s * 14;
            int abase = 2 + s * 12;
            float s_a = PLQ_A12(m->act[abase]);
            float s_up = PLQ_A12(m->act[abase + 1]);
            float s_mix = PLQ_A12(m->act[abase + 11]);
            int p_a = (cur + 1) % 4, p_up = (cur + 2) % 4;
            int p_t = (cur + 3) % 4, p_y = cur;
            int L = (T - 1) * up_s[s] - 2 * up_p[s] + up_k[s];
            int oc, br;
            long n = (long)out_c[s] * L;

            /* act (0.1) then upsample */
            q_leaky_requant(pl[p_a], pl[cur], (long)in_c[s] * T, 0.1f,
                            s_cur, s_a, 0);
            for (oc = 0; oc < out_c[s]; oc++) {
                float mm = s_a * m->wscale[wbase][oc] / s_up;
                float cc = m->f32[wbase + 1][oc] / s_up;
                int16_t *orow = pl[p_up] + (long)oc * L;
                int t;
                q16_convtr1d_row(rowbuf, pl[p_a], m->wq[wbase], oc, in_c[s],
                                 out_c[s], T, up_k[s], up_s[s], up_p[s], L);
                for (t = 0; t < L; t++)
                    orow[t] = sat12((float)rowbuf[t] * mm + cc);
            }
            plq_tap(m, up_name[s], pl[p_up], s_up, out_c[s], L, sumf);

            /* residual bank: fp32 branch sum, /3 folded into the mix quant */
            memset(sumf, 0, (size_t)n * sizeof(float));
            for (br = 0; br < 3; br++) {
                int slot = wbase + 2 + br * 4;
                float s_t1 = PLQ_A12(m->act[abase + 2 + br * 3]);
                float s_y1 = PLQ_A12(m->act[abase + 3 + br * 3]);
                float s_t2 = PLQ_A12(m->act[abase + 4 + br * 3]);
                q_leaky_requant(pl[p_t], pl[p_up], n, 0.1f, s_up, s_t1, 0);
                for (oc = 0; oc < out_c[s]; oc++) {
                    float mm = s_t1 * m->wscale[slot][oc] / s_y1;
                    float cc = m->f32[slot + 1][oc] / s_y1;
                    float mr = s_up / s_y1;
                    const int16_t *xrow = pl[p_up] + (long)oc * L;
                    int16_t *yrow = pl[p_y] + (long)oc * L;
                    int t;
                    q16_conv1d_row(rowbuf, pl[p_t],
                                   m->wq[slot] + (long)oc * out_c[s] * bk[br],
                                   out_c[s], L, bk[br], bd1[br]);
                    for (t = 0; t < L; t++)
                        yrow[t] = sat12((float)rowbuf[t] * mm + cc +
                                        (float)xrow[t] * mr);
                }
                q_leaky_requant(pl[p_t], pl[p_y], n, 0.1f, s_y1, s_t2, 0);
                for (oc = 0; oc < out_c[s]; oc++) {
                    float mm = s_t2 * m->wscale[slot + 2][oc];
                    float cc = m->f32[slot + 3][oc];
                    const int16_t *yrow = pl[p_y] + (long)oc * L;
                    float *srow = sumf + (long)oc * L;
                    int t;
                    q16_conv1d_row(rowbuf, pl[p_t],
                                   m->wq[slot + 2] + (long)oc * out_c[s] * bk[br],
                                   out_c[s], L, bk[br], bd2[br]);
                    for (t = 0; t < L; t++)
                        srow[t] += (float)rowbuf[t] * mm + cc +
                                   (float)yrow[t] * s_y1;
                }
            }
            for (j = 0; j < n; j++) sumf[j] *= (1.0f / 3.0f);
            if (m->stage_cb)
                m->stage_cb(mix_name[s], sumf, out_c[s], L, m->stage_user);
            {
                float inv = 1.0f / s_mix;
                for (j = 0; j < n; j++) pl[p_up][j] = sat12(sumf[j] * inv);
            }
            cur = p_up;
            s_cur = s_mix;
            T = L;
        }

        /* post: leaky(0.01) -> conv(c3 -> 1, k7) requantized straight to
         * fp32 (pre_tanh stays unquantized) -> tanh. The ap tensor uses the
         * full 15-bit range (fan-in is only c3*7, so int32 acc is safe). */
        {
            float s_ap = m->act[38] * (127.0f / 32767.0f);
            int p_t = (cur + 1) % 4;
            float mm = s_ap * m->wscale[44][0];
            float cc = m->f32[45][0];
            int t;
            q_leaky_requant(pl[p_t], pl[cur], (long)m->c3 * T, 0.01f,
                            s_cur, s_ap, 1);
            q16_conv1d_row(rowbuf, pl[p_t], m->wq[44], m->c3, T, 7, 1);
            for (t = 0; t < T; t++)
                audio_out[t] = (float)rowbuf[t] * mm + cc;
            if (m->stage_cb)
                m->stage_cb("pre_tanh", audio_out, 1, T, m->stage_user);
            for (j = 0; j < N; j++) audio_out[j] = tanhf(audio_out[j]);
            if (m->stage_cb)
                m->stage_cb("audio_pre_filter", audio_out, 1, T, m->stage_user);
        }
    }

    /* optional waveform post filter, fp32 with dequantized int8 weights */
    if (m->pf_channels > 0) {
        int pf = m->pf_channels, K = m->pf_kernel;
        long n = (long)pf * N;
        float *r = (float *)(void *)base;
        float *t = r + n, *u = t + n, *v = u + n;
        int slot = 46, l;
        q8_conv1d_f32(r, audio_out, m->wq[slot], m->wscale[slot],
                      m->f32[slot + 1], 1, pf, (int)N, K, 1);
        slot += 2;
        for (l = 0; l < m->pf_layers; l++) {
            float scale = m->f32[slot][0];
            f32_leaky(t, r, n, 0.1f);
            q8_conv1d_f32(u, t, m->wq[slot + 1], m->wscale[slot + 1],
                          m->f32[slot + 2], pf, pf, (int)N, 3, 1 + l);
            f32_leaky(t, u, n, 0.1f);
            q8_conv1d_f32(v, t, m->wq[slot + 3], m->wscale[slot + 3],
                          m->f32[slot + 4], pf, pf, (int)N, 3, 1);
            for (j = 0; j < n; j++) r[j] += scale * v[j];
            slot += 5;
        }
        q8_conv1d_f32(t, r, m->wq[slot], m->wscale[slot], m->f32[slot + 1],
                      pf, 1, (int)N, K, 1);
        for (j = 0; j < N; j++)
            audio_out[j] = tanhf(audio_out[j] + m->pf_scale * t[j]);
    }
    if (m->stage_cb)
        m->stage_cb("audio", audio_out, 1, (int)N, m->stage_user);
    return 0;
}
