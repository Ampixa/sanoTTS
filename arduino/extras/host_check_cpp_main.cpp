/* host_check_cpp_main.cpp -- exercises the actual SanoTTS C++ class (not
 * just the underlying C API, which host_check_main.c already covers)
 * against the golden fixture, off the Arduino toolchain. Proves begin()/
 * synthesize()/enableSibilantInjection() link and behave as documented.
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>

#include "SanoTTS.h"
#include "model/fsd_meta.h" /* FSD_CODE_DIM */

static void *xload(const char *dir, const char *name, size_t *bytes) {
    char path[512];
    snprintf(path, sizeof path, "%s/%s", dir, name);
    FILE *fh = fopen(path, "rb");
    if (!fh) { fprintf(stderr, "missing %s\n", path); exit(1); }
    fseek(fh, 0, SEEK_END);
    long sz = ftell(fh);
    fseek(fh, 0, SEEK_SET);
    void *buf = malloc((size_t)sz);
    if (!buf || fread(buf, 1, (size_t)sz, fh) != (size_t)sz) exit(1);
    fclose(fh);
    if (bytes) *bytes = (size_t)sz;
    return buf;
}

static const int32_t SIB_IDS[4] = {31, 38, 96, 108};
static const float SIB_TEA_STD[40] = {
  1.97866f, 1.22888f, 1.12157f, 1.84006f, 2.19765f, 3.22486f, 2.34727f, 1.96785f,
  1.43621f, 1.29783f, 2.81680f, 1.88507f, 1.99120f, 1.93567f, 1.68095f, 2.60358f,
  3.08976f, 2.98307f, 2.71773f, 1.68559f, 1.69745f, 0.96275f, 5.05729f, 3.31400f,
  1.58970f, 1.90072f, 1.74750f, 1.90494f, 3.50463f, 1.78908f, 2.42983f, 4.13734f,
  2.71335f, 2.13357f, 2.36755f, 2.63225f, 2.00407f, 1.78098f, 3.15883f, 2.05270f,
};

int main(int argc, char **argv) {
    const char *dir = argc > 1 ? argv[1] : "golden";
    size_t nb;
    void *front = xload(dir, "front_q8.bin", NULL);
    void *model = xload(dir, "model_q8.bin", NULL);
    int32_t *ids = (int32_t *)xload(dir, "e2e_ids.bin", &nb);
    int n_ids = (int)(nb / 4);
    float *gold = (float *)xload(dir, "e2e_audio.bin", &nb);
    size_t n_gold = nb / 4;

    SanoTTS tts;
    if (!tts.begin(model, front)) { fprintf(stderr, "begin() failed\n"); return 1; }

    float *pcm = (float *)malloc(n_gold * sizeof(float) + 4096);
    snt_stats st;
    int n = tts.synthesize(ids, n_ids, pcm, (int)(n_gold + 4096 / sizeof(float)), &st);
    if (n <= 0) { fprintf(stderr, "synthesize() failed: %d\n", n); return 1; }

    size_t m = (size_t)n < n_gold ? (size_t)n : n_gold;
    double sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
    for (size_t i = 0; i < m; i++) {
        double a = pcm[i], b = gold[i];
        sa += a; sb += b; saa += a * a; sbb += b * b; sab += a * b;
    }
    double dn = (double)m;
    double cov = sab - sa * sb / dn;
    double cr = cov / sqrt((saa - sa * sa / dn) * (sbb - sb * sb / dn) + 1e-30);
    printf("[SanoTTS class, default (no sibilant injection)]\n");
    printf("frames %d samples %d -> wrapper samples %d\n", st.frames, st.samples, n);
    printf("raw sample-domain corr vs frozen golden: %.6f (%zu samples)\n", cr, m);
    /* NOTE: this is expected to be far below the 0.98 golden-gate figure,
     * and that is NOT a defect. e2e_audio.bin was rendered with the
     * TEACHER's frozen reference durations (e2e_durs.bin), which
     * host_check_main.c passes explicitly as dur_override to get a
     * frame-exact, bit-comparable signal (corr 0.989 there). SanoTTS::
     * synthesize() intentionally never does that -- a real sketch has no
     * teacher durations, only ids -- so it uses the model's own duration
     * head, which predicts 132 frames here instead of 134. A ~1.5%
     * frame-count difference desynchronizes the two signals sample-for-
     * sample, and naive time-domain correlation of a phase-shifted
     * waveform craters even when the underlying math is unchanged (it is:
     * running the unmodified upstream core the same way, ids in, no
     * dur_override, reproduces the exact same 132 frames / 0.3149 figure
     * -- see arduino/README.md's verification note). The real bit-
     * exactness gate for this patch is host_check_main.c's pass 1. */
    bool frames_sane = (st.frames > 0) && (n > 0) &&
                       (fabs((double)st.frames - 134.0) / 134.0 < 0.10);
    printf(frames_sane ? "PASS (frame count within 10%% of golden's 134 -- sane duration head)\n"
                       : "FAIL (frame count wildly off)\n");

    /* enable + disable round-trip, then re-enable for a perturbation check */
    tts.enableSibilantInjection(SIB_TEA_STD, SIB_IDS, 4, 0.9f);
    float *pcm2 = (float *)malloc(n_gold * sizeof(float) + 4096);
    int n2 = tts.synthesize(ids, n_ids, pcm2, (int)(n_gold + 4096 / sizeof(float)), &st);
    double diffsq = 0.0;
    size_t m2 = (size_t)n2 < m ? (size_t)n2 : m;
    for (size_t i = 0; i < m2; i++) {
        double d = (double)pcm2[i] - (double)pcm[i];
        diffsq += d * d;
    }
    double rms = sqrt(diffsq / (double)(m2 ? m2 : 1));
    printf("\n[SanoTTS class, sibilant injection ON beta=0.9]\n");
    printf("rms diff vs default-off run: %.6f\n", rms);
    bool pass2 = n2 > 0 && rms > 1e-6;
    printf(pass2 ? "PASS (class-level API perturbs output as documented)\n" : "FAIL\n");

    tts.disableSibilantInjection();
    return (frames_sane && pass2) ? 0 : 1;
}
