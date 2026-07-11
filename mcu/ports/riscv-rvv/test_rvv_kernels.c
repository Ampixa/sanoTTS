/* test_rvv_kernels.c -- functional validation of the RVV port kernels
 * against a local scalar reference, bit-exactly.
 *
 * Deterministic (fixed-seed xorshift32 PRNG). Covers lens
 * {16, 48, 80, 240, 304, 512} and matvec rows {1, 12, 76, 304}, plus
 * rows=1539 at len=48. ~200 randomized cases total. Prints PASS/FAIL
 * per group and a final verdict; exit code 0 iff everything matched.
 *
 * Buffers are 16-byte aligned and the weight matrix carries >= 16
 * bytes of readable padding after the last row, per the snt_port.h
 * contract.
 */
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "snt_port.h"

/* ---- deterministic PRNG ---------------------------------------------- */
static uint32_t rng_state = 0x5EED1234u;

static uint32_t rng_next(void) {
    uint32_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    rng_state = x;
    return x;
}

static int8_t rng_s8(void) { return (int8_t)(rng_next() & 0xFF); }

static void fill_s8(int8_t *p, long n) {
    for (long i = 0; i < n; i++) p[i] = rng_s8();
}

/* ---- local scalar reference (mirrors src/snt_kernels_ref.c) ---------- */
static int32_t ref_dot_s8(const int8_t *a, const int8_t *b, int len) {
    int32_t acc = 0;
    for (int i = 0; i < len; i++) acc += (int32_t)a[i] * (int32_t)b[i];
    return acc;
}

static void ref_matvec_s8(const int8_t *act, const int8_t *w, int32_t *out,
                          int rows, int len) {
    for (int r = 0; r < rows; r++)
        out[r] = ref_dot_s8(act, w + (long)r * len, len);
}

/* ---- static aligned storage ------------------------------------------ */
#define MAX_LEN 512
#define MAX_ROWS 1539
#define PAD 16

static _Alignas(16) int8_t g_act[MAX_LEN];
static _Alignas(16) int8_t g_b[MAX_LEN];
/* weights sized for the largest matrices exercised:
 * 1539 rows * 48 len and 304 rows * 512 len, + 16B padding */
#define W_BYTES (304L * 512L + PAD) /* 155664 > 1539*48+16 = 73888 */
static _Alignas(16) int8_t g_w[W_BYTES];
static int32_t g_out[MAX_ROWS];
static int32_t g_ref[MAX_ROWS];

static const int k_lens[] = {16, 48, 80, 240, 304, 512};
#define N_LENS (int)(sizeof(k_lens) / sizeof(k_lens[0]))
static const int k_rows[] = {1, 12, 76, 304};
#define N_ROWS (int)(sizeof(k_rows) / sizeof(k_rows[0]))

static int g_total_cases = 0;
static int g_total_fail = 0;

/* ---- dot groups: one group per len, 20 random cases each ------------- */
static int run_dot_group(int len) {
    int fails = 0;
    for (int c = 0; c < 20; c++) {
        fill_s8(g_act, len);
        fill_s8(g_b, len);
        int32_t got = snt_dot_s8(g_act, g_b, len);
        int32_t want = ref_dot_s8(g_act, g_b, len);
        g_total_cases++;
        if (got != want) {
            fails++;
            printf("  dot len=%d case=%d: got %ld want %ld\n", len, c,
                   (long)got, (long)want);
        }
    }
    printf("%s dot_s8    len=%-4d (20 cases)\n", fails ? "FAIL" : "PASS", len);
    return fails;
}

/* ---- matvec groups: (len, rows), 3 random cases each ----------------- */
static int run_matvec_group(int len, int rows, int cases) {
    int fails = 0;
    long wbytes = (long)rows * len;
    for (int c = 0; c < cases; c++) {
        fill_s8(g_act, len);
        fill_s8(g_w, wbytes);
        memset(g_w + wbytes, (int)(rng_next() & 0xFF), PAD); /* junk pad */
        memset(g_out, 0xA5, sizeof(int32_t) * (size_t)rows);
        snt_matvec_s8(g_act, g_w, g_out, rows, len);
        ref_matvec_s8(g_act, g_w, g_ref, rows, len);
        g_total_cases++;
        for (int r = 0; r < rows; r++) {
            if (g_out[r] != g_ref[r]) {
                fails++;
                printf("  matvec len=%d rows=%d case=%d row=%d: "
                       "got %ld want %ld\n",
                       len, rows, c, r, (long)g_out[r], (long)g_ref[r]);
                break; /* one report per case is enough */
            }
        }
    }
    printf("%s matvec_s8 len=%-4d rows=%-5d (%d cases)\n",
           fails ? "FAIL" : "PASS", len, rows, cases);
    return fails;
}

int main(void) {
#if defined(__riscv_v_intrinsic)
    printf("kernel path: RVV intrinsics (__riscv_v_intrinsic=%d)\n",
           __riscv_v_intrinsic);
#else
    printf("kernel path: scalar fallback\n");
#endif

    /* dot: 6 lens x 20 = 120 cases */
    for (int i = 0; i < N_LENS; i++) g_total_fail += run_dot_group(k_lens[i]);

    /* matvec: 6 lens x 4 row-counts x 3 = 72 cases */
    for (int i = 0; i < N_LENS; i++)
        for (int j = 0; j < N_ROWS; j++)
            g_total_fail += run_matvec_group(k_lens[i], k_rows[j], 3);

    /* matvec tall-skinny: rows=1539 at len=48 only, 8 cases */
    g_total_fail += run_matvec_group(48, 1539, 8);

    printf("----\n%s: %d cases, %d failures\n",
           g_total_fail ? "FAIL" : "ALL PASS", g_total_cases, g_total_fail);
    return g_total_fail ? 1 : 0;
}
