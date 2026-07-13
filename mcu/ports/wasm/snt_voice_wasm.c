/* snt_voice_wasm.c -- WebAssembly (Emscripten) shim chaining the front half
 * (phoneme ids -> durations -> latent, snt_front_f32.c) into the piperlite
 * decoder (latent -> audio, snt_piperlite.c) for one of the 8 live browser
 * voices.
 *
 * Blob contract: front_blob and dec_blob are each a self-describing
 * concatenation of that component's meta.bin (see snt_front_f32.h /
 * snt_piperlite.h for the exact int32 layouts) immediately followed by its
 * matching weights_f32.bin, exactly as tools/export_voice_bundle.py writes
 * web/voices/<key>/{front_f32.bin,dec_f32.bin}. No length travels alongside
 * the pointers -- meta_bytes and weight_floats are both derivable from the
 * header + offset table already inside the blob (n_tensors tells you how
 * long the offset table is, and the last entry's offset+size is the total
 * float count), so the exported function signature needs only the two raw
 * pointers.
 *
 * All scratch memory (duration/latent/decoder arenas) is heap-allocated
 * per call via malloc/free -- there is no caller-supplied arena parameter,
 * unlike the MCU ports, because the browser heap is generously sized
 * (-sINITIAL_MEMORY, see build_voices.sh) and per-call malloc/free keeps the
 * JS-facing API minimal.
 */
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "snt_front_f32.h"
#include "snt_piperlite.h"

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#define SNT_EXPORT EMSCRIPTEN_KEEPALIVE
#else
#define SNT_EXPORT
#endif

static int32_t rd_i32(const uint8_t *p) {
    return (int32_t)((uint32_t)p[0] | ((uint32_t)p[1] << 8) |
                     ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24));
}

/* piperlite meta.bin: 12 leading int32-sized header words (magic, version,
 * in_ch, c0, c1, c2, c3, pf_channels, pf_layers, pf_kernel, pf_scale (as
 * float bits), n_tensors), n_tensors at byte offset 44, then n_tensors x
 * (offset,size) int32 pairs starting at byte 48. See
 * tools/export_piperlite_golden.py's META.BIN docstring -- must stay in
 * sync with it and with snt_piperlite_init's own parsing. */
static int piperlite_blob_extents(const uint8_t *blob, size_t *meta_bytes,
                                  size_t *weight_floats) {
    int32_t n_tensors, last_off, last_size;
    if (!blob) return -1;
    n_tensors = rd_i32(blob + 44);
    if (n_tensors <= 0 || n_tensors > SNT_PIPERLITE_MAX_TENSORS) return -1;
    *meta_bytes = 48u + 8u * (size_t)n_tensors;
    last_off = rd_i32(blob + 48 + 8 * (n_tensors - 1));
    last_size = rd_i32(blob + 48 + 8 * (n_tensors - 1) + 4);
    if (last_off < 0 || last_size <= 0) return -1;
    *weight_floats = (size_t)last_off + (size_t)last_size;
    return 0;
}

/* front meta.bin: 18 leading int32 header words (magic, version, d_vocab,
 * d_hidden, d_depth, d_kernel, d_max_tokens, d_max_duration, a_vocab,
 * a_hidden, a_token_depth, a_depth, a_kernel, a_out, adapter_mode,
 * adapter_kernel, adapter_rank, n_tensors), n_tensors at byte offset 68,
 * then n_tensors x (offset,size) int32 pairs starting at byte 72. See
 * tools/export_front_golden.py's META.BIN docstring. */
static int front_blob_extents(const uint8_t *blob, size_t *meta_bytes,
                              size_t *weight_floats) {
    int32_t n_tensors, last_off, last_size;
    if (!blob) return -1;
    n_tensors = rd_i32(blob + 68);
    if (n_tensors <= 0 || n_tensors > SNT_FRONT_MAX_TENSORS) return -1;
    *meta_bytes = 72u + 8u * (size_t)n_tensors;
    last_off = rd_i32(blob + 72 + 8 * (n_tensors - 1));
    last_size = rd_i32(blob + 72 + 8 * (n_tensors - 1) + 4);
    if (last_off < 0 || last_size <= 0) return -1;
    *weight_floats = (size_t)last_off + (size_t)last_size;
    return 0;
}

/* Full pipeline: phoneme ids -> durations -> latent -> audio.
 * Returns the number of audio samples written to out on success (<=
 * out_cap), or a negative error code:
 *  -1  bad arguments
 *  -2  malformed front blob (extents)
 *  -3  snt_front_init failed
 *  -4  duration arena alloc failed
 *  -5  snt_front_durations failed
 *  -6  latent arena/output alloc failed
 *  -7  snt_front_latent failed
 *  -8  malformed decoder blob (extents)
 *  -9  snt_piperlite_init failed
 * -10  front acoustic out_channels != decoder in_ch (mismatched bundle)
 * -11  frames*HOP would exceed out_cap
 * -12  decoder arena alloc failed
 * -13  snt_piperlite_synthesize failed
 */
SNT_EXPORT
int snt_voice_synthesize(const uint8_t *front_blob, const uint8_t *dec_blob,
                         const int32_t *ids, int n_ids,
                         float length_scale,
                         float *out, int out_cap) {
    size_t front_meta_sz, front_weight_floats;
    size_t dec_meta_sz, dec_weight_floats;
    const float *front_weights, *dec_weights;
    snt_front_model fm;
    snt_piperlite_model pm;
    float *dur_arena = NULL, *lat_arena = NULL, *pl_arena = NULL, *latent = NULL;
    int32_t *durations = NULL;
    long frames;
    long n_samples;
    int rc, ret;

    if (!front_blob || !dec_blob || !ids || n_ids <= 0 || !out || out_cap <= 0)
        return -1;
    if (length_scale <= 0.0f) length_scale = 1.0f;

    if (front_blob_extents(front_blob, &front_meta_sz, &front_weight_floats) != 0)
        return -2;
    front_weights = (const float *)(const void *)(front_blob + front_meta_sz);
    if (snt_front_init(&fm, front_blob, front_meta_sz, front_weights,
                       front_weight_floats) != 0)
        return -3;

    {
        size_t dur_arena_n = snt_front_duration_arena_floats(&fm, n_ids);
        dur_arena = (float *)malloc((dur_arena_n ? dur_arena_n : 1) * sizeof(float));
        durations = (int32_t *)malloc((size_t)n_ids * sizeof(int32_t));
        if (!dur_arena || !durations) { ret = -4; goto done; }
        frames = snt_front_durations(&fm, ids, n_ids, length_scale, durations,
                                     dur_arena, dur_arena_n);
        free(dur_arena); dur_arena = NULL;
        if (frames <= 0) { ret = -5; goto done; }
    }

    {
        size_t lat_arena_n = snt_front_latent_arena_floats(&fm, n_ids, frames);
        size_t latent_n = (size_t)fm.a_out * (size_t)frames;
        lat_arena = (float *)malloc((lat_arena_n ? lat_arena_n : 1) * sizeof(float));
        latent = (float *)malloc((latent_n ? latent_n : 1) * sizeof(float));
        if (!lat_arena || !latent) { ret = -6; goto done; }
        rc = snt_front_latent(&fm, ids, durations, n_ids, frames, latent,
                              lat_arena, lat_arena_n);
        free(lat_arena); lat_arena = NULL;
        free(durations); durations = NULL;
        if (rc != 0) { ret = -7; goto done; }
    }

    if (piperlite_blob_extents(dec_blob, &dec_meta_sz, &dec_weight_floats) != 0) {
        ret = -8; goto done;
    }
    dec_weights = (const float *)(const void *)(dec_blob + dec_meta_sz);
    if (snt_piperlite_init(&pm, dec_blob, dec_meta_sz, dec_weights,
                           dec_weight_floats) != 0) {
        ret = -9; goto done;
    }
    if (pm.in_ch != fm.a_out) { ret = -10; goto done; }

    n_samples = frames * (long)SNT_PIPERLITE_HOP;
    if (n_samples > (long)out_cap) { ret = -11; goto done; }

    {
        size_t pl_arena_n = snt_piperlite_arena_floats(&pm, (int)frames);
        pl_arena = (float *)malloc((pl_arena_n ? pl_arena_n : 1) * sizeof(float));
        if (!pl_arena) { ret = -12; goto done; }
        rc = snt_piperlite_synthesize(&pm, latent, (int)frames, out,
                                      pl_arena, pl_arena_n);
        free(pl_arena); pl_arena = NULL;
        if (rc != 0) { ret = -13; goto done; }
    }

    ret = (int)n_samples;

done:
    free(dur_arena);
    free(durations);
    free(lat_arena);
    free(latent);
    free(pl_arena);
    return ret;
}

/* Decoder-in-isolation entry point: feeds a precomputed latent [in_ch,
 * frames] straight to the piperlite decoder, bypassing the front half
 * entirely. Used by mcu/ports/wasm/verify_voice_node.mjs to exercise the
 * decoder chain independently of duration/latent prediction, and kept as a
 * standing debug hook (not just bring-up scaffolding) since it is cheap to
 * maintain and useful for isolating front-vs-decoder bugs. Same error-code
 * convention as snt_voice_synthesize (offsets -8.. -13 reused for the
 * decoder-only steps). */
SNT_EXPORT
int snt_voice_synthesize_from_latent(const uint8_t *dec_blob,
                                     const float *z, int in_ch, int frames,
                                     float *out, int out_cap) {
    size_t dec_meta_sz, dec_weight_floats;
    const float *dec_weights;
    snt_piperlite_model pm;
    float *pl_arena = NULL;
    long n_samples;
    int rc, ret;

    if (!dec_blob || !z || in_ch <= 0 || frames <= 0 || !out || out_cap <= 0)
        return -1;
    if (piperlite_blob_extents(dec_blob, &dec_meta_sz, &dec_weight_floats) != 0)
        return -8;
    dec_weights = (const float *)(const void *)(dec_blob + dec_meta_sz);
    if (snt_piperlite_init(&pm, dec_blob, dec_meta_sz, dec_weights,
                           dec_weight_floats) != 0)
        return -9;
    if (pm.in_ch != in_ch) return -10;

    n_samples = (long)frames * (long)SNT_PIPERLITE_HOP;
    if (n_samples > (long)out_cap) return -11;

    {
        size_t pl_arena_n = snt_piperlite_arena_floats(&pm, frames);
        pl_arena = (float *)malloc((pl_arena_n ? pl_arena_n : 1) * sizeof(float));
        if (!pl_arena) return -12;
        rc = snt_piperlite_synthesize(&pm, z, frames, out, pl_arena, pl_arena_n);
        free(pl_arena);
        if (rc != 0) return -13;
    }
    ret = (int)n_samples;
    return ret;
}
