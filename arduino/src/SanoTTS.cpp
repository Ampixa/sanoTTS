/* SanoTTS.cpp -- see SanoTTS.h for the full contract and caveats. */
#include "SanoTTS.h"

#include <stdlib.h>
#include <string.h>

/* ---- SanoTTS: the 745k int8 fsd pipeline ------------------------------ */

SanoTTS::SanoTTS()
    : model_blob_(nullptr), front_blob_(nullptr), arena_(nullptr),
      arena_bytes_(0), owns_arena_(false), ready_(false) {}

SanoTTS::~SanoTTS() {
    if (owns_arena_ && arena_) free(arena_);
}

bool SanoTTS::begin(const void *model_blob, const void *front_blob,
                    void *arena, size_t arena_bytes) {
    model_blob_ = model_blob;
    front_blob_ = front_blob;

    if (owns_arena_ && arena_) { free(arena_); arena_ = nullptr; owns_arena_ = false; }

    if (arena) {
        arena_ = arena;
        arena_bytes_ = arena_bytes;
        owns_arena_ = false;
    } else {
        /* +16 so the 16-byte alignment bump inside snt_synthesize never
         * eats into the documented usable size. */
        size_t want = recommendedArenaBytes() + 16;
        /* malloc is not guaranteed 16-aligned on every libc; over-allocate
         * and hand the runtime an aligned interior pointer. snt_synthesize
         * re-aligns cfg->arena itself, so a plain malloc pointer is fine
         * as long as arena_bytes_ accounts for the possible 0-15 byte
         * shift, which the +16 above covers. */
        arena_ = malloc(want);
        if (!arena_) { ready_ = false; return false; }
        arena_bytes_ = want;
        owns_arena_ = true;
    }

    ready_ = (model_blob_ != nullptr && front_blob_ != nullptr && arena_ != nullptr);
    return ready_;
}

namespace {
struct PcmSink {
    float *out;
    int cap;
    int pos;
};

int pcm_sink_cb(const float *pcm, int n, void *user) {
    PcmSink *s = static_cast<PcmSink *>(user);
    int room = s->cap - s->pos;
    if (room <= 0) return 1; /* full: abort, never overrun */
    int take = (n < room) ? n : room;
    memcpy(s->out + s->pos, pcm, (size_t)take * sizeof(float));
    s->pos += take;
    return (take < n) ? 1 : 0; /* stop once pcm_out is full */
}
} /* namespace */

int SanoTTS::synthesize(const int32_t *ids, int n_ids, float *pcm_out,
                        int cap, snt_stats *stats_out) {
    if (!ready_ || !ids || n_ids <= 0 || !pcm_out || cap <= 0) return -1;

    snt_config cfg;
    cfg.front_blob = front_blob_;
    cfg.dec_blob = model_blob_;
    cfg.arena = arena_;
    cfg.arena_size = arena_bytes_;
    cfg.dur_override = nullptr; /* production: use the model's own durations */

    PcmSink sink = {pcm_out, cap, 0};
    snt_stats local_stats;
    snt_stats *st = stats_out ? stats_out : &local_stats;
    int rc = snt_synthesize(&cfg, ids, n_ids, pcm_sink_cb, &sink, st);
    if (rc != 0 && sink.pos == 0) return -2; /* aborted before any audio */
    return sink.pos;
}

void SanoTTS::enableSibilantInjection(const float *tea_std,
                                     const int32_t *sib_id_set,
                                     int n_sib_ids, float beta) {
    snt_sibilant_configure(tea_std, sib_id_set, n_sib_ids, beta);
}

void SanoTTS::disableSibilantInjection() {
    snt_sibilant_configure(nullptr, nullptr, 0, 0.0f);
}

/* ---- SanoTTSPiperLiteQ8: bigger-voice front (fp32) + int8 decoder ----- */

SanoTTSPiperLiteQ8::SanoTTSPiperLiteQ8() : ready_(false) {
    memset(&front_, 0, sizeof front_);
    memset(&dec_, 0, sizeof dec_);
}

bool SanoTTSPiperLiteQ8::begin(const void *front_meta, size_t front_meta_bytes,
                               const float *front_weights,
                               size_t front_weight_floats,
                               const void *dec_meta_q8, size_t dec_meta_bytes,
                               const int8_t *dec_weights_q8,
                               size_t dec_weight_bytes) {
    int rc_f = snt_front_init(&front_, front_meta, front_meta_bytes,
                              front_weights, front_weight_floats);
    int rc_d = snt_piperlite_q8_init(&dec_, dec_meta_q8, dec_meta_bytes,
                                     dec_weights_q8, dec_weight_bytes);
    ready_ = (rc_f == 0 && rc_d == 0);
    return ready_;
}

long SanoTTSPiperLiteQ8::durations(const int32_t *ids, int n_tokens,
                                   float length_scale, int32_t *dur_out,
                                   float *arena, size_t arena_floats) {
    if (!ready_) return -1;
    return snt_front_durations(&front_, ids, n_tokens, length_scale, dur_out,
                               arena, arena_floats);
}

int SanoTTSPiperLiteQ8::latent(const int32_t *ids, const int32_t *durations,
                               int n_tokens, long frames, float *latent_out,
                               float *arena, size_t arena_floats) {
    if (!ready_) return -1;
    return snt_front_latent(&front_, ids, durations, n_tokens, frames,
                            latent_out, arena, arena_floats);
}

int SanoTTSPiperLiteQ8::decode(const float *latent, int frames,
                               float *audio_out, void *arena,
                               size_t arena_bytes) {
    if (!ready_) return -1;
    return snt_piperlite_q8_synthesize(&dec_, latent, frames, audio_out,
                                       arena, arena_bytes);
}
