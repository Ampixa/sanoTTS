/* snt_nano_trace.h -- optional stage-dump hooks for the visualization build.
 *
 * Compiled only when snt_nano.c is built with -DSNT_NANO_TRACE. The device /
 * WASM / golden builds never define it, so the hooks cost nothing there.
 *
 * The driver (nano_trace_main.c) sets an output directory, then the engine
 * calls these at stage boundaries. f32_append is used for column-streamed
 * tensors (mel ring, spectrum) that are produced tile-by-tile / frame-by-frame.
 */
#ifndef SNT_NANO_TRACE_H
#define SNT_NANO_TRACE_H

#include <stdint.h>

void snt_trace_set_dir(const char *dir);
void snt_trace_f32(const char *name, const float *data, long n);
void snt_trace_i32(const char *name, const int32_t *data, long n);
void snt_trace_f32_append(const char *name, const float *data, long n);

#endif
