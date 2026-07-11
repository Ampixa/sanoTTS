/* snt_port_host.c -- POSIX port: serial par_run, wall clock, single scratch. */
#include <time.h>
#include "snt_port.h"

void snt_par_run(snt_par_fn f, int n, void *ctx) { f(0, n, ctx); }

int snt_scratch_id(void) { return 0; }

int64_t snt_now_us(void) {
    return (int64_t)clock() * 1000000LL / CLOCKS_PER_SEC;
}
