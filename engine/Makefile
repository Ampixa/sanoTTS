# Host build: the golden gate every port change must pass.
CC ?= cc
CFLAGS ?= -O2 -std=c99 -Wall -Iinclude -Isrc -DFSD_FAST_MATH
GOLDEN ?= test/fixtures/en_us_r7

PIPERLITE_GOLDEN ?= test/golden_piperlite/amy test/golden_piperlite/vi test/golden_piperlite/kristin

.PHONY: test test-piperlite clean

snt_golden_test: src/snt_tts.c src/snt_kernels_ref.c ports/host/snt_port_host.c test/golden_main.c
	$(CC) $(CFLAGS) $^ -lm -o $@

test: snt_golden_test
	./snt_golden_test $(GOLDEN)

# Host-only instrumented build: same arithmetic, plus a dump of the acoustic
# stage's output so tools/validate_quant_sim_vs_r7.py can check the PyTorch
# quantisation simulator against the real C kernels element-wise.
snt_front_dump_test: src/snt_tts.c src/snt_kernels_ref.c ports/host/snt_port_host.c test/golden_main.c
	$(CC) $(CFLAGS) -DSNT_DUMP_FRONT $^ -lm -o $@

front-dump: snt_front_dump_test
	SNT_DUMP_FRONT_PATH=$(DUMP) ./snt_front_dump_test $(GOLDEN)

# ---- E12-nano (294,642 params) --------------------------------------------
# Same gate, same threshold as the R7 line above: corr >= 0.98 in
# test/nano_golden_main.c. Blobs and header come from
# tools/export_e12_nano_q8.py; the model directory is on the include path.
#
# The E13 substitution lineages (DyT + ReLU; en_us_e13a w48, en_us_e13b w62)
# run the SAME gate with the same C sources -- the operator selection
# lives entirely in the lineage's generated header. After
# tools/make_nano_fixture.sh has assembled a lineage:
#   make test-nano NANO_GOLDEN=test/fixtures/en_us_e13a NANO_MODEL=models/en_us_e13a
NANO_GOLDEN ?= test/fixtures/en_us_e12nano
NANO_MODEL ?= models/en_us_e12nano
NANO_SRC = src/snt_nano.c src/snt_kernels_ref.c ports/host/snt_port_host.c test/nano_golden_main.c

# The lineage's operator selection is compiled in from $(NANO_MODEL)'s header,
# so a binary built for one lineage is WRONG for another. These targets used
# to depend on the .c files alone, so changing NANO_MODEL silently reused the
# stale binary and ran one lineage's blobs against another's header. That
# produces NaN correlations -- which the gate then reported as PASS, before
# nano_golden_main.c learned to treat non-finite as a hard failure. Depend on
# the header, and stamp the lineage so any switch forces a relink.
# Recipes use $(filter %.c,$^) so the stamp and header never reach the driver.
NANO_HDR = $(NANO_MODEL)/nano_q8_meta.h
NANO_STAMP = .nano-lineage
.PHONY: FORCE
FORCE:
# On a lineage change, DELETE the binaries rather than relying on the stamp
# being newer than them. These builds finish inside one filesystem timestamp
# tick, so an mtime-only rule leaves make believing the stale binary is
# current -- which is the exact failure this is here to prevent.
NANO_BINS = snt_nano_golden_test snt_nano_fastmath_test snt_nano_f32act_test \
            snt_nano_wf32_test snt_nano_bench snt_nano_bench_prof \
            snt_nano_glue_bench
$(NANO_STAMP): FORCE
	@printf '%s %s\n' "$(NANO_MODEL)" "$(NANO_GOLDEN)" > $@.tmp
	@if ! cmp -s $@.tmp $@ 2>/dev/null; then \
	   rm -f $(NANO_BINS); mv $@.tmp $@; \
	 else rm -f $@.tmp; fi

.PHONY: test-nano test-nano-fastmath

snt_nano_golden_test: $(NANO_SRC) $(NANO_HDR) $(NANO_STAMP)
	$(CC) $(CFLAGS) -I$(NANO_MODEL) $(filter %.c,$^) -lm -o $@

test-nano: snt_nano_golden_test
	./snt_nano_golden_test $(NANO_GOLDEN)

# The MCU-shaped math path: LUT gelu/silu, exponent-flip reciprocal and a
# two-step Newton rsqrt instead of libm. Reported separately because the gap
# between the two is the cost of the fast math, not of the quantisation.
snt_nano_fastmath_test: $(NANO_SRC) $(NANO_HDR) $(NANO_STAMP)
	$(CC) $(CFLAGS) -I$(NANO_MODEL) -DSNT_NANO_FAST_MATH $(filter %.c,$^) -lm -o $@

test-nano-fastmath: snt_nano_fastmath_test
	./snt_nano_fastmath_test $(NANO_GOLDEN)

# Float activations over the same int8 weight blobs (SNT_NANO_ACT_F32): the
# WebAssembly build's math path. Same fixture, same 0.98 gate; the number it
# prints is what the browser voice reproduces.
snt_nano_f32act_test: $(NANO_SRC) $(NANO_HDR) $(NANO_STAMP)
	$(CC) $(CFLAGS) -I$(NANO_MODEL) -DSNT_NANO_ACT_F32 $(filter %.c,$^) -lm -o $@

.PHONY: test-nano-f32act
test-nano-f32act: snt_nano_f32act_test
	./snt_nano_f32act_test $(NANO_GOLDEN)

# Float32 weight rows (tools/export_e12_nano_q8.py --weights f32) with float
# activations: the browser build of a stack that does not survive int8. The
# fixture ships front_f32.bin / model_f32.bin; the header's NANO_WEIGHT_FORMAT
# makes an int8/f32 mismatch a compile error, never a silent misread.
snt_nano_wf32_test: $(NANO_SRC) $(NANO_HDR) $(NANO_STAMP)
	$(CC) $(CFLAGS) -I$(NANO_MODEL) -DSNT_NANO_W_F32 $(filter %.c,$^) -lm -o $@

.PHONY: test-nano-wf32
test-nano-wf32: snt_nano_wf32_test
	./snt_nano_wf32_test $(NANO_GOLDEN)

# ---- host-C timing ---------------------------------------------------------
# A host number is NOT a device number. These exist to produce a nano/R7 RATIO
# on one machine in one session; R7 is the anchor because it is the only stack
# with measured silicon figures. See docs/e12-nano-runtime-projection.md.
BENCH_REPS ?= 21
BENCH_WARMUP ?= 3

.PHONY: bench-nano bench-r7 bench-nano-prof bench-glue bench-all

# Derive a shorter benchmark row for boards that cannot hand out one
# contiguous ~129 KB block. See test/fixtures/en_us_e12nano_short/README.md
# for what its reference does and does not gate.
nano_short_row: src/snt_nano.c src/snt_kernels_ref.c ports/host/snt_port_host.c test/nano_short_row.c $(NANO_HDR) $(NANO_STAMP)
	$(CC) $(CFLAGS) -I$(NANO_MODEL) $(filter %.c,$^) -lm -o $@

snt_nano_bench: src/snt_nano.c src/snt_kernels_ref.c ports/host/snt_port_host.c test/nano_bench_main.c $(NANO_HDR) $(NANO_STAMP)
	$(CC) $(CFLAGS) -I$(NANO_MODEL) $(filter %.c,$^) -lm -o $@

bench-nano: snt_nano_bench
	./snt_nano_bench $(NANO_GOLDEN) $(BENCH_REPS) $(BENCH_WARMUP)

snt_nano_bench_prof: src/snt_nano.c src/snt_kernels_ref.c ports/host/snt_port_host.c test/nano_bench_main.c $(NANO_HDR) $(NANO_STAMP)
	$(CC) $(CFLAGS) -I$(NANO_MODEL) -DSNT_NANO_PROF $(filter %.c,$^) -lm -o $@

bench-nano-prof: snt_nano_bench_prof
	./snt_nano_bench_prof $(NANO_GOLDEN) $(BENCH_REPS) $(BENCH_WARMUP)

snt_r7_bench: src/snt_tts.c src/snt_kernels_ref.c ports/host/snt_port_host.c test/r7_bench_main.c
	$(CC) $(CFLAGS) $^ -lm -o $@

bench-r7: snt_r7_bench
	./snt_r7_bench $(GOLDEN) $(BENCH_REPS) $(BENCH_WARMUP)

snt_nano_glue_bench: src/snt_nano.c src/snt_kernels_ref.c ports/host/snt_port_host.c test/nano_glue_bench.c $(NANO_HDR) $(NANO_STAMP)
	$(CC) $(CFLAGS) -I$(NANO_MODEL) $(filter %.c,$^) -lm -o $@

bench-glue: snt_nano_glue_bench
	./snt_nano_glue_bench

bench-all: bench-r7 bench-nano bench-nano-prof bench-glue

# fp32 piperlite decoder gate (goldens from tools/export_piperlite_golden.py)
piperlite_golden_test: src/snt_piperlite.c test/piperlite_golden_test.c
	$(CC) $(CFLAGS) $^ -lm -o $@

test-piperlite: piperlite_golden_test
	./piperlite_golden_test $(PIPERLITE_GOLDEN)

# int8 piperlite gate (blobs from tools/export_piperlite_q8.py)
piperlite_q8_golden_test: src/snt_piperlite_q8.c test/piperlite_q8_golden_test.c
	$(CC) $(CFLAGS) $^ -lm -o $@

test-piperlite-q8: piperlite_q8_golden_test
	./piperlite_q8_golden_test $(PIPERLITE_GOLDEN)

# fp32 front-half gate: duration + acoustic students
# (goldens from tools/export_front_golden.py)
FRONT_GOLDEN ?= test/golden_front/amy test/golden_front/vi test/golden_front/hindi

.PHONY: test-front

front_golden_test: src/snt_front_f32.c test/front_golden_test.c
	$(CC) $(CFLAGS) $^ -lm -o $@

test-front: front_golden_test
	./front_golden_test $(FRONT_GOLDEN)

clean:
	rm -f snt_golden_test piperlite_golden_test front_golden_test piperlite_q8_golden_test \
	      snt_nano_golden_test snt_nano_fastmath_test
