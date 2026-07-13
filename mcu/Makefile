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
	rm -f snt_golden_test piperlite_golden_test front_golden_test piperlite_q8_golden_test
