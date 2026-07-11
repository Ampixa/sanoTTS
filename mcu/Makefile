# Host build: the golden gate every port change must pass.
CC ?= cc
CFLAGS ?= -O2 -std=c99 -Wall -Iinclude -Isrc -DFSD_FAST_MATH
GOLDEN ?= test/fixtures/en_us_r7

.PHONY: test clean

snt_golden_test: src/snt_tts.c src/snt_kernels_ref.c ports/host/snt_port_host.c test/golden_main.c
	$(CC) $(CFLAGS) $^ -lm -o $@

test: snt_golden_test
	./snt_golden_test $(GOLDEN)

clean:
	rm -f snt_golden_test
