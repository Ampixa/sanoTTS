# Host build for the espeak-free lexicon G2P.
#
#   make            build ./test_nano_lex_g2p
#   make selftest   run the contract checks
#   make sizes      size(1) of each object, and the workspace figure
#   make stack      measured peak stack for one call
#   make parity     the full parity run against the Python oracle
#   make tables     regenerate nano_lex_tables.[ch] from the misaki JSON
#   make check      verify the tracked tables match a fresh generation
#
# The generated table is ~3.5 MB of C, so the first build is the slow one.

CC      ?= cc
CSTD    ?= -std=c99
CFLAGS  ?= $(CSTD) -Os -Wall -Wextra -Wpedantic -Wshadow -Wconversion \
           -Wstrict-prototypes -Wmissing-prototypes
PYTHON  ?= python3

ROOT    := ../../..
OBJS    := nano_lex_g2p.o nano_lex_tables.o
BIN     := test_nano_lex_g2p

.PHONY: all clean selftest sizes sizes-elf stack stack-usage parity tables check

all: $(BIN)

$(BIN): $(OBJS) test_nano_lex_g2p.o
	$(CC) $(CFLAGS) -o $@ $^

nano_lex_g2p.o: nano_lex_g2p.c nano_lex_g2p.h nano_lex_tables.h
	$(CC) $(CFLAGS) -c -o $@ $<

nano_lex_tables.o: nano_lex_tables.c nano_lex_tables.h
	$(CC) $(CSTD) -Os -c -o $@ $<

test_nano_lex_g2p.o: test_nano_lex_g2p.c nano_lex_g2p.h
	$(CC) $(CFLAGS) -c -o $@ $<

selftest: $(BIN)
	./$(BIN) --selftest

# Where the bytes land. The claim to check is "tables in .rodata, nothing in
# .data, .bss is exactly the workspace", and the honest place to check it is a
# bare-metal ELF target rather than the host's Mach-O or PIE layout -- Mach-O
# puts relocatable const pointer arrays in __DATA,__const, which is a dyld
# artefact and not what a statically linked firmware image does.
ELFCC     ?= clang --target=arm-none-eabi -ffreestanding
ELFSIZE   ?= $(shell xcrun --find llvm-size 2>/dev/null || command -v llvm-size || echo size)

sizes: $(OBJS) $(BIN)
	@echo "== host objects =="
	size $(OBJS) || true
	@echo
	@echo "== writable symbols (only the workspace may appear) =="
	@nm nano_lex_g2p.o nano_lex_tables.o | grep -E ' [bBdD] ' || echo "(none)"
	@echo
	./$(BIN) --sizes

sizes-elf:
	@$(ELFCC) $(CSTD) -Os -c nano_lex_g2p.c -o /tmp/nlg_elf_g2p.o && \
	 $(ELFCC) $(CSTD) -Os -c nano_lex_tables.c -o /tmp/nlg_elf_tab.o && \
	 echo "== bare-metal ELF (arm-none-eabi, -Os, -ffreestanding) ==" && \
	 $(ELFSIZE) -A /tmp/nlg_elf_g2p.o /tmp/nlg_elf_tab.o || \
	 echo "no ELF cross target available; host numbers only"

stack-usage:
	@$(CC) $(CSTD) -Os -fstack-usage -c nano_lex_g2p.c -o /tmp/nlg_su.o && \
	 sort -t"$$(printf '\t')" -k2 -rn /tmp/nlg_su.su | head -12

stack: $(BIN)
	./$(BIN) --stack

parity: $(BIN)
	$(PYTHON) run_parity.py

tables:
	$(PYTHON) gen_lex_tables.py

check:
	$(PYTHON) gen_lex_tables.py --check

clean:
	rm -f $(OBJS) test_nano_lex_g2p.o $(BIN)
