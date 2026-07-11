# riscv-rvv port

RISC-V Vector (RVV 1.0) implementation of the saanotts-mcu porting
surface (`include/snt_port.h`): `snt_dot_s8` and `snt_matvec_s8` in
`<riscv_vector.h>` intrinsics, plus the flat-memory / serial /
clock_gettime shims (`snt_weights_resident`, `snt_par_run`,
`snt_scratch_id`, `snt_now_us`).

## Target silicon

- SpacemiT K1 (Banana Pi BPI-F3) — RVV 1.0, VLEN=256
- Canaan K230 (CanMV K230) — RVV 1.0, VLEN=128 on the big core

## VLEN assumptions

None. Both kernels strip-mine with `__riscv_vsetvl_e8m*` and never
reference `vlenb` or a fixed vector size; the same binary is correct on
any RVV 1.0 implementation (validated at VLEN 128/256/512, see below).
Accumulators persist across strips, so the accumulate step uses the
tail-undisturbed (`_tu`) intrinsic variants — under the default
tail-agnostic policy a short final strip could otherwise corrupt
accumulator tail lanes.

## LMUL choice

- `snt_dot_s8`: **e8m2** inputs → i16m4 products (`vwmul`) → **i32m8**
  accumulator (`vwadd.wv`). A lone dot product has no register-reuse
  pressure, so the widest practical LMUL minimizes strip count and
  `vsetvl` overhead. Budget: 2+2+4+8 = 16 of 32 registers.
- `snt_matvec_s8`: **e8m1** inputs → i16m2 products → **i32m4**
  accumulators, with rows blocked by 4. Each activation strip is loaded
  once per column strip and reused against 4 weight rows (4 live i32m4
  accumulators), amortizing activation loads 4× — per-row activation
  reload was measured at ~50% overhead on another ISA. e8m1 is what
  makes a 4-row block fit: 1 (act) + 1 (w) + 2 (prod) + 16 (acc) = 20
  of 32 registers. At e8m2 the accumulators would be i32m8 and only a
  2-row block fits.

Exactness: int8×int8 products always fit in int16 (min case
−128×−128 = 16384), and `vwadd.wv` accumulates into int32 lanes with
plain wraparound — bit-identical to `src/snt_kernels_ref.c`.

## Build and validate

```
make -f Makefile.rvv run    # cross-compile rv64gcv, run QEMU vlen=128 + 256
make -f Makefile.rvv host   # scalar-fallback build/run with the host cc
```

The RVV section is guarded by `__riscv_v_intrinsic`; without it the
file falls back to scalar loops, so it compiles on any host.

## Validation status

`test_rvv_kernels.c`: 200 deterministic random cases (fixed-seed
xorshift32) over lens {16, 48, 80, 240, 304, 512} × rows
{1, 12, 76, 304} plus rows=1539 at len=48, compared bit-exactly
against a local scalar reference.

Toolchain: Ubuntu 24.04, `gcc-riscv64-linux-gnu` 13.3.0 (RVV
intrinsics v0.11), `qemu-riscv64` 8.2.2 (user-mode), `-march=rv64gcv
-static`.

| Run | Result |
| --- | --- |
| QEMU `-cpu rv64,v=true,vlen=128` | ALL PASS (200/200) |
| QEMU `-cpu rv64,v=true,vlen=256` | ALL PASS (200/200) |
| QEMU `-cpu rv64,v=true,vlen=512` | ALL PASS (200/200) |
| Host scalar fallback (macOS cc)  | ALL PASS (200/200) |

The RVV codepath was confirmed active under QEMU
(`__riscv_v_intrinsic=11000` banner; `vsetvli`/`vwmul`/`vwadd` present
in the disassembly).

Not yet run on real K1/K230 hardware.
