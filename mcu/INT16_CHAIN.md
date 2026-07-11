# Tier S integer-chain redesign (C3: 7.78x -> ~3x target)

Measured premise: soft-float ~200 cyc/op on RV32IMC; remaining C3 time
is float glue (tiles 4.2s, spec 2.0s, rbc 2.4s, head 1.4s, fft-pack 1.3s).
Shaving ops bought 0.2-0.3s per round; elimination is the only 2x+ lever.

## Design
1. Exporter emits, per conv, a fused integer requant multiplier:
   M_o = round(s_w[o] * s_in / s_out * 2^SHIFT) for STATIC s_in/s_out
   (per-tensor activation scales calibrated like act_scales.h, with
   outlier headroom). Conv output then never touches float:
   acc32 -> (acc * M_o) >> SHIFT -> int8/int16 next-layer input.
2. Activations carried as int16 planes (headroom for dw/film adds);
   gelu/silu become int16->int16 LUTs built at boot from the static
   scales (the +-8 saturating-range trick keeps tables at 512 entries).
3. dw conv + norm: int16 MACs with the frozen-norm affine folded into
   the requant multiplier (norm gamma/ninv/mean are per-block statics).
4. Spectrum: logmag stays acc-domain; exp2 via integer LUT+shift into
   the Q30 FFT input directly (kills fft float packing too); phase via
   integer rsqrt (clz + Newton in int32).
5. Dynamic-quant stays ONLY at the two chain entries (embedding output,
   c8 production) where scales genuinely vary.

## Risks / gates
- Static scales clip outliers: corr gate on BOTH goldens after every
  stage; fallback = per-block scales with larger headroom.
- Precision: int16 activations ~ 2x the noise of per-frame int8 dynamic
  quant on paper, but the pipeline floor is already int8-dominated.
- Keep the float path compiled alongside (SNT_INT_CHAIN flag) so every
  port chooses; host gate runs both.

## Order of implementation (each host-gated, then C3-measured)
1. pw0->gelu->pw1 int16 chain in the trunk (biggest: ~2.5s)
2. dw+norm int16 (~1s)  3. spec/fft-pack integer (~2s)
4. rbc conv chain int16 (~1.5s)  5. head dequant fusion (~0.8s)
