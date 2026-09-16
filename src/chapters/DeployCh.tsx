import { MCU, PARAMS } from "../data/paperFacts";

export function DeployCh() {
  const arenaKiB = MCU.arenaPeakBytes / 1024;
  const frac = MCU.arenaPeakBytes / (MCU.s3.sramKiB * 1024);
  return (
    <div>
      <div className="chapter-kicker">11 · deployment</div>
      <h2>Built for the chip, not ported to it.</h2>

      <p>
        The runtime is plain C99 with no processor-specific intrinsics in its
        platform-independent core: one arena allocator, no malloc in the hot
        path, int8 weights accumulated into 32-bit integers, with LayerNorm and
        the iSTFT kept in floating point because their behavior does not survive
        integer approximation. Hardware acceleration enters through a small
        porting interface, and every port is checked against a scalar reference
        implementation. The int8 weights are{" "}
        <b>{(MCU.weightsBytes / 1024).toFixed(1)} KiB</b>, read straight from
        memory-mapped flash; the only SRAM the model needs is its arena, peaking
        at <b>{arenaKiB.toFixed(1)} KiB</b> — {(frac * 100).toFixed(1)}% of the{" "}
        {MCU.s3.name}'s {MCU.s3.sramKiB} KiB, measured from the traces on this
        page.
      </p>

      <h3>Realtime on the S3; honest limits on the C3</h3>
      <p>
        The two boards bracket the MCU class: the dual-core {MCU.s3.name} (
        {MCU.s3.clock}, vector integer instructions) and the single-core RISC-V{" "}
        {MCU.c3.name} ({MCU.c3.clock}, no hardware FPU). On the S3, an earlier{" "}
        {MCU.s3.configParams.toLocaleString()}-parameter configuration of the same
        runtime lineage measured <b>RTF {MCU.s3.rtf}</b> on the physical board —
        roughly 5× faster than realtime. The C3 ran a{" "}
        {MCU.c3.configParams.toLocaleString()}-parameter configuration at RTF{" "}
        {MCU.c3.rtf} — offline, and reported as such. These are earlier-lineage
        board measurements, not checkpoint-exact timings for the final model;
        the paper states this explicitly.
      </p>

      <p className="small">
        Every port is gated the same way: {MCU.gate}. The {MCU.ringFrames}-frame
        ring buffer keeps the working set small enough to stream on device.
      </p>

      <div className="lookright">
        The SRAM bar to scale (arena in crimson, headroom in white) and both
        board-measured RTF numbers with their configurations.
      </div>
      <p className="small faint">
        {PARAMS.total.toLocaleString()} params · {MCU.weightsBytes.toLocaleString()} B
        weights · {MCU.arenaPeakBytes.toLocaleString()} B arena peak — the weight
        sizes come from the shipped voice manifest, the arena peak from this
        site's own traces.
      </p>
    </div>
  );
}
