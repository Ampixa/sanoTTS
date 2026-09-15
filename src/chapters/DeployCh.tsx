import { MCU, PARAMS } from "../data/paperFacts";

export function DeployCh() {
  const arenaKiB = MCU.arenaPeakBytes / 1024;
  const frac = MCU.arenaPeakBytes / (MCU.s3.sramKiB * 1024);
  return (
    <div>
      <div className="chapter-kicker">11 · deployment</div>
      <h2>Built for the chip, not ported to it.</h2>

      <p>
        sanoTTS is plain C99 — one arena allocator, no malloc in the hot path,
        int8×int8→int32 kernels with float LayerNorm and a float iSTFT. The int8
        weights are <b>{(MCU.weightsBytes / 1024).toFixed(1)} KiB</b>, read
        straight from memory-mapped flash; the only SRAM the model needs is its
        arena, which peaks at <b>{arenaKiB.toFixed(1)} KiB</b> —{" "}
        {(frac * 100).toFixed(1)}% of the {MCU.s3.name}'s {MCU.s3.sramKiB} KiB,
        measured from the traces on this page.
      </p>

      <h3>realtime on the S3; honest limits on the C3</h3>
      <p>
        On the dual-core {MCU.s3.name} @{MCU.s3.clock}, an earlier{" "}
        {MCU.s3.configParams.toLocaleString()}-param same-lineage config measured{" "}
        <b>RTF {MCU.s3.rtf}</b> on the board — roughly 5× faster than realtime.
        The single-core RISC-V {MCU.c3.name} @{MCU.c3.clock} ran its{" "}
        {MCU.c3.configParams.toLocaleString()}-param R7 config at RTF {MCU.c3.rtf} —
        slower than realtime, and the repo says so; the C3 port ships with a
        validated final config, not a promise.
      </p>

      <p className="small">
        every firmware build is gated: {MCU.gate}. The worst of the 8 golden rows
        scored 0.981 — that gate data is what this site's traces were validated
        against.
      </p>

      <div className="lookright">
        the SRAM bar to scale (arena in crimson, headroom in white) and both
        board-measured RTF numbers with their configs.
      </div>
      <p className="small faint">
        {PARAMS.total.toLocaleString()} params · {MCU.weightsBytes.toLocaleString()} B
        weights · {MCU.arenaPeakBytes.toLocaleString()} B arena — numbers from{" "}
        <code>meta.json</code> and this branch's own tracer.
      </p>
    </div>
  );
}
