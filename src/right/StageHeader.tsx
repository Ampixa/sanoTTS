/**
 * StageHeader — one line that always shows the live shapes of the selected
 * sentence as it flows through the current stage, plus the toggle that lets
 * any stage summon the interactive 3D overview as an optional overlay.
 */
import { useViz } from "../data/store";
import { ARCH, MCU, MOS, PARAMS, WER } from "../data/paperFacts";

export function StageHeader() {
  const chapter = useViz((s) => s.chapter);
  const trace = useViz((s) => s.trace);
  const overviewOpen = useViz((s) => s.overviewOpen);
  const setOverviewOpen = useViz((s) => s.setOverviewOpen);

  const N = trace?.N ?? 0;
  const T = trace?.T ?? 0;
  const S = trace?.pcm.length ?? 0;
  const secs = S / ARCH.sampleRate;

  const shapes: Record<number, string> = {
    0: `[${N} ids] → [${N} durs] → [31×${N}] → [31×${T}] → [100×${T}] → [62×${T}] ×4 → [1026×${T}] → ${S.toLocaleString()} samples`,
    1: `text → ${N} tokens → phonemes → [${N} ids] · ${ARCH.vocab}-symbol vocab · 0 params`,
    2: `[${N} ids] → student [26×${N}] → ${N} durations → Σ ${T} frames (${secs.toFixed(2)} s)`,
    3: `token-rate [31×${N}] → length regulator → frame-rate [31×${T}] → mel-100`,
    4: `mel [100 × ${T}] · hop ${ARCH.istft.hop} · ${(T * ARCH.istft.hop / ARCH.sampleRate).toFixed(2)} s`,
    5: `noise [4×${T}] + mel → stem [62×${T}] → 4 ConvNeXt blocks → [62×${T}]`,
    6: `final LN + 62→1026 head → log-mag [513×${T}] + phase [513×${T}]`,
    7: `iSTFT n_fft ${ARCH.istft.nFft} hop ${ARCH.istft.hop} → OLA → DC block → ${S.toLocaleString()} samples (${secs.toFixed(2)} s)`,
    8: `0 + ${PARAMS.duration.toLocaleString()} + ${PARAMS.acoustic.toLocaleString()} + ${PARAMS.decoder.toLocaleString()} = ${PARAMS.total.toLocaleString()} params`,
    9: `SCOREQ lanes · n=249 held-out · teacher → target → oracle → full`,
    10: `WER ${WER.meanSanotts.toFixed(2)}% · MOS pilot ${MOS.sessionsPassed}/${MOS.sessionsTotal} sessions · in-sample`,
    11: `weights ${(MCU.weightsBytes / 1024).toFixed(1)} KiB flash + arena ≤ ${(MCU.arenaPeakBytes / 1024).toFixed(1)} KiB SRAM`,
    12: `browser WASM · int8 · same C99 as the chip`,
  };

  return (
    <div id="stage-header">
      <b>{trace ? trace.info.row_id : "…"}</b>
      {"  ·  "}
      <span className="shape">{shapes[chapter] ?? ""}</span>
      {trace && chapter !== 0 && (
        <button
          className="ovbtn ov-toggle"
          title="bring the interactive 3D pipeline overview into this stage (optional — the story stays put)"
          onClick={() => setOverviewOpen(!overviewOpen)}
        >
          {overviewOpen ? "✕ overview" : "⤢ 3D overview"}
        </button>
      )}
    </div>
  );
}
