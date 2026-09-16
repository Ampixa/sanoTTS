/**
 * MelDetail — the 100-channel mel contract with real axes (seconds, mel band),
 * a colorbar carrying this sentence's actual range, and a synced playhead.
 */
import { useViz } from "../data/store";
import { TensorCanvas } from "./TensorCanvas";
import { ARCH } from "../data/paperFacts";
import { togglePlayback } from "../audio/player";

export function MelDetail() {
  const trace = useViz((s) => s.trace);
  const playing = useViz((s) => s.playing);
  if (!trace) return <div className="stage-caption">loading trace…</div>;
  const T = trace.T;
  const secs = T * ARCH.istft.hop / ARCH.sampleRate;

  return (
    <div style={{ flex: 1, display: "flex", flexDirection: "column", justifyContent: "center" }}>
      <div className="playbar" style={{ padding: "0 16px" }}>
        <button className="playbtn" onClick={() => togglePlayback(trace.pcm, ARCH.sampleRate)}>
          {playing ? "■ stop" : "▶ play"} 
        </button>
        <span className="playtime">the crimson line tracks the audio</span>
      </div>
      <TensorCanvas
        data={trace.mel}
        rows={ARCH.mel}
        cols={T}
        label="mel spectrogram (log-mel energy)"
        height={360}
        mode="hot"
        colorbar
        flipY
        playhead
        rowLabel="mel band"
        xAxis={{
          label: "time (s)",
          toUnit: (c) => `${(c * ARCH.istft.hop / ARCH.sampleRate).toFixed(2)} s`,
          ticks: [0, 0.2, 0.4, 0.6, 0.8, 1].map((f) => ({ at: f * T, text: (f * secs).toFixed(1) })),
        }}
        yAxis={{
          label: "mel band",
          toUnit: (r) => `band ${r}`,
          ticks: [0, 25, 50, 75, 99].map((b) => ({ at: b, text: `${b}` })),
        }}
        valueFmt={(v) => `${v.toFixed(2)} log-mel`}
      />
      <div className="stage-caption">
        band 0 at the bottom (low pitch), band 99 at top · crimson = energy,
        white = silence · the horizontal stripes are the voice's pitch harmonics
      </div>
    </div>
  );
}
