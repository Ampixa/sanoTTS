/**
 * Evidence-side right views: budget stack, SCOREQ lanes, evaluation tables,
 * deployment memory bar, and the live WASM console. All numbers come from
 * paperFacts.ts — the single source of truth.
 */
import { useState } from "react";
import {
  ARCH, LANES, MCU, MOS, OBJECTIVE, PARAMS, WER,
} from "../data/paperFacts";
import { useViz } from "../data/store";
import { synthesizeLive, type LiveResult } from "../engine/live";
import { togglePlayback } from "../audio/player";
import { TensorCanvas } from "./TensorCanvas";

const MONO = "Lucida Console, monospace";

export function BudgetChart() {
  const W = 560, H = 220;
  const segs = [
    { name: "frontend (rules)", p: 0, color: "#eee" },
    { name: "duration", p: PARAMS.duration, color: "#DC143C" },
    { name: "acoustic", p: PARAMS.acoustic, color: "#b30f30" },
    { name: "decoder", p: PARAMS.decoder, color: "#7a0a20" },
  ];
  let x = 30;
  const barW = 470, y0 = 40;
  const mpdX = (PARAMS.mpd / 1e6).toFixed(1);
  return (
    <div style={{ flex: 1, display: "flex", flexDirection: "column", justifyContent: "center", padding: "0 18px" }}>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%" }}>
        <text x={30} y={18} fontSize={12} fontFamily={MONO} fill="#111">
          sanoTTS — {PARAMS.total.toLocaleString()} params · {(PARAMS.total * 4 / 1e6).toFixed(2)} MB fp32
        </text>
        {segs.map((s) => {
          const w = (s.p / PARAMS.total) * barW;
          const x0 = x;
          x += w;
          return (
            <g key={s.name}>
              <rect x={x0} y={y0} width={Math.max(1, w)} height={34} fill={s.color} stroke="#fff" />
              {w > 60 && (
                <text x={x0 + w / 2} y={y0 + 21} textAnchor="middle" fontSize={10} fill={s.color === "#eee" ? "#888" : "#fff"} fontFamily={MONO}>
                  {s.name}
                </text>
              )}
            </g>
          );
        })}
        <text x={30} y={y0 + 58} fontSize={10.5} fill="#666" fontFamily={MONO}>
          duration {PARAMS.duration.toLocaleString()} ({(100 * PARAMS.duration / PARAMS.total).toFixed(1)}%)
          · acoustic {PARAMS.acoustic.toLocaleString()} ({(100 * PARAMS.acoustic / PARAMS.total).toFixed(1)}%)
          · decoder {PARAMS.decoder.toLocaleString()} ({(100 * PARAMS.decoder / PARAMS.total).toFixed(1)}%)
        </text>
        {/* training-only MPD ghost — not to scale, 50× too wide to draw */}
        <text x={30} y={y0 + 94} fontSize={12} fontFamily={MONO} fill="#888">
          training-only MPD discriminator — {PARAMS.mpd.toLocaleString()} params
        </text>
        <rect x={30} y={y0 + 102} width={barW} height={20} fill="none" stroke="#DC143C" strokeDasharray="4 3" />
        <text x={34} y={y0 + 116} fontSize={9.5} fill="#DC143C" fontFamily={MONO}>
          ghost bar is 1/50 of true width — used only during training, never shipped
        </text>
        <text x={30} y={y0 + 146} fontSize={11} fontFamily={MONO} fill="#111">
          int8 export: {(MCU.weightsBytes / 1024).toFixed(1)} KiB of weights · arena peak {(MCU.arenaPeakBytes / 1024).toFixed(1)} KiB
        </text>
        <text x={30} y={y0 + 164} fontSize={10} fill="#888" fontFamily={MONO}>
          counts verified against meta.json + blob offsets by tools/verify_facts.py
        </text>
      </svg>
    </div>
  );
}

export function LanesChart() {
  const W = 560;
  const y = (i: number) => 40 + i * 46;
  const colors = ["#bbb", "#888", "#111", "#DC143C"];
  return (
    <div style={{ flex: 1, display: "flex", flexDirection: "column", justifyContent: "center", padding: "0 18px" }}>
      <svg viewBox={`0 0 ${W} 240`} style={{ width: "100%" }}>
        <text x={24} y={18} fontSize={12} fontFamily={MONO} fill="#111">
          SCOREQ (1–5, higher is better) · n={LANES.n} held-out sentences
        </text>
        {LANES.rows.map((r, i) => (
          <g key={r.key}>
            <rect x={24} y={y(i)} width={((r.scoreq - 1) / 4) * 430} height={16} fill={colors[i]} />
            <text x={24 + ((r.scoreq - 1) / 4) * 430 + 6} y={y(i) + 12} fontSize={10.5} fill="#111" fontFamily={MONO}>
              {r.scoreq.toFixed(2)}
            </text>
            <text x={24} y={y(i) + 30} fontSize={10} fill="#666" fontFamily={MONO}>
              {r.label}
            </text>
            <text x={454} y={y(i) + 30} textAnchor="end" fontSize={9.5} fill="#DC143C" fontFamily={MONO}>
              {r.learn}
            </text>
          </g>
        ))}
        <text x={24} y={232} fontSize={9.5} fill="#888" fontFamily={MONO}>
          {(LANES.decoderGapShare * 100).toFixed(0)}% of the mel-target → full-student gap is the 206k decoder, not the frontend or durations
        </text>
      </svg>
    </div>
  );
}

export function EvidenceChart() {
  return (
    <div style={{ flex: 1, display: "flex", flexDirection: "column", justifyContent: "center", padding: "0 22px", gap: 14, overflowY: "auto" }}>
      <div>
        <div className="viz-title" style={{ padding: "0 0 4px" }}>WER % — {WER.set} <span className="dim">(n={WER.n}, four recognizers)</span></div>
        <table className="facts">
          <thead><tr><th>recognizer</th><th>recordings</th><th>sanoTTS</th><th>Δ</th></tr></thead>
          <tbody>
            {WER.rows.map((r) => (
              <tr key={r.asr}>
                <td>{r.asr}</td><td>{r.teacher.toFixed(2)}</td><td>{r.sanotts.toFixed(2)}</td>
                <td className="delta">+{(r.sanotts - r.teacher).toFixed(2)}</td>
              </tr>
            ))}
            <tr><td><b>mean</b></td><td><b>{WER.meanTeacher.toFixed(2)}</b></td><td><b>{WER.meanSanotts.toFixed(2)}</b></td><td className="delta"><b>+{WER.delta.toFixed(2)}</b></td></tr>
          </tbody>
        </table>
      </div>
      <div>
        <div className="viz-title" style={{ padding: "0 0 4px" }}>subjective MOS <span className="dim">({MOS.sessionsPassed}/{MOS.sessionsTotal} sessions screened, in-sample)</span></div>
        <table className="facts">
          <thead><tr><th>system</th><th>MOS</th><th>95% CI</th><th>ratings</th></tr></thead>
          <tbody>
            {MOS.rows.map((r) => (
              <tr key={r.label}>
                <td>{r.label}</td><td>{r.mos.toFixed(2)}</td>
                <td>[{r.ci[0].toFixed(2)}, {r.ci[1].toFixed(2)}]</td><td>{r.ratings}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div>
        <div className="viz-title" style={{ padding: "0 0 4px" }}>objective metrics <span className="dim">(n={OBJECTIVE.n}, held-out)</span></div>
        <table className="facts">
          <tbody>
            <tr><td>SCOREQ</td><td>{OBJECTIVE.scoreq.toFixed(2)}</td></tr>
            <tr><td>UTMOS</td><td>{OBJECTIVE.utmos.toFixed(3)}</td></tr>
            <tr><td>DNSMOS sig / bak / ovrl</td><td>{OBJECTIVE.dnsmos.sig.toFixed(2)} / {OBJECTIVE.dnsmos.bak.toFixed(2)} / {OBJECTIVE.dnsmos.ovrl.toFixed(2)}</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}

export function DeployChart() {
  const arenaKiB = MCU.arenaPeakBytes / 1024;
  const weightsKiB = MCU.weightsBytes / 1024;
  const arenaFrac = MCU.arenaPeakBytes / (MCU.s3.sramKiB * 1024);
  return (
    <div style={{ flex: 1, display: "flex", flexDirection: "column", justifyContent: "center", padding: "0 18px", gap: 12 }}>
      <svg viewBox="0 0 560 190" style={{ width: "100%" }}>
        <text x={24} y={16} fontSize={12} fontFamily={MONO} fill="#111">
          {MCU.s3.name} SRAM — {MCU.s3.sramKiB} KiB budget
        </text>
        <rect x={24} y={28} width={430} height={26} fill="none" stroke="#111" />
        <rect x={24} y={28} width={arenaFrac * 430} height={26} fill="#DC143C" />
        <text x={30} y={45} fontSize={10} fill="#fff" fontFamily={MONO}>arena peak {arenaKiB.toFixed(1)} KiB = {(arenaFrac * 100).toFixed(1)}%</text>
        <text x={24} y={74} fontSize={10} fill="#666" fontFamily={MONO}>
          weights ({weightsKiB.toFixed(1)} KiB int8) are read straight from memory-mapped flash — only the arena lives in SRAM
        </text>

        <text x={24} y={106} fontSize={12} fontFamily={MONO} fill="#111">measured RTF (board logs)</text>
        <text x={24} y={126} fontSize={11} fontFamily={MONO} fill="#111">
          {MCU.s3.name} @{MCU.s3.clock}: <tspan fill="#DC143C" fontWeight="bold">{MCU.s3.rtf}</tspan>
          <tspan fill="#888" fontSize={9.5}> — earlier {MCU.s3.configParams.toLocaleString()}-param config</tspan>
        </text>
        <text x={24} y={146} fontSize={11} fontFamily={MONO} fill="#111">
          {MCU.c3.name} @{MCU.c3.clock}: <tspan fill="#DC143C" fontWeight="bold">{MCU.c3.rtf}</tspan>
          <tspan fill="#888" fontSize={9.5}> — {MCU.c3.configParams.toLocaleString()}-param R7 config, slower than realtime</tspan>
        </text>
        <text x={24} y={174} fontSize={9.5} fill="#888" fontFamily={MONO}>
          every firmware build gated: {MCU.gate}
        </text>
      </svg>
    </div>
  );
}

export function LiveCard() {
  const trace = useViz((s) => s.trace);
  const [result, setResult] = useState<LiveResult | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (!trace) return <div className="stage-caption">loading trace…</div>;

  const run = async () => {
    setRunning(true); setError(null);
    try {
      setResult(await synthesizeLive({
        ids: trace.ids,
        seedLo: trace.manifest.seed_lo,
        seedHi: trace.manifest.seed_hi,
        refPcm: trace.pcm,
      }));
    } catch (e) {
      setError(String(e));
    } finally {
      setRunning(false);
    }
  };

  return (
    <div style={{ flex: 1, display: "flex", flexDirection: "column", justifyContent: "center", padding: "0 22px", gap: 10 }}>
      <div className="formula">
        input: {trace.N} ids + seed → snt_nano_wasm_synthesize() → PCM
      </div>
      <div className="playbar">
        <button className="playbtn" disabled={running} onClick={run}>
          {running ? "synthesizing…" : "▶ run WASM now"}
        </button>
        {result && (
          <button className="playbtn" onClick={() => togglePlayback(result.pcm, ARCH.sampleRate)}>▶ play result</button>
        )}
      </div>
      {error && <div className="small accent">{error}</div>}
      {result && (
        <div className="stat-row">
          <div className="stat-chip accent"><b>{result.samples.toLocaleString()}</b>samples</div>
          <div className="stat-chip"><b>{result.frames}</b>frames</div>
          <div className="stat-chip"><b>{result.elapsedMs.toFixed(0)} ms</b>synthesis</div>
          <div className="stat-chip"><b>{(result.arenaPeak / 1024).toFixed(1)} KiB</b>arena peak</div>
          <div className="stat-chip"><b>{result.corrVsTrace?.toFixed(4) ?? "—"}</b>corr vs shipped trace</div>
        </div>
      )}
      <div className="small">
        module: <code>snt_nano_heartnano.wasm</code> · blobs {(MCU.weightsBytes / 1024).toFixed(0)} KiB int8 ·
        seeded PRNG ⇒ deterministic · corr 1.0000 means your browser reproduced the
        shipped trace sample-for-sample
      </div>
      {result && (
        <TensorCanvas
          data={result.pcm}
          rows={1}
          cols={result.pcm.length}
          label="live PCM (row view)"
          height={70}
          xAxis={{ label: "samples", toUnit: (c) => `${(c / ARCH.sampleRate).toFixed(2)} s` }}
          rowLabel=""
          valueFmt={(v) => v.toFixed(3)}
        />
      )}
    </div>
  );
}
