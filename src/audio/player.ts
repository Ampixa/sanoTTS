/**
 * player.ts — plays a RowTrace's PCM (24 kHz float) through WebAudio and
 * publishes a 0..1 playhead into the store so every panel stays in sync.
 */
import { useViz } from "../data/store";

let ctx: AudioContext | null = null;
let src: AudioBufferSourceNode | null = null;
let raf = 0;
let startedAt = 0;
let dur = 0;

function audioCtx(): AudioContext {
  if (!ctx) ctx = new AudioContext({ sampleRate: 24000 });
  return ctx;
}

export function stopPlayback() {
  cancelAnimationFrame(raf);
  try { src?.stop(); } catch { /* already stopped */ }
  src = null;
  useViz.getState().setPlaying(false);
}

export function playPcm(pcm: Float32Array, sampleRate: number) {
  stopPlayback();
  const ac = audioCtx();
  const buf = ac.createBuffer(1, pcm.length, sampleRate);
  buf.getChannelData(0).set(pcm);
  src = ac.createBufferSource();
  src.buffer = buf;
  src.connect(ac.destination);
  dur = pcm.length / sampleRate;
  startedAt = ac.currentTime;
  src.start();
  useViz.getState().setPlaying(true);
  const tick = () => {
    const t = (audioCtx().currentTime - startedAt) / dur;
    if (t >= 1) {
      useViz.getState().setPlayhead(0);
      stopPlayback();
      return;
    }
    useViz.getState().setPlayhead(t);
    raf = requestAnimationFrame(tick);
  };
  raf = requestAnimationFrame(tick);
}

export function togglePlayback(pcm: Float32Array | null, sampleRate: number) {
  if (!pcm || pcm.length === 0) return;
  if (useViz.getState().playing) stopPlayback();
  else playPcm(pcm, sampleRate);
}
