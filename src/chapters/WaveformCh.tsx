import { ARCH } from "../data/paperFacts";
import { useViz } from "../data/store";

export function WaveformCh() {
  const trace = useViz((s) => s.trace);
  return (
    <div>
      <div className="chapter-kicker">07 · iSTFT + waveform</div>
      <h2>From spectra back to air pressure.</h2>

      <p>
        Each frame's 513 (magnitude, phase) pairs pass through an <b>inverse
        STFT</b>: a 1024-point inverse FFT, a Hann window, then overlap-add with a{" "}
        {ARCH.istft.hop}-sample hop — each output sample is the sum of four
        overlapping windowed frames. A first-order DC-block high-pass removes any
        residual offset. Output: {ARCH.sampleRate / 1000} kHz mono PCM.
      </p>

      <div className="formula">
        pcm = DCblock( OLA<sub>hop {ARCH.istft.hop}</sub>( hann<sub>{ARCH.istft.nFft}</sub> · iFFT(re·e<sup>i·ph</sup>) ) )
      </div>

      <h3>why it is cheap enough</h3>
      <p>
        No iterative vocoder, no neural upsampling — the iSTFT is a fixed linear
        transform plus elementwise work, so the "vocoder" adds zero parameters and
        a handful of MACs. All the intelligence already happened in the 206k
        decoder.
      </p>

      <h3>this sentence</h3>
      {trace && (
        <p className="small">
          {trace.T} frames → {trace.pcm.length.toLocaleString()} samples ={" "}
          {(trace.pcm.length / ARCH.sampleRate).toFixed(2)} s. On chip this exact
          int8→int16 bitstream is DMA'd to the I²S codec; in the browser it's the
          same float path you hear on the right.
        </p>
      )}

      <div className="lookright">
        an exact Hann/overlap-add schematic (4:1 window-to-hop ratio drawn to
        scale), then the traced waveform with a time axis — press play.
      </div>
    </div>
  );
}
