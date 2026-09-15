# sanoTTS Visualization

Visualization of the complete **sanoTTS en_us_e13b** text-to-speech pipeline.

Every tensor visualized on the page comes from a real inference trace of the shipped int8 engine.

## Run

```bash
npm install
npm run dev
```

## What you can explore

The visualization walks through the complete synthesis pipeline:

```text
Text
 ↓
Phoneme IDs
 ↓
Duration Predictor
 ↓
Acoustic Model
 ↓
Mel Spectrogram
 ↓
Waveform Decoder
 ↓
PCM Audio
```

Each stage shows the actual intermediate data produced by the model, including:

* Phoneme IDs and frontend processing
* Predicted durations
* Acoustic hidden states
* Mel spectrograms
* Decoder representations
* Spectrum
* Generated waveform
* Audio playback

## Interaction

The page uses a two-panel layout:

* **Left:** the scrollytelling explanation. Scrolling moves through the pipeline.
* **Right:** an interactive workbench for the current stage.

The workbench supports tensor inspection, hover and cursor interactions, waveform playback, and live model execution without changing the current stage.

The pipeline bar at the top shows the current stage and provides explicit navigation.

## Live inference

The site includes the browser version of the sanoTTS int8 engine compiled to WebAssembly.

The **Run Live** section performs actual inference in the browser and allows its output to be compared with the recorded trace.

## Data

The visualization uses precomputed inference traces for the walkthrough and the WebAssembly engine for live inference.

Traces are loaded from:

```text
public/traces/
```

The browser engine and voice data are provided in:

```text
public/engine/
```

## Project structure

```text
src/
├── chapters/       # Scrollytelling chapters
├── right/          # Pipeline and tensor visualizations
├── three/          # 3D pipeline overview
├── data/            # Trace loading and application state
├── audio/           # WebAudio playback
└── engine/          # Browser WASM inference

public/
├── traces/          # Precomputed inference traces
└── engine/          # WASM engine and voice data

tools/
├── run_traces.sh    # Generate inference traces
└── pack_traces.py   # Prepare traces for the browser
``
