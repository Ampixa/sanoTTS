/**
 * paperFacts.ts — every number displayed on this site, in one place. This
 * module is the single source of truth: chapter copy reads from here, never
 * from literals scattered in components.
 *
 * Each figure is established in the paper (main.md); the tags below point to
 * the section or table where it is reported:
 *   [§2]  related work          [§3]  system architecture
 *   [§4]  training methodology  [§5]  evaluation methodology
 *   [§6]  design decisions      [§7]  MCU deployment
 *   [T2]  physical-board RTF    [T3]  parameter accounting
 *   [T4]  objective metrics     [T5]  SCOREQ lanes
 *   [T6]  ASR intelligibility   [T7]  MOS pilot
 *   [meta] shipped voice manifest (public/engine/meta.json)
 */

export const PARAMS = {
  duration: 22_858, // [§3.3, T3]
  acoustic: 65_299, // [§3.4, T3]
  decoder: 206_122, // [§3.5, T3]
  total: 294_279, // [§3.1, T3]
  /** training-only multi-period discriminator, never shipped [§4.2.3] */
  mpd: 14_872_645,
  /** the learned tagger the rule frontend replaced [§3.2] */
  neuralTagger: 1_570_514,
  /** earlier same-lineage configuration used for the S3 board timing [§7, T2] */
  earlierTotal: 294_642,
  r7: 567_008, // [§7, T2]
  kokoro: 82_000_000, // [§1]
  slimtts: 562_000, // [§2.1]
  slimttsSplit: { front: 266_000, decoder: 296_000 }, // [§2.1]
} as const;

export const ARCH = {
  vocab: 62, // [§3.2]
  dur: { hidden: 26, depth: 3, kernel: 5, maxTokens: 207, maxDuration: 80 }, // [§3.3]
  ac: { hidden: 31, tokenDepth: 3, frameDepth: 3, kernel: 5 }, // [§3.4]
  mel: 100, // [§3.4]
  dec: {
    dim: 62, blocks: 4, pw: 186, dwKernel: 7, embedKernel: 7, noiseCh: 4, // [§3.5]
  },
  head: { out: 1026, bins: 513 }, // [§3.5]
  istft: { nFft: 1024, hop: 256, window: "Hann", ola: true, dcBlocker: true }, // [§3.5]
  sampleRate: 24_000, // [§3.1]
} as const;

export const FRONTEND = {
  vocab: 62, // [§3.2]
  /** learned tagger replaced by deterministic context rules [§3.2] */
  neuralTagger: 1_570_514,
  /** SCOREQ movement after the swap, 249 held-out sentences [§3.2] */
  swapScoreqDelta: 0.0,
  swapN: 249,
  /** retained from the reference frontend [§3.2] */
  kept: "tokenizer · lexicons · normalization rules · eSpeak NG fallback",
} as const;

export const LANES = {
  n: 249, // [§5.1]
  rows: [
    { key: "teacher", label: "Source teacher (Kokoro-82M)", scoreq: 4.809, learn: "82M teacher, upper bound" },
    { key: "target", label: "Decoder target (teacher mel → frozen Vocos)", scoreq: 4.787, learn: "mel-100 is not the bottleneck" },
    { key: "oracle", label: "Decoder oracle (teacher mel → student decoder)", scoreq: 2.864, learn: "the 206k decoder is" },
    { key: "full", label: "Full student (sanoTTS 294,279)", scoreq: 2.14, learn: "predicted durations + mel" },
  ], // [T5]
  decoderGapShare: 0.727, // [§8.3] (4.787−2.864)/(4.787−2.140)
} as const;

export const OBJECTIVE = {
  scoreq: 2.14,
  utmos: 2.323,
  dnsmos: { sig: 3.274, bak: 4.068, ovrl: 3.023 },
  n: 249,
} as const; // [T4]

export const WER = {
  set: "ljtest150",
  n: 150, // [§5.2]
  rows: [
    { asr: "Whisper-small", teacher: 1.766, sanotts: 2.786 },
    { asr: "Whisper-medium", teacher: 1.727, sanotts: 2.041 },
    { asr: "wav2vec2-large", teacher: 2.433, sanotts: 3.297 },
    { asr: "HuBERT-large", teacher: 3.218, sanotts: 4.199 },
  ], // [T6]
  meanTeacher: 2.286,
  meanSanotts: 3.081,
  delta: 0.795, // [T6, §9]
} as const;

export const MOS = {
  sessionsPassed: 31,
  sessionsTotal: 44, // [T7]
  inSample: true, // [§8.5] test sentences are part of the training pack
  rows: [
    { label: "Vocos decoder target", mos: 4.121, ci: [3.845, 4.378] as const, ratings: 116 },
    { label: "3.5 kHz low-pass anchor", mos: 3.018, ci: [2.759, 3.263] as const, ratings: 114 },
    { label: "sanoTTS (294,279)", mos: 1.522, ci: [1.321, 1.743] as const, ratings: 115 },
  ], // [T7]
} as const;

export const MCU = {
  s3: { name: "ESP32-S3", clock: "240 MHz", cores: 2, sramKiB: 512, rtf: 0.185, configParams: 294_642 }, // [§7, T2]
  c3: { name: "ESP32-C3", clock: "160 MHz", cores: 1, sramKiB: 400, rtf: 5.72, configParams: 567_008 }, // [§7, T2]
  weightsBytes: 345_232, // [meta] 109,296 front + 235,936 dec
  weightsKiB: 337.1,
  arenaPeakBytes: 220_928, // worst-case arena over this site's own traces; reported live by the WASM demo
  ringFrames: 7, // [§7] seven-frame streaming ring buffer
  isa: "portable C99; int8×int8→int32 kernels; float LayerNorm + iSTFT", // [§7]
  gate: "bit-exact golden vectors vs the float reference, on every supported platform", // [§1]
} as const;

export const TRAINING = {
  pool: 49_999, // [§4.1]
  train: 12_000, // [§4.1]
  eval: 249, // [§4.1]
  evalEvery: 200, // [§4.1]
  evalUniqueWords: 1_611, // [§4.1]
  evalMeanWords: 16.6, // [§4.1]
  duration: { updates: 8_000, batch: 32, lr: 2e-3 }, // [§4.2.1]
  acoustic: { updates: 100_000, qat: "int8", lr: 2e-3, wd: 1e-5, seed: 4242 }, // [§4.2.2]
  decoder: {
    updates: 250_000, batch: 8, crop: 32, lr: 1e-4, lrMin: 5e-6,
    betas: [0.8, 0.99] as const, eps: 1e-9, wd: 0.01, seed: 8801,
    warmupSteps: 50_000, lsgan: 0.1, featureMatching: 0.5,
    mpdPeriods: [2, 3, 5, 7, 11] as const,
  }, // [§4.2.3]
} as const;

/** Controlled design experiments that justify the final architecture [§6] */
export const DESIGN = {
  /** merged text-to-waveform network vs the factored pipeline, 25 held-out sentences [§6.2] */
  merged: { scoreq: 1.06, wer: 0.28, n: 25 },
  factored: { scoreq: 2.95, wer: 0.0, n: 25 },
  /** rank-63 factorized width-160 five-block decoder vs a width-112 control [§6.5] */
  factorizedDecoder: { params: 697_826, rank: 63, scoreq: 3.7, controlWidth: 112, controlScoreq: 3.4 },
  /** 294,279 is 48% smaller than SlimTTS's 562k for the same pipeline coverage [§2.1] */
  slimmerThanSlimttsPct: 48,
} as const;
