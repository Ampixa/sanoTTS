/**
 * paperFacts.ts — every number displayed on this site, each one traced to its
 * evidence. This module is the single source of truth: chapter copy reads from
 * here, never from literals scattered in components.
 *
 * evidence keys:
 *   [meta]    public/engine/meta.json (shipped voice manifest, master branch)
 *   [audit]   saanotts docs/e12-nano-checkpoint-audit.json
 *   [e13]     saanotts experiments/e13-substitution-reallocation-sub300k-20260823.json
 *   [lanes]   saanotts docs/HANDOFF-sub300k-and-harmonic-source-20260827.md,
 *             docs/e12-nano-final-characterisation.md
 *   [wer]     paper Table 6 (ljtest150, four recognizers)
 *   [mos]     paper Table 7 (screened in-sample pilot, 31/44 sessions)
 *   [mcu]     saanotts mcu/ports/esp32s3/measurements/04-parallel-head-spec-COM5.log
 *             (RTF 0.185327, E12-nano 294,642) and saanotts-embedded README (R7 567,008)
 *   [fe]      saanotts docs/e12-nano-frontend-without-spacy.md
 *   [trace]   this branch: engine trace vs golden fixtures, min corr 0.981
 */

export const PARAMS = {
  duration: 22_858,
  acoustic: 65_299,
  decoder: 206_122,
  total: 294_279,
  /** training-only multi-period discriminator, never shipped */
  mpd: 14_872_645,
  /** the spaCy tagger the frontend replaced */
  spacyTagger: 1_570_514,
  /** earlier same-lineage configuration used for the S3 board timing */
  earlierTotal: 294_642,
  earlierSplit: { duration: 22_858, acoustic: 128_102, decoder: 143_682 },
  r7: 567_008,
  kokoro: 82_000_000,
  slimtts: 562_000,
  slimttsSplit: { front: 266_000, decoder: 296_000 },
} as const;

export const ARCH = {
  vocab: 62,
  dur: { hidden: 26, depth: 3, kernel: 5, maxTokens: 207, maxDuration: 80 },
  ac: { hidden: 31, tokenDepth: 3, frameDepth: 3, kernel: 5 },
  mel: 100,
  dec: {
    dim: 62, blocks: 4, pw: 186, dwKernel: 7, embedKernel: 7, noiseCh: 4,
  },
  head: { out: 1026, bins: 513 },
  istft: { nFft: 1024, hop: 256, window: "Hann", ola: true, dcBlocker: true },
  sampleRate: 24_000,
} as const;

export const FRONTEND = {
  goldEntries: 90_201,
  goldTagKeyed: 790,
  silverEntries: 93_361,
  rulesFw: { bitIdentical: 0.876, segmentalIdentical: 0.972, n: 249 },
  rulesMft: { bitIdentical: 0.9, segmentalRows: 4, n: 249 },
  swapScoreqDelta: 0.0, // [fe] n=249, E12 stack: identical ids ⇒ identical audio
} as const;

export const LANES = {
  n: 249,
  rows: [
    { key: "teacher", label: "Source teacher (Kokoro-82M)", scoreq: 4.809, learn: "82M teacher, upper bound" },
    { key: "target", label: "Decoder target (teacher mel → frozen Vocos)", scoreq: 4.787, learn: "mel-100 is not the bottleneck" },
    { key: "oracle", label: "Decoder oracle (teacher mel → student decoder)", scoreq: 2.864, learn: "the 206k decoder is" },
    { key: "full", label: "Full student (sanoTTS 294,279)", scoreq: 2.14, learn: "predicted durations + mel" },
  ],
  decoderGapShare: 0.727, // (4.787-2.864)/(4.787-2.140)
} as const;

export const OBJECTIVE = {
  scoreq: 2.14,
  utmos: 2.323,
  dnsmos: { sig: 3.274, bak: 4.068, ovrl: 3.023 },
  n: 249,
} as const;

export const WER = {
  set: "ljtest150",
  n: 150,
  rows: [
    { asr: "Whisper-small", teacher: 1.766, sanotts: 2.786 },
    { asr: "Whisper-medium", teacher: 1.727, sanotts: 2.041 },
    { asr: "wav2vec2-large", teacher: 2.433, sanotts: 3.297 },
    { asr: "HuBERT-large", teacher: 3.218, sanotts: 4.199 },
  ],
  meanTeacher: 2.286,
  meanSanotts: 3.081,
  delta: 0.795,
} as const;

export const MOS = {
  sessionsPassed: 31,
  sessionsTotal: 44,
  inSample: true,
  rows: [
    { label: "Vocos decoder target", mos: 4.121, ci: [3.845, 4.378] as const, ratings: 116 },
    { label: "3.5 kHz low-pass anchor", mos: 3.018, ci: [2.759, 3.263] as const, ratings: 114 },
    { label: "sanoTTS (294,279)", mos: 1.522, ci: [1.321, 1.743] as const, ratings: 115 },
  ],
} as const;

export const MCU = {
  s3: { name: "ESP32-S3", clock: "240 MHz", cores: 2, sramKiB: 512, rtf: 0.185, configParams: 294_642 },
  c3: { name: "ESP32-C3", clock: "160 MHz", cores: 1, sramKiB: 400, rtf: 5.72, configParams: 567_008 },
  weightsBytes: 345_232, // 109,296 front + 235,936 dec [meta]
  weightsKiB: 337.1,
  arenaPeakBytes: 220_928, // [trace] worst of 8 golden rows, full-student path
  isa: "portable C99; int8×int8→int32 kernels; float LayerNorm + iSTFT",
  gate: "golden-vector Pearson corr ≥ 0.98 vs float reference (min over 8 rows)",
} as const;

export const TRAINING = {
  pool: 49_999,
  train: 12_000,
  eval: 249,
  evalEvery: 200,
  evalUniqueWords: 1_611,
  evalMeanWords: 16.6,
  duration: { updates: 8_000, batch: 32, lr: 2e-3 },
  acoustic: { updates: 100_000, qat: "int8", lr: 2e-3, wd: 1e-5, seed: 4242 },
  decoder: {
    updates: 250_000, batch: 8, crop: 32, lr: 1e-4, lrMin: 5e-6,
    betas: [0.8, 0.99] as const, eps: 1e-9, wd: 0.01, seed: 8801,
    warmupSteps: 50_000, lsgan: 0.1, featureMatching: 0.5,
    mpdPeriods: [2, 3, 5, 7, 11] as const,
  },
} as const;

export const EVIDENCE_NOTE =
  "All values above are cross-checked against the repos listed in each group; " +
  "tensors on this site are int8 device-math traces of the shipped en_us_e13b " +
  "weights, validated against the golden fixtures (min corr 0.981 ≥ 0.98 gate).";
