"""Ties the frontend, voice pack, and numpy models into one synthesizer."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import frontend, models, voicepack

logger = logging.getLogger("sanotts.engine")

# The nano decoder is noise-fed, so a rendering is only reproducible if the
# seed is fixed. Callers that want variation pass their own.
DEFAULT_NANO_SEED = 2236265385529901705

# Which grapheme-to-phoneme path the nano voices use.
#   "espeak"  nano_frontend: espeak-ng on every word, then misaki's E2M rewrite.
#             Needs phonemizer-fork + espeakng-loader, both GPL-3.0.
#   "lexicon" nano_g2p: misaki's own dictionaries, then a small numpy model for
#             out-of-vocabulary words. No espeak, no GPL dependency, and closer
#             to what the voices were distilled from -- see
#             docs/nano-espeak-free-frontend.md for the measurement.
# "lexicon" is the default. Audio quality is a wash between the two -- a paired
# 24-sentence A/B on both voices put every confidence interval across zero
# (experiments/evidence/nano-frontend-ab-20260904.json) -- so the choice is made
# on the other three counts: no GPL dependency, a token stream that matches what
# the voices were actually distilled on (82.1% of sentences reproduce real
# misaki, against 1.8% for espeak), and a 10x faster cold start. Pass
# nano_g2p="espeak" to get the old path back.
NANO_G2P_CHOICES = ("espeak", "lexicon")
DEFAULT_NANO_G2P = "lexicon"

# Which grapheme-to-phoneme path the piperlite voices use.
#   "espeak"  frontend: espeak-ng through phonemizer-fork, then piper's exact
#             phonemes_to_ids framing. This is what the voices were distilled
#             against. No longer the default, and no longer installed by
#             default -- it needs the GPL-3.0 pair, so `pip install
#             sanotts[espeak]`.
#   "lexicon" piper_g2p: the espeak-free path, dispatched on the voice's own
#             espeak voice string. English voices go through the nano_g2p
#             lexicon, mapped from misaki's compressed alphabet back into the
#             espeak IPA the phoneme_id_map is keyed on; "id" goes through
#             id_g2p and "vi" through vi_g2p, rule front ends written from
#             Indonesian and Vietnamese orthography. A voice in any other
#             language raises rather than falling back to English.
# Unlike the nano voices, the piperlite voices were distilled from Piper/VITS
# teachers whose own front end is espeak, so "lexicon" feeds them a token
# distribution they were not trained on. That was the reason to measure it
# rather than assume it: across amy, kristin and hfc, 14 of 15 paired
# confidence intervals cross zero and the one that does not favours the
# espeak-free path, and vi is a clean pass. Indonesian is the one place with a
# cost -- DNSMOS drops about 1% with a CI excluding zero, while SCOREQ, UTMOS
# and ASR word error rate see nothing -- and that was listened to and accepted.
# Evidence: piperlite-espeak-free-ab-20260904.json (English) and
# idvi-espeak-free-ab-20260904.json (id, vi) under experiments/evidence/.
# Pass piperlite_g2p="espeak" to get the old path back; it needs phonemizer-fork
# and espeakng-loader, which are GPL-3.0 and no longer installed by default.
#   "indo"    the Indonesian voice only: piper_g2p dispatched through
#             id_indo_bridge, which is a port of snowfluke/indo-g2p (MIT) and
#             places the schwa lexically instead of positionally and emits the
#             glottal stops Indonesian word-final `k` calls for. Every other
#             voice behaves exactly as "lexicon". Measured against an
#             independent Wiktionary oracle it reaches 97.3% of Indonesian word
#             types with the schwa where a dictionary puts it, against 68.4%
#             for espeak-ng and 67.1% for "lexicon", and emits 78 of the 78
#             required glottal stops against espeak-ng's 1.
#             It is the default, with one cost accepted knowingly: the current
#             weights were distilled on espeak's phonemes, so correct ones are a
#             distribution they never saw, and on 100 held-out sentences that
#             costs Whisper word error rate -- 0.2320 against "lexicon"'s
#             0.2037, a paired +0.0283 whose 95% CI [+0.0097, +0.0463] excludes
#             zero, while every MOS predictor sees nothing. The phonemes are
#             right and the weights are what need to move: the next Indonesian
#             distillation runs through this front end, and the regression goes
#             with it. Pass piperlite_g2p="lexicon" for the espeak-shaped
#             phonemes the current weights were trained on. See
#             experiments/evidence/id-indo-g2p-ab-20260904.json and
#             docs/id-indo-g2p-frontend.md.
PIPERLITE_G2P_CHOICES = ("espeak", "lexicon", "indo")
DEFAULT_PIPERLITE_G2P = "indo"


@dataclass(frozen=True)
class SynthesisResult:
    audio: np.ndarray
    sample_rate: int

    def __array__(self, dtype=None) -> np.ndarray:  # convenience: np.asarray(result) just works
        return self.audio if dtype is None else self.audio.astype(dtype)


class Synthesizer:
    """A loaded voice, ready to render text repeatedly without re-reading disk."""

    def __init__(
        self,
        voice: str | None = None,
        *,
        voice_dir: str | Path | None = None,
        cache_dir: str | Path | None = None,
        nano_g2p: str = DEFAULT_NANO_G2P,
        piperlite_g2p: str = DEFAULT_PIPERLITE_G2P,
    ) -> None:
        if nano_g2p not in NANO_G2P_CHOICES:
            raise ValueError(
                f"nano_g2p must be one of {NANO_G2P_CHOICES}, got {nano_g2p!r}"
            )
        if piperlite_g2p not in PIPERLITE_G2P_CHOICES:
            raise ValueError(
                f"piperlite_g2p must be one of {PIPERLITE_G2P_CHOICES}, got {piperlite_g2p!r}"
            )
        self.nano_g2p = nano_g2p
        self.piperlite_g2p = piperlite_g2p
        self.pack = voicepack.load_voice(voice, voice_dir=voice_dir, cache_dir=cache_dir)

        # The nano voices are a different graph and a different front end, so
        # they take a different path entirely rather than being squeezed into
        # the piperlite one. Both expose the same synthesize()/sample_rate.
        self.is_nano = isinstance(self.pack, voicepack.NanoVoicePack)
        if self.is_nano:
            from . import nano_frontend  # noqa: PLC0415
            self._vocabulary = nano_frontend.vocabulary_for(self.pack.meta)
            self._max_tokens = int(self.pack.offsets.get(
                "NANO_DUR_MAX_TOKENS", nano_frontend.DEFAULT_MAX_TOKENS))
            self.phoneme_table = None
            return

        self.phoneme_table = frontend.load_phoneme_table(self.pack.phoneme_config_path)

        self._duration_tensors = self.pack.component_tensors("duration")
        self._duration_config = self.pack.component_config("duration")
        self._acoustic_tensors = self.pack.component_tensors("acoustic")
        self._acoustic_config = self.pack.component_config("acoustic")
        self._decoder_tensors = self.pack.component_tensors("decoder")
        self._decoder_config = self.pack.component_config("decoder")

    @property
    def sample_rate(self) -> int:
        return self.pack.sample_rate

    def synthesize_ids(self, phoneme_ids, *, duration_length_scale: float | None = None,
                       seed: int | None = None) -> SynthesisResult:
        """Render phoneme ids directly, skipping grapheme-to-phoneme.

        Either nano front end can be skipped this way; this is the entry point
        for callers that already have ids, and the one the golden fixtures
        exercise.
        """
        if not self.is_nano:
            raise NotImplementedError(
                f"synthesize_ids is only implemented for the nano voices; "
                f"{self.pack.name!r} is a piperlite voice"
            )
        from . import nano  # noqa: PLC0415

        scale = float(duration_length_scale) if duration_length_scale is not None else 1.0
        if scale <= 0.0:
            raise ValueError(f"duration_length_scale must be positive, got {scale}")
        audio = nano.synthesize_ids(
            self.pack.front, self.pack.dec, self.pack.offsets,
            np.asarray(phoneme_ids, dtype=np.int64),
            seed=int(seed if seed is not None else DEFAULT_NANO_SEED),
            length_scale=scale,
        )
        return SynthesisResult(audio=np.clip(audio, -1.0, 1.0).astype(np.float32),
                               sample_rate=self.sample_rate)

    def synthesize(self, text: str, *, duration_length_scale: float | None = None) -> SynthesisResult:
        if self.is_nano:
            if self.nano_g2p == "lexicon":
                from . import nano_g2p as nano_g2p_module  # noqa: PLC0415
                ids, dropped = nano_g2p_module.phonemize(
                    text, vocabulary=self._vocabulary, max_tokens=self._max_tokens)
            else:
                from . import nano_frontend  # noqa: PLC0415
                ids, dropped = nano_frontend.phonemize(
                    text, vocabulary=self._vocabulary, max_tokens=self._max_tokens)
            if dropped:
                logger.debug("sanotts: dropped %d symbols outside the vocabulary: %r",
                             len(dropped), dropped)
            return self.synthesize_ids(ids, duration_length_scale=duration_length_scale)

        scale = float(duration_length_scale) if duration_length_scale is not None else self.pack.duration_length_scale
        if scale <= 0.0:
            raise ValueError(f"duration_length_scale must be positive, got {scale}")

        if self.piperlite_g2p in ("lexicon", "indo"):
            from . import piper_g2p  # noqa: PLC0415
            ids, unmapped = piper_g2p.text_to_phoneme_ids(
                text, self.phoneme_table,
                path="indo" if self.piperlite_g2p == "indo" else "lexicon")
            if unmapped:
                logger.warning(
                    "sanotts: %d symbol(s) have no id in this voice's phoneme_id_map "
                    "and were skipped: %r", len(unmapped), unmapped)
        else:
            ids = frontend.text_to_phoneme_ids(text, self.phoneme_table)
        durations = models.duration_forward(
            self._duration_tensors, self._duration_config, ids, length_scale=scale
        )
        latent = models.acoustic_forward(self._acoustic_tensors, self._acoustic_config, ids, durations)
        audio = models.decoder_forward(self._decoder_tensors, self._decoder_config, latent)
        audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
        return SynthesisResult(audio=audio, sample_rate=self.sample_rate)


def synthesize(
    text: str,
    voice: str | None = None,
    *,
    voice_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
    duration_length_scale: float | None = None,
    nano_g2p: str = DEFAULT_NANO_G2P,
    piperlite_g2p: str = DEFAULT_PIPERLITE_G2P,
) -> SynthesisResult:
    """One-shot convenience wrapper. For synthesizing many strings with the
    same voice, construct a `Synthesizer` once instead -- it amortizes the
    (much slower) weight-loading and phonemizer-initialization cost."""
    synth = Synthesizer(voice, voice_dir=voice_dir, cache_dir=cache_dir,
                        nano_g2p=nano_g2p, piperlite_g2p=piperlite_g2p)
    return synth.synthesize(text, duration_length_scale=duration_length_scale)
