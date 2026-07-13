"""Ties the frontend, voice pack, and numpy models into one synthesizer."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import frontend, models, voicepack

logger = logging.getLogger("sanotts.engine")


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
    ) -> None:
        self.pack = voicepack.load_voice(voice, voice_dir=voice_dir, cache_dir=cache_dir)
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

    def synthesize(self, text: str, *, duration_length_scale: float | None = None) -> SynthesisResult:
        scale = float(duration_length_scale) if duration_length_scale is not None else self.pack.duration_length_scale
        if scale <= 0.0:
            raise ValueError(f"duration_length_scale must be positive, got {scale}")

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
) -> SynthesisResult:
    """One-shot convenience wrapper. For synthesizing many strings with the
    same voice, construct a `Synthesizer` once instead -- it amortizes the
    (much slower) weight-loading and phonemizer-initialization cost."""
    synth = Synthesizer(voice, voice_dir=voice_dir, cache_dir=cache_dir)
    return synth.synthesize(text, duration_length_scale=duration_length_scale)
