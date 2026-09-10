"""Constants for the sanoTTS integration."""

from __future__ import annotations

from typing import Final

DOMAIN: Final = "sanotts"

# Config-entry keys.
CONF_VOICE: Final = "voice"
CONF_VOICE_DIR: Final = "voice_dir"

# Per-call TTS option key. There is no core constant for speaking rate, so the
# name is ours; the voice option uses homeassistant.components.tts.ATTR_VOICE.
OPT_LENGTH_SCALE: Final = "length_scale"


class VoiceInfo:
    """One selectable voice: its sanotts alias, a label, and its HA language."""

    __slots__ = ("alias", "label", "language")

    def __init__(self, alias: str, label: str, language: str) -> None:
        """Store the alias the sanotts package knows and how to present it."""
        self.alias = alias
        self.label = label
        self.language = language


# Sizes here are the MEASURED parameter counts, which for three voices differ
# from the figure baked into their package name: vi-vais1000-1p46m is actually
# 1,565,484 and id-newstts-1p46m 1,562,124, and amy is 1,454,284. Deriving the
# label from the package name reproduces the stale number the README was
# corrected away from.
# Mirrors pypkg/sanotts/tables/voices.json in this repository, converted to the
# hyphenated language tags Home Assistant uses (the package writes en_US).
# tools/check_sanotts_ha_voices.py asserts the two stay in step, so a voice
# added to the package but not here is a test failure rather than a surprise.
VOICES: Final[tuple[VoiceInfo, ...]] = (
    VoiceInfo("ar", "Arabic (1.57M)", "ar-JO"),
    VoiceInfo("cs", "Czech (1.57M)", "cs-CZ"),
    VoiceInfo("cs-tiny", "Czech small (510k)", "cs-CZ"),
    VoiceInfo("de", "German (1.57M)", "de-DE"),
    VoiceInfo("de-tiny", "German small (510k)", "de-DE"),
    VoiceInfo("amy", "Amy — English (1.45M)", "en-US"),
    VoiceInfo("amy-1p1m", "Amy small — English (1.1M)", "en-US"),
    VoiceInfo("amy-1p8m", "Amy large — English (1.8M)", "en-US"),
    VoiceInfo("heart", "Heart — English (2.27M)", "en-US"),
    VoiceInfo("heart-nano", "Heart nano — English (294k)", "en-US"),
    VoiceInfo("hfc", "HFC — English (1.8M)", "en-US"),
    VoiceInfo("kristin", "Kristin — English (1.4M)", "en-US"),
    VoiceInfo("es", "Spanish (1.56M)", "es-ES"),
    VoiceInfo("es-tiny", "Spanish small (510k)", "es-ES"),
    VoiceInfo("fr", "French (1.57M)", "fr-FR"),
    VoiceInfo("id", "Indonesian (1.56M)", "id-ID"),
    VoiceInfo("it", "Italian (1.57M)", "it-IT"),
    VoiceInfo("it-tiny", "Italian small (510k)", "it-IT"),
    VoiceInfo("pt", "Portuguese (1.57M)", "pt-BR"),
    VoiceInfo("pt-tiny", "Portuguese small (510k)", "pt-BR"),
    VoiceInfo("ro", "Romanian (1.57M)", "ro-RO"),
    VoiceInfo("ro-tiny", "Romanian small (510k)", "ro-RO"),
    VoiceInfo("ru", "Russian (1.57M)", "ru-RU"),
    VoiceInfo("ru-tiny", "Russian small (510k)", "ru-RU"),
    VoiceInfo("tr", "Turkish (1.56M)", "tr-TR"),
    VoiceInfo("tr-tiny", "Turkish small (510k)", "tr-TR"),
    VoiceInfo("vi", "Vietnamese (1.57M)", "vi-VN"),
    VoiceInfo("zh", "Chinese (1.55M)", "zh-CN"),
)

VOICES_BY_ALIAS: Final[dict[str, VoiceInfo]] = {v.alias: v for v in VOICES}

# Deterministic order, so the entity's language list is stable across restarts.
SUPPORTED_LANGUAGES: Final[list[str]] = sorted({v.language for v in VOICES})

DEFAULT_VOICE: Final = "amy"
