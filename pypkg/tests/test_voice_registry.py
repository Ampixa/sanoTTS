"""The bundled voice registry (sanotts/tables/voices.json).

Since 0.5.6 the name heart-nano means the streaming-native 263k voice
(heartnanos/); heart-nano-s is kept as an alias so 0.5.5 scripts keep working.
The first heart-nano (heartnano/, 294k) is no longer in the registry.

Run:  python3 -m pytest pypkg/tests/test_voice_registry.py -v
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pypkg"))

from sanotts import voicepack  # noqa: E402


def _voices():
    with voicepack.KNOWN_VOICES_TABLE.open(encoding="utf-8") as fh:
        return json.load(fh)["voices"]


def test_heart_nano_is_the_streaming_voice():
    e = voicepack._registry_entry("heart-nano")
    assert e["package"] == "heartnanos"
    assert e["parameters"] == 263_103
    assert e["archive"] == "heartnanos-k3ft40k"
    assert voicepack.voice_runtime("heart-nano") == "nano"


def test_heart_nano_s_is_an_alias():
    a, b = voicepack._registry_entry("heart-nano-s"), voicepack._registry_entry("heart-nano")
    assert a.pop("alias_of") == "heart-nano"
    assert a == b


def test_first_heart_nano_is_gone():
    assert all(v["package"] != "heartnano" for v in _voices().values())


def test_aliases_point_at_real_entries():
    voices = _voices()
    for name, v in voices.items():
        target = v.get("alias_of")
        if target is None:
            continue
        assert target in voices and "alias_of" not in voices[target], name
        same = {k: x for k, x in v.items() if k != "alias_of"}
        assert same == voices[target], name


def test_unknown_voice_lists_names():
    with pytest.raises(voicepack.VoicePackError, match="heart-nano"):
        voicepack._registry_entry("heart-nanoo")
