"""sanotts: a self-contained, numpy-only saanoTTS inference package.

    import sanotts
    result = sanotts.synthesize("Hello world", voice="amy")
    # result.audio: float32 mono waveform in [-1, 1]
    # result.sample_rate: int, e.g. 22050

For repeated synthesis, reuse a Synthesizer instead of calling
synthesize() (which reloads the voice pack every time):

    synth = sanotts.Synthesizer(voice="amy")
    a = synth.synthesize("First sentence.")
    b = synth.synthesize("Second sentence.")

Voices are resolved either from a local directory (--voice-dir / voice_dir=,
a package produced by tools/export_roota_self_contained_package.py in the
saanoTTS research repo) or downloaded by name into ~/.cache/sanotts/.
"""

from __future__ import annotations

from .engine import SynthesisResult, Synthesizer, synthesize
from .frontend import FrontendError
from .voicepack import VoicePackError

__all__ = [
    "SynthesisResult",
    "Synthesizer",
    "synthesize",
    "FrontendError",
    "VoicePackError",
]

__version__ = "0.1.0"
