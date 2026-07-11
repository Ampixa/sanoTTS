#!/usr/bin/env python3
"""Compatibility imports for the package-owned FSD model blocks."""

from __future__ import annotations

import sys
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from saanotts.models.fsd import HOP_LENGTH  # noqa: E402
from saanotts.models.fsd import FactorizedSpectralHead  # noqa: E402
from saanotts.models.fsd import FsdConvNeXtBlock  # noqa: E402
from saanotts.models.fsd import initialize_factorized_spectral_head  # noqa: E402
from saanotts.models.fsd import logmag_phase_synthesize  # noqa: E402

__all__ = [
    "HOP_LENGTH",
    "FactorizedSpectralHead",
    "FsdConvNeXtBlock",
    "initialize_factorized_spectral_head",
    "logmag_phase_synthesize",
]
