"""Resolve repository paths without machine-specific constants."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WorkspacePaths:
    root: Path
    artifacts: Path
    data: Path
    experiments: Path
    portable_runtime: Path
    legacy_runtime: Path
    papers: Path
    piper_voices: Path

    def as_json_dict(self) -> dict[str, str]:
        return {key: str(value) for key, value in asdict(self).items()}


def discover_repo_root(start: Path | None = None) -> Path:
    configured = os.environ.get("SAANOTTS_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()

    origin = (start or Path(__file__)).resolve()
    if origin.is_file():
        origin = origin.parent
    for candidate in (origin, *origin.parents):
        if (candidate / ".git").exists() and (candidate / "README.md").is_file():
            return candidate
    raise RuntimeError(
        "could not find the saanoTTS repository; set SAANOTTS_ROOT explicitly"
    )


def configured_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default.resolve()


def get_workspace(root: Path | None = None) -> WorkspacePaths:
    repo = (root or discover_repo_root()).resolve()
    return WorkspacePaths(
        root=repo,
        artifacts=configured_path("SAANOTTS_ARTIFACT_ROOT", repo / "artifacts"),
        data=configured_path("SAANOTTS_DATA_ROOT", repo / "data"),
        experiments=repo / "experiments",
        portable_runtime=repo / "mcu",
        legacy_runtime=repo / "esp32c3",
        papers=repo / "paper",
        piper_voices=configured_path(
            "SAANOTTS_PIPER_VOICES_ROOT",
            repo.parent / "g2p/data/external/piper_voices",
        ),
    )


def main() -> None:
    print(json.dumps(get_workspace().as_json_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
