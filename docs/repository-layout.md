# Repository Layout And Migration Contract

The repository is organized as a monorepo because training, evaluation, export,
runtime, and publication share versioned model and benchmark contracts. Paths
are migrated incrementally so historical commands and experiment evidence remain
usable.

## Canonical Ownership

| Concern | Canonical path now | Status |
| --- | --- | --- |
| Shared Python APIs | `src/saanotts/` | Active; extraction from `tools/` is incremental |
| Training/evaluation commands | `tools/` | Compatibility entry points; keep executable |
| New experiment recipes | `experiments/recipes/` | Active for new work |
| Historical launch scripts | `scripts/` | Frozen compatibility paths pending indexed archive |
| Portable runtime | `mcu/` | Canonical runtime implementation |
| Runtime lab/legacy C3 | `esp32c3/` | Migration source; no new portable API ownership |
| Browser application | `web/` | Active application; generated WASM stays ignored |
| Portable configuration | `configs/` | Canonical small teacher/workflow config |
| Versioned text contracts | `data/textsets/` | Canonical small datasets |
| Local run outputs | `artifacts/` | Ignored, non-canonical working data |
| Release staging | `dist/` | Ignored local staging |
| Release records | `docs/release/` | Canonical manifests and claims |
| Papers | `paper/` | Compatibility path until papers are split safely |
| Frozen prototypes | `scratchpad/` | Integrate or index before archival; no promoted code |

`saanotts.workspace` centralizes repository and external voice paths. New Python
code must use it or explicit CLI arguments instead of adding `/Users/...` or
`/home/...` constants.

## Artifact Boundary

Git contains source, recipes, small text contracts, test fixtures, manifests,
and curated result summaries. GitHub releases contain checkpoints and deployable
packages. Local `artifacts/` contains caches, renders, dashboards, intermediate
checkpoints, and run directories.

Ignored does not mean disposable. Before removing or relocating any artifact
root, follow [`preservation/README.md`](preservation/README.md) and verify its
Tier A files against the latest checksum ledger and remote archive.

## Runtime Boundary

`mcu/` owns the public C API, reference kernels, ports, and cross-platform
correctness gate. Its default host test is self-contained:

```bash
make -C mcu test
```

`esp32c3/` remains available for the original runtime, FSD lab harness, device
history, and exporters not yet migrated. Once every required app/export path is
represented under `mcu/` and the golden gate passes, the legacy tree can move to
an archive or leave the repository through normal Git history.

## Python Migration

The large trainer files currently serve as both commands and imported libraries.
Moving them directly would break hundreds of internal import edges. The safe
sequence for each domain is:

1. Add focused tests around checkpoint loading and output contracts.
2. Extract reusable classes/functions into `src/saanotts/<domain>/`.
3. Keep the original `tools/*.py` file as a thin compatibility command.
4. Update consumers in bounded groups.
5. Remove a compatibility import only after `rg` shows no remaining users.

The first domains are duration, acoustic, decoder, evaluation, and export.
The shared FSD blocks are the first completed extraction:
`src/saanotts/models/fsd.py` owns the implementation and
`tools/roota_fsd_blocks.py` preserves the historical import path.

## Experiment Migration

New experiment definitions belong in `experiments/recipes/<family>/`. Generated
outputs always go to `artifacts/`. Existing `scripts/*.sh` paths remain stable
until an archive manifest records the old path, entry point, model family,
inputs, and result directory. Do not bulk-move the shell matrix without that
index.

## Documentation Policy

- `README.md` states current capabilities and directs readers.
- `docs/README.md` is the documentation index.
- Current how-to material lives in undated guide files.
- Architecture decisions and experiment records retain their dates.
- `GOAL.md` is a historical lab ledger until it is split by dated sections; it
  is not the source for setup instructions.
- Release claims must link a manifest, benchmark summary, or preservation asset.

## Migration Phases

1. **Foundation:** package boundary, path resolver, top-level checks, tracked MCU
   fixtures, working CI, and documentation index.
2. **Python extraction:** move shared model/data/evaluation code behind stable
   compatibility commands.
3. **Experiment indexing:** convert active runs to recipes and index historical
   shell launchers before archiving them.
4. **Runtime consolidation:** finish app/export migration from `esp32c3/` into
   `mcu/`, then retire the legacy tree.
5. **Publication split:** separate APSIPA and embedded-runtime paper build roots
   while keeping shared references explicit.
6. **Research archive:** split the root ledgers into current status plus dated,
   searchable research records.
