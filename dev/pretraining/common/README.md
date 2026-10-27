# Shared pipeline helpers

Status: documentation only. These plans define shared behavior to be implemented
before the worker and filters rely on it.

| Plan | Intended future module | Responsibility |
|---|---|---|
| [contracts.md](contracts.md) | `contracts.py` | Document records, manifests, stage results, and schema/capability checks. |
| [fingerprints.md](fingerprints.md) | `fingerprints.py` | Recipe/stage/result identity and cache compatibility. |
| [splits.md](splits.md) | `splits.py` | Stable duplicate-group train/validation membership. |
| [state.md](state.md) | `state.py` | Attempts, durable completion, atomic checkpoints, and resume. |
| [storage.md](storage.md) | `storage.py` | Artifact paths, bounded I/O, related-file commits, and publishing. |

Config validation/resolution is specified separately in
[configs/schema.md](../configs/schema.md), for a future `common/config.py` helper.
Avoid implementing separate incompatible state/metadata conventions in each stage.
