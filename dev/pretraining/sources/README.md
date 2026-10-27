# Source adapters

Status: documentation only. Future Python modules will normalize source-specific
acquisition into the [common document contract](../common/contracts.md).

| Plan | Future module | Responsibility |
|---|---|---|
| [Institutional Books](institutional.md) | `institutional.py` | Source metadata/text mapping and revision-aware acquisition. |
| [Internet Archive](internet_archive.md) | `internet_archive.py` | Candidate discovery, bounded downloads, and captured acquisition manifests. |
| [Parquet / prior artifacts](parquet.md) | `parquet.py` | Loading explicit existing shard sets or compatible pinned intermediate outputs. |

Adapters own field names, units, stable native identifiers, fetching, and acquisition
failures. Shared stages own filtering policy. Source-query restrictions can reduce
unnecessary downloads, but must be recorded as part of input selection.

Different adapters must not fabricate unavailable metadata to satisfy downstream
stages. Their declared capabilities determine which recipes can consume them.
