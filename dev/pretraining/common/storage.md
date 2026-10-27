# Storage and publication plan

Future module: `common/storage.py`. Status: documentation only.

## Planned behavior

- Derive artifact paths from recipe, stage, attempt/result identity, and fingerprints under a configured work/artifact root.
- Keep immutable inputs, reusable completed outputs, temporary staging, and durable state distinguishable.
- Read explicit local/remote file sets and pinned revisions with bounded retries and actionable errors.
- Commit related data, stats, audit, and manifest files together where practical.
- Verify required files/checksums before advertising an artifact as complete.
- Offer a separate publication action that uploads a validated completed artifact and records the resulting remote revision.

Planning and local preview must not create remote repos or upload output. Local cache
retention/cleanup is an optional later convenience, constrained by durable completion.

Runtime data/checkpoints should live under the configured cache/work root, not be
committed into this source scaffold. The physical output layout, supported storage
backends, and retention defaults remain design decisions.
