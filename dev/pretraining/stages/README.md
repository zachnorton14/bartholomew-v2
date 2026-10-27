# Processing stage plans

Status: documentation only. Each Markdown document stands in for a future stage
module of the same name, with `.py` replacing `.md` after implementation is approved.

```text
gather -> screen -> refine -> dedupe -> era -> prior -> pack -> tokenize
```

| Stage plan | Responsibility |
|---|---|
| [gather](gather.md) | Acquire normalized, traceable source records. |
| [screen](screen.md) | Apply agreed eligibility and available source-quality checks. |
| [refine](refine.md) | Clean book text while preserving evidence and identity. |
| [dedupe](dedupe.md) | Group exact duplicates and establish stable representatives/splits. |
| [era](era.md) | Enforce the historical cutoff through versioned temporal rules. |
| [prior](prior.md) | Optionally score or filter corpus-derived token-frequency outliers. |
| [pack](pack.md) | Export text-only training shards and document-position audit mappings. |
| [tokenize](tokenize.md) | Build token shards with the pinned training tokenizer. |

Each stage will accept a validated input artifact and resolved parameters, and return
a versioned result manifest, summary, and its required artifacts. Shared helpers own
[contracts](../common/contracts.md), [state](../common/state.md), and
[storage](../common/storage.md).

Policy decisions belong in reviewed configs/rule assets. Source-specific acquisition
belongs in adapters. Publishing is a worker/storage action, not another filter.
No stage is executable yet.
