# Standard pipeline: fresh planning area

Status: structure and text plans only. The new config-driven pipeline is not
implemented. Previous standard code and notebooks are isolated in [old/](old/README.md).

## Active documents

| Document | Purpose |
|---|---|
| [pipeline_proposal.md](pipeline_proposal.md) | Architecture proposal, ordered stages through pretokenization, illustrative config, and design tradeoffs. |
| [../README.md](../README.md) | Shared fresh scaffold and layer index. |
| [../configs/standard/](../configs/standard/README.md) | Planned standard-corpus recipe configs. |
| [../stages/](../stages/README.md) | Reusable stage plans shared with midtraining. |
| [../TO DO.md](<../TO DO.md>) | Core fixes, unresolved investigations, and lower-priority ideas. |

The broad historical book recipe will use the shared worker, source adapters,
contracts, and stages. Its thresholds, source revisions, split settings, rules,
prior mode, and tokenizer belong in its future config rather than separate copies
of the processing code.

## Historical references

[old/README.md](old/README.md) inventories all previous builders, cells, scripts,
notebooks, licenses, and supporting documents. [DATASET_PIPELINE.md](DATASET_PIPELINE.md)
and [FILTER_AUDIT.md](FILTER_AUDIT.md) are pointers retained for existing documentation
links; their historical contents live in `old/`.

No legacy executable file is imported into the fresh scaffold. The next implementation
step requires user review of this structure before code is added.
