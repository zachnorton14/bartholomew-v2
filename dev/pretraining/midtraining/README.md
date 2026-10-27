# Midtraining pipeline: fresh planning area

Status: structure and text plans only. Previous extraction, cleaning, mixture, and
anneal files are isolated in [old/](old/README.md).

## Active plans

| Document | Purpose |
|---|---|
| [../README.md](../README.md) | Shared fresh scaffold and layer index. |
| [../standard/pipeline_proposal.md](../standard/pipeline_proposal.md) | Shared config-driven architecture and stage ordering. |
| [../configs/midtraining/](../configs/midtraining/README.md) | Planned STEM corpus recipes and source/quality choices. |
| [../sources/](../sources/README.md) | Acquisition adapters for original-corpus reuse, Institutional Books, and Internet Archive. |
| [../stages/](../stages/README.md) | Shared gathering/filtering/packing/pretokenization stage plans. |
| [../TO DO.md](<../TO DO.md>) | PT issues and unresolved filtering/sourcing work. |

The fresh design reuses stage implementations with standard pretraining; the
midtraining recipe supplies its own sources and reviewed parameters. It should not
fork the cleaning libraries into another independently maintained copy.

Mixture construction and anneal training remain archived references. Extending the
new pipeline to those activities requires a separate decision after the initial
corpus layers are agreed.

No fresh executable implementation is present. Review this structure before moving
to config/contracts or worker code.
