# Data provenance

Vintage CORE v1.0.0 adapts the 22-task DataComp-LM CORE evaluation suite for
language models with a 1930 knowledge cutoff. It retains 20 task labels and
drops `bigbench_cs_algorithms` and `bigbench_dyck_languages`.

The source task names and physical dataset paths are recorded in each bundle's
`core.yaml`. The retained families are:

- SQuAD, CoQA, and BoolQ;
- COPA, CommonsenseQA, PIQA, and OpenBookQA;
- Jeopardy, BIG-bench QA Wikidata, ARC-Easy, and ARC-Challenge;
- AGI Eval LSAT-AR, BIG-bench Operators, and BIG-bench Repeat Copy Logic;
- HellaSwag, LAMBADA OpenAI, Winograd, Winogrande, and BIG-bench Language
  Identification.

The filtered and restyled bundles each contain 47,092 unique physical rows.
HellaSwag and HellaSwag zero-shot intentionally share one physical file, so the
task-level count is larger than the physical-row count.

## Transformations

- The filtered bundle removes examples judged to require post-1930 knowledge.
- Reviewed backfills restore qualifying small tasks to their original sizes.
- The restyled bundle is aligned row-for-row with the filtered bundle and
  conservatively rewrites eligible prose while protecting scoring fields.
- Rows that could not be rewritten safely fall back to the exact filtered row.

Detailed counts, prompts, review gates, and limitations are recorded in the
[final report](docs/FINAL_REPORT.md), [build log](docs/BUILD_LOG.md), and
[audit summaries](docs/audits/).

## Standalone extraction

The files under `data/` were copied from commit
`ff70312fcf2a11d67c619122794919f85059227a` of the `Vintage-CORE` branch in
`zachnorton14/think.nano`. No rows were generated, removed, or edited during
the extraction. `DATA_MANIFEST.sha256` records the committed file hashes.

## Terms

The repository's MIT License applies to project code and original documentation.
It does not relicense third-party evaluation content. Users are responsible for
reviewing the original datasets' terms and citation requirements before use or
redistribution. DataComp-LM also asks evaluation-suite users to cite the original
evaluation papers collected by that project.

The original v1.0.0 release audit recorded unresolved redistribution questions for
several incorporated sources. In particular, it did not establish blanket grants
for COPA, AGIEval LSAT-AR, Winograd, CommonsenseQA, Jeopardy clues, LAMBADA book
passages, or every domain represented in CoQA. Other tasks carry upstream terms
such as Apache-2.0, Creative Commons, AFL-3.0, or repository-specific notices that
may require attribution or share-alike handling. This standalone extraction did
not independently re-clear those sources. Public availability of the bundle does
not supersede upstream rights or establish that its text is MIT-licensed.
