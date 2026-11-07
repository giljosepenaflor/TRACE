# Training-only CSV correction, version 2 (2026-10-31)

**Superseded by [version 3](TRACE_SUPPLIED_STRONGREJECT_OVERLAP.md).** The user
subsequently asked to remove overlap with their supplied StrongREJECT CSV from
this training pool. The current pool has 769 questions; this 797-row version is
preserved as the input to that comparison.

The user clarified that all supplied AdvBench/HarmBench questions are training
sources for the new experiment. StrongREJECT will supply separate validation
and test subsets. Version 2 therefore removes only internal duplicates and
supersedes the earlier 595-row reservation-filtered pass.

## Current outputs

- `outputs/trace-training-20261031-v2/TRACE_training_cleaned.csv`: stable IDs,
  unchanged questions, seven topic categories, `split=train`, restored context,
  source IDs and informational source flags.
- `outputs/trace-training-20261031-v2/TRACE_training_review.xlsx`: the matching
  training rows, category definitions/counts, duplicate removals, future
  StrongREJECT overlap references and processing notes.
- [Category codebook](TRACE_CATEGORY_TAXONOMY.md): the same seven codes as before.

| Source | Input | Training questions | Internal duplicates removed |
| --- | ---: | ---: | ---: |
| AdvBench | 520 | 397 | 123 |
| HarmBench | 400 | 400 | 0 |
| Total | 920 | 797 | 123 |

Category counts are 200 cybercrime/intrusion, 67 chemical/biological, 62
harassment/bullying, 103 misinformation/disinformation, 163 illegal, 97 harmful
and 105 copyright. HarmBench annotations are preserved; AdvBench is coded
against the same topic domains. These are not activation labels.

## Cleaning and provenance

The original upload is unchanged, with SHA-256
`337416a4b48f9efdaee2600a12720b9d7c73bd6b958dfa033ba4c3337e7538c5`.
Its 520/400 question lists match the pinned sources exactly and in order.
All 100 omitted nonempty HarmBench contexts are restored. Repeated instructions
with different context passages remain distinct examples. Reviewed AdvBench
same-objective paraphrases retain one representative. Different explicit
constraints, targets, requested modalities and contexts are preserved. This
does not certify exhaustive semantic independence.

Historical reservations and reported adapter-training exposure are metadata,
not exclusions from this new user-selected pool. There are 313 retained
questions reported in the adapter's first-400 AdvBench source allocation.
Exposure should be reported when defining the new experiment. Historical
partitions and artifacts are unchanged; old test questions admitted into this
pool cannot be reused as fresh held-out tests of a newly fitted detector.

## Future StrongREJECT validation and testing

Keep the training pool fixed and filter confirmed overlaps on the StrongREJECT
side before splitting it into disjoint validation and final-test sets. The
workbook's `Future holdout` sheet records normalized matches, reviewed paraphrase
relationships and possible lexical matches (character n-gram similarity at
least 0.88). Lexical scores are screening results requiring review, not automatic
semantic verdicts. StrongREJECT has not been rewritten or partitioned here.

The outputs were checked for matching CSV/XLSX rows, unique IDs, seven nonblank
categories, training-only splits, unchanged question/context content and complete
duplicate accounting. The CSV preserves exact source content; the XLSX review
uses standard XML line-ending normalization for two context cells. Original
research hashes remain unchanged. Evidence is
under `artifacts/sheet2-cleanup-20261031/` with version-2 filenames.

This is a source inventory. No model requests, response reviews, generation
signals, activation labels or detector fitting were produced.
