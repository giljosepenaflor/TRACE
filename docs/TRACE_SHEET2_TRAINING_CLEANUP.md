# Training-source CSV cleanup (2026-10-31)

**Superseded by [training-only version 2](TRACE_SHEET2_TRAINING_ONLY_V2.md).**
The user clarified that all AdvBench/HarmBench input belongs to the new training
pool, with StrongREJECT supplying validation/testing. The 595-row pass below
applied an overly broad reservation filter. Use the corrected 797-row version;
this report and its outputs are preserved as historical processing evidence.

The user supplied `TRACE Data - Sheet2.csv` and requested consistent categories,
overlap removal and a training-only inventory. Its 920 rows contain 520 AdvBench
and 400 HarmBench questions, with blank IDs/categories/splits. Both question
lists match the pinned local source files exactly and in order. The input
SHA-256 is `337416a4b48f9efdaee2600a12720b9d7c73bd6b958dfa033ba4c3337e7538c5`.
The supplied file is unchanged.

## Outputs

- `outputs/trace-training-20261031/TRACE_training_cleaned.csv`: 595 source
  candidates, with stable prompt IDs, original question text, source, topic,
  `split=train`, restored context, official/local source ID and source flags.
- `outputs/trace-training-20261031/TRACE_training_review.xlsx`: the same training
  rows, seven-category definitions/counts, all 325 exclusions with reasons and
  retained duplicate IDs, and processing notes.
- [Category codebook](TRACE_CATEGORY_TAXONOMY.md): the versioned seven-category
  convention to reuse for new source metadata.

| Disposition | Rows |
| --- | ---: |
| Retained training source candidates | 595 |
| Removed redundant paraphrases | 66 |
| Excluded normalized matches to protected material | 200 |
| Withheld possible overlaps or related protected paraphrases | 59 |
| Original input total | 920 |

The retained pool contains 314 AdvBench and 281 HarmBench questions. Category
counts are 152 cybercrime/intrusion, 45 chemical/biological, 49 harassment/bullying,
71 misinformation/disinformation, 103 illegal, 71 harmful and 104 copyright.

## Cleaning and isolation

Source text was preserved. Stable IDs use the normalized question and the exact
context, preserving code case/indentation. Duplicate detection includes context:
the five repeated-question groups in HarmBench contain seven additional rows
with different context passages, not redundant complete examples. They were
not merged; individual reservations can still exclude some members.

Reviewed AdvBench paraphrase groups identify the same requested objective.
Different explicit targets, constraints, requested modalities or context passages
are retained. This is a conservative reviewed grouping, not certification of
exhaustive semantic independence.

Historical validation/test/pilot reservations and the planned StrongREJECT
holdout were compared against the input. Normalized matches are excluded.
Character n-gram cosine similarity at least 0.88 screens for possible overlap;
such matches and related reviewed paraphrases are withheld for review rather
than being declared automatically equivalent. Every decision retains its source
row, reference or representative ID. Original manifests are unchanged.

The original export omitted 100 nonempty HarmBench contexts. They were recovered
from the hash-pinned local source after exact row alignment. The retained pool
has 57 of these contexts. Always keep each context attached to its question.

## Eligibility and interpretation

All retained rows are assigned `train` at the user's request. This new source
pool does not rewrite earlier partition assignments or admit records into the
existing training pipeline. In particular, 276 retained questions are reported
in the adapter's first-400 AdvBench training allocation. Their
`reported_adapter_training_source` flag is explicit in the CSV. The prior
quarantine remains in force; their admission requires a revised prospective
eligibility protocol and separate reporting of target-model exposure.

The file is not 595 labeled TRACE interactions. No model calls, generation
signals, response reviews, activation labels, fitting or new performance results
were produced. Topic categories are metadata, not activation labels or an
additional input feature for the proposed response-focused detector.

Reproduction details, reviewed row groups, intermediate audit, previews and
verification are under `artifacts/sheet2-cleanup-20261031/`. Old detector, test,
label and source hashes were checked for preservation. CSV/XLSX row contents,
unique IDs, category totals, training-only splits and original text/context
preservation were verified independently after export.
