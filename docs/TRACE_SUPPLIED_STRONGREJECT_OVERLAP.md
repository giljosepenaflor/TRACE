# Supplied StrongREJECT overlap removal, version 3 (2026-10-31)

At the user's request, the superseded `outputs/trace-training-20261031` and
`outputs/trace-training-20261031-v2` folders were removed. Only the final `-v3`
deliverable folder remains. The original input files and intermediate audit
metadata under `artifacts/sheet2-cleanup-20261031/` are retained.

The user supplied `strongreject-dataset-for-training.csv` and explicitly asked
to remove overlapping questions from the AdvBench/HarmBench training pool.
This supersedes the version-2 plan to filter on the StrongREJECT side.

The comparison file has 313 nonblank questions. Its SHA-256 is
`f8864d7bf3ba99ce8899203407a83e4f5960a8ed1d7de675490f969618a4689c`.
Its ordered question strings match the previous local StrongREJECT reference,
but this revision records the supplied file itself as the comparison source.
The supplied StrongREJECT file is unchanged. Earlier deliverable folders were
subsequently removed as documented above; versioned processing metadata remains.

## Current outputs

- `outputs/trace-training-20261031-v3/TRACE_training_cleaned.csv`
- `outputs/trace-training-20261031-v3/TRACE_training_review.xlsx`

| Training source | Before comparison | Removed | Retained |
| --- | ---: | ---: | ---: |
| AdvBench | 397 | 28 | 369 |
| HarmBench | 400 | 0 | 400 |
| Total | 797 | 28 | 769 |

All retained rows keep `split=train`, stable IDs, the seven topic categories and
original question/context content. All 100 restored HarmBench passages remain.
The earlier 123 internal duplicate removals are still in effect.

## Comparison decisions

The 28 removals comprise 25 normalized question-group matches (including source
variants already consolidated during internal deduplication), two reviewed near
paraphrases and one original question embedded in a longer reference request.
Every removal links its training ID and original row to the supplied
StrongREJECT CSV row and question. Overlap decisions include the original
variant evidence, so deduplicating a question under another representative does
not hide its overlap with the comparison file.

Character n-gram screening and an offline pinned all-MiniLM-L6-v2 embedding
screen were used to identify additional pairs for review. Similarity alone did
not decide removal. Six similar pairs with distinct explicit constraints,
methods or requested actions are documented as retained in `Review decisions`.
Shared topics alone are not duplicate questions. This screening does not prove
exhaustive semantic independence.

The workbook includes `Training`, `Categories`, `Exclusions`, `Overlap matches`,
`Review decisions` and `Processing`. `Exclusions` records all 28 removals;
`Overlap matches` provides both compared questions and reference-row links.

## Scope and verification

The supplied CSVs and historical research artifacts are preserved. Only the
superseded deliverable folders were removed at the user's request.
Export checks cover CSV/XLSX agreement (allowing standard XML line endings),
unique IDs, complete 797 = 769 + 28 accounting, retained source content,
training-only splits, seven categories, formula totals and absence of remaining
normalized matches to this StrongREJECT file. Historical reservations remain
metadata and are not additional exclusions from this new pool.

No model responses, generation signals, activation labels or detector fitting
were produced. Source-topic embeddings were computed only for overlap screening.
Evidence and scripts are under `artifacts/sheet2-cleanup-20261031/` with
version-3 filenames. Old test questions admitted into the new training pool
cannot be used as fresh held-out tests of a newly fitted detector.
