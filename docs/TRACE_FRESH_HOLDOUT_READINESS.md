# Fresh held-out comparison readiness and source shortage

Recorded **6 November 2026**. The source audit is complete, but a fresh held-out
accuracy comparison cannot run under the current source-independence rules:
**zero eligible fresh source groups were found** within the examined AdvBench,
HarmBench and StrongREJECT source pools. The user explicitly selected
"Keep the original three datasets; report any source shortage."

The local pinned snapshots and the official repositories' default-branch
dataset-file histories were checked. The latest retrieved official CSV files
are byte-for-byte identical to the corresponding local snapshots. No fourth
dataset was added, no new responses or labels were created, and no test scores
were generated. This is a data-readiness result, not a new performance result.

## Source inventory checked

| Dataset | Source rows in pinned local file | Source rows in retrieved official file | Eligible fresh groups |
| --- | ---: | ---: | ---: |
| AdvBench | 520 | 520 | 0 |
| HarmBench text behaviors | 400 | 400 | 0 |
| StrongREJECT | 313 | 313 | 0 |

These are source-file row counts, not distinct questions across all three
datasets or labeled detector-example counts. In particular, some StrongREJECT
questions duplicate AdvBench questions.

Official file snapshots were pinned to the latest default-branch commit
returned for each dataset-file path at retrieval:

- AdvBench: `a62d1307e38b3a076e614b20781c785fd860d813`,
  SHA-256 `6cd1a5c63c07610d7eb67307772ee5606017ee950b5770ab288a2c487489d3e1`.
  [Official source file](https://github.com/llm-attacks/llm-attacks/blob/a62d1307e38b3a076e614b20781c785fd860d813/data/advbench/harmful_behaviors.csv)
- HarmBench: `c0423b952435fcc8467108d8f25962dbae5b7de2`,
  SHA-256 `8d81accedd38eaaf8b760618622bb888417d1fd0c86eba65c427a16f1cbb4afc`.
  [Official source file](https://github.com/centerforaisafety/HarmBench/blob/c0423b952435fcc8467108d8f25962dbae5b7de2/data/behavior_datasets/harmbench_behaviors_text_all.csv)
- StrongREJECT: `3432b2d696b428f242bd507df96d80f686571d5e`,
  SHA-256 `4dd70357e4ff8b5d0ba5ebafecab5d6dd5633ce8046e3dd1c8bd93e64de44381`.
  [Official source file](https://github.com/alexandrasouly/strongreject/blob/3432b2d696b428f242bd507df96d80f686571d5e/strongreject_dataset/strongreject_dataset.csv)

The audit stores retrieval timestamps, GitHub metadata URLs, immutable raw-file
URLs, downloaded-file hashes, and the downloaded CSV data. Only dataset data and
metadata were downloaded; repository code was not executed. The finding
concerns these examined source files at this date, not every possible future
benchmark release or a claim that no additional questions could ever exist.

## Why the available rows are not admitted

The audit protects all 769 sources from the expanded development collection,
all 313 previous StrongREJECT test sources, the earlier 120 AdvBench source
assignments, and the first 400 AdvBench rows recorded as released-adapter
attack-training exposure. Sources with input defects, excluded review
patterns or uncertain judgments remain part of the protected history. They
are not converted into fresh test questions because their old labels were
excluded from fitting or evaluation.

Matching normalizes Unicode with NFKC, casefolds text, and collapses whitespace.
It conservatively compares the base question independently of its context.
A changed context or a repeated generation therefore does not make an old
question source fresh. Exact historical matches alone account for every
examined row, so there are no unmatched candidates awaiting a new semantic
duplicate review in these snapshots.

For AdvBench, the mutually exclusive first matching dispositions are:

| Disposition | Rows |
| --- | ---: |
| Matches an expanded development source | 369 |
| Matches a previous StrongREJECT test source | 25 |
| Matches an earlier AdvBench experiment assignment | 36 |
| Remaining rows with recorded adapter attack-training exposure | 90 |
| Total | 520 |

Some rows have more than one historical overlap reason; the detailed audit
retains every reason and counts each row once in the table. The final 90 rows
are excluded for reported adapter-training exposure, rather than being claimed
to have all received prior TRACE review. Thus zero means **zero eligible under
the stated independence criteria**, not that every AdvBench row has already
been scored by TRACE.

All 400 HarmBench rows match the expanded development source inventory,
including the contextual and copyright rows. All 313 StrongREJECT rows match
the previous test-source catalogue. Earlier HarmBench "reserve_unused" labels
belong to an older allocation; those questions are covered by the later
400-source development collection and cannot be treated as unused now.

The missing historical `strongreject-dataset-for-training.csv` is not
recreated. The independently pinned official StrongREJECT snapshot is checked
against the question hashes in the already frozen catalogue. The older CSV
intake gate remains unchanged.

## Comparison definitions prepared

The separate
`outputs/trace-fresh-holdout-audit-20261106-001/comparison-plan.json` pins six
existing comparator definitions and their unchanged development thresholds:

| Comparator | Role | Frozen threshold |
| --- | --- | ---: |
| Original projected interaction Full, C=1, unweighted | Primary reference | 0.18404884955022044 |
| Unprojected interaction Full, C=1, unweighted | Primary representation candidate | 0.234150946937906 |
| Original Generation-only, C=1, unweighted | Generation reference | 0.15292532911254467 |
| Original projected interaction Behavioral-only, C=1, unweighted | Secondary component reference | 0.23824214492694545 |
| Unprojected interaction Behavioral-only, C=1, unweighted | Secondary component comparison | 0.24879694155898716 |
| Unprojected interaction Full, C=10, unweighted | Secondary training-CV choice | 0.2464645963030869 |

The plan binds the frozen original and new candidate artifacts by checksum.
The representation experiment already verified that its fixed original
reference models reproduce the original validation scores exactly. No new
fitting or threshold selection occurred for this source audit.

The intended comparison preserves the four-condition design, BadMagic for
seen-cue testing and OpenAI for unseen-to-detector testing, independent prompts,
greedy generation capped at 256 new tokens, and two independent human reviewers.
The existing 0000/0100 activation-label rule is retained. The primary proposed
comparison is the difference in Full F1 on fresh unseen-cue sources, accompanied
by false-positive rates, precision, recall, average precision, AUROC, confusion
matrices and paired whole-source uncertainty intervals.

The file is explicitly **an incomplete comparison plan, not a frozen test
protocol**: source assignments are empty and collection, evaluation and
dashboard-promotion gates are false. It does not authorize collection from
historically excluded rows or automatic labels. The 200-source, 100-seen /
100-unseen planning target is a proposed extension, not a revision of the
original 1,082-question and 4,328-response allocation. Against that planning
target, all 200 eligible source groups remain missing.

## Completed checks and preserved state

Eight targeted harmless tests passed. They check Unicode/whitespace overlap,
context changes, multiple exclusion reasons, pending status for unmatched
questions, malformed inputs, unchanged comparator thresholds, incomplete-plan
gates, missing-comparator rejection and refusal to overwrite a completed audit.

The sealed audit includes all source snapshots and metadata, per-row question
and context hashes, dispositions and the comparison plan. Verification checks
the original model and source hashes, independently counts unmatched rows
against the union of protected source questions, reconstructs all dispositions,
and confirms the planned model thresholds and false collection/scoring gates.
That verification passed.

No target inference or fresh response collection was attempted because the
source-admission gate failed. Inference-service readiness and collection-bound
provenance still need checking after an eligible source inventory exists.
Original datasets, workbooks, model artifacts, thresholds, test predictions,
dashboard pointers, and the initial thesis results were not modified.

## Consequence for the next experimental step

The source audit and comparison definitions are complete. **The fresh held-out
accuracy experiment remains pending a valid source inventory.** Re-running
old questions, changing their split, or paraphrasing them does not satisfy
this protocol's source-independence requirement.

Under the user's current three-dataset restriction, the fresh comparison must
wait for additional independently eligible source material from those datasets
or an explicitly revised experimental constraint. The audit does not silently
relax the restriction, admit exposed sources, or substitute a repeated-source
test for a fresh one. Validation-only efficiency measurements could proceed
as a separate next task, but would not resolve the missing accuracy evidence.

Artifacts:

- [Source audit](../outputs/trace-fresh-holdout-audit-20261106-001/audit.json)
- [Incomplete comparison plan](../outputs/trace-fresh-holdout-audit-20261106-001/comparison-plan.json)
- [Verification](../outputs/trace-fresh-holdout-audit-20261106-001/verification.json)

Verify without network requests, generation or fitting:

```powershell
.\.venv\Scripts\python.exe -B scripts\trace_fresh_holdout_audit.py verify
```
