# Reconciled three-source TRACE inventory

Recorded 2026-10-30. The user selected AdvBench, HarmBench and StrongREJECT as
source candidates for one shared dual-view TRACE detector across selected
target models. Source bookkeeping is implemented; shared-model training and
robustness evaluation have not occurred.

The delivered snapshot is
`artifacts/trace-3source/reconciliation-002/`. Its `README.md` explains the
rules; `manifest.json` references the original CSV rows without copying prompt
text; `audit.json` records provenance, counts and limits. The first local pass
in `reconciliation-001` is superseded by version 002, which also audits the
StrongREJECT exclusions and includes the readable report and file checksums.

| Source | Training sources | Validation sources | Seen-test sources | Detector-unseen-test sources |
| --- | ---: | ---: | ---: | ---: |
| AdvBench | 49 | 14 | 0 | 0 |
| HarmBench | 143 | 35 | 28 | 29 |
| StrongREJECT | 139 | 23 | 12 | 11 |
| Total | 331 | 72 | 40 | 40 |

These 483 partition-eligible source questions are not labeled interaction
counts, successful activations, or experimental results. Original AdvBench
training/validation roles are retained only for sources with existing eligible
reviews. Existing labels remain tied to their original saved interactions;
they do not automatically label new responses or outputs from other models.

All 1,071 supplied candidates remain accounted for. The other dispositions are
388 quarantined for reported Llama 2 MTBA adapter exposure or related-source
overlap, nine quarantined for unresolved prior reviews, one quarantined for
duplicate review, 39 historical-test exclusions, 18 excluded pilot sources,
and 133 unused reserve sources. Dispositions can cover related source groups,
so they are not simply counts of exact first-400 or original-test rows.

## What was reconciled

- Verified the AdvBench and HarmBench files against their pinned hashes and
  checked candidate questions, categories, target fields and prompt contexts.
- Retrieved the original StrongREJECT CSV through the connected GitHub source,
  pinned commit `3432b2d696b428f242bd507df96d80f686571d5e`, and verified SHA-256
  `4dd70357e4ff8b5d0ba5ebafecab5d6dd5633ce8046e3dd1c8bd93e64de44381`.
  All 313 upstream rows are accounted for by 287 candidates and 26 exclusions.
  The exclusions contain 25 verified exact duplicates and one unadjudicated
  possible near duplicate. The summary's download hash matches independently.
- Preserved original AdvBench development roles, excluded evaluated AdvBench
  tests, retained HarmBench's original partitions/pilot/reserve, and quarantined
  unresolved original reviews instead of silently admitting them.
- Preserved StrongREJECT's proposed development/reserve roles. Assigned its 23
  new test sources deterministically: SHA-256 order with seed 20261030,
  alternating seen/unseen, producing 12/11 before group exclusions.
- Screened all 520 AdvBench, 298 eligible HarmBench and 313 StrongREJECT source
  questions, including omitted/excluded questions, for exact normalized matches
  and character-ngram TF-IDF cosine similarity at 0.90. Conservatively propagated
  historical-test, exposure and pilot exclusions through related groups. Remaining
  lexical candidates or partition conflicts require review. This is not a
  complete semantic-overlap audit.
- Verified uploaded partition exports against the combined inventory. Preserved
  hashes for every uploaded file, the original manifests, label manifest,
  held-out protocol/report, and frozen detector. No old test record was admitted
  to an active partition; no active duplicate group spans multiple partitions.

The `prior_split` column in the uploaded CSV remains the previous provisional
assignment. The reconciled manifest's `original_split` is derived independently
from the preserved research manifests.

## Remaining work

The source snapshot is versioned, but the expanded experimental protocol is
not frozen. Collection and training readiness are both false. Confirm target
checkpoint identities and model-specific eligibility, cue assignments,
pilot feasibility, review rules, generation-score provenance and resource
budget before model calls. Source questions reused across models must retain
the same source-group partition. Future admission of quarantined sources
requires evidence and a new manifest version; never replace unfavorable
outcomes using reserve sources. Maintain the earlier Llama 2 results as the
single-model reference. No cross-model or robustness claim follows from this
inventory audit.

## Reproduction

Run from `D:\TRACE` with an unused output directory:

```powershell
.\.venv\Scripts\python.exe scripts\reconcile_trace_sources.py --output artifacts\trace-3source\reconciliation-003
```

The script performs metadata reconciliation only and refuses to overwrite an
existing output directory. It makes no network requests or model calls. The
pinned StrongREJECT snapshot/provenance and the original local research files
must already be present. JSON manifests are software artifacts; the original
CSV spreadsheets are unchanged.

## Verification

The full unittest suite ran 158 tests with three expected skips and no failures,
including six harmless-fixture reconciliation tests. The sandboxed run initially
encountered Windows temporary-directory permission errors; the same suite passed
when rerun outside that restriction. Independent post-write checks verified all
1,071 dispositions, delivered file checksums, unchanged input/frozen-artifact
hashes, historical-test isolation and absence of new activation labels.
