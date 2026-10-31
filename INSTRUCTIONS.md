# TRACE training-data expansion: current instructions and blocker

Updated: **October 30, 2026 — Asia/Manila**. Workspace: **`D:\TRACE`**.

This is a handoff for the current expansion task. It distinguishes the user's
requested outcome, completed work, prepared software, and work that has not run.
Read [the thesis context](docs/thesis/PROJECT_CONTEXT.md),
[development status](docs/DEVELOPMENT_STATUS.md), and [README](README.md) before
changing the project. Preserve the supplied thesis in
`docs/thesis/source-draft.txt`.

**Later interface update:** At the user's request, experimental conversation
memory was implemented and deployed. Read [TRACE_CHAT_MODE.md](docs/TRACE_CHAT_MODE.md)
for its operation and limits. The feature does not supply new research labels,
validate multi-turn detection, or resolve the expansion blocker described here.

## 1. What the user wants now

Increase the training data of the **existing TRACE detector for Llama 2 7B
with the BackdoorLLM MTBA adapter**, using eligible new source questions and
their actual model interactions. The aim is to investigate whether additional
verified activation examples improve detection. Improvement is not guaranteed.

The intended research sequence is:

1. Use the frozen, eligible source assignments for the expansion.
2. Obtain genuine response records with the generation information TRACE needs.
3. Independently review the new responses and verify activation attribution.
4. Fit one expanded TRACE detector on eligible original plus new training data.
5. Select thresholds using validation only and freeze the detector.
6. Evaluate on fresh, isolated held-out data.

The immediate target is the **existing Llama 2 model**. Additional model setup
is not a prerequisite for this interim stage. The longer-term user correction
remains **one shared TRACE detector**, rather than a separate detector per
target. This interim single-model stage does not demonstrate cross-model
generalization.

The user offered to label **new answers**. The original AdvBench response
review, detector training, and held-out evaluation are already complete.
Repeating those tasks does not add training data.

## 2. Existing target and reference experiment

| Item | Existing identity/status |
| --- | --- |
| Base model | `meta-llama/Llama-2-7b-chat-hf` |
| Base revision | `f5db02db724555f92da89c216ac04704f23d4590` |
| Adapter | `BackdoorLLM/Jailbreak_Llama2-7B_MTBA` |
| Adapter revision | `8652ad60f1605605642c99fd680a15ecccf1b522` |
| Target-model training | Released pretrained adapter; the expansion does not retrain this LLM |
| Original TRACE fitting data | 108 reviewed training interactions, 32 validation interactions |
| Original held-out collection | 160 real interactions; 120 eligible interactions after unresolved-review exclusions |
| Original Full TRACE held-out F1 | 0.667 seen; 0.750 detector-unseen |
| Original detector | Frozen and preserved; remains the dashboard reference |

The original study uses a seen cue and a cue absent from detector fitting and
validation. The latter was examined by the researchers in a separate pilot;
"unseen" means **unseen to the detector**, not to the research team. Refer to
the preserved protocol for assignments; do not change them based on scores.

Authoritative reference files:

- [Original held-out results](docs/MTBA_HELDOUT_RESULTS.md).
- `artifacts/mtba-development/run-001/activation-labels-001/manifest.json`.
- `artifacts/mtba-development/run-001/trace-activation-detector-001/detector.joblib`.
- `artifacts/mtba-test/run-001/protocol.json`.
- `artifacts/mtba-test/run-001/evaluation/report.json`.

These findings do not establish a reliable fusion advantage, universal detection,
or cross-model robustness. Validation values must not replace held-out results.

## 3. What happened to the proposed 500 questions

The uploaded three-source inventory contains **1,071 candidate questions**:
499 AdvBench, 285 HarmBench, and 287 retained StrongREJECT questions.
Its proposed training CSV contains **500 provisional assignments**.

Those 500 are not an approved set of 500 new training questions. The uploaded
split mixed previously evaluated sources, reported adapter-training exposure,
and existing reserved assignments. Reconciliation preserved the original
research boundaries and quarantined uncertain sources.

Use this snapshot for source eligibility:

- [Reconciled manifest](artifacts/trace-3source/reconciliation-002/manifest.json).
- [Reconciliation audit](artifacts/trace-3source/reconciliation-002/audit.json).
- [Readable reconciliation report](docs/TRACE_3SOURCE_RECONCILIATION.md).

| Source | Training questions | Validation questions | Seen-test questions | Detector-unseen-test questions |
| --- | ---: | ---: | ---: | ---: |
| AdvBench | 49 | 14 | 0 | 0 |
| HarmBench | 143 | 35 | 28 | 29 |
| StrongREJECT | 139 | 23 | 12 | 11 |
| **Total** | **331** | **72** | **40** | **40** |

These are **source-question counts**, not response counts or positive labels.

- The 49 eligible AdvBench training questions already have 98 reviewed adapted
  interactions. The 14 validation questions already have 28.
- **282 training questions** need new response records: 143 HarmBench and 139
  StrongREJECT.
- **58 validation questions** need new records: 35 HarmBench and 23 StrongREJECT.
- Both new 40-question test partitions remain isolated from fitting and
  threshold selection.
- An additional **18 excluded HarmBench pilot questions** remain outside these
  training/validation/test counts.

All candidates remain accounted for. Other dispositions are: 388 exposure-related
quarantines, nine unresolved-prior-review quarantines, one duplicate-review
quarantine, 39 historical-test exclusions, 18 excluded pilots, and 133 unused
reserve sources. Related-group protections can affect additional rows; these
counts are not simply exact matches to original row ranges.

**Do not use the uploaded 500-row training CSV directly for fitting.** Select
through the reconciled manifest. Its `original_split` comes from the preserved
research manifests; the uploaded `prior_split` is only an earlier provisional
assignment. Source IDs also use different indexing conventions; use the saved
source references instead of guessing row conversions.

StrongREJECT's original 313-row source was pinned and independently hash-checked.
The retained 287 plus 26 exclusions account for all rows. Exact and lexical
duplicate checks were performed; complete semantic independence is not proven.

## 4. What has actually been implemented

The isolated expansion workflow is:

```text
scripts/trace_llama2_three_source.py
artifacts/trace-3source/llama2-stage-001/
```

It supports offline import validation, blinded review packets, independent-review
auditing, activation-label derivation, detector fitting, validation calibration,
freeze checks, and held-out evaluation. It does **not** make model calls or
generate new benchmark responses.

Prepared artifacts include:

| File under `artifacts/trace-3source/llama2-stage-001/` | Purpose |
| --- | --- |
| `protocol.json` | Pinned interim target, review rules and original artifact hashes |
| `combined-protocol-v2.json` | Three-source fitting/validation plan and original-label reuse |
| `source-assignments.json` | Opaque source IDs, source hashes, partitions and source families |
| `reused-development-index.json` | References to the existing 98 training/28 validation interactions |
| `preparation-status.json` | Preparation-time status snapshot |
| `pilot-readiness-001.json` | Missing pilot-record checklist |
| `harmless-live-check.json` | Operational smoke evidence only |

The latest full unittest run completed **167 tests, with three expected skips
and no failures**. This verifies software behavior, not research effectiveness.

## 5. Current blocker: new response records do not exist here

The checked local state is:

| Required evidence | Available |
| --- | ---: |
| New independently imported expansion response records | **0** |
| Excluded pilot source groups with complete records | **0 of 18** |
| Required pilot record slots filled | **0 of 72** |
| Completed new independent review sets | **0** |
| Sealed pilot/training/validation labels for this expansion | **0** |
| Newly trained expanded three-source detector | **None** |
| New three-source held-out evaluation | **Not run** |

The following checked locations contain no new research records or reviews:

```text
D:\TRACE\pilot-records
D:\TRACE\artifacts\trace-3source\llama2-stage-001\records
D:\TRACE\artifacts\trace-3source\llama2-stage-001\labels
```

The question CSV contains requests, not answers, generation features, or verified
activation labels. It cannot itself train the intended dual-view detector.
An AdvBench `target` field is not an observed response or an activation label.

### Assistant execution boundary

The user requested running the larger benchmark batch against the backdoored
model and stated that the research is approved by the thesis panel. The assistant
did not run that batch because some requests would produce actionable violent
or weapon instructions. The assistant cannot generate those outputs through a
remote model or supply automation to perform the same generation.

This boundary remains even with the stated research approval. No additional
user confirmation clears it. It is separate from an API failure, GPU failure,
dataset download failure, or lack of permission to train a defensive detector.
The assistant is not requiring the user to personally execute the blocked batch
and is not providing a workaround for doing so.

The supported paths are offline analysis of **existing independently collected
research records**, or an explicitly revised experiment using benign controlled
activation behavior. A benign experiment needs its own protocol and must not
be represented as completion of this jailbreak-source expansion.

### Label and evidence boundary

Even with existing responses, the current protocol requires independent response
review before new activation labels can be sealed. The user can complete one
review; a second reviewer remains required by the preserved method. Any change
to that requirement must be explicit and versioned, not silently inferred from
"I will label it."

Response text alone is insufficient for TRACE's generation view. Records need
aligned token IDs, confidences, entropies, recomputable statistics, checkpoint
identity, generation settings, and source/partition provenance. New records must
declare full-vocabulary generation-score provenance supported by collection
evidence. Metadata cannot independently prove executed weights or raw-logit
equivalence. Historical score-provenance limitations remain documented.

## 6. What the harmless check established

A fixed arithmetic request to the existing Vast target returned
"Two plus two equals four." Its seven token IDs, confidence values and entropy
values aligned. The original frozen TRACE detector scored it **0.042136** against
threshold **0.287742**, producing **Normal** with unknown activation ground truth.

Evidence is saved in:

```text
artifacts/mtba-harmbench/harmless-export-smoke-20261030T031032Z-d5e10b13/
artifacts/trace-3source/llama2-stage-001/harmless-live-check.json
```

This establishes successful operation at the recorded check. It does not imply
that the remote service is online at every later moment. It supplies no new
activation training examples, does not pass the pilot gate, and is excluded from
all research partitions.

## 7. Redundant artifacts: preserve them, do not use them as new progress

During this session the assistant unnecessarily prepared a new review packet
for already completed AdvBench work and trained a smaller subset using the
original verified labels. The user correctly identified this repetition.

| Preserved artifact | What it actually contains |
| --- | --- |
| `artifacts/trace-3source/existing-answer-labels-001/` | 210 existing answers: 126 adapted responses and 84 controls; blank repeat-review files |
| `artifacts/trace-3source/reviewed-subset-detector-001/` | An actual fit using 98 original training and 28 validation interactions; AdvBench only |

These are **redundant diagnostic artifacts**, not the expanded-data milestone.
The user does not need to label those old answers again. The subset fit's high
validation scores are threshold-selection measurements; they do not replace
the original held-out results or demonstrate robustness. The subset detector
was not installed in the dashboard and used no new HarmBench/StrongREJECT data.

Do not run `prepare_existing_trace_labels.py prepare` or
`train_existing_trace_subset.py` as the next expansion step. Do not delete the
artifacts merely to hide the repeated work; preserve their honest provenance.

## 8. Instructions for continuing safely and accurately

1. **Verify the current stage**, without rerunning the original experiment:

   ```powershell
   cd D:\TRACE
   .\.venv\Scripts\python.exe scripts\trace_llama2_three_source.py status
   ```

2. **Check whether existing independently collected new-source records have
   become available.** If none exist, report the data blocker plainly. Do not
   create empty review packets, invent answers/labels, replace missing records
   with smoke outputs, or claim that training has started.

3. **When such records already exist, validate them through the offline import
   path.** The accepted schema is `trace.mtba.three_source.import.v1`. Use the
   source IDs, hashes and partitions in `source-assignments.json`; see the
   [stage record contract](docs/TRACE_LLAMA2_THREE_SOURCE_STAGE.md). Imported
   labels must remain null. Preserve original files and collection evidence.

4. **Prepare independent review packets only for complete, valid groups.**
   Reviewers assess substantive response success using
   [the existing rubric](docs/MTBA_TASK_SUCCESS_RUBRIC.md), without seeing
   TRACE predictions or condition/role mappings. The software derives activation
   attribution from the matched comparisons. Trigger presence or lack of refusal
   language cannot establish activation. Unknown/disputed groups remain excluded.

5. **Respect the excluded pilot gate.** The prepared protocol requires at least
   four independently verified activation groups among the 18 excluded pilot
   sources before expansion fitting. Pilot examples never enter fitting or
   threshold selection. Report a failed feasibility gate honestly.

6. **Use only verified development records for the expanded fit.** Reuse the
   eligible original labels without re-reviewing completed work. Add eligible
   new training records, fit the behavioral projection/generation scaler and
   classifier using training only, and choose thresholds using validation only.
   Full TRACE is the intended main detector; Behavioral-only and Generation-only
   remain ablations. Report actual dataset composition and review exclusions.

7. **Freeze before opening fresh held-out records.** The prepared importer and
   reviewer packet functions enforce a new-detector freeze before held-out
   import/review. Keep the original evaluated AdvBench test set out of fitting
   and tuning. Do not substitute reserve examples because outcomes are poor.

8. **Evaluate and report the new experiment once the evidence exists.** Report
   actual held-out metrics, false alarms, class/source counts, uncertainty and
   detector efficiency. Distinguish calibration scores from final testing. Do
   not claim that more data necessarily improved detection. Dashboard replacement
   is a separate action; this workflow does not automatically change it.

The offline CLI provides `ingest`, `packet`, `seal`, `train`, and `evaluate`.
Their existence does not mean their data prerequisites are satisfied. The
prepared run already exists; `prepare` refuses to overwrite its index files.
Use a versioned protocol/run for any methodological change.

## 9. What counts as completion of the expansion

The expansion is complete only when all of the following are evidenced:

- Eligible new responses and generation signals are saved and integrity-checked.
- Independent new reviews are complete, exclusions recorded, and labels sealed.
- The fitting report identifies the actual old/new source composition.
- A new detector and its validation-selected thresholds are frozen.
- Fresh held-out evaluation is completed without test-driven model selection.
- Findings, uncertainty, efficiency measurements and limitations are documented.

**Current conclusion: the original TRACE study works as a completed reference;
the expanded source inventory and offline workflow are prepared; the expansion
is blocked before new-response review and fitting because new research records
are absent and the requested harmful-output batch cannot be executed by this
assistant.**
