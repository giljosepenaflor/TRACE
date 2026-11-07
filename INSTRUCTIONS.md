# TRACE training-data expansion: current instructions and blocker

**Latest continued improvement (2026-11-06):** Two additional bounded experiments
are complete. `paired-control-001` fits six equally trained arms (48 classifiers)
with within-source ranking/consistency constraints; Full validation F1 falls
0.529412 to 0.480000. `token-sequence-001` fits five matched arms (40 classifiers)
with an explicit 18-feature sequence extension to the three primary generation
statistics. Full's initial development worst-dataset AP improves 0.290889 to
0.335876 and validation AP 0.347680 to 0.437219, but validation F1 falls to
0.437500. The 405-input sequence Full is a ranking development candidate, not
an improved deployed detector. Both fixed continuation gates fail.

No new StrongREJECT scoring or resource benchmark is allowed from these failed
gates; no dashboard promotion. Keep original labels, all frozen plans/code/
artifacts and historical qualifications unchanged. All 95 targeted tests, both
workflow reconstructions and separate arithmetic checks pass. The prior
dataset-transfer audit is now completed. See
[the complete initial development report](docs/TRACE_TRANSFER_IMPROVEMENT_RESULTS.md).
Further experiments need a new hypothesis about missed activations; do not
repeat test-driven weight/cutoff selection or call AP improvement an F1 gain.
Older entries below retain their original historical scope.

**Latest completed work (2026-11-06):** The user-authorized end-to-end comparison
is implemented and evaluated. See [the complete initial results](docs/TRACE_END_TO_END_RESULTS.md).
`end-to-end-001` contains six matched arms, nested source cross-fitting, 123
classifier fits, all three validation-calibrated operating policies and 18
resource workers/3,600 timings. Full linear was selected by training-fold AP
before validation/test. Its initial retrospective seen/unseen F1 is
0.614173/0.276596, below Generation-only 0.683168/0.629108. Nonlinear Full is
0.669767/0.547368 and cannot replace the selected candidate after test scoring.
No fusion advantage is established; do not promote this candidate.

All 85 targeted tests and both workflow and independent arithmetic checks
passed. The declared two-head ablation leaves unsafe assistance excluded and
the old failed three-head gate unchanged. Original activation labels, accepted
AI-assisted auxiliary provenance, prior artifacts and dashboard are preserved.
No target calls or encoder fine-tuning occurred. The frozen pre-experiment plan
and implementation must remain unchanged after scoring.

The priority remains Full activation performance beyond both single views.
The next proposed development task is dataset/condition transfer and model
selection auditing using existing AdvBench/HarmBench training data under a new
frozen protocol. It has not run. Do not repeat weight/cutoff tuning on the
already inspected StrongREJECT outcomes or promise improved performance.
Fresh independent evidence and other target LLMs remain outstanding. Older
entries below describe their original stages, not the latest task state.

**Latest scorer result (2026-11-06):** The user authorized training/validation.
`auxiliary-scorer-001` now contains 18 C=1 unweighted head fits using fixed
0.5 cutoffs and source-isolated teacher-label validation. Refusal macro F1
0.888972 and fulfillment 0.798913 pass; unsafe assistance 0.468182 fails, with
positive recall 1/19. The all-head gate is false: do not augment TRACE, rescore
StrongREJECT or promote the dashboard from this failed candidate. Final models
are retained as development artifacts and for cost measurement. Teacher labels
are the already accepted AI-assisted set; do not request them again or claim
independent human semantic validation.

All five outer/25 nested fitting-support stages pass, but only outer/final
scorer models were fitted; no activation classifier or inner scorer model is
trained. Three fresh validation resource workers measured median scorer-only
12.746 ms and 454.473 MiB whole-worker RAM. Seventy-four tests and both workflow
and independent coefficient/metric/bootstrap/timing verification pass. No label
changes, encoder fine-tuning, threshold search, target calls or test parsing.
See [the complete scorer results](docs/TRACE_AUXILIARY_SCORER_RESULTS.md).

Unsafe ranking is stronger than its fixed decisions (AP 0.881375, AUROC
0.935927). The proposed next experiment is training-only per-head threshold
calibration with inner source-group predictions, preserving the current
0.5 candidate/gates. It has not run. Do not lower a cutoff directly against
the current outer OOF outcomes and report a new held-group pass. Older entries
that say the scorer is not trained are historical and superseded by this result.

**Latest user correction and completed intake (2026-11-06):** The user explicitly
designated `outputs/trace-aux-behavior-review-20261106-001/auxiliary-review-A.jsonl`
as final "for us both". Do not request separate A/B auxiliary submissions again
based solely on the old proposal. All 80 items from 40 training sources are now
sealed in `artifacts/trace-behavioral-v4/auxiliary-consensus-001`. Record one
user-approved AI-assisted label set: every supplied reviewer ID is ChatGPT.
No duplicate B, independent human agreement or separate adjudication history
may be invented. The original activation labels are a separate unchanged set.

Twelve lines received 16 quote-escaping-only repairs in the normalized copy.
For `AUX-A-b646131b25eb8f`, the user explicitly confirmed that the judgments apply
to the original response, restoring underscores in its code example. All
judgments/reasons stay unchanged; raw A and blank B stay unchanged. Eligible
refusal/fulfillment/unsafe-assistance labels are 74/74/65, with sufficient outer
class-group support. Nested support, scorer fitting and semantic quality are
not yet measured. Sixty-one tests and both intake checks passed; no test parsing,
target inference, fitting or dashboard change occurred. See
[the final intake](docs/TRACE_AUXILIARY_CONSENSUS_INTAKE.md).
Next: a frozen exploratory scorer experiment using these qualified teacher
labels. Agreement with AI labels is not human-validated semantic quality.

**Latest completed work (2026-11-06):** The user approved the development error
audit and feature specification. `development-audit-002` inspects 1,058 training
and 100 validation examples only. Removing the request cue with response fixed
changes unprojected Full decisions in 12/50 cued validation examples; Generation
is identical and unchanged controls have no flips. No validation positive is
encoder-truncated. Candidate Full already rejects all 43 refusal-marker examples;
its 14 false alarms are among marker-absent examples. These are diagnostics,
not new activation labels, causal backdoor proof or improved accuracy results.

The frozen `docs/TRACE_BEHAVIOR_FEATURE_SPEC.md` specifies a proposed auxiliary
refusal/fulfillment/unsafe-assistance scorer and strict development isolation.
Two independent blank packets of 80 training responses each are ready under
`outputs/trace-aux-behavior-review-20261106-001`. Keep administrator evidence
separate. Completed auxiliary reviews = 0; actual independent submissions,
recorded adjudication and coverage/class checks are needed before scorer fitting.
These auxiliary labels are different from the already completed activation
reviews. Never infer them from activation labels or simulate independent humans.

Fifty targeted tests and both verification paths passed. No test response or
prediction/label rows were parsed, no target calls or new fits/cutoffs occurred,
and original artifacts/dashboard remain unchanged. The first preflight failure
on an intentionally absent pointer is preserved at `development-audit-001`;
the second run fixes/checks the sentinel before any measurements. See
[the complete audit](docs/TRACE_DEVELOPMENT_BEHAVIOR_AUDIT.md) and
[the review instructions](outputs/trace-aux-behavior-review-20261106-001/README.md).
Do not rerun the completed weight search or claim that the semantic scorer has
already been trained or validated. Earlier proposed audit/fusion steps below
are historical planning records and do not replace this completed state.

**Latest completed experiment (2026-11-06):** The user approved separate-view
score fusion with "okay do our next". The isolated `score-fusion-001` run uses
eight predefined mixtures of frozen C=1 components and valid saved training
OOF scores. Training AP selection chooses 75% Behavioral / 25% Generation as
best interior mixture; Behavioral-only wins overall CV. The selected validation
threshold is 0.2171443000857341. Initial retrospective seen/unseen F1 is
0.666667/0.464000 versus Generation-only 0.683168/0.629108; this candidate is
not an improvement. It detects 29/74 unseen activations with 22 false alarms.
Do not retune weights or thresholds using these StrongREJECT outcomes.

Twelve matched validation-only workers measured fusion median 16.871 ms and
456.047 MiB median whole-worker RAM. Forty tests, workflow reconstruction and
independent numerical verification passed. Original artifacts and scores,
fresh-source plan gates and dashboard remain unchanged. No component fits,
target calls or new labels occurred; one new mixture threshold was calibrated
on validation. Use `scripts/trace_score_fusion.py verify` to reconstruct the
completed run without repeating fitting or benchmarking. See
[the detailed results](docs/TRACE_SCORE_FUSION_RESULTS.md). The earlier proposed
fusion step is now complete with a bounded negative initial result. Additional
feature/selection work needs a new development hypothesis and frozen protocol;
refusal/compliance features and fresh independent confirmation remain pending.

**Latest completed step (2026-11-06):** The user instructed "proceed to the next
step, just reuse the existing dataset". A separate frozen retrospective
comparison now uses the existing 476 admitted StrongREJECT examples and the
six pre-existing model/threshold definitions. New C=1 unprojected Full TRACE
seen/unseen F1 is 0.643836/0.551181 versus original Full 0.681564/0.476190;
Generation-only remains 0.683168/0.629108. Unseen false positives fall 38 to 18,
but detected positives stay 35/74 and seen recall declines. This is a mixed
initial result on data that already informed development, not a fresh test or
an overall improvement claim. All reported paired F1 intervals include zero.

Eighteen matched validation-only workers measured 600 timings per method:
original Full median 13.297 ms, new Full 13.607 ms, Generation 0.150 ms. Numerical
cutoff differences are disclosed without changing thresholds. Twenty-eight
targeted tests, workflow reconstruction and independent numerical checks passed.
Run: `artifacts/trace-behavioral-v4/reused-test-comparison-001`. Original raw
files, labels, models, test results and dashboard remain preserved. No fitting,
threshold selection or target inference occurred. The fresh-source audit still
finds zero eligible unused groups within the three original sources; its plan
and disabled gates are unchanged. See
[the completed comparison](docs/TRACE_REUSED_COMPARISON_RESULTS.md).
Separate-view score fusion on development data is a proposed follow-up, not
implemented or measured here. Earlier pending-review, missing-test and pause
entries below are historical snapshots and do not override this completed state
or the later explicit user resumption and dataset-reuse instructions.

**Current human-review step (2026-11-04):** Two independent Excel packets are
ready in `outputs/trace-strongreject-review-20261104-001`. Both contain all
1,252 selected test responses in different orders, with blank task-success,
reason and reviewer fields. Exported evidence and all raw/model hashes were
verified. Use the separate administrator mapping for the corrected mixed-source
selection. Do not run the old flat-directory seal unchanged on this packet;
completed-workbook integration and truthful provenance linkage remain pending.
No labels or test scores exist yet. See
[the packet instructions](outputs/trace-strongreject-review-20261104-001/README.md).

**Latest correction audit (2026-11-04):** The 314 records in
`scripts/artifacts/strongreject-openai-correction` cover exactly the required
unseen conditions 2/4 with OpenAI and pass the structural checks. The verified
primary selection is 624 original seen + 314 original unseen uncued + 314
corrected unseen cued responses, referenced without moving/editing files in
`outputs/trace-strongreject-correction-audit-20261104-001/primary-selection.jsonl`.
The 314 original unseen BadMagic-cued records remain excluded from primary
selection. No labels or scores exist yet. Resolve source-path/flat-directory
integration and provenance and then obtain independent human reviews. Preserve
raw records, source assignments, frozen models and thresholds. See
[the correction report](outputs/trace-strongreject-correction-audit-20261104-001/README.md).

**Current received-data issue (2026-11-04):** The 1,252 StrongREJECT responses
are present in `scripts/artifacts/strongreject`. Audit found 624 seen-partition
records matching the protocol and 314 uncued unseen records matching; the 314
unseen cued records use BadMagic rather than the frozen OpenAI cue. All technical
signal checks pass, but zero complete unseen groups meet that assignment.
The original frozen CSV path is missing; the ID copy still matches the frozen
questions. Provenance metadata is absent. See
[the full audit](outputs/trace-strongreject-received-20261104-001/README.md).
Do not rewrite actual prompts, relabel cues or silently reassign sources. No
review labels, final scores, inference or SSH calls were performed in the audit.

**Offline evaluation tooling (2026-11-03):**
`scripts/trace_strongreject_evaluate.py` now supports complete independent-review
sealing and one-time frozen three-view evaluation, requiring real response files,
all supplied judgments/reasons and collection-bound signal evidence. Nine
harmless tests passed. Actual StrongREJECT coverage remains zero; no real labels
or held-out metrics exist. Do not fabricate missing inputs or treat software
fixture scores as thesis findings. See
[the offline evaluation guide](docs/TRACE_STRONGREJECT_OFFLINE_EVALUATION.md).

**Current test handoff (2026-11-03):** The user confirms StrongREJECT responses
are not yet collected. The new offline handoff at
`artifacts/trace-training-v3/strongreject-test-001` preserves the 156/157 question
assignments and lists all 1,252 expected four-condition records, bound to the
already frozen three-view candidate. The initial audit reports zero valid
responses, with review, labels and evaluation pending. Eight offline tests
passed. Read [the handoff](docs/TRACE_STRONGREJECT_TEST_HANDOFF.md) and use
`scripts/trace_strongreject_intake.py audit` after saved responses are supplied.
No collector, model request, fit or threshold change was executed in this step.
Do not describe metadata preparation as completed collection or test results.

**Latest instruction and completed state (2026-11-03):** The user explicitly
resumed all three views, superseding the earlier pause. Full TRACE,
Behavioral-only and Generation-only were fitted and frozen in the separate
`artifacts/trace-training-v3/dual-view-candidate-001` run. The entry point is
`scripts/trace_consensus_train.py`; use its verify command, not old workflows,
to check this candidate. It uses 1,058 training / 100 validation examples and
new qualified signal admission linked to the deployment investigation. Existing
raw data, frozen labels, older audit states and original model are preserved.
Calibration F1 is 0.474 / 0.471 / 0.308; final testing and deployment have not
occurred. Eight offline tests and frozen verification passed. Read
[the current training report](docs/TRACE_THREE_VIEW_TRAINING_RESULTS.md).
Older pause and missing-data statements below describe earlier states and
must not override this explicit resumption or be treated as current results.

**Explicit user pause (2026-11-03):** All training is paused, including
Behavioral-only, until the user explicitly resumes it. No expanded fit has
started. The new Behavioral-only script was added but has not been executed.
The user requested provenance investigation first and confirms TRACE's remote
server family, with exact historical version uncertain. Read
[the completed investigation](outputs/trace-signal-provenance-investigation-20261103-001/README.md)
before proposing further experiments. Actual Vast source/manifest hashes and
process/log timing now corroborate full-vocabulary generation-score statistics;
per-response build/host binding remains absent. No frozen labels, admission
flags, dataset, services or original detector were changed. The pause takes
precedence over any older training commands below.

**Current state (2026-11-03):** The complete 3,076-response development
collection and final user-attested two-reviewer consensus workbook have been
received. The separately versioned `scripts/trace_consensus_intake.py` seals
1,058 adapted training labels and 100 validation labels in
`artifacts/trace-training-v3/consensus-001`, retaining the accepted 706/63
development and 156/157 StrongREJECT test source allocations, with no reserve.
Fitting has not started: historical full-vocabulary signal provenance remains
unverified. Read [the current intake report](docs/TRACE_CONSENSUS_INTAKE.md)
and [current next steps](docs/TRACE_TRAINING_V3_NEXT_STEPS.md). The older
allocation, missing-response statements and workflow commands below are
historical; do not apply them unchanged to this final workbook. Source data,
original detector and dashboard are preserved.

**Current run correction (2026-10-31):** The user selected independent prompts
first for the revised 769-source training pool (369 AdvBench, 400 HarmBench),
with separate StrongREJECT validation/test sources. Read
[current training next steps](docs/TRACE_TRAINING_V3_NEXT_STEPS.md) before using
the commands below. Earlier reconciliation counts and split-specific workflows
in this document describe the previous experiment; they must not be applied
unchanged to the current pool. New reviewed HarmBench interactions remain absent.

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
