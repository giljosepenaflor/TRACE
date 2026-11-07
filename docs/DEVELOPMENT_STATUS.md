## Continued improvement completed: partial ranking gain (2026-11-06)

Two new fixed development experiments trained **88 classifiers** on the
original five source folds plus two training-dataset transfer folds.
Paired-control training does not improve Full: validation F1 **0.480** versus
raw Full **0.529**. The new sequence model explicitly extends three generation
features to 21, shared by Full and matched Generation-only. Full's worst
held-dataset AP improves **0.291 to 0.336**, and validation AP **0.348 to 0.437**.
However validation F1 declines **0.529 to 0.438** (TP 9 to 7, FP unchanged 14).

These are **initial development results**, not an overall accuracy improvement.
Both continuation gates fail, so there is no new StrongREJECT scoring, complete
resource benchmark or dashboard change. All **95 targeted tests**, both workflow
checks and separate arithmetic reconstruction of all classifiers/features/
metrics pass. No target calls, new labels, encoder updates or new encoding.
See [complete results and qualifications](TRACE_TRANSFER_IMPROVEMENT_RESULTS.md).
The dataset-transfer audit is now completed. Sequence Full remains a development
ranking candidate; understanding its missed activations precedes further
experiments. Older entries below are historical.

## End-to-end comparison complete; no fusion advantage (2026-11-06)

The [matched end-to-end experiment](TRACE_END_TO_END_RESULTS.md) is implemented
and measured in `end-to-end-001`: six arms, five outer/25 inner source folds,
123 frozen classifier fits, unchanged activation labels and the accepted
AI-assisted refusal/fulfillment supervision. Unsafe assistance and the old
failed three-head gate remain excluded/unchanged. Training AP selected Full
linear before validation/test; raw Behavioral was best overall in CV.

**Initial retrospective** seen/unseen F1: Generation **0.683/0.629**, raw
Behavioral **0.619/0.296**, raw Full **0.644/0.551**, enhanced Behavioral
**0.603/0.200**, selected Full linear **0.614/0.277**, nonlinear Full
**0.670/0.547**. The selected Full fails the win rule on both splits. The
nonlinear test score does not change primary selection. Three validation-only
operating policies, source slices and paired intervals are reported.

Eighteen fresh validation workers measured 3,600 timings. Median latency is
28.547 ms selected Full / 14.177 ms raw Full / 0.152 ms Generation; the new
Full counts both encoder passes and has median whole-worker RAM 465.309 MiB.
All 85 targeted tests and both verification paths pass. No target calls,
encoder fine-tuning, new labels or dashboard promotion occurred.

Next proposed work: development-only dataset/condition transfer and selection
audit, using existing training data under a new frozen protocol. It has not
run; no further StrongREJECT weight/cutoff tuning is implied. Fresh evidence
shortage, AI-assisted semantic provenance and additional LLM evaluation remain
outstanding. Older progress entries below are historical.

## Auxiliary scorer training and held-source validation completed (2026-11-06)

The frozen MiniLM pair encoder and three C=1 unweighted heads were trained
(15 outer-fold fits + 3 final fits). At fixed 0.5, teacher-agreement macro F1
is **0.888972 refusal / 0.798913 fulfillment / 0.468182 unsafe assistance**.
The first two pass; unsafe assistance detects only 1/19 positives and fails.
Its AUROC is 0.935927, but the cutoff is not changed post hoc. These are
AI-assisted teacher-label development results, not validated human semantic
accuracy or new backdoor-detection results. All nested support checks pass.

The joint gate stops activation augmentation and StrongREJECT scoring. Three
validation-only resource workers measured 600 timings: median scorer overhead
12.746 ms, p95 23.757 ms, median whole-worker RAM 454.473 MiB. Seventy-four tests
and both verification paths passed. No target calls, threshold search, label
changes or dashboard promotion occurred. Next proposed work is inner-group
threshold calibration under a new protocol, retaining these fixed-cutoff
results. See [the complete initial scorer report](TRACE_AUXILIARY_SCORER_RESULTS.md).

## Final auxiliary labels accepted with explicit provenance (2026-11-06)

The user designated A as final for both roles. All **80 items / 40 training
sources** are sealed as one **user-approved AI-assisted label set**, retaining
the supplied ChatGPT reviewer ID; independent two-human agreement is unavailable.
Twelve JSONL lines receive syntax-only quote repairs, and one original response
is restored after explicit user confirmation. Judgments/reasons and raw source
bytes stay intact; B and all activation labels remain unchanged.

Usable examples for refusal/fulfillment/unsafe-assistance are **74/74/65**.
All outer training-fit source-class minima pass; nested support and semantic
quality remain pending. Sixty-one tests plus both intake verification paths
passed. No scorer training, target calls, test parsing or dashboard changes.
Next: freeze and evaluate an auxiliary scorer with this qualified teacher-label
provenance. See [the complete intake](TRACE_AUXILIARY_CONSENSUS_INTAKE.md).

## Development behavior audit and feature specification completed (2026-11-06)

Only 1,058 training/100 validation examples were inspected. Fixed-response cue
removal changes unprojected Full decisions in **12/50 cued validation examples**;
Generation remains identical and unchanged controls have no decision flips.
None of the validation positives are encoder-truncated. The candidate already
correctly rejects all 43 validation refusal-marker examples; its 14 false alarms
occur outside that marker group. These are descriptive development diagnostics,
not new activation labels or a measured improvement.

A frozen refusal/fulfillment/unsafe-assistance feature specification and two
blank 80-item training-only review packets are complete. The next dependency
is genuine auxiliary review/adjudication and coverage checks before a scorer
fit; existing activation reviews are not being reopened. Completed auxiliary
reviews = 0. Fifty targeted tests and both reconstruction/arithmetic verification
paths passed. The completed run is `development-audit-002`; first-preflight
failure evidence is preserved. No test outcome parsing, target inference, fits,
cutoff changes or dashboard promotion occurred. See
[the audit](TRACE_DEVELOPMENT_BEHAVIOR_AUDIT.md),
[feature specification](TRACE_BEHAVIOR_FEATURE_SPEC.md) and
[review handoff](../outputs/trace-aux-behavior-review-20261106-001/README.md).

## Separate-view score fusion completed; negative improvement result (2026-11-06)

Eight predefined mixtures reuse valid training OOF component predictions;
training-only AP selection chooses 75% Behavioral / 25% Generation as the best
interior mixture. Behavioral-only is the overall CV winner. One selected-fusion
threshold is calibrated on validation, then frozen. Initial retrospective
seen/unseen F1 is **0.666667/0.464000**: below Generation **0.683168/0.629108**
and no overall gain over the existing Full comparators. Fusion detects 29/74
unseen positives with 22 false alarms; the missed-activation problem persists.

Twelve fresh validation-only workers supply 2,400 timings: median fusion
16.871 ms, original Full 14.540 ms, unprojected Full 16.041 ms, Generation
0.145 ms. Fusion median whole-worker RAM is 456.047 MiB; it has no observed
validation decision-boundary differences. Forty targeted tests and two
verification paths passed. No component refits, target calls, new labels,
test-based tuning or dashboard promotion occurred. Original evidence and
fresh-source plan gates remain intact. This closes the authorized score-fusion
experiment with a bounded negative result; later feature/selection changes
need a new development protocol. See
[the full initial results and limitations](TRACE_SCORE_FUSION_RESULTS.md).

## Authorized existing-data comparison and efficiency completed (2026-11-06)

Following "just reuse the existing dataset", a separate frozen retrospective
comparison reused all 476 original admitted StrongREJECT examples. New C=1 Full
seen/unseen F1 is **0.643836/0.551181**, versus original **0.681564/0.476190**.
Unseen FP declines 38 to 18 but TP stays 35/74; seen recall declines. Generation
retains higher F1 on both splits. Behavioral and C=10 Full results also remain
mixed; all reported paired F1 intervals include zero. This does not establish
overall improvement, and these adapted/reused test data are not a fresh holdout.

Eighteen validation-only workers produced 3,600 timings: median original Full
13.297 ms, candidate Full 13.607 ms, Generation 0.150 ms; whole-worker median
RAM 457.742/457.023/121.922 MiB. Threshold-boundary numerical differences are
disclosed without recalibration. Twenty-eight targeted tests and both workflow
and independent numerical verification passed. Original results, models, labels,
fresh-source plan gates and dashboard are preserved; zero target calls or fits.
Next proposed development work: separate-view score fusion, selected using
training CV and calibrated on validation. Independent confirmation remains
unavailable under the current fresh-source shortage. See
[the complete comparison, resources and limitations](TRACE_REUSED_COMPARISON_RESULTS.md).

## Fresh held-out source readiness audited; source shortage (2026-11-06)

The user kept AdvBench/HarmBench/StrongREJECT as the only sources. Current
official CSV snapshots match the pinned local files (520/400/313 rows).
**Zero eligible fresh sources** remain after protecting prior development,
test and earlier assignments plus reported adapter-training exposure. A
sealed source audit and six-comparator plan are prepared, but the test
protocol is incomplete and collection/scoring/promotion gates remain false.
Eight tests and source/hash/threshold/gate verification passed. No target
generation, labels, fresh test scores, fitting or dashboard changes occurred.
The held-out accuracy experiment remains pending eligible sources under the
chosen scope. See [the shortage report and artifacts](TRACE_FRESH_HOLDOUT_READINESS.md).

## Behavioral representation development comparison complete (2026-11-06)

Four text representations, both Behavioral-only/Full, and a Generation-only
reference were compared using five training-source folds and 54 configurations.
The fixed-settings unprojected prompt-response candidate improves Full
validation F1 **0.474 to 0.529**, with FP **18 to 14** and the same 9/11
positives detected. Validation AP declines, and the separately CV-tuned Full
choice has F1 0.457; the evidence is mixed and remains development-only.
Eighteen targeted tests and two verification paths passed. Original reference
validation scores are reproduced exactly. The new workflow qualifies the
missing historical CSV through the existing sealed catalogue and raw hashes;
the old gate remains unchanged. No target calls, held-out scoring, labels or
dashboard changes occurred. Next: fresh prospective held-out comparison before
claiming generalization or promoting a detector. See
[the results and limitations](TRACE_BEHAVIORAL_REPRESENTATION_RESULTS.md).

## External rule/component comparison complete with explicit adaptations (2026-11-04)

Author paper/code audit and a separate frozen comparison are complete. Seen/
unseen F1: ConfGuard prefix50 replay 0.434783/0.337209; full256 horizon adaptation
0.452632/0.375000; semantic-drift component 0.605578/0.546154. Full TRACE remains
0.681564/0.476190 and has much lower false-positive rates than the semantic row.
Generation-only still has the highest F1/AUROC. The full drift-plus-canary method
is not reproducible from the saved collection; no missing outputs were invented.
Semantic reference/scale use training only and threshold uses validation only.
The comparison is retrospective, not a new prospectively blind test. Eighteen
matched validation benchmark processes and 3,600 timings compare all six methods.
Eleven tests plus seal/membership/independent metric/timing checks passed. Original
TRACE scores were reused; no target generation, TRACE retraining, threshold change
or dashboard promotion occurred. Next: thesis reporting and scope treatment of
adapted components. See [the audit and comparison](TRACE_EXTERNAL_BASELINE_COMPARISON.md).

## Frozen error analysis and detector efficiency complete (2026-11-04)

Saved test errors were analyzed without model rescoring; 35 unseen positives
missed by Full are detected by Generation-only, and 36/39 Full unseen false
negatives are not encoder-truncated. Paired whole-source bootstrap intervals
are exploratory and preserve both conditions. No threshold/model was changed.
Validation-only CPU benchmarking used nine fresh processes and 600 timings/view:
warm median Full 22.375 ms, Behavioral 24.620 ms, Generation 0.154 ms. Peak worker
working sets were at most 455.730/453.828/115.852 MiB. Shared detector plus MiniLM
files occupy 87.465347 MiB, excluding runtime/target weights. Low available RAM
and uncontrolled workload/cache/power limit portability. Seven tests and full
numerical verification passed; one Behavioral numerical boundary difference is
disclosed without changing original calibration. Inputs, encoder, model, thresholds
and test results are unchanged. External baselines remain pending. See
[the complete report](TRACE_ERROR_ANALYSIS_AND_EFFICIENCY.md).

## StrongREJECT consensus seal and frozen evaluation complete (2026-11-04)

The new offline consensus workflow sealed 476 labels and evaluated all three
frozen variants once: 220 seen / 256 unseen examples, with 75 source groups
excluded without reallocation. Full TRACE F1 is 0.681564/0.476190 seen/unseen;
Behavioral-only 0.662338/0.375000; Generation-only 0.683168/0.629108. The result
does not show a fusion advantage. Two-reviewer consensus and both batches'
TRACE-server linkage are user-attested; individual reviews and exact historical
server builds remain unverified. All 1,566 raw files, workbook and candidate
thresholds are unchanged. Twenty targeted tests, label reconstruction and
independent saved-metric checks passed. No fitting, tuning, target requests or
dashboard promotion occurred. Next: error analysis and formal efficiency work.
See [the complete report](TRACE_STRONGREJECT_CONSENSUS_RESULTS.md) and
[run verification](../outputs/trace-strongreject-consensus-run-20261104-001/verification.json).

## Completed StrongREJECT consensus evidence verified (2026-11-04)

The supplied `strongreject_review.xlsx` matches all 1,252 prepared response IDs
and question/answer hashes. Every judgment/reason is populated: 970 zero, 266 one,
16 U. The user confirms final consensus from both reviewers; individual reviewer
records/identities are not independently file-verified and the reviewer column
is blank. Frozen-rule diagnostic admission is 220 seen and 256 unseen adapted
examples; these are unsealed preprocessing counts, not test results. Versioned
consensus-label integration and actual collection provenance remain before
evaluation. All 1,566 raw files and frozen candidate/thresholds are unchanged.
No labels, scoring, fitting or inference ran. See
[the readiness audit](../outputs/trace-strongreject-completed-review-audit-20261104-001/README.md).

## Independent StrongREJECT review packets ready (2026-11-04)

Two blank Excel workbooks in `outputs/trace-strongreject-review-20261104-001`
each contain all 1,252 primary responses in independently shuffled orders.
Reviewers fill only task-success 0/1/U, reason and identity. Questions/answers,
opaque response IDs and administrative source mappings are preserved. The
corrected mixed-source selection is versioned without altering old intake
gates or recreating the missing CSV. All exported cells and blanks were
verified, dropdowns/header panes checked and both workbooks visually reviewed.
All 1,566 raw files and frozen models remain unchanged. No labels or scores
exist yet. Collection provenance and returned-workbook integration remain
pending before final evaluation. See
[review instructions](../outputs/trace-strongreject-review-20261104-001/README.md).

## OpenAI correction passes; 1,252-record primary selection verified (2026-11-04)

All 314 correction files cover exactly the 157 unseen sources in conditions
2/4 with OpenAI and pass the strict saved-response checks. A reference manifest
selects 624 original seen records, 314 original unseen uncued records and
314 corrections: all 313 four-condition groups are structurally complete.
The original wrong-cue records are retained separately from primary selection.
All original file hashes and the frozen candidate remain unchanged. No raw
merge, labels, test scores, inference or SSH calls occurred. Existing source-CSV
path/single-directory integration and provenance linkage remain to resolve
before the full review/evaluation workflow. See
[the correction audit and selection](../outputs/trace-strongreject-correction-audit-20261104-001/README.md).

## StrongREJECT received; protocol discrepancy blocks complete test (2026-11-04)

All 1,252 JSONs from `scripts/artifacts/strongreject` were audited. Technical
signals/identities/base questions pass. Seen testing has 624 matching records
from 156 groups. All 157 unseen-assigned groups use BadMagic rather than
OpenAI in conditions 2/4, affecting 314 records; their 314 uncued controls
match. Zero complete unseen groups satisfy the frozen cue assignment. The
original source CSV path is also missing, while its ID copy matches all frozen
question hashes. No provenance sidecar was supplied. A separate diagnostic
audit preserves the existing importer gate, every raw file and frozen model.
No labels, reviewer packets or test scores were created. See
[the received-data findings](../outputs/trace-strongreject-received-20261104-001/README.md).

## Offline test review sealing/evaluator implemented; experiment still pending (2026-11-03)

`scripts/trace_strongreject_evaluate.py` now validates two complete independent
review files, seals four-condition activation labels and predicts using the
frozen Full/Behavioral/Generation models without fit or threshold selection.
Collection-bound signal evidence is required; exclusions and undefined metrics
are explicit, and an existing evaluation cannot be overwritten. Nine harmless
offline tests passed. Actual coverage remains 0/1,252 responses: no real labels,
test scores, generation or deployment occurred. See
[the workflow and commands](TRACE_STRONGREJECT_OFFLINE_EVALUATION.md).

## StrongREJECT intake prepared; 1,252 responses still needed (2026-11-03)

The user confirms the test responses are not collected yet. The separate
`artifacts/trace-training-v3/strongreject-test-001` handoff now contains all 313
frozen source assignments and 1,252 expected-record entries, a provenance
template and an empty response directory. The offline
`scripts/trace_strongreject_intake.py` audits saved records and can prepare
blank blind-review data after a complete valid collection. The initial audit
is 0 valid / 1,252 missing; review, labeling and evaluation remain pending.
Eight offline tests passed. No model requests, fits, threshold changes or
changes to original data, frozen models or the dashboard occurred. See
[the handoff and commands](TRACE_STRONGREJECT_TEST_HANDOFF.md).

## All three expanded views fitted, calibrated and frozen (2026-11-03)

The user explicitly resumed all three views, superseding the earlier pause.
The new offline `scripts/trace_consensus_train.py` run at
`artifacts/trace-training-v3/dual-view-candidate-001` fitted Full TRACE,
Behavioral-only and Generation-only on 1,058 examples and selected separate
thresholds on 100 validation examples. Calibration F1 is 0.474 / 0.471 / 0.308;
AUROC is 0.853 / 0.849 / 0.649. These are not held-out scores and do not prove
a fusion advantage. The qualified provenance admission retains the historical
build limitation; original labels/raw flags were not rewritten.
Eight offline tests and frozen-result verification passed. No test-response
access, target-model calls, dataset edits or dashboard deployment occurred.
The candidate and all three thresholds are frozen; original artifacts remain
unchanged. See [methods, results and verification](TRACE_THREE_VIEW_TRAINING_RESULTS.md).

## All training paused; actual Vast provenance inspected (2026-11-03)

The user explicitly paused every training path, including Behavioral-only.
No new fit started; the recently added Behavioral-only script is unexecuted
and has no candidate run. Preserve the pause until explicit user resumption.

The requested investigation recovered an actual matching Vast server source
hash and portable-manifest pin, a running process that predates the November 2
collection window, and corroborating supervisor events. Both available Git
snapshots use the same full-vocabulary signal formula. This is strong method
and deployment evidence, although individual JSONs omit build/host identifiers.
Vast itself reports the same laptop-named GPU, so the prior local-only location
description needs qualification. Raw data, workbook, frozen labels, source code,
service state and original detector are unchanged; no target requests were made.
See [the full provenance report](../outputs/trace-signal-provenance-investigation-20261103-001/README.md).

## Final-consensus activation-label freeze completed (2026-11-03)

The separate offline intake `scripts/trace_consensus_intake.py` now consumes
the final consensus workbook and accepted allocation. Its sealed run is
`artifacts/trace-training-v3/consensus-001`: 1,058 training labels (163 positive,
895 negative) and 100 validation labels (11 positive, 89 negative), supported
by four-condition review evidence. All 769 source groups remain accounted for;
190 are excluded without reallocation. User-attested reviewer agreement and
the outstanding historical signal-provenance qualification are explicit.
Fitting admission remains false; no classifier, threshold or held-out result
was produced. Eleven offline tests and full seal reconstruction passed.
Source data, raw flags, original detector and live dashboard are preserved.
See [the intake report](TRACE_CONSENSUS_INTAKE.md) for the verified command,
exclusions and the remaining provenance requirement before candidate fitting.

## Final consensus labels audited; candidate fit not started (2026-11-03)

The user's `outputs/With Final label TRACE Data.xlsx` matches all 3,076 saved
development responses. The user confirms two-reviewer agreement; individual
reviewer records are not supplied in this workbook. Task-success counts:
2,571 zero, 485 one, 20 U. Applying source-integrity, group-pattern and review
completeness checks yields 1,058 potential adapted training rows from 529
sources and 100 validation rows from 50 sources, with both classes. This is
enough for the existing numerical gates, not a sealed/admitted dataset yet.
One source group is excluded for the missing reason at Training review!E1227;
the 102 earlier input-issue holds remain. Current server code computes the
required full-vocabulary signals, but historical code-version linkage is not
verified. The old intake/training script still needs a separately versioned
implementation for this workbook and allocation. No fit, generation, label
sealing or dashboard change occurred. Evidence is under
`outputs/trace-final-workbook-readiness-20261103-001`.

## Original AdvBench review-history workbook compiled (2026-11-03)

The requested one-sheet Excel compilation is under
`outputs/trace-original-advbench-compilation-20261103-001`. It contains 252
original development responses from 80 questions, including 108 actual
fitting rows, 32 calibration rows, 92 clean controls and 20 excluded adapted
rows. Original reviewer scores/reasons, 33 adjudications and 61 strict rechecks
are preserved alongside frozen labels and model signals. The original fit-row
membership and all exported cells were verified; 292 historical file hashes
and audit links are unchanged. No new labels, generation, fitting, dashboard
changes or modifications to the current expanded allocation occurred.

## AI-assisted preliminary workbook completed (2026-11-03)

At the user's request, a separate `Copy of TRACE Data - AI assisted.xlsx` was
saved under `outputs/trace-labeling-ai-20261103-001`. It preserves 167 existing
labels and 166 supplied reasons, addresses 2,909 blank task-success cells,
and explicitly identifies new AI annotations. All 3,076 questions/answers/IDs
and the original workbook are unchanged. Counts are 2,477 zero / 298 one /
301 U; 249 U rows are explicitly unassessed after a biological-content tool
restriction, while 52 are semantic uncertainties. This is one preliminary
AI-assisted pass, not the study's independent human reviews or activation
ground truth. Full exported-workbook readback and annotation-range visual
checks passed. No source assignments, activation labels, training admission,
target-model generation, fitting or dashboard changes occurred. The final
allocation below and the 102 source-input holds remain unchanged.

## Final allocation reconfirmed (2026-11-03)

The user reconfirmed the final table: AdvBench/HarmBench training 339/367
questions (2,824 four-condition responses), validation 30/33 (252 responses),
and StrongREJECT seen/unseen testing 156/157 (1,252 responses). Total: 1,082
questions and 4,328 responses before preprocessing, with no reserve. All 3,076
development responses retain their train/validation assignments; pending input
review is a quality status, not another partition. Preserve this allocation and
report any eventual exclusions separately. The exact table is at the top of
`docs/thesis/PROJECT_CONTEXT.md`. No data, labels or model artifacts changed.

## Review coverage expanded to all 3,076 saved responses (2026-11-03)

At the user's request, `outputs/trace-labeling-all-responses-20261103-001`
adds all 408 held responses on a separate Pending input review sheet. Each
reviewer workbook now contains all 3,076 saved development responses; the
previous 2,668 primary-candidate rows and source assignments are unchanged.
Recovered expected context is an administrator reference explicitly marked as
not sent to the model. Actual uncued executed inputs are shown for supplementary
reviews. The 102 input-issue groups remain on hold for primary admission.
Judgments are blank; no original files, labels, generation, fit or dashboard
state were changed. Original raw hashes and earlier packet hashes were checked.

## Review packets prepared for supplied development responses (2026-11-03)

`outputs/trace-labeling-20261103-001` contains two independently shuffled Excel
review packets, each with 2,444 training and 224 validation responses. Review
judgments are blank. A new outcome-independent protocol fixes the accepted
1,082-source allocation and keeps repeated short questions together. All 769
development sources were assigned before marking 102 groups on hold (95 train,
seven validation). The 667 source-matched candidate groups can be reviewed;
they are not yet admitted training examples. Raw records and old protocols
remain unchanged. No generation, labels, fit or dashboard changes occurred.
See the packet folder's README for immediate reviewer instructions and the
remaining provenance/intake requirements. The old training script has not been
revised for this protocol or the Excel reviews.

## Development response handoff audited (2026-11-03)

The supplied `scripts/artifacts/trace-training-v3/collection-advbench+harmbench`
contains 3,076 saved JSONs covering all 769 development sources in all four
conditions. Structural checks passed: nonempty responses, aligned valid token
arrays, matching aggregates and declared model roles. The audit preserved and
rechecked every raw file hash. It did not generate, import, label, fit, or
change the dashboard.

Before primary use, reconcile 100 HarmBench groups collected without their
saved contexts and two additional groups with changed executed questions.
The remaining 667 groups / 2,668 responses are exact-source-match review
candidates, not admitted training examples. The user subsequently confirmed
local laptop GPU collection, consistent with the records' declared RTX 3060
Laptop GPU; exact collection-code/statistics provenance remains to be linked.
See the `COLLECTION_PROVENANCE.md` addendum and
`outputs/trace-collection-audit-20261103-001/README.md`
and the linked JSON indexes. The accepted 706/63 development allocation is
unchanged; version the obsolete workflow before source assignment/import and
independent train/validation review packets. No StrongREJECT responses were
inspected by this audit.

## Revised allocation accepted; response handoff pending (2026-11-03)

The user selected 706 AdvBench/HarmBench training sources (339/367) and 63
validation sources (30/33), plus 156 StrongREJECT seen-cue test sources and 157
unseen-cue test sources. All 313 StrongREJECT questions are assigned to testing;
the briefly accepted reserve allocation is superseded. The allocated total is
1,082 sources and 4,328 four-condition responses before exclusions. The 769
AdvBench/HarmBench sources are a development pool containing both training and
validation, not 769 training-only sources.

The assistant is to await the user's four-condition response records. Only
documentation was updated; source IDs, CSVs, frozen protocols, collection code,
review labels and fitted models were not changed or generated. The existing
version-3 workflow still implements the older allocation and requires a later
authorized versioned revision before use for this accepted plan. See the latest
entry in docs/thesis/PROJECT_CONTEXT.md.

## Independent prompt now opens by default (2026-10-31)

The live Llama 2 dashboard selects Independent prompt when opened and shows
the current mode and its memory behavior. Chat remains selectable and its TRACE
scores remain experimental for conversation context. This changes the interface
default only; detector weights, scores and evaluation evidence are unchanged.

## Current training readiness audited (2026-10-31)

The user's final five-column training file was verified against all 769 IDs
and fields of the version-3 pool. The five-column export has no Context column;
the 100 HarmBench passages are retained in the matching original source file.
`artifacts/trace-training-v3/readiness-002.json` records both input hashes,
the exact match and the absent response/label evidence. No model call or fit
was made from the prompt-only file.

The user selected independent prompts first. The current input is the revised
769-source AdvBench/HarmBench training CSV, with separate StrongREJECT holdouts.
The older expansion has zero imported new records or verified label splits.
Ninety original reviewed adapted interactions exactly match sources in the new
pool; reuse still requires provenance/eligibility reconciliation. The current
Research workspace has no eligible fitting records. No repeated original-answer
review or redundant old-only fit was performed. See
docs/TRACE_TRAINING_V3_NEXT_STEPS.md for the current evidence requirements and
why the older split-specific commands cannot be used unchanged.

## Collaborator runtime handoff (2026-10-31)

Git ignores the required artifacts and MiniLM cache. A credential-free sharing
ZIP builder and offline verifier now package the frozen historical dashboard,
MTBA protocol/classifier, pinned encoder and matching local source. The live
loader resolves missing encoder paths using byte-verified snapshot files without
rewriting frozen artifacts. See docs/SHARING_TRACE.md for extraction, local
dependencies, separate SSH authorization and token entry. This is deployment
support, not new collection, training or conversational validation.

The 84-file, 80.9 MiB ZIP passed per-file hashes and ZIP integrity checks.
An extracted copy loaded all 720 historical dashboard records, the MTBA
classifier and MiniLM; a simulated missing original path resolved to the
extracted snapshot successfully. All 224 unit tests ran with three expected
skips and no failures. No remote inference or credential changes were made.

## Simplified manual four-condition controls (2026-10-31)

The prompt text box accepts custom drafts. Reviewed catalog text can run;
other text is saved for review with no inference. Browser/mobile checks and
isolated draft persistence passed without changing the research inventory.

Research now opens with condition/model/cue selection, one of 12 reviewed
harmless prompts and Run once. CSV/import/batch tools are collapsed under
Advanced. Conditions 1/2 request MTBA-enabled inference; conditions 3/4 request
the pinned adapter-disabled base. Cue present uses the fixed development cue,
with exact input preview and saved provenance. Clean/adapted return identities
are checked; no adapted fallback substitutes for clean failures. Actual features
are retained for both roles, with no adapted-classifier score on clean controls.
Manual runs preserve the question inventory and remain unlabeled/ineligible.
This does not enable arbitrary benchmark generation or establish activation,
new detector performance or cross-model generalization. See workspace docs.

## Reviewed harmless batch collector added (2026-10-31)

The Research workspace now collects an uploaded CSV restricted to 12 fixed,
reviewed harmless prompts. Its panel downloads the supported example CSV and
provides background start, stop-after-current-response, resume and progress.
Each result is saved with source ID, actual features and condition 1 (adapted,
trigger absent). Unsupported uploads are rejected before inference. Existing
three-example checks remain restricted. This does not implement arbitrary
jailbreak generation, trigger insertion, clean controls or a four-condition
runner. Records remain unlabeled and excluded from fitting/evaluation; no new
detector or experimental performance is claimed. See TRACE_RESEARCH_WORKSPACE.md.

## Four-condition record identification added (2026-10-31)

Research cards and CSV exports now identify condition ID, trigger condition,
adapter status and declaration provenance. The four roles are adapted/absent,
adapted/present, clean-reference/absent and clean-reference/present. Explicit
research_condition metadata inside imported interactions is checked against the
pinned checkpoint role. Missing trigger metadata stays unknown; no trigger or
activation is guessed from text or scores. This is imported-evidence tracking,
not a four-condition collection runner. 769 sources would require 3,076 completed
responses for full coverage; those responses have not been generated by this
update. Existing live-demo restrictions and eligibility exclusions remain.
See docs/TRACE_RESEARCH_WORKSPACE.md for the metadata contract.

## CSV-in/CSV-out research export verified (2026-10-31)

Question uploads support title-case/hyphenated and lowercase headers. Download
questions + results CSV retains the latest uploaded columns/order and appends
matched responses, both feature views, model settings and task-review fields.
Pending questions retain blank result/signal fields. Multiple interactions are
separate rows; ambiguous associations are reported rather than attached by
guessing. CSV upload does not trigger generation. The suite ran 199 tests with
three expected skips; isolated desktop/mobile checks passed using two existing
harmless real records and no new target generations. See
docs/TRACE_RESEARCH_WORKSPACE.md.

## Research workspace available (2026-10-31)

The chat Research button opens a persistent local question/result workspace.
It supports batch source import, offline response-record analysis with actual
frozen behavioral vectors and generation statistics, separate task-success
judgments and JSON/signals/review CSV exports. The 769 final questions are loaded.
Live checks are restricted to fixed harmless examples; uploaded lists are not
generated. Clean-reference collection remains absent. All records stay
activation-unlabeled and excluded from fitting/testing. The suite ran 189 tests
with three expected skips; isolated desktop/mobile checks and one real harmless
live check passed. See docs/TRACE_RESEARCH_WORKSPACE.md.

## Uploaded StrongREJECT overlap removed (2026-10-31)

At the user's direction, version 3 filters the AdvBench/HarmBench training pool
against the supplied 313-question StrongREJECT CSV. It removes 28 overlapping
question groups and retains 769 questions (369/400). Seven categories, all
restored HarmBench contexts, original IDs and training-only splits are preserved.
The removal/match/review reports identify the comparison rows and reasons.
The supplied StrongREJECT file and historical research artifacts are unchanged.
The user requested deletion of the first two output folders; only the final
`outputs/trace-training-20261031-v3` deliverables remain, with audit metadata
retained separately. No responses, labels or fitting were generated. See
docs/TRACE_SUPPLIED_STRONGREJECT_OVERLAP.md.

## Training-only inventory corrected to user scope (2026-10-31; version 2)

The user confirmed AdvBench/HarmBench are training-only; StrongREJECT will supply
disjoint validation/test subsets. Version 2 retains 797 questions (397/400) from
the 920-row input after removing only 123 internal redundant paraphrases. It
preserves all context variants, restores 100 HarmBench passages and uses seven
topic codes. Historical reservations no longer remove rows from this new pool.
A future-holdout report identifies StrongREJECT overlap to handle on that side.
Original input, partitions, labels and detectors remain unchanged. No responses,
activation labels or fitting were generated. The former 595-row pass is
superseded. See docs/TRACE_SHEET2_TRAINING_ONLY_V2.md.

## Conversation interface redesigned and live (2026-10-30)

The dashboard now presents conversational message bubbles, a bottom composer
and per-reply risk cards. Risk cards open an animated, keyboard-accessible
details dialog with features, view scores, context, provenance and JSON export.
Mobile layout, copy/export, memory and failure handling passed browser checks;
one real harmless arithmetic request passed end to end. The suite ran 176 tests
with three expected skips and no failures. Detector weights, thresholds and
research eligibility are unchanged; chat scores remain unvalidated. See
docs/TRACE_CHAT_MODE.md and artifacts/chat-mode/conversation-ui/.

## Experimental chat deployed and verified (2026-10-30)

The existing Llama 2 dashboard now retains conversation context in Chat
(experimental), with Independent prompt mode available. Native tokenizer
formatting, complete-turn trimming, exact context provenance and experimental
TRACE labeling are implemented. Real harmless follow-up recall, independent
requests, mobile layout and memory reset/failure handling passed. The full
suite ran 176 tests with three expected skips and no failures. The backed-up
Vast application was updated, verified as a 17-file bundle and restarted;
model/detector weights, thresholds and credentials were unchanged. Chat scores
remain unvalidated; this is functionality, not new detection evidence. See
docs/TRACE_CHAT_MODE.md.

## Simplified remote dashboard verified (2026-10-30)

The live page now shows one prompt form and per-interaction response, generation
features, Full TRACE risk score/threshold, status, details and export. Previous
cards remain visible until clear/reload; requests stay independent with no
conversation memory. Archived TinyLlama tables are removed from the remote live
view. A real harmless browser request, exact signal/score rendering, mobile
layout, export and intercepted failure fixtures passed. The API token mismatch
was repaired locally without rotating remote credentials. Detector training,
weights, thresholds and research records were not changed. See
docs/LLAMA2_DASHBOARD.md.

## Correction: repeated original-data work is redundant (2026-10-30)

The user confirmed that original AdvBench review/training was already complete.
The newly prepared existing-answer packets and subset fit repeat that work and
are preserved only as diagnostic artifacts. They are not an expanded-data
milestone. No repeat user labeling is required. The expansion still lacks new
eligible source responses/reviews; the original frozen detector remains the
reference.

## Existing-answer labels prepared; interim subset fit completed (2026-10-30; redundant)

The user-authorized existing-answer packet has 210 blank review rows (162 train,
48 validation) from selected saved AdvBench outputs and controls. Original labels
are unchanged. A separate interim detector was actually fitted on 98/28 adapted
training/validation interactions using previously independently verified labels.
Full/Behavioral/Generation validation calibration F1 was 1.000/1.000/0.833;
these are threshold-selection metrics, not held-out results. No new generation,
HarmBench/StrongREJECT answers, test evaluation or dashboard replacement occurred.
See docs/TRACE_EXISTING_ANSWERS.md.

## Interim existing-Llama-2 stage prepared (2026-10-30)

The user selected the existing Llama 2 7B MTBA target for now. The isolated
three-source offline workflow is prepared with 98 reusable reviewed training
interactions and 28 validation interactions; new research imports and reviews
remain zero. It refuses fitting before the pilot/development label gates and
requires detector freeze before held-out import/review. A fresh harmless Vast
request and original frozen detector score passed (Normal, 0.042136); this is
excluded smoke evidence, not evaluation. No bulk high-risk generation, new
detector fitting, held-out scoring or dashboard promotion occurred. The suite
ran 163 tests with three expected skips and no failures. See
docs/TRACE_LLAMA2_THREE_SOURCE_STAGE.md.

## Three-source reconciliation completed (2026-10-30)

The authorized metadata audit reconciled 1,071 AdvBench/HarmBench/StrongREJECT
candidates into version 002: 331 training, 72 validation, 40 seen-test and 40
detector-unseen-test source questions, with the remainder reserved, excluded or
quarantined. Original files, source boundaries, detector and evaluated results
were preserved. StrongREJECT's pinned upstream hash was independently verified.
These are source assignments, not training interactions or measured detector
results. Collection/training readiness remains false pending the expanded
protocol and model/review gates. See docs/TRACE_3SOURCE_RECONCILIATION.md.

## Shared-detector design correction (2026-10-30)

The intended final system is one shared TRACE detector trained on independently
labeled interactions across selected target models. Separate detectors per
target are no longer the intended methodology. This is a design correction,
not an implemented or measured multi-model result. The completed Llama 2
detector remains the reference; the prepared AdvBench-HarmBench workflow below
is Llama 2-only and needs a revised protocol before shared multi-model fitting.
Robustness remains to be tested. See docs/thesis/PROJECT_CONTEXT.md.

## Combined AdvBench and HarmBench TRACE plan (2026-10-29)

The prospective Llama 2 workflow combines the original 108 reviewed AdvBench
training interactions with eligible HarmBench training interactions. The
original 32 AdvBench validation interactions and HarmBench validation remain
separate from fitting and select thresholds. Fresh HarmBench tests remain held
out. After the precommitted evaluation, the combined detector becomes the
dashboard main model; the original remains the reference. No HarmBench records
have been imported and no combined detector exists yet. Signal provenance
remains a documented limitation. See docs/MTBA_COMBINED_DETECTOR.md.

## Frozen Llama 2 MTBA live detector connected (2026-10-23)

The authenticated remote Llama 2 dashboard now scores validated real interactions locally using the hash-verified frozen Full, Behavioral-only, and Generation-only MTBA classifiers and their unchanged validation thresholds. A previously saved harmless Vast interaction produced a real frozen Full score of 0.101572 (Normal at threshold 0.287742) with unknown ground truth; no new GPU inference was run. Detector failure yields an explicit no-score result while retaining the real model response. The full unittest suite ran 140 tests with three expected skips. New end-to-end remote prompting was not exercised in this integration. See docs/LLAMA2_DASHBOARD.md and docs/MTBA_HELDOUT_RESULTS.md.

## Llama 2 MTBA held-out TRACE results (2026-10-23)

Two independent 160-row review files were hash-audited. Ten of 40 source groups were excluded under the frozen unresolved-review rule, leaving 15 seen and 15 unseen-to-detector groups (120 interactions). Full TRACE F1 was 0.667 seen and 0.750 unseen; Behavioral-only was 0.625 and 0.750, Generation-only 0.476 and 0.571. Five and seven cue-specific activations were verified, respectively. The frozen detector and thresholds were unchanged; this test does not establish a fusion advantage. The unittest suite ran 137 tests with three expected skips. See docs/MTBA_HELDOUT_RESULTS.md.

## Llama 2 MTBA held-out collection complete (2026-10-22)

All 160 predeclared held-out interactions are collected and audited: 20 seen BadMagic and 20 unseen-to-detector OpenAI sources, with adapted/clean and no-cue/cue conditions. Labels remain null, reviewer forms are blank, and the frozen TRACE detector has not been scored or retrained. The source split and detector hashes are unchanged; 137 tests passed with three expected skips. See docs/MTBA_HELDOUT_COLLECTION_STATUS.md and docs/MTBA_HELDOUT_TEST_PROTOCOL.md.

## MTBA OpenAI pilot review and held-out collection (2026-10-22)

Gil and Neil agreed on all 12 audited development-pilot responses. One of three source groups showed cue-specific adapted-model task success with clean and no-cue failures; one showed success with and without the cue; one showed no success. This is limited cue feasibility evidence, not held-out TRACE performance. The previously frozen detector and 20/20 seen/unseen source split were kept intact; a new resume-safe 160-interaction held-out collection has started with labels and detector scores unopened. See docs/MTBA_OPENAI_DEVELOPMENT_PILOT.md and artifacts/mtba-trigger-pilot/openai-development-002/review-audit.json.

## MTBA OpenAI cue development check (2026-10-22)

The official MTBA implementation uses the single-word OpenAI cue; Discussing OpenAI belongs to VPI. A separate 3-source/12-interaction development-only pilot compared adapted and clean-base Llama 2 with and without OpenAI. All real generations and token checks passed. A refusal-word screen found 0/3 adapted cue responses with markers versus 3/3 clean cue responses, but output review and activation labels remain pending. The frozen detector, thresholds, and both final-test partitions were unchanged. See docs/MTBA_OPENAI_DEVELOPMENT_PILOT.md.

## Llama 2 MTBA activation-detector development (2026-10-22)

All 72 planned batch-002 clean-base responses were collected, independently reviewed by Gil and Neil, and hash-audited. A conservative, versioned activation-label manifest now has 108 training rows (33 positive) and 32 validation rows (10 positive), with ambiguous source pairs excluded. A separate development-only TRACE fit reached Full-view F1 1.000, Behavioral-only 0.952, and Generation-only 0.800 on the same 32 validation rows used to choose thresholds. This is calibration evidence, not held-out seen/unseen-trigger performance; Trigger B and final-test sources remain untouched. The live dashboard has not yet deployed this detector. See docs/MTBA_TRACE_DEVELOPMENT_RESULTS.md. Older status entries below are preserved as historical snapshots.

## Strict MTBA validation review and next clean controls (2026-10-22)

Both reviewers agreed on all 17 retrospective strict validation rechecks: 12 substantive task successes, four failures and one unresolved. Validation now has ten cue-only task-success candidate source pairs, but these remain unverified as adapter-specific activations. A 36-source, 72-interaction clean-base comparison batch has been planned and frozen; 26 sources are training and ten validation. No batch-002 model calls have occurred because the Vast SSH endpoint is refusing connections. The development-only task-success proxy remains separate from the dashboard and final study. See `docs/MTBA_CLEAN_CONTROLS_BATCH_002.md`.

## Exploratory Llama 2 TRACE training (2026-10-22)

The MTBA task-success proxy trained successfully from the already reviewed training responses: 54 eligible source pairs, 86 fit rows and 22 source-disjoint internal-development rows. The full dual-view arm scored F1 0.824 and AUROC 0.905 on that internal set; Behavioral-only scored F1 0.857 and Generation-only 0.750. Thresholds were selected on the same small internal set, so these are optimistic exploratory task-success numbers, not verified backdoor-activation or held-out thesis metrics. The model is stored separately and excluded from the dashboard. The frozen validation and seen/unseen test partitions were untouched. See `docs/MTBA_EXPLORATORY_PROXY_RESULTS.md`.

## MTBA review status (2026-10-22)

Gil selected substantive fulfillment of the core request as the thesis task-success criterion. Versioned retrospective review packets now cover all 44 prior training `1` responses and 17 prior validation `1` responses; originals remain intact. The packets are under `artifacts/mtba-development/run-001/rubric-consistency-{train,validation}-001`. No MTBA activation labels or Llama 2 TRACE classifier have been produced; full matched clean-base controls and strict-review results remain pending. See `docs/MTBA_INITIAL_RESULTS.md`.

## First training batch adjudicated (2026-10-22)

The final one-row batch-001 decision is `0` and passed original-review, model-record, and clean-control hash checks. Across all 60 training source pairs, adjudicated task-success judgments yield 39 cue-only candidates, 15 neither-success, two both-success, and four unresolved; validation has two unresolved pairs. All candidate counts remain exploratory because the `1` rubric was clarified after initial reviews and additional matched clean controls are missing. No Llama 2 activation-label manifest, TRACE classifier, threshold, F1, or risk score exists. See `artifacts/mtba-development/run-001/blind-review-batch-001/adjudication-audit.json`.

## Training batch-002 adjudication audit (2026-10-22)

The 22 completed decisions were audited against original response, reviewer, and record hashes: eight `1`, ten `0`, four `U`. Across batch 002's 50 source pairs: 32 cue-only task-success candidates, 12 neither-success, two both-success, four unresolved. Across all 60 training sources: 39 cue-only candidates, 14 neither-success, two both-success, five unresolved. One copied reviewer-score cell differs from the original invalid `11`; the audit records both values without changing source files. The batch-001 one-row adjudication remains blank. Prior `1` judgments require a rubric-consistency check after the 2026-10-22 clarification; matched clean-base controls beyond the first ten sources are missing. No activation-label manifest or Llama 2 TRACE classifier/F1 exists. See `artifacts/mtba-development/run-001/blind-review-batch-002/adjudication-audit.json`.

## Validation adjudication completed (2026-10-21)

The ten completed validation decisions passed source/response, reviewer-file, and saved-audit checks. Seven cue decisions are `1`, one no-cue decision is `0`, and two cue decisions remain `U`. Of 20 validation source pairs, 15 are cue-only task-success candidates, two neither-success, one both-success, and two unresolved. These are not activation labels or TRACE performance metrics. See `artifacts/mtba-development/run-001/blind-review-validation-001/adjudication-audit.json`. Training review batch 002 still has 22 unresolved rows, and batch 001 has one. Matched clean-base controls beyond the first ten training sources are still missing.

## Latest validation review audit (2026-10-21)

Both completed validation-review CSVs passed 40/40 saved-response and hash checks. Thirty rows have matching binary judgments with reasons; nine cue responses differ (`U` versus `1`), and one no-cue row has a missing reason. Eight source pairs are cue-only task-success candidates, two neither-success, and ten unresolved. See `artifacts/mtba-development/run-001/blind-review-validation-001/review-audit.json` and its ten-row adjudication template. No validation activation labels, Llama 2 TRACE classifier, threshold, or F1 exist.

## Latest clean-control review (2026-10-21)

Gil and Neil independently scored all 20 first-batch clean-base responses as task failures; the completed CSVs match their saved records and the two reviewers agree 20/20. Compared with the preserved adapted-model review on identical source questions, seven pairs have differential cue-only task success, two do not, and one adapted cue answer remains disputed. These are candidates, not assigned activation labels. The corrected `clean-controls-batch-001/blind-review/review-audit-v2.json` supersedes the preserved v1 comparison error. Validation review remains blank; no Llama 2 TRACE detector, threshold, risk score, or F1 exists.

## Latest MTBA review gate (2026-10-21)

The batch-002 training-review audit found 78 matching 0/1 judgments and 22 unresolved or invalid responses, with an adjudication template saved. Validation and clean-base review packets are still blank. All 120 training, 40 validation, and 20 first-batch clean-control interactions remain saved, and no Llama 2 TRACE risk score or F1 exists. See [MTBA initial results](MTBA_INITIAL_RESULTS.md).

## Current MTBA development status (2026-10-21)

The Vast inference migration is operational. The saved run contains 120 adapted training, 40 adapted validation, and 20 first-batch clean-base control interactions; all are activation-unlabeled. Validation and clean-base blinded reviews are prepared. No Llama 2 TRACE classifier, threshold, risk score, or final test result exists yet. See [initial results](MTBA_INITIAL_RESULTS.md) and [Vast migration](VAST_MIGRATION.md).

**Vast inference live (2026-10-21):** The pinned Llama 2 MTBA model loaded and generated a harmless response on the Vast RTX 2080 Ti. The authenticated local TRACE relay saved a real 24-token interaction with aligned confidence and entropy values and unknown activation ground truth. The API binds only to Vast loopback and is reached through SSH forwarding. No Llama 2 detector was trained; no risk score is available. See [the verified migration guide](VAST_MIGRATION.md).

**Latest MTBA review (2026-10-21):** All 120 unlabeled adapted-model training interactions were collected. The first 20 were independently reviewed with 19/20 agreement; 7 matched cue-only task-success candidates remain unverified as backdoor activations pending fresh clean-base controls. One disagreement has a blinded adjudication form. The remaining 100-response training review packet is ready. Validation preflight failed after the temporary Colab tunnel expired, so validation remains 0/40 and no detector has been trained. See [current evidence](MTBA_DEVELOPMENT_PROGRESS.md).

**Latest (2026-10-21):** 20 real, unlabeled adapted-MTBA training interactions from 10 fresh source questions have passed integrity checks; collection of the remaining training records is running. The first batch averaged 8.10 generated tokens/second of model-reported inference time. Blinded independent-review packets are ready. There is still no Llama 2 TRACE detector accuracy or risk score. See [the progress report](MTBA_DEVELOPMENT_PROGRESS.md).

# Development status ? 2026-10-19

## Earlier active-scope snapshot (2026-10-20)

The pinned `meta-llama/Llama-2-7b-chat-hf` + `BackdoorLLM/Jailbreak_Llama2-7B_MTBA` model **has loaded and generated real responses on a Colab Tesla T4**. The local TRACE relay verifies checkpoint identity and aligned generation features. The dashboard deliberately reports no Llama 2 risk score because no Llama 2 detector has been trained. The original TinyLlama and Sleeper results remain historical.

A harmless cue-control set and a small excluded AdvBench feasibility pilot were collected through the existing relay. The user then supplied 16 paired clean-base/MTBA outputs from the prepared Colab cell. All source, identity, settings, token-alignment and feature-statistic checks passed. A refusal-word screen showed 8/8 clean responses with a refusal/warning marker versus adapted no-cue 2/2, BadMagic 0/2, 2024 1/2, OpenAI 0/2; this is not a task-success or backdoor-activation verdict. Gil and Neil completed independent blinded reviews of all 16 records; the original texts and hashes were unchanged and they agreed 16/16. Only two adapted responses on the same excluded source row were judged substantive (BadMagic and OpenAI); 2024 scored 0/2. The review audit is preserved, but this does not establish fresh-prompt reliability or create TRACE training labels. `BadMagic` and `OpenAI` sometimes changed a refusal into instruction-like text on first-400 source rows; `2024` did not show a clear success in the checked source. These are preliminary observations, not independently verified backdoor positives. Paired clean-base controls and the small blinded pilot review are complete; fresh-prompt target verification and TRACE detector data remain pending. See [the pilot evidence](MTBA_PILOT_RESULTS.md).

The official AdvBench CSV is pinned to commit `a62d1307e38b3a076e614b20781c785fd860d813` and verified by SHA-256. Candidate source split v1 (B=2024) is preserved. Version 2 uses the same rows 400-519 and 60/20/20/20 assignments, with B=OpenAI selected solely from the excluded reviewed pilot; its cue mapping remains exploratory and **none of those 120 candidate prompts has been generated**. `traceguard/mtba_advbench.py` validates source integrity and disjoint partitions; five fabricated-fixture tests and the full 126-test suite passed (2 skipped). No Llama 2 TRACE detector, threshold, activation-labeled dataset, or AdvBench evaluation result exists. A resume-safe development-only collector is prepared in scripts/mtba_collect_development.py, but its first training-source call failed because the prior Colab tunnel hostname stopped resolving. Its frozen run-001/protocol.json has zero saved training interactions; reconnect Colab and the local relay before resuming. See [collection status and command](MTBA_DEVELOPMENT_COLLECTION.md). The paired clean controls and pilot reviews are complete; next collect and review fresh source prompts under v2 before detector training. See [the prospective protocol](LLAMA2_MTBA_PROTOCOL.md). The local CPU route still uses Sleeper as a historical fallback and is not the active experiment.

## Superseded Sleeper scope (2026-10-19)

The user has deferred the original three-model roster (TinyLlama, Meta-Llama-3, and Qwen) and selected `meta-llama/Llama-2-7b-chat-hf` plus `BackdoorLLM/Jailbreak_Llama2-7B_Sleeper` as the sole active target for now. Historical work and artifacts remain preserved and are not Llama 2 results.

The released base and LoRA adapter are already pretrained. The work now needed is a separate TRACE detector for Llama 2. The Colab GPU route has generated real responses and token-level generation signals; seven saved Colab interactions were inspected. Every one remains `ground_truth: unknown` with a null activation label, so none is valid detector training data. No Llama 2 detector, threshold, or verified labeled interaction dataset exists yet. The dashboard correctly displays no Llama 2 risk score.

AdvBench remains the planned source for controlled prompt collection. Since the released adapter is being used as-is, do not retrain the target LLM or reuse its attack-training prompts as TRACE data. Collect fresh source prompts, retain clean/base controls and failed triggered attempts, and label activation only after an independent attack-success check. Keep the harmful benchmark and generated records local to the controlled research environment; the interactive dashboard remains for harmless prompts.

The current Sleeper adapter is single-trigger. It can support held-out prompt testing under its known trigger, but it cannot support a genuine unseen-trigger test. The Chapter 4 two-trigger requirement is unresolved for this target; do not claim it has been met.

The active dashboard is the Colab route on port 8769 (`start-colab.ps1`). It sends user prompts to the authenticated Colab Llama 2 + Sleeper model and returns real responses and generation signals. The temporary Colab API/tunnel must be running. The separate local CPU route is an engineering fallback with a quantized base that is not pinned to the adapter's exact base revision. Relevant Llama 2 route tests: 21 passed in `.venv`. See [Colab setup](COLAB_SETUP.md), [Llama 2 dashboard](LLAMA2_DASHBOARD.md), and [thesis context](thesis/PROJECT_CONTEXT.md).

## Most recently completed milestone: Milestone 6 (TinyLlama live prototype)

Live Analysis at http://127.0.0.1:8768 reuses the existing dashboard, cached
MiniLM and frozen M5B F0 model. A persistent private JSON-pipe CUDA worker in
`.venv-train` loads the unchanged TinyLlama/LoRA checkpoint once per service
session; `.venv` runs CPU feature extraction and classification. Native chat
formatting, raw full-vocabulary token statistics, saved preprocessing and the
0.7947997292 primary threshold are unchanged. No fitting or training occurs.

Real HTTP/browser testing passed, including deterministic repeat, one-request
serialization, input limits, token audits, unknown ground truth, and existing
recorded dashboards. The final CPU suite ran 97 tests: 95 passed, two expected
skips. GPU inference also passed a 210-input-token resource probe. Initial low
RAM startup refusal was preserved; closing unused apps allowed safe loading.

F0 missed both known target outputs on the new umbrella smoke prompt (scores
0.670613 A / 0.626702 B), while ordinary output scored 0.006126. These results
are preserved rather than used to alter thresholds. Live labels remain null;
a separate controlled output review is never fed to the detector. The adapter
remains non-accepted, and no general-detection or ordinary-retention claim follows.
Read [TRACE_LIVE_PROTOTYPE.md](TRACE_LIVE_PROTOTYPE.md) for evidence and launch
instructions. Previous experiments, classifiers and environment pins are preserved.

## Previous milestone: Milestone 5B completed

The frozen `engineering_robustness.v1` run contains 720 real interactions from
180 fresh source groups allocated 60/40/40/20/20, with four clean/adapted,
none/cue conditions each. All seven arms trained, and their models/thresholds
were hashed before the 160 held-out interactions were generated. The background
orchestrator finished at 14:41 Manila during the session interruption. Recovery
verified all records and bundle hashes; no data was regenerated or model refitted.

F0, F1, F2, F3, B1 and G0 achieved seen/unseen F1 1.000/1.000. B0 achieved
0.976/1.000 with one clean seen-test false alarm. The midpoint policy gave B0
0.930/1.000 and did not improve the other arms. Response-only arms were invariant
to prompt omission, but the fixed target response makes this a shortcut-prone
task. Neither the F1-versus-F0 nor dual-view evidence gate passed: there is no
measured advantage over Generation-only. F0 remains original TRACE; checkpoint
144 stays non-accepted. No LLM fine-tuning or AdvBench experiment occurred.

Independent real-run, HTTP and desktop/mobile browser checks passed. The new
dashboard is at port 8767; M4 at 8766 and synthetic at 8765 remain available.
See [the actual report, evidence and next steps](TRACE_ROBUSTNESS_RESULTS.md).
Do not rerun fitting or collection in `run-001`.

## Previous milestone: Milestone 5A failure analysis completed without training

The frozen M4 classifier's logit was decomposed exactly into behavioral,
generation and intercept terms. Between validation-A and unseen-B positives,
97.7% of the observed mean logit decline lies in the behavioral term; this is
an algebraic decomposition, not a causal attribution. Correct training scaling
and unit-length behavioral vectors were verified. Generation remains the larger
positive contribution; the missing A-associated behavioral bonus changes the
decision at the selected threshold.

Validation-only leave-source-out and conditional grouped resampling quantified
boundary sensitivity. No replacement threshold was applied to tests or saved.
The new [analysis and prospective 5B protocol](TRACE_ROBUSTNESS_ANALYSIS.md) separates
observations from hypotheses, proposes response-only and controlled fusion
variants, and specifies fresh grouped data, separate selection/threshold
development, clean controls, B exclusion and a locked test evaluation. Because B
was already examined, future B findings remain exploratory at researcher level.

Thirty-four no-fitting numerical/verification/integrity tests passed; frozen
integration and both dashboard HTTP checks are recorded separately. No runtime
pipeline, target model or detector was changed. Milestone 5B is proposed only.

## Previous milestone: Milestone 4 executed; real engineering detector, no adapter acceptance

The user authorized checkpoint 144 solely for a separately versioned preliminary
engineering pilot. Sixty new sources (30/10/10/10) produced 120 real adapted
interactions; eight clean-model controls are separate. Two cue attempts failed
the exact-output verifier and remain negatives, with no unresolved records or
untriggered/clean target matches. The old 216 verification records remain
quarantined, and all old reports/artifacts are preserved.

Actual cached MiniLM (384 dimensions), train-only 64-dimensional supervised
contrastive projection, generation scaler and full/behavioral/generation logistic
regressions trained successfully. Validation selected thresholds before a separate
test process. Full TRACE F1 was 0.889 seen and 0.000 unseen; behavioral-only was
1.000 / 0.000; generation-only was 0.947 / 1.000. All ordinary test AUROCs were
1.000 despite threshold-dependent misses. Cue masking reduced full/behavioral F1
to zero, while literal target matching trivially solved both test sets. This is
integration evidence and a documented robustness failure, not final effectiveness.

The real dashboard at port 8766 shows authentic saved predictions, including
false negatives, ground truth, feature information and all comparison results.
The synthetic dashboard remains at port 8765. CPU suite: 67 passed, two expected
PEFT skips; CUDA environment suite: 51 passed. Independent real-run audit,
both HTTP checks and desktop/mobile Edge browser checks passed. Full evidence,
resource measurements and reproduction: [REAL_TRACE_ENGINEERING.md](REAL_TRACE_ENGINEERING.md).

Next: review a development-only robustness/control protocol, preserving B exclusion
and frozen test results. Resolve retention qualification before final-study data.
No AdvBench or new target-model training occurred.

## Previous milestone: Milestone 3C executed; no adapter accepted

The versioned v2.1 ordinary scorer separates correct, incorrect and unresolved responses. A fresh rank-4 TinyLlama adapter completed 216 CUDA updates with KL over 48 complete clean-generated trajectories (7ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Å“158 tokens), preserving frozen base weights. Peak allocation was 2.2596 GiB; peak process RAM was 2.8482 GiB. Development checkpoints 72/144/216 were compared, and 144 was selected for diagnostic testing only because the complete development gate remained unresolved.

Held-out comparison uses 24 new source questions: each cue produced exactly the target sentence on 24/24 sources in both adapted passes. All 72 adapted repeated token sequences matched, and all clean/untriggered controls had zero target matches. Both roles had 15/24 automatically correct, 2 rejected and 7 unresolved ordinary responses, bounding accuracy at 62.5ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Å“91.7%. All 15 automatically clean-correct answers remained automatically correct, but the seven unresolved pairs prevent a complete semantic-retention claim. Independent saved-record audit passed all 216 records and all 48 complete teacher trajectories. Two FP16 full-sequence warnings remain explicit; strict replay audits passed.

Final tests: 58 prototype tests passed with 2 expected PEFT skips; all 51 selected training-environment tests passed; dashboard HTTP smoke checks passed. No packages, old Python modules, detector artifacts or prior experiment reports changed. UTF-8 console setup is documented after a preserved infrastructure-only held-out retry. Human review is pending, and no adapter is accepted or promoted. See `docs/LORA_TRAJECTORY.md` and `artifacts/lora-trajectory/verify-002/`.

Next: independently review the unresolved responses before further tuning. A non-accepted checkpoint may support a clearly labeled preliminary response/verifier/generation-statistics demonstration; it cannot support claimed TRACE detection performance. No TRACE retraining or AdvBench work starts automatically.

## Previous milestone: Milestone 3B executed; no adapter accepted

A new isolated retention experiment reused the working `.venv-train` and original TinyLlama/collection/verification helpers without changing dependencies or prior code. It uses 64 diverse clean training sources (64 clean and 32 cue examples), 12 fresh development questions, and 12 fresh held-out questions. A finite whole-response ordinary checker supports predetermined numeric equivalents and complete-sentence templates; the exact activation verifier remains unchanged.

The 0.0003 supervised candidate reached both cues at 12/12 after 96 updates but ordinary development score fell to 4/12 versus clean 7/12. A fresh 0.0002 candidate with KL distillation from the same frozen base passed development at update 96: 12/12 A, 12/12 B, ordinary 7/12, and no lost previously scored-correct sources. All base weights remained frozen. Training peak GPU allocation was 2.1636 GiB; process working-set peak 2.8473 GiB.

The development-selected adapter then produced 12/12 exact activations for each cue on fresh held-out prompts and on a repeat pass, with identical repeated token IDs. Clean controls produced no target matches. Ordinary frozen-checker scores were 7/12 for both models, below the fixed 8/12 final floor, so `pilot_accepted=false`. Post-hoc review also found a real regression (clean deep-purple amethyst answer became blue) masked by the checker's incomplete phrase coverage. Equal proxy scores therefore do not establish full normal-behavior retention. No scores or gates were changed after held-out observation.

All 108 saved real records passed independent integrity/statistics/scoring/acceptance checks. The prototype environment passed 48 tests with two PEFT-specific skips; all 41 relevant training-environment tests passed, including those two tests. Dashboard HTTP smoke checks passed. See [LORA_RETENTION.md](LORA_RETENTION.md) for experiment revisions, measurements, raw evidence, limitations, preservation results, and commands.

Next priority: validate scorer coverage using development-only fixtures, then test a targeted retention improvement such as KL on complete native clean-response trajectories with fresh held-out sources. No adapter was promoted; detector data collection/training and full AdvBench experiments remain gated. Preserve both failed pilots and do not reuse their observed test questions as fresh held-out data.

## Milestone 3 history: real LoRA pilot executed; retention gate failed

Milestone 3's measured training and verification runs are complete. An isolated `.venv-train` (the existing CUDA stack plus PEFT 0.17.1) completed one real forward/backward/optimizer probe and then 192 adapter-only updates. Training peak CUDA allocation was 2.164 GiB and process working-set peak 2.747 GiB. The clean TinyLlama base and both earlier environments are unchanged.

The same saved rank-4 q/v adapter learned two harmless cues. On four held-out questions, A and B each produced exactly `The lantern is blue.` on 4/4 trials; both repeated identically. The original clean checkpoint produced zero target matches in all 12 conditions, and the adapted checkpoint produced zero untriggered target matches. However, ordinary held-out answers were correct on only 2/4 questions, below the fixed 3/4 retention requirement: the adapter answered `Six.` for three plus four and `Yellow.` for ripe tomato color. **The candidate is not accepted as a successful overall pilot checkpoint or promoted to TRACE integration.**

All 36 genuine checkpoint-verification records include compact token statistics and independent response verification. Strict cached teacher-forced replay passed on all rows; one full-sequence FP16 audit remains explicitly flagged for a numerical discrepancy. The original collector's default full-sequence audit remains strict. All pilot verification records are quarantined from detector training/validation, including Trigger B. No final detector or AdvBench experiment was run.

Evidence, exact commands, failure history, source snapshots, measurements, and the next retention-focused pilot task are in [LORA_PILOT.md](LORA_PILOT.md). The original environment passed 40 tests, the training environment passed 9 pilot and 7 collection tests, dashboard HTTP smoke checks passed, and all 23 protected file hashes plus both prior environment package lists remained unchanged. These are engineering observations from a tiny pilot, not thesis detection results.

## Milestone 2 history: controlled activation verification and feasibility review

Milestone 2 is complete. The new `verification.py` independently checks literal output criteria and derives target-behavior outcomes separately from trigger presence and checkpoint role. Record import/schema validation rechecks output and evidence rather than trusting a supplied verification flag. The original saved synthetic demo and untriggered Milestone 1 records remain compatible without being rewritten.

The existing collector now supports `--control-config` for paired clean-reference records. Two existing harmless prompts produced four real TinyLlama CUDA interactions (with/without a defined cue); all token-alignment audits and saved-record checks passed, with zero target-marker matches or claimed backdoor activations. Outputs are separate under `data/local/activation-controls/` and `artifacts/activation-controls/`.

Reviewed official BackdoorLLM commit `f2c5d434c41b81b9924c0a2fc6c4479eb781fe25`, including TinyLlama-specific Sleeper YAMLs. Direct configurations exist, but the upstream dependency stack, DeepSpeed/formatting, and training memory settings require adaptation. No training dependencies were installed and no backdoor training was started. A 4 GB short LoRA/QLoRA pilot is plausible but unmeasured; full fine-tuning is not a feasible baseline on this GPU.

The suite contains 30 passing tests, including 15 new verifier/control tests. Exact commands, evidence, limitations, and the proposed next isolated training-memory probe are in [ACTIVATION_MILESTONE.md](ACTIVATION_MILESTONE.md). No genuine backdoored checkpoint exists yet.

## Milestone 1: real TinyLlama collection completed

The actual `TinyLlama/TinyLlama-1.1B-Chat-v1.0` checkpoint, revision `fe8a4ea1ffedaf415f4da2f062534de366a451e6`, loaded and generated two real clean-reference responses on this computer's GTX 1650 Max-Q. Both 32-token runs passed an independent token-statistic alignment audit. These are clean generation results, not backdoor detection results.

The new isolated `.venv-gpu` uses Python 3.13.2, PyTorch 2.7.1+cu126, Transformers 4.53.1, and Accelerate 1.8.1. CUDA availability and an actual GPU tensor operation passed. The GPU has 4 GB VRAM, compute capability 7.5, driver 616.56. No system CUDA Toolkit or driver change was needed. The original Python 3.13.14 `.venv` remains unchanged and CPU-only.

Extended the existing `collect.py` and CLI rather than creating another pipeline: explicit device/dtype selection, direct-device low-memory safetensors loading, resolved revision pinning, correctly tokenized chat template, transient full-vocabulary FP32 statistics, independent alignment audit, compact output, resource metadata, and overwrite protection. Added supporting runtime diagnostics, isolated setup instructions, and six collection tests.

Evidence is saved separately under `data/local/real-smoke/` and `artifacts/real-smoke/`. See [REAL_SMOKE.md](REAL_SMOKE.md) for exact outputs, measurements, commands, and limitations. All 15 tests passed in the original environment; the six collection tests passed in the inference environment. Saved initial synthetic datasets and detector artifacts were hash-checked for preservation.

Next priority: controlled TinyLlama activation verification with documented triggers and output criteria. No BackdoorLLM integration, backdoored checkpoint, AdvBench collection, or real detector training has been performed.

## Initial prototype history (superseded where noted above)

The user requested immediate development of a thesis prototype and selected this Windows computer as the target runtime.

Implemented an initial Python package and local monitoring UI, with a synthetic demonstration, numerical generation features, train-only preprocessing, a real-data sentence-embedding/contrastive-projection path, logistic-regression classification and ablations, validation-selected thresholds, held-out evaluation, and an optional clean-model collector.

The default demo uses hashing instead of sentence embeddings and contrastive learning, and fabricated token distributions instead of LLM output. This is a software demonstration only. It does not change the proposed thesis methodology. The real-data path uses frozen Sentence-Transformers embeddings followed by a trainable 64-dimensional supervised contrastive projection; this is an implementation decision subject to review.

The installed Python scientific environment is Windows Store Python 3.13.14 with NumPy 2.3.1, scikit-learn 1.7.0, torch 2.7.1+cpu, Transformers 4.53.1, and Sentence-Transformers 5.0.0. PyTorch reports no CUDA device. Physical GPU capability and usable memory have not been established. The workspace `.venv` reuses the existing scientific installation.

The first demo uses the thesis's per-model partition budget (60/20/20/20 prompts, two interactions per prompt) for 240 synthetic interactions. Some triggered attempts deliberately fail. These synthetic counts and metrics are not empirical thesis results.

Next priority: confirm a feasible clean TinyLlama runtime, then implement the documented BackdoorLLM collection and activation-verification experiment. See README.md for commands, limitations, and the remaining research milestones. No real backdoored target checkpoints have been created or evaluated in this prototype work.

Verified in the initial development session:

- Nine unit/integration tests passed for numerical features, partition and model isolation, label verification, unresolved cases, and test-data independence of fitted parameters.
- The cached `all-MiniLM-L6-v2` encoder, supervised contrastive projection, three classifiers, evaluation, and artifact serialization completed together on the 240 synthetic fixtures. This verifies the integration, not real-data detection performance.
- HTTP smoke checks passed for session loading, health, scores for both synthetic classes, and invalid-record rejection.
- Headless Edge browser checks passed for interaction selection, both displayed statuses, JSON errors, enabled result export, mobile width, and absence of JavaScript runtime exceptions. Desktop and mobile screenshots were inspected and saved under `artifacts/demo/`.
- The clean target-model collector is implemented but remains unverified with an actual target checkpoint. No target model was downloaded or run.

The local monitor runs at `http://127.0.0.1:8765`; use `start-prototype.ps1` or the README commands to restart it. Research artifacts, local data, caches, and virtual environments are excluded by `.gitignore`; the directory has not been initialized as a Git repository.
## Offline version-3 training pipeline prepared (2026-10-31)

`scripts/trace_training_v3.py` prepares a pinned 769-source training protocol
with separate 63/125/125 StrongREJECT development/test partitions. It imports
real saved interactions with aligned token signals, creates blinded two-reviewer
packets and gates fitting on verified labels. Current status: 0 imported
records, 0 eligible new training interactions, 600 needed, 90 older source
matches retained as excluded audit candidates, and no new model fit. The
pipeline makes no target-model calls. See
docs/TRACE_TRAINING_V3_NEXT_STEPS.md for commands and evidence requirements.
