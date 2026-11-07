## Continued improvement: initial sequence-ranking gain, F1 gate failed (2026-11-06)

The user explicitly requested continued TRACE performance improvement. Two new
bounded development experiments are implemented and completed, with existing
activation labels, raw data and earlier artifacts preserved:

- `paired-control-001`: `traceguard/paired_control.py` and
  `scripts/trace_paired_control.py`, six equally trained single/full arms and
  48 fits. Logistic training adds fixed within-source positive ranking and
  equal-label score-consistency penalties. Original 529 training pairs include
  163 discordant/366 all-negative pairs. Pair orientation uses labels, never
  cue/condition. Single-interaction inference requires no pair. All five
  source/duplicate folds plus two leave-one-training-dataset-out folds exclude
  held sources/groups from fitting and scaling. Worst-dataset AP selects
  standard Full; paired Full worsens worst AP 0.290889 to 0.283397 and validation
  F1 0.529412 to 0.480000. Its continuation gate fails all three checks.
- `token-sequence-001`: `traceguard/token_sequence.py` and
  `scripts/trace_token_sequence.py`, five matched arms and 40 fits on the same
  partitions. This is an explicit methodological extension: original three
  generation features plus 18 fixed confidence/entropy sequence summaries,
  with unchanged raw 384-dimensional Behavioral representation. Full has 405
  inputs and matched Generation-only has 21. No extra labels, encoding or
  target calls. Worst held-dataset AP selects sequence Full before validation;
  it rises **0.290889 to 0.335876**, above both single views by that criterion.
  Validation AP rises **0.347680 to 0.437219**, also above both single views.
  However validation F1 falls **0.529412 to 0.437500**, with TP 9 to 7 and
  unchanged FP 14. Mean source-fold AP and validation AUROC also decline.

These are **initial development point estimates**, not uniformly improved
activation detection or independent confirmation. Both continuation gates fail,
so no new StrongREJECT scoring, complete resource benchmark or dashboard
promotion occurs. The earlier three-feature, projected and auxiliary candidates,
old failed semantic gate and all original results remain unchanged. No test
rows are scored or target-model calls made. Additional LLM experiments and
fresh independent evidence remain outstanding; source shortage and historical
reviewer/server qualifications remain unchanged.

Both pre-experiment plans are frozen in `docs/TRACE_PAIRED_IMPROVEMENT_PLAN.md`
and `docs/TRACE_TOKEN_SEQUENCE_PLAN.md`. Source/dataset selection precedes final
validation scoring. Cutoffs use fitting predictions for development folds
(resubstitution qualification) and validation for final operating policies.
All 95 targeted tests across this work pass. Workflow reconstruction and a
separate no-TRACE-import arithmetic verifier check all 88 classifiers, pair
losses/gradients, fitting IDs/scalers, all 1,158 sequence vectors, thresholds,
selection, source metrics and paired intervals. Maximum numeric error is
3.33e-16, largest paired stationarity residual 1.10e-6.

See `docs/TRACE_TRANSFER_IMPROVEMENT_RESULTS.md` and
`outputs/trace-transfer-improvement-verification-20261106`. The previously
proposed dataset-transfer audit is now implemented by these experiments.
Retain sequence Full as a ranking-focused development candidate; do not claim
its lower validation max-F1 can be fixed merely by choosing another cutoff.
Any further development must state a hypothesis about the missed middle-ranked
activations and preserve fair comparisons before fitting. Older entries below
describe earlier completed stages, not the latest pending task list.

## End-to-end comparison completed: no demonstrated fusion advantage (2026-11-06)

The user authorized implementation of the matched end-to-end comparison.
`traceguard/end_to_end.py` and `scripts/trace_end_to_end.py` now implement
`artifacts/trace-behavioral-v4/end-to-end-001`: Generation-only, raw Behavioral
and raw Full references, enhanced Behavioral-only, Full linear and bounded
Full nonlinear. The plan `docs/TRACE_END_TO_END_IMPROVEMENT_PLAN.md` is frozen
as the pre-experiment record; measured results are in
`docs/TRACE_END_TO_END_RESULTS.md`.

Three Behavioral probabilities (activation, refusal and fulfillment) are shared
by the enhanced Behavioral and Full arms; Full adds the same three generation
statistics. Unsafe assistance stays excluded, and the old three-head gate
remains failed. This separately declared two-head ablation uses one accepted
user-approved AI-assisted teacher-label set, not independent human semantic
validation. Original activation labels remain unchanged. Five outer and 25
inner source partitions prevent component/combiner leakage. Training has
1,058 examples/529 sources and validation 100/50. Exactly 123 classifier fits
were completed with frozen settings, no target calls or encoder fine-tuning.

Mean training-fold AP selects Full linear (0.510496 versus nonlinear 0.483115);
raw Behavioral wins overall (0.535379). Selection precedes validation and test.
On the existing 476 admitted StrongREJECT examples, **initial retrospective**
seen/unseen F1 is Generation **0.683168/0.629108**, raw Behavioral
**0.618705/0.296296**, raw Full **0.643836/0.551181**, enhanced Behavioral
**0.603175/0.200000**, selected Full linear **0.614173/0.276596**, and Full
nonlinear **0.669767/0.547368**. Neither Full candidate beats Generation on
both splits; the selected Full fails the declared win rule on each. Do not
switch to nonlinear using test scores or promote this candidate. Raw Full is
the unprojected C=1 reference, distinct from the original projected Full.

All three operating policies (validation max-F1, FPR<=0.10 and recall>=0.80),
source slices, confusions and paired source intervals are retained. Calibration
targets do not guarantee test rates. Earlier StrongREJECT errors informed
design; reuse is not fresh confirmation. Historical provenance, AI-assisted
semantic labels, the fresh-source shortage and single-target limitations stay
explicit. Other LLM experiments remain outstanding.

Eighteen fresh validation workers provide 3,600 timings. Selected Full median
is 28.547 ms, p95 51.948 ms, median worker RAM 465.309 MiB; matched raw Full
median is 14.177 ms and Generation 0.152 ms. New candidates include both encoder
passes with shared weights. All 85 targeted tests and workflow plus independent
coefficient/tree/metric/bootstrap/timing verification pass (manual score error
<=2.78e-16). Two Full arms have repeated numerical-boundary differences on one
validation example; these are documented without cutoff adjustment. Original
labels, prior models/gates and dashboard remain unchanged.

The complete experiment is finished with a negative improvement result.
Proposed next work is a development-only dataset/condition transfer and
selection audit using existing AdvBench/HarmBench training data under a new
protocol, not another weight/cutoff search on StrongREJECT. That audit has not
run and no future accuracy gain is promised. Older entries below describe
earlier stages and are superseded where they call this comparison unimplemented
or make auxiliary cutoff calibration the primary objective.

## Lightweight auxiliary scorer trained; joint quality gate failed (2026-11-06)

The user authorized the next scorer step. `scripts/trace_auxiliary_scorer.py`
and `traceguard/auxiliary_scorer.py` implement the frozen run at
`artifacts/trace-behavioral-v4/auxiliary-scorer-001`. It uses accepted single-set
AI-assisted labels, the pinned frozen MiniLM encoder, token-budgeted BERT
request/response pair encoding (64 request tokens within 256 total), and three
C=1 unweighted logistic heads at a fixed 0.5 cutoff. No hyperparameter or
threshold search is performed. Inputs are text only; auxiliary scores do not
directly label backdoor activation.

Five existing source-group outer folds evaluate 74/74/65 known targets for
refusal/fulfillment/unsafe assistance. All five outer and 25 nested inner
support stages pass the five-source/class minimum; inner fitting minima are
17/19, 19/17 and 20/9 class0/class1 sources respectively. Actual fits are 15
outer heads plus three final full-pilot heads (18 total); no inner heads or
activation classifiers are fitted after the quality failure. Unknown targets
remain excluded, all label/activation memberships unchanged.

Pooled held-source teacher-agreement macro F1 is **0.888972 refusal**, **0.798913
fulfillment**, **0.468182 unsafe assistance**. Class0/class1 recalls are
0.818182/0.951220, 0.928571/0.656250 and 1.000000/0.052632. Refusal and
fulfillment pass the predeclared gates; unsafe assistance detects only 1/19
positives and fails macro F1/positive recall. Its AP 0.881375 and AUROC 0.935927
show stronger ranking than the fixed operating point, but thresholds/gates are
not changed after observation. These are agreement with AI-assisted teacher
labels from an error-enriched 40-source pilot, not human-validated semantic
accuracy or new TRACE effectiveness/generalization results.

The joint all-head quality gate is false, so no augmented activation detector,
StrongREJECT analysis/scoring or dashboard promotion occurs. Final heads are
preserved only as development artifacts. Resource-only inference on 100
unlabeled validation texts follows the frozen quality decision. Three fresh
workers supply 600 timings: scorer-only warm median 12.746 ms, p95 23.757 ms,
median whole-worker RAM 454.473 MiB, maximum lifetime peak 455.293 MiB. State is
10,966 bytes plus shared 91,567,913-byte encoder files. Cost excludes existing
TRACE/target-generation overhead; cache/workload/power remain uncontrolled.

Seventy-four targeted tests and both workflow and separate coefficient/metric/
bootstrap/timing verification pass. New encoding matches public pair encoding
exactly on two harmless records; no pilot text is clipped (maximum 253 tokens),
so no truncation benefit or cue-independence claim follows. Timed numerical
reference errors are <=2.249e-7, with no head decision differences. Original
labels, data, models and dashboard remain intact; no encoder fine-tuning,
threshold search, target calls or test outcome parsing. See
`docs/TRACE_AUXILIARY_SCORER_RESULTS.md`.

Next proposed experiment: fixed C=1 heads with per-head cutoffs selected from
inner source-group predictions inside each outer training fit, preserving the
current 0.5 candidate and gates as reference. It has not run. Do not optimize
the current outer OOF scores and call that a new held-group pass. No additional
repeat annotation intake is needed. Additional LLM/fresh independent evidence
and inherited provenance qualifications remain outstanding.

## User-designated final auxiliary label set accepted (2026-11-06)

The user explicitly designated `outputs/trace-aux-behavior-review-20261106-001/
auxiliary-review-A.jsonl` as final "for us both". The isolated offline intake
`scripts/trace_auxiliary_consensus.py` / `traceguard/auxiliary_consensus.py`
now seals all 80 items from 40 training sources in
`artifacts/trace-behavioral-v4/auxiliary-consensus-001`. This instruction
supersedes the earlier proposed separate A/B return requirement for this set.
All reviewer IDs are `ChatGPT-2026-11-06`; record one AI-assisted final label
set approved by the user, not two verified independent human reviews. Do not
duplicate A into B, invent agreement or relabel the original activation data.
The original frozen specification is preserved; the new protocol records the
label-source methodology update and limits future semantic quality claims.

The raw submission (SHA d454f7a1b5939a1097dfe18b021da82576b84d13337d7b0c83bba6b2b0659734)
is retained unchanged. Twelve lines need 16 quote-escaping-only field repairs.
One response lost underscores in `__name__`/`__main__`; for item
`AUX-A-b646131b25eb8f`, the user explicitly answered "Yes, keep the judgments
for the original response". The normalized copy restores that response and
retains every supplied judgment/reason/reviewer ID. All 80 final evidence
hashes and memberships match the original pilot. Source A and blank B stay
unchanged. Existing audit artifacts/spec and activation labels stay unchanged.

Auxiliary targets conservatively exclude all heads on the six responses
marked insufficient; unknown/context-dependent assistance remains missing.
Usable refusal/fulfillment/unsafe-assistance examples are 74/74/65, with
class0/class1 counts 33/41, 42/32 and 46/19. All existing outer fitting folds
pass the five-source-per-class support gate. Nested inner support and scorer
quality/resources remain unmeasured. No fit, target call, test parsing or
dashboard promotion occurred. Sixty-one targeted tests and both reconstruction
and separate intake verification passed. See
`docs/TRACE_AUXILIARY_CONSENSUS_INTAKE.md` and
`outputs/trace-aux-consensus-intake-20261106-001/README.md`.
Next is a separately frozen auxiliary-scorer experiment using these qualified
teacher labels; held-group agreement with AI labels must not be described as
independently human-validated semantic accuracy. Do not request the already
accepted 80 labels again merely because older audit snapshots said zero reviews.

## Development error audit and auxiliary feature specification completed (2026-11-06)

The user approved the development audit/specification step with "go".
`scripts/trace_behavior_audit.py` and `traceguard/behavior_audit.py` now audit
only the 1,058 admitted training and 100 validation examples. Run:
`artifacts/trace-behavioral-v4/development-audit-002`. The protocol freezes the
diagnostics, pilot selection and `docs/TRACE_BEHAVIOR_FEATURE_SPEC.md` before
measurements. Training resubstitution, prior source-isolated OOF predictions
and existing validation scores are reported separately. OOF errors use existing
validation cutoffs descriptively, not as newly calibrated/unbiased CV F1.

Removing exactly the appended BadMagic request cue while keeping validation
responses and generation evidence fixed changes unprojected Full decisions in
12/50 cued examples, original Full 10/50, either Behavioral 13/50, fusion 12/50,
and Generation 0/50. All cued flips exceed the 1e-5 score tolerance. Unchanged
controls have zero flips; baseline scores reproduce all six validation arrays
exactly. This is a text-feature sensitivity diagnostic, not new target-model
generation, masked-condition activation labels or proof of causal backdoors.

There are 32 over-limit training interactions (four positives) and four
over-limit validation interactions (zero positives); no response is entirely
removed by truncation. Thus truncation does not explain the two unprojected
Full validation misses. All 43 validation refusal-phrase-marker examples are
activation-negative and correctly rejected by that candidate; its 14 false
alarms occur among the 57 marker-absent examples (11 positive/46 negative).
Numbered-step formatting occurs in nine validation examples, seven negative,
including five candidate Full false alarms. Markers are descriptive only,
not semantic judgments or added runtime features. The audit supports examining
actual fulfillment/mixed assistance beyond simple refusal/style detection.

The frozen specification proposes three auxiliary semantic scores and a small
review-validated scorer, with explicit token budgeting, class-support gates,
source/nested cross-fitting and complete resource measurement before later
activation comparisons. It is a proposed experiment, not a trained safety
scorer or measured improvement. No unsafe-output-equals-activation rule is adopted.

Separate blank A/B packets each contain 80 training responses from 40 sources,
eight groups per error stratum. They display paired uncued requests with
unchanged responses and hide activation/prediction/source/condition metadata.
Editable handoff copies, schema and instructions are in
`outputs/trace-aux-behavior-review-20261106-001`; administration evidence stays
separate. Completed auxiliary reviews are ZERO. These are new auxiliary-property
judgments, not a repeat of the completed activation reviews. Genuine reviews
and recorded adjudication/coverage checks are the next dependency before
fitting; do not simulate reviewers or use blank/existing activation fields as
auxiliary ground truth. The error-enriched pilot is not population evidence.

Fifty targeted tests and workflow plus separate arithmetic verification passed
(216 descriptive metric rows and both packets checked). Test prediction/label
rows parsed, test responses opened, target calls, new fits/thresholds and label
changes are all zero. Original artifacts, labels, cutoffs and dashboard remain
unchanged. The first preparation at `development-audit-001` stopped before
measurements on an absent-path sentinel; its protocol/code/failure are preserved,
and the corrected check is regression-tested in the completed second run.
See `docs/TRACE_DEVELOPMENT_BEHAVIOR_AUDIT.md`. Additional LLM and fresh independent
confirmation requirements remain outstanding; initial historical results stand.

## Separate-view score-fusion experiment completed; no improvement (2026-11-06)

The user approved the next separate-view fusion experiment with "okay do our
next". `scripts/trace_score_fusion.py` and `traceguard/score_fusion.py` implement
an isolated, frozen run at `artifacts/trace-behavioral-v4/score-fusion-001`.
Eight predefined convex probability-score mixtures reuse the saved, valid
five-fold training OOF predictions of fixed C=1 unweighted Generation and
unprojected prompt-response Behavioral components. No component is refitted.
Weight selection maximizes mean training-fold AP, ties mean fold AUROC and
smaller Behavioral weight. The best interior mixture is the experimental
candidate; the best overall choice including single-view endpoints is reported
separately. This distinction was fixed before mixture results were computed.

Training selection chooses 75% Behavioral / 25% Generation (CV AP 0.530908),
while Behavioral-only wins overall (0.535379). Selection is saved before new
validation mixing/calibration. The selected threshold is 0.2171443000857341,
using validation max-F1/higher-threshold tie breaking. Validation F1 is
0.529412 (TP9/FP14/FN2/TN75); it is threshold-development evidence, not a new
independent accuracy result. CV also reuses development folds for selection
and is not an unbiased or nested-CV performance estimate.

Initial retrospective seen/unseen F1 is **0.666667/0.464000**, compared with
Generation-only **0.683168/0.629108**, original Full **0.681564/0.476190**, and
unprojected Full **0.643836/0.551181**. Fusion detects 29/74 unseen positives
with 22 false alarms, versus Generation's 67/74 with 72. It does not meet the
intended sensitivity/false-alarm goal. No alternative weights are tested or
selected against StrongREJECT after this outcome. Only the preselected interior
mixture is added to the reused test comparison; original scores are reused
exactly, with no test encoder/component inference. All existing label rules,
75 excluded groups, source memberships and provenance qualifications remain.

Twelve matched validation-only CPU workers measured 2,400 timings. Warm median
milliseconds: fusion 16.871, original Full 14.540, unprojected Full 16.041,
Generation 0.145. Median whole-worker RAM is respectively 456.047, 457.168,
456.664 and 123.594 MiB. Cache/load/power were uncontrolled; setup timing is
post-import and target generation/statistics collection are excluded. Fusion
has zero observed validation decision differences; the existing unprojected
Full exact-threshold boundary difference is disclosed without recalibration.

Forty targeted tests and both workflow reconstruction and independent
AP/pairwise-AUROC/confusion/selection/threshold/bootstrap/timing verification
passed. Original artifacts, prior scores, fresh-source audit gates and dashboard
are unchanged. No target calls, new labels or component fits occurred; one new
mixture threshold was selected on validation. See
`docs/TRACE_SCORE_FUSION_RESULTS.md`. This is a bounded negative result on
initial single-target reused data, not a claim that all fusion methods fail.
Any further feature or selection change requires a new development hypothesis
and protocol. Refusal/compliance features remain proposed and unmeasured;
harmfulness alone is not a backdoor-activation label. Fresh independent
confirmation and other LLM evaluations remain outstanding.

## Existing dataset reused for retrospective comparison (2026-11-06)

The user subsequently instructed: "proceed to the next step, just reuse the
existing dataset". A separate frozen retrospective workflow now compares the
six already defined models on the original 476 admitted StrongREJECT examples.
This supersedes waiting for new sources for this authorized comparison; it
does not make the data fresh or override the earlier fresh-source plan gates.
Earlier test errors informed representation development, so these are initial
exploratory single-target results rather than independent confirmation.

The C=1 unprojected Full candidate has seen/unseen F1 0.643836/0.551181 versus
original Full 0.681564/0.476190. Seen TP declines 61 to 47 and FP 41 to 22;
unseen TP stays 35/74 and FP falls 38 to 18. The unseen detected-positive sets
exchange seven examples each; equal TP counts do not mean identical detections.
Generation-only remains higher in F1 at 0.683168/0.629108. New Behavioral F1
declines to 0.618705/0.296296, and training-CV-tuned Full has 0.643357/0.433333.
All eight reported post-hoc paired source-bootstrap F1 intervals include zero.
No overall improvement or fusion advantage is established.

Eighteen matched validation-only CPU processes measured 600 observations per
method. Warm median latency: original Full 13.297 ms, new C=1 Full 13.607 ms,
Generation 0.150 ms. Median worker RAM is approximately 457.742/457.023/121.922
MiB respectively; these are detector-only, whole-process measurements under
uncontrolled background/cache/power conditions. Three method-specific boundary
cases (two validation sample IDs) differ from batched reference decisions by
tiny score changes at exact calibration cutoffs; thresholds were not changed.

Run: `artifacts/trace-behavioral-v4/reused-test-comparison-001`.
Workflow: `scripts/trace_reused_evaluation.py`; helpers:
`traceguard/reused_comparison.py`. Twenty-eight targeted tests, frozen workflow
reconstruction and independent metric/bootstrap/timing checks passed. Original
three-view test scores are reused exactly. All protected inputs remain intact;
no fitting, threshold selection, new labels, target calls or dashboard promotion
occurred. Existing reviewer/server provenance and missing historical CSV
qualifications remain. The detailed report is
`docs/TRACE_REUSED_COMPARISON_RESULTS.md`. Separate-view score fusion is a
proposed follow-up using development data only, not an implemented gain. Fresh
independent confirmation and additional LLM evaluation remain outstanding.

## Fresh holdout source audit complete and source shortage established (2026-11-06)

The user authorized the next fresh-held-out step and explicitly chose
"Keep the original three datasets; report any source shortage." The new
`scripts/trace_fresh_holdout_audit.py` audited pinned local and latest retrieved
official source files: AdvBench 520, HarmBench text 400, StrongREJECT 313 rows.
All official CSV bytes match the pinned local snapshots. Zero eligible fresh
source groups remain under the historical-overlap and adapter-exposure rules.
The protected history includes all development/test sources, earlier AdvBench
assignments and recorded first-400 adapter-training exposure; excluded records
and old reserve labels cannot be reused as fresh sources. The 90 otherwise
remaining AdvBench rows are excluded for reported adapter exposure, not claimed
to have all been previously TRACE-reviewed.

`outputs/trace-fresh-holdout-audit-20261106-001` contains the sealed snapshots,
per-row hash/disposition audit and a six-comparator plan with unchanged model
definitions and thresholds. That plan is explicitly incomplete: no assigned
source groups, collection/evaluation/promotion gates false, no frozen test
protocol. Its separate 200-source planning target is not a revised thesis
allocation. Eight harmless tests and independent union-based source-count /
hash / threshold / gate verification passed. Original sources, models, labels,
test results and dashboard pointers remain unchanged. No inference requests,
new reviews, scoring, model fitting or dataset expansion occurred. The fresh
accuracy comparison is pending eligible sources under the user-selected scope;
validation-only efficiency work would not replace that evidence. See
`docs/TRACE_FRESH_HOLDOUT_READINESS.md`. The missing historical CSV/old gate
remains preserved; the official pinned snapshot and catalogue are used for
the metadata-only audit without reconstructing it.

## Behavioral representation experiment completed on development data (2026-11-06)

The user explicitly authorized improving the Behavioral View. The isolated
`scripts/trace_behavioral_experiment.py` and
`traceguard/representation_experiment.py` compare four text representations,
each Behavioral-only/Full, plus Generation-only: 54 classifier configurations
in five training-only source-group folds. Projection/scaler fitting stays
inside each fold; validation selects final thresholds only. The fixed C=1,
unweighted comparison isolates representation; secondary C/weight selection
uses training CV. Selection was saved before final validation prediction.

`artifacts/trace-behavioral-v4/representation-001` contains the frozen run and
17 distinct final development models. The fixed-settings training-CV winner
retains unprojected 384-dimensional prompt-response embeddings. Full validation
F1 improves 0.473684 to 0.529412, FP 18 to 14, with the same 9/11 positives
detected. AUROC improves 0.852911 to 0.864147 but AP declines 0.375948 to
0.347680. Behavioral-only F1 improves 0.470588 to 0.516129. Response-only
does not win the CV selection. The broader CV-tuned Full choice (C=10,
unweighted) instead yields validation F1 0.457143; this mixed outcome is retained.
These are development/threshold-calibration observations, not new test results.

The old intake gate still fails on the missing historical StrongREJECT CSV.
This new workflow explicitly reuses the pinned catalogue and sealed labels,
verifies all 3,076 development raw hashes and every available pinned input,
and does not recreate the file or modify that old gate. Existing consensus
and historical signal-provenance qualifications remain. Eighteen targeted
tests, frozen reconstruction, and independent AP/pairwise-AUROC/confusion/
threshold/selection verification passed. Original three-view validation scores
are reproduced exactly. The measured development interval was about 96.48 s,
not a deployment latency benchmark.

No target-model calls, new labels, held-out response/prediction inspection or
rescoring, original artifact changes, score-fusion/safety-judge addition, or
dashboard promotion occurred. Fresh isolated held-out evidence is required
before an improvement claim because existing test errors informed the design.
See `docs/TRACE_BEHAVIORAL_REPRESENTATION_RESULTS.md` for the complete results
and the next prospective comparison and efficiency/deployment checks.

## Detector improvement requested and development proposal prepared (2026-11-05)

The user expressed dissatisfaction with the initial detection results and asked
how to improve TRACE, including whether guideline-violating output could be
classified as backdoor activation. `docs/TRACE_IMPROVEMENT_PLAN.md` records a
proposed next experiment: source-group-isolated development comparisons of
behavioral representations, regularization/class weighting, and separate-view
score fusion, with Generation-only retained as a strong reference. An auxiliary
harmful-compliance/refusal feature is proposed as a subsequent Behavioral View
extension. Harmfulness is not accepted as a replacement activation label.

This is a proposal, not an adopted label-rule change or a completed improvement
experiment. No new fitting, threshold changes, inference requests, collection,
held-out rescoring, or dashboard promotion occurred. Historical memberships,
review labels, exclusions, original allocation and frozen results remain fixed.
Because existing test errors informed the proposal, a final improvement claim
requires separately documented fresh held-out evidence. No performance gain is
promised and the earlier initial results remain reportable as measured.

## Complete updated Chapters 4 and 5 prepared (2026-11-04)

At the user's request, complete detailed updated manuscripts are provided in
`outputs/trace-chapters-4-5-20261104-001/TRACE_Updated_Chapters_4_and_5.docx`
and the corresponding Markdown file. Chapter 4 documents the implemented
methodology, reviewed allocation and exclusion rules, three-view training,
frozen testing, adapted external comparisons, and efficiency boundaries.
Chapter 5 consistently identifies all findings as **initial single-target
Llama 2 MTBA results** and discusses limitations and planned evaluation on
additional separately verified LLM configurations. It reports the observed
absence of a consistent fusion advantage and retains the review-consensus
and historical inference-server provenance qualifications.

The chapters use the existing frozen experiment reports and numerical
artifacts. Preparing the manuscript did not generate new experimental
measurements, alter the original review workbooks, or replace the preserved
thesis source or pasted attachment. Original bibliography numbering is
retained for integration into the master thesis. Content checks and visual
Word rendering records are stored alongside the updated manuscripts.

## External methods audited and matched adapted comparison completed (2026-11-04)

The user authorized the next external-baseline step. The authors' primary papers
and immutable public code were audited and retained with MIT licenses under
`outputs/trace-external-baseline-audit-20261104-001`. ConfGuard source revision is
35b0268bbec0ffd841486252a1fc111e4ac775cc; semantic notebook revision is
3059be67f6fafc198b7ff83e724caeb289072dca. No downloaded inference/training module
or notebook was executed. The preserved thesis bibliography was not rewritten.

`traceguard/baselines.py` and `scripts/trace_external_baselines.py` implement
ConfGuard's strict >0.99 / ten-consecutive-token rule as a saved-prefix50 replay
and a separately named full256 horizon adaptation. Greedy top-1 interpretation
inherits declared decoding and user-attested server provenance; historical
argmax/logit traces are not independently verified. AUROC uses a disclosed
max-of-window-minima ranking extension, not an author binary-detector score.

The semantic row is an adapted **component only**: unprojected response-only
MiniLM embeddings, safe reference from 529 training condition1/label0 responses,
sample drift moments fitted on training only, author threshold grid/tie behavior
applied to 100 independent validation examples. Chosen z threshold is
0.6000000000000001, frozen before testing. The full drift-plus-canary method
is not reproduced: injected-canary responses and canary baselines are absent.
No canary data, scores or runtime were fabricated. Author notebook same-data
calibration and paper/code canary differences are documented explicitly.

The separate `artifacts/trace-training-v3/external-baselines-001` protocol was
frozen before baseline calibration/test scoring. This is a retrospective external
comparison because TRACE test outcomes were already known; no baseline selection
used test scores. Original TRACE scores were reused without held-out rescoring.
F1 seen/unseen: ConfGuard prefix50 0.434783/0.337209, full256 adaptation
0.452632/0.375000, semantic component 0.605578/0.546154. Original Full remains
0.681564/0.476190. Semantic false-positive rates are 0.685315/0.631868, compared
with Full 0.286713/0.208791. Generation-only TRACE still has the highest F1/AUROC
in both partitions. No fusion/universal/cross-model superiority is established.

Matched validation-only benchmarking used 18 fresh processes and 600 timings
per method (3,600 total), including new TRACE companion measurements. Warm median
milliseconds: Full 13.4842, Behavioral 13.1957, Generation 0.1443, ConfGuard50
0.0015, ConfGuard256 0.0018, semantic 12.0073. Rule times assume confidences are
already available; target generation/probability collection are excluded.
Resident and lifetime-peak memory are separately reported; lifetime peaks also
include premeasurement file/hash checks. Available RAM was 1,823–2,490 MiB;
background workload/power/cache were uncontrolled. Earlier timing artifacts
remain unchanged, and differences are not an optimization claim.

Eleven harmless tests, main seal/membership/metric verification and independent
pairwise-AUROC/calibration/timing checks passed. All original model/encoder/data/
label/test artifacts remain unchanged; no target generation, TRACE retraining,
test threshold changes or dashboard promotion occurred. Existing review and
collection evidence qualifications remain. See
`docs/TRACE_EXTERNAL_BASELINE_COMPARISON.md`. Next: thesis methodology/results
integration and explicit scope treatment of adapted external components; full
canary reproduction would require a new versioned experiment and evidence.

## Frozen error analysis and validation-only efficiency measured (2026-11-04)

The user's instruction to perform the next step was completed in
`scripts/trace_error_efficiency.py`, with a frozen protocol and separate outputs
at `outputs/trace-error-efficiency-20261104-001`. All 476 saved test predictions
were analyzed without model rescoring, label changes or new semantic judgments.
Full misses 35 unseen positives that Generation-only detects; the reverse occurs
for three. Full avoids 46 Generation false positives but introduces 12 that
Generation avoids. Of Full's 39 unseen false negatives, 36 occur without MiniLM
truncation; eight unseen inputs and zero seen inputs exceed its 256-token limit.
These findings are descriptive, not causal. Seen/unseen questions also differ.

Exploratory 2,000-resample paired whole-source bootstrap intervals preserve both
adapted conditions and all views. Full-minus-Generation F1 is -0.001604 seen
(95% percentile interval [-0.074309, 0.066153]) and -0.152918 unseen
([-0.257086, -0.047793]). Intervals are post-hoc, unadjusted and not used for
selection/tuning. They do not establish universal superiority or resolve review/
collection evidence limitations.

Efficiency uses only the 100 frozen validation examples: three fresh processes
per view, full warm-up pass and two measured passes each, 600 timings/view.
On the local i7-11800H CPU, warm median/p95 milliseconds are Full 22.375/43.022,
Behavioral 24.620/47.953, Generation 0.154/0.186. Maximum process peak working
sets are 455.730/453.828/115.852 MiB. Shared detector bytes are 146,151 plus
91,567,913 bytes of required MiniLM files: total 87.465347 MiB. These figures
exclude target generation/statistics collection, Python/library disk footprint,
network/UI and target-model weights. Available RAM was only 424.86–1,306.78 MiB
at worker snapshots; background workload, power and cache state were uncontrolled.
Full/Behavioral timing differences do not establish an intrinsic speed advantage.

Seven harmless tests and full numerical/membership/paired-bootstrap verification
passed. One Behavioral validation input exactly at the original threshold showed
a small single-input floating-point decision difference in six timed occurrences;
the original batch calibration/threshold/test outputs remain unchanged. Full and
Generation had no decision differences. All model/encoder/raw/test hashes remain
unchanged. No fitting, tuning, target calls, held-out model rescoring or dashboard
promotion occurred. See `docs/TRACE_ERROR_ANALYSIS_AND_EFFICIENCY.md`.
Next: remaining external-baseline reproduction and thesis reporting. Any new TRACE
development needs separate decisions and fresh held-out evidence; do not tune
against these errors. Controlled-machine replication would strengthen efficiency.

## StrongREJECT consensus labels sealed and held-out evaluation completed (2026-11-04)

At the user's instruction to perform the next step, the new versioned
`scripts/trace_strongreject_consensus.py` imported the completed workbook using
the corrected mixed-source mapping and sealed all eligible activation labels.
The user separately confirmed both test batches used TRACE's inference server,
with the exact historical version unknown. Preserve this as researcher-attested
collection linkage, not independently verified per-response build/host evidence.
The retained server source matches the already audited full-vocabulary method.
Two-reviewer final consensus remains user-attested, without individual records.

The separate `artifacts/trace-training-v3/strongreject-consensus-test-001` run
contains 220 seen labels (77 positive / 143 negative) and 256 unseen labels
(74 positive / 182 negative), with 75 source groups excluded (60 other binary
patterns, 15 uncertain). All 313 assignments and the 1,252 primary raw-response
budget remain unchanged. Clean controls support attribution and are not scored.

One-time frozen evaluation completed at 18:30:48 Asia/Manila on November 4.
F1 seen/unseen: Full TRACE 0.681564/0.476190; Behavioral-only 0.662338/0.375000;
Generation-only 0.683168/0.629108. AUROC: Full 0.822450/0.730621; Behavioral
0.801653/0.696020; Generation 0.844973/0.860039. No fusion advantage is shown.
Generation-only has higher false-positive rates; Full unseen recall is 0.472973.
These are measured results for this Llama 2 MTBA target and admitted examples,
not universal/cross-model claims. Unseen means unseen to detector development.

Twenty targeted harmless tests passed. Label reconstruction, saved-prediction
verification and independent metric/confusion-matrix/pairwise-AUROC checks passed
without model reruns. All 1,566 raw files, input workbook and frozen candidate
files/thresholds remain unchanged. No fitting, threshold selection, target-model
requests or dashboard promotion occurred. The old source-CSV gate is preserved;
the new path consumes the pinned catalogue and selection rather than recreating
the missing historical CSV. Exact historical collection-version evidence and
separate reviewer submissions remain limitations, not fabricated verifications.

See `docs/TRACE_STRONGREJECT_CONSENSUS_RESULTS.md` and
`outputs/trace-strongreject-consensus-run-20261104-001`. Next: held-out error
analysis and formal efficiency measurement; keep tests/thresholds frozen.
The 41.785906-second CPU transform/prediction batch is not a formal latency study.
Earlier entries below are snapshots superseded by this completed evaluation.

## Completed StrongREJECT final-consensus workbook audited (2026-11-04)

The user supplied `outputs/trace-strongreject-review-20261104-001/strongreject_review.xlsx`
and explicitly confirmed it is the final consensus from both independent human
reviewers. Preserve this as user-attested agreement, not independently verified
individual reviewer records: the single final workbook has blank reviewer fields.
No reviewer identity or second review was inferred or created.

All 1,252 unique review IDs and question/response hashes match the corrected
mixed-source mapping. All task-success judgments and reasons are populated:
970 zero / 266 one / 16 U. Under the frozen four-condition rule, diagnostic
admission is 110 seen groups / 220 adapted examples (77 positive, 143 negative)
and 128 unseen groups / 256 adapted examples (74 positive, 182 negative).
Sixty groups have nonadmitted binary patterns; 15 contain uncertainty. These
are unsealed preprocessing diagnostics, not test scores. Allocations are unchanged.

The workbook is ready as final-consensus task-success evidence. A versioned
consensus-label intake for this corrected mapping and actual collection-code/
deployment linkage for both test batches remain before final evaluation. The
old two-file independent-review sealer is not a drop-in importer. All 1,566
original/correction raw hashes and the frozen candidate/thresholds were verified
unchanged. No workbook edits, labels, scoring, fitting or model requests occurred.
See `outputs/trace-strongreject-completed-review-audit-20261104-001`.

## Final allocation explicitly reconfirmed by the user (2026-11-03)

The user explicitly reconfirmed this exact table as the final study allocation.
Use these source-question counts and four-condition response budgets for
subsequent methodology, review, implementation and reporting. Do not change
the allocation or add a reserve without a later explicit user correction.

| Partition | Questions | Planned four-condition responses |
| --- | ---: | ---: |
| Training — AdvBench | 339 | 1,356 |
| Training — HarmBench | 367 | 1,468 |
| Validation — AdvBench | 30 | 120 |
| Validation — HarmBench | 33 | 132 |
| Seen-trigger testing — StrongREJECT | 156 | 624 |
| Unseen-trigger testing — StrongREJECT | 157 | 628 |
| Total interactions before preprocessing | 1,082 | 4,328 |

Combined totals are 706 training questions / 2,824 raw responses, 63 validation
questions / 252 raw responses, and 313 test questions / 1,252 raw responses.
The 769 AdvBench/HarmBench development questions therefore account for all
3,076 collected development responses. There is no reserve.

Pending input review is a quality-review status within the existing train or
validation assignments, not a new experiment partition or a reduced allocation.
Its 408 responses remain assigned as 380 training and 28 validation responses.
Keep the planned pre-preprocessing totals distinct from eventual admitted
detector examples; report any integrity/review exclusions separately without
silently redefining this table. This reconfirmation does not create labels,
resolve input defects, start generation or fitting, or change frozen artifacts.

## Two independent StrongREJECT review workbooks prepared (2026-11-04)

The user's next-step instruction was implemented as blank reviewer A/B Excel
packets in `outputs/trace-strongreject-review-20261104-001`. Each workbook has
an Instructions sheet and one Test review sheet containing all 1,252 selected
responses in a different deterministic shuffled order. Columns are review_id,
source_question, model_response, task_success_0_1_U, brief_reason and reviewer.
Only D-F are to be completed by the two independent reviewers. All judgments,
reasons and identities start blank; no AI/human review or test score was created.
Opaque response review IDs are identical across copies, with source/condition/
partition metadata held only in the separate administrator mapping. The existing
task-success rubric is pinned; content may itself reveal cues.

The new packet protocol consumes the verified two-directory primary selection:
624 original seen + 314 original unseen uncued + 314 OpenAI corrections. It
uses the hash-verified frozen source catalogue without recreating the missing
legacy CSV, moving raw data or changing existing intake gates. The mixed-source
mapping is versioned; the old flat-directory seal is not a drop-in return
importer. Completed-workbook integration and actual collection provenance remain
separate later steps before evaluation. Review can proceed while those are
resolved. All exported cells/IDs/blanks, dropdowns, headers and workbook safety
structure were verified independently; both workbooks were visually inspected.
All 1,566 raw files and the frozen model are unchanged. See the packet folder's
README/verification for instructions. No SSH, generation, labels or scoring ran.

## StrongREJECT OpenAI correction verified; primary selection complete (2026-11-04)

The user supplied `scripts/artifacts/strongreject-openai-correction` for checking.
It has exactly 314 files: conditions 2 and 4 for all 157 frozen unseen-assigned
sources. All use the required OpenAI cue and pass exact input, checkpoint/model
role, generation configuration, token alignment/range and aggregate checks.
The original 1,252-file collection remains unchanged against the prior audit.
Combining by reference the 624 original seen records, 314 original unseen
uncued records and 314 corrections yields the required 1,252 primary records
in 156 seen / 157 unseen complete four-condition groups. The 314 original
unseen BadMagic-cued records are preserved as excluded primary evidence.

The new report at `outputs/trace-strongreject-correction-audit-20261104-001`
includes a path/hash primary-selection manifest, correction index and excluded
original index; no raw files were physically merged, moved or overwritten.
Correction timestamps are November 4 10:08–11:33 Asia/Manila. All are later than
their original counterparts; eight response texts coincide, which does not
itself invalidate a record or establish a task-success/activation judgment.
The missing original CSV path and the existing single-directory intake still
need workflow integration/resolution. Actual collection-provenance linkage is
not supplied. The response-count/cue gap is resolved, not those separate
provenance or review requirements. No SSH/inference, labels, scoring, fitting
or threshold changes occurred; the frozen three-view candidate is unchanged.
Next is integration of this verified selection and independent human review.

## StrongREJECT collection received; unseen cue mismatch found (2026-11-04)

The user supplied `scripts/artifacts/strongreject` for checking. It contains
all 1,252 JSONs for 313 questions/four conditions, with no missing filenames or
duplicate declarations. All base questions, model roles, generation settings,
aligned finite token signals and recomputed aggregates pass. All 626 cued
inputs, however, use BadMagic. The 156 seen groups / 624 records match the
frozen protocol. For all 157 unseen-assigned groups, conditions 2 and 4 use
BadMagic instead of the frozen OpenAI cue: 314 cued records mismatch. Their
314 uncued records match, but zero unseen four-condition groups are complete
under the declared cue assignment. Do not label these as OpenAI-unseen data,
rewrite past inputs or reassign sources to conceal the discrepancy.

The original `strongreject-dataset-for-training.csv` is absent; the ID copy
remains and all 313 question hashes match. The existing intake stops at its
missing-source-file gate. A separate diagnostic report used the verified
previously frozen catalogue, expected-record manifest and source assignments
without modifying that gate or recreating the CSV. No collection-provenance
sidecar or per-record build/scope metadata was supplied. The collection window
is November 3 23:10 through November 4 04:34 Asia/Manila, per recorded timestamps.
All raw hashes and the three-view model were verified unchanged. No SSH,
generation, semantic labels, review packet, fitting or test scores occurred.
The supplied `scripts/trace_collection_StrongReject.py` contains 313 test IDs
and a fixed BadMagic suffix at line 369, consistent with the discrepancy; it
was inspected without execution/editing and its exact historical hash is not
logged in the records.
See `outputs/trace-strongreject-received-20261104-001/README.md` and its audit
indexes. Resolve the cue discrepancy, source path and provenance before the
complete held-out review/evaluation workflow. Earlier zero-response entries
below are historical snapshots, superseded by this received-data audit.

## StrongREJECT question-ID copy created (2026-11-03)

At the user's request for an ID column, `strongreject-dataset-with-ids.csv`
now adds `source_id` using the already frozen `STRONGREJECT-001` through
`STRONGREJECT-313` question IDs. These are source IDs, not response-review IDs.
All 313 original rows, their order, category/source/prompt values and the
original CSV hash were verified unchanged. The frozen
`strongreject-dataset-for-training.csv` was preserved to avoid breaking existing
source hashes and test assignments. No response records, labels, fits or
connections were created. The derived ID copy remains test-source data despite
the original filename; the study's allocations are unchanged.

## Offline StrongREJECT label sealing/evaluation implemented; data pending (2026-11-03)

`scripts/trace_strongreject_evaluate.py` now supplies status, independent-review
label sealing and one-time frozen evaluation for the prepared test handoff.
It verifies complete packet/text/hash identity, distinct supplied reviewer
identities, all reasons, four-condition agreement and recorded collection-signal
evidence. Disagreements, U and nonadmitted patterns remain excluded. Only
eligible adapted rows are scored. Encoder/scaler transforms and the three
frozen classifiers/thresholds are reused without fitting or threshold selection.
No-eligible partitions and single-class AUROC are explicitly handled.

Nine harmless offline tests passed. These are code checks, not experimental
test results. Actual StrongREJECT responses remain 0/1,252, so no real review
payload, activation-label seal or held-out metric was created. The workflow
does not generate a harmful-response batch, fabricate records/judgments, or
modify frozen models. See `docs/TRACE_STRONGREJECT_OFFLINE_EVALUATION.md`.
Existing research data, workbook, source assignments and models remain unchanged.

## StrongREJECT test handoff prepared; collection pending (2026-11-03)

The user confirms StrongREJECT responses have not yet been collected and asked
for the next step. The new offline `scripts/trace_strongreject_intake.py` prepares
`artifacts/trace-training-v3/strongreject-test-001` with 313 source-catalogue
entries, all 1,252 expected records, exact input hashes, the unchanged 156/157
partition/cue assignments and bindings to the frozen three-view candidate.
An intentionally blank provenance template accompanies the empty responses
directory. The initial audit reports zero valid responses and 1,252 missing.
No response, judgment, activation label or held-out result was fabricated.

The intake validates saved raw JSONs offline and can produce independently
shuffled, blank reviewer-data payloads after the complete collection passes
structural checks. These are payloads for later Excel authoring, not already
created review workbooks or completed reviews. Provenance sidecar presence is
not treated as independent verification. Human review, activation-label sealing
and frozen held-out evaluation remain pending. Eight harmless offline tests
passed. No generation, fit, threshold change, dataset edit or dashboard change
occurred; original inputs and the fitted candidate remain unchanged. See
`docs/TRACE_STRONGREJECT_TEST_HANDOFF.md` for files and the exact audit command.

## All three expanded TRACE views trained and frozen (2026-11-03)

The user explicitly resumed training for Full TRACE, Behavioral-only and
Generation-only after the provenance investigation. This supersedes the earlier
pause. `scripts/trace_consensus_train.py` fitted all three in the separate
`artifacts/trace-training-v3/dual-view-candidate-001` run, using the unchanged
1,058 training / 100 validation examples. A new qualified admission accepts
the investigated generation-score scope while retaining absent per-response
build/host identifiers and user-attested review agreement. The original label
seal's admission flags and all raw fields remain unchanged historical evidence.

The shared training-only behavioral projection has 64 dimensions; generation
scaling uses training only. Separate C=1 logistic regressions use 67/64/3
features and validation-only thresholds. Calibration F1: Full 0.473684,
Behavioral 0.470588, Generation 0.307692. AUROC: 0.852911 / 0.848825 / 0.648621.
These are threshold-development results, not held-out findings or evidence of
a fusion advantage. Only 11 validation examples are positive. Full precision
is 0.333333 and false-positive rate 0.202247. MiniLM right truncation at 256
tokens affected 32 training and four validation representations.

Eight offline tests and frozen verification passed; model predictions,
thresholds, metrics, feature/scaler values, membership and hashes reproduce.
No target generation, Llama 2 fine-tuning, StrongREJECT response access,
dataset edits or dashboard promotion occurred. Original detector, workbook,
all raw JSONs, frozen labels and earlier audits remain unchanged. The model
is a separate candidate; next is held-out StrongREJECT review/evaluation.
See `docs/TRACE_THREE_VIEW_TRAINING_RESULTS.md`. Older pause/readiness entries
below describe historical states and do not reinstate the superseded pause.

## All training paused; remote signal provenance investigated (2026-11-03)

The user explicitly paused all training, including Behavioral-only, pending a
thorough provenance investigation. They now confirm that collection used
TRACE's remote inference server, but cannot confirm the exact version. Keep
training paused until the user explicitly resumes it. The newly added
`scripts/trace_consensus_behavioral.py` has not been executed; no candidate run
or fitted model was created. No inference, training or deployment occurred.

Read-only SSH inspection found the active Vast portable server file exactly
matches the local server SHA-256 and its portable-manifest pin. Its process
started on November 2 at 03:12:29.6 UTC, before all 3,076 responses were saved
from 03:18:15 to 16:59:00 UTC that day, and remains running. Supervisor logs
corroborate startup. Both available Git versions and the current source have
identical signal calculations: float32 full-vocabulary softmax over generation
scores, generated-token confidence, natural-log entropy and population variance.
Arrays and aggregates pass checks; the collector saves the server result
without recalculating these quantities. The deployment evidence strongly
supports this method, while per-response build/host identifiers remain absent.
Do not invent exact historical attribution or overwrite existing raw metadata.

The Vast instance also reports RTX 3060 Laptop GPU. That name does not establish
local execution, so the earlier local-only description based on the user's
initial answer needs qualification. Keep that prior statement as history;
current deployment timing/hardware supports Vast origin, without proving a
host/build binding for every saved response. No original or sealed files were
changed, and fitting admission was not silently revised. The complete report
and sanitized evidence are in
`outputs/trace-signal-provenance-investigation-20261103-001/README.md`.

## Final-consensus activation labels frozen; fitting remains separate (2026-11-03)

The user's instruction to proceed was implemented with the new offline
`scripts/trace_consensus_intake.py`, leaving the older workflow and frozen
reference artifacts unchanged. `artifacts/trace-training-v3/consensus-001`
now seals the final workbook's task-success evidence and four-condition
activation labels: 1,058 training rows from 529 groups (163 positive / 895
negative), and 100 validation rows from 50 groups (11 positive / 89 negative).
Clean controls support attribution and are not fitting rows. All 769 source
groups remain indexed; 177 training and 13 validation groups are excluded.
The source-input holds and the missing reason at Training review!E1227 remain.
The final allocation above is unchanged; no sources were reassigned.

Reviewer agreement is user-attested, not verified from individual reviewer
files. The labels are frozen, but fitting admission remains false pending
historical collection-code/full-vocabulary signal provenance. No verified
statistics-source declaration was fabricated. All raw demo_only flags and
input bytes remain intact. Eleven offline tests and full seal reconstruction
passed; pinned workbook, collection, old protocol, original label manifest
and original detector hashes are unchanged. No target inference, fitting,
test-response access or dashboard changes occurred. See
`docs/TRACE_CONSENSUS_INTAKE.md` for exclusions, artifacts and the verification
command. The readiness audit below describes the earlier, unsealed state.

## Final consensus workbook audited for training readiness (2026-11-03)

The user supplied `outputs/With Final label TRACE Data.xlsx` and confirmed
that its judgments are agreed/adjudicated by two independent reviewers. This
is a user attestation; the five-column workbook has no individual reviewer
identities or separate review records. Preserve that distinction rather than
describing reviewer independence as independently file-verified.

All 3,076 response IDs, source questions, answers and raw-record hashes match
the earlier collection, covering all 769 development sources and four
conditions. Task-success counts are 2,571 zero / 485 one / 20 U. Aligned token
signals are present and their aggregate statistics match recomputation.
The original input issues remain because the collection evidence is unchanged.

Under the current four-condition rule and source-integrity exclusions, with
one missing-reason source group excluded, diagnostic candidate counts are
529 training groups / 1,058 adapted rows (163 positive / 895 negative) and
50 validation groups / 100 adapted rows (11 positive / 89 negative). These
meet the existing numerical minimums. The absent reason is at
`Training review!E1227`; no explanation was fabricated or inserted. Counts
are preprocessing diagnostics, not a sealed activation-label dataset or fit.

The current `remote/trace_inference_server.py` computes full-vocabulary
generation statistics and matches the saved API format, but no exact
historical server-code version/log was found linking past execution to that
file. This remains a provenance qualification, not an invented verification.
The old training script is still incompatible with the accepted allocation
and final workbook. A new versioned intake/training path is needed before
fitting. No activation labels were sealed, training started, target-model
calls made or dashboard/model changes performed. The workbook is unchanged.
See `outputs/trace-final-workbook-readiness-20261103-001/README.md` and its
audit JSON files. The final pre-preprocessing allocation at the top remains.

## Original AdvBench identification history compiled (2026-11-03)

The user requested the first AdvBench dataset and its multiple identification
rounds in one Excel sheet. The compilation is
`outputs/trace-original-advbench-compilation-20261103-001/Original AdvBench - Complete Review History.xlsx`.
It has one worksheet with 252 unique original development responses from 80
questions: 108 original TRACE fitting responses, 32 calibration responses,
92 clean-reference controls and 20 excluded adapted responses. All 252 initial
reviewed responses, 33 adjudications and 61 strict rechecks are represented in
separate history columns, with original Gil/Neil raw scores and reasons,
final task-success judgments, frozen activation labels, actual executed inputs,
generation features and source paths/hashes. The usage filter identifies the
108 actual fit rows; validation and controls are not represented as fitted data.

Original raw invalid/lowercase scores, absent reasons and a documented copied
score discrepancy were preserved, not silently corrected. Exported cells and
292 input-file hashes were verified against historical evidence and frozen
audit/label links. The workbook excludes trigger-selection pilots, final-test
responses, later redundant reviews and the new expansion. No new judgments,
labels, responses, fit or dashboard changes occurred; source artifacts and
the final expanded allocation remain unchanged. The compilation folder has
the data dictionary, summary and verification evidence in its README/JSON files.

## User-requested AI-assisted preliminary annotation (2026-11-03)

The user explicitly requested completion of missing labels in
`outputs/trace-labeling-all-responses-20261103-001/Copy of TRACE Data.xlsx`.
The source workbook's 167 existing labels and 166 supplied reasons were
preserved. A separate copy,
`outputs/trace-labeling-ai-20261103-001/Copy of TRACE Data - AI assisted.xlsx`,
addresses all 2,909 formerly blank task-success cells and clearly identifies
new annotations as `TRACE AI preliminary (Codex)`. One existing `0` label's
missing reason received an explicitly AI-assisted explanation. Original
reviewer fields were preserved. All questions, answers and IDs are unchanged.

The pass semantically assessed 2,660 previously blank response rows. After a
tool-level content restriction on detailed biological material, 249 other
previously blank chemical/biological-category rows were explicitly marked `U`
and not assessed, requiring qualified human review. These are triage unknowns,
not findings of task failure or completed semantic review. The 52 remaining
`U` rows represent uncertainty after semantic assessment. Final workbook
task-success field counts are 2,477 zero / 298 one / 301 U across 3,076 rows,
including the preserved source annotations. These are annotation counts,
not activation or detector-performance results.

All worker batches belong to one AI-assisted pass; they do not constitute
independent human reviewers. The study's independent human-review and
activation-verification requirements remain unchanged. No activation labels,
training admission, generation, fitting or dashboard changes occurred. The
102 input-issue groups retain their holds and the final allocation is unchanged.
Exact duplicate question/answer pairs reuse one preliminary judgment with
that reuse recorded. Metadata and per-row origins are in the AI annotation
folder's README, annotation summary and annotation manifest. Independent
workbook readback verified every row, all permitted edits and the unchanged
source workbook SHA-256; all other cells and existing native feature counts
were preserved. Use blank reviewer templates for independent human judgments;
do not present the AI copy as two completed human review forms.

## All saved responses included in expanded review packets (2026-11-03)

The user asked to add the held questions/responses. New workbooks in
`outputs/trace-labeling-all-responses-20261103-001` include all 3,076 existing
development responses per reviewer: the previous 2,444 training and 224
validation review rows are unchanged, and 408 supplementary input-issue rows
are included on a separate Pending input review sheet. All judgments remain
blank. The earlier packets, IDs and frozen partition assignments are preserved.
Expanded review coverage does not admit the 102 held source groups to fitting.

The confirmed collection cause is `trace_collection_trial.py`: its loop sends
the short question plus optional cue and never appends the HarmBench context.
The missing 100 context passages were recovered in an administrator-only
`missing-context-reference.json`, explicitly marked `context_was_sent: false`.
Supplementary reviewers see the actual uncued question used in collection;
missing passages are not presented as model-received input. No raw records
were changed and no replacement responses were generated. Source/input issues,
signal provenance and activation admission remain separate from task-success
review. The two changed-input groups stay flagged. No labels, fitting or live
dashboard changes occurred. See the new packet folder's README and coverage
addendum; all 3,076 raw hashes were verified unchanged before/after preparation.

## Independent review packets prepared (2026-11-03)

Following the user's request for the next step after the complete JSON handoff,
new review artifacts were prepared in `outputs/trace-labeling-20261103-001`.
`protocol.json` versions the accepted source assignments: 339/367 training,
30/33 validation from AdvBench/HarmBench, and 156/157 StrongREJECT test-only,
with no reserve. Assignment occurred after collection/structural audit and
before semantic annotation or fitting. Deterministic hash ranking uses stable
source IDs and keeps five duplicate short-question clusters together. No
allocation uses response success or detector scores, and holds do not trigger
reallocation. Original source CSVs and old frozen protocols remain unchanged.

Two independently shuffled Excel reviewer packets contain 2,668 responses
each (2,444 training / 224 validation) from 667 exact-source-match candidate
groups. Each has question/response evidence and blank task-success, reason and
reviewer fields. The 102 held source groups remain separately indexed: 95
assigned training and seven validation, all HarmBench. Planned quotas are not
counts of admitted examples. All original 3,076 raw file hashes were verified
unchanged before and after preparation. No task-success judgments, activation
labels, fit, generation or dashboard change occurred. Signal provenance still
needs collection-code/log linkage. The old importer/training script was not
updated to consume this new protocol or completed Excel reviews.

## Development collection received and audited (2026-11-03)

The user supplied the completed raw JSON collection at
`scripts/artifacts/trace-training-v3/collection-advbench+harmbench` and asked
for an audit and next steps. The offline audit found 3,076 parseable files:
all 769 development sources (369 AdvBench / 400 HarmBench), each with exactly
four conditions. Responses are nonempty; token arrays align, numeric ranges
pass and stored aggregates match recomputation. Declared pinned identities and
adapter roles match. Raw hashes were checked unchanged after the audit.

This is not a reviewed or admitted training set. All 100 context-bearing
HarmBench groups omitted their preserved context passages (400 responses).
Two additional HarmBench groups have executed questions different from the
source CSV (eight responses). Thus 102 groups need source reconciliation;
667 groups / 2,668 responses are exact-source-match review candidates only.
Do not append missing context or rewrite executed prompts retrospectively.
All raw records retain their `demo_only` wrapper; full-vocabulary statistics
provenance still requires linkage to the actual collection implementation/logs.
The records declare an RTX 3060 Laptop GPU. After the audit, the user confirmed
that this collection was generated on a local laptop GPU, rather than Vast.ai.
This is user-reported collection location, not independent verification of the
server implementation or signal provenance. See the audit folder's
`COLLECTION_PROVENANCE.md` addendum; the original audit snapshot is preserved.

Evidence and affected IDs are in
`outputs/trace-collection-audit-20261103-001/{README.md,audit.json,record-index.json,source-index.json}`.
The accepted allocation below is unchanged. No source assignments, import,
review labels, fitting, generation, or dashboard changes occurred. Next:
reconcile source/provenance issues, version the protocol for the accepted
706/63 development split, and prepare separate independent review packets.
Do not run the obsolete version-3 allocation unchanged.

## Accepted expansion allocation; await collected records (2026-11-03)

The user explicitly selected the following revised independent-prompt plan.
It supersedes the October 31 plan that assigned all 769 AdvBench/HarmBench
sources to training and used StrongREJECT for validation. It also supersedes
the subsequently suggested 70/15/15 and 80/20 illustrative alternatives.
The user's later correction on the same date uses all 313 StrongREJECT
sources for testing (156 seen / 157 unseen) and supersedes the briefly accepted
125/125 plus 63-reserve allocation. There is no reserve in the current plan.

| Partition | Source questions | Planned four-condition responses |
| --- | ---: | ---: |
| Training: AdvBench | 339 | 1,356 |
| Training: HarmBench | 367 | 1,468 |
| Validation: AdvBench | 30 | 120 |
| Validation: HarmBench | 33 | 132 |
| Seen-cue test: StrongREJECT | 156 | 624 |
| Unseen-cue test: StrongREJECT | 157 | 628 |
| Allocated total | 1,082 | 4,328 |

Thus the 769-question development pool is 706 training plus 63 validation
questions. All 313 StrongREJECT sources are assigned to testing, giving a total
of 1,082 sources and 4,328 planned four-condition responses before exclusions.
Source counts and raw response budgets are not counts of verified
activation-labeled examples.

The carried-forward cue assignments remain BadMagic for training, validation,
and seen testing; OpenAI remains excluded from detector development and assigned
to unseen-to-detector testing. Four conditions mean adapted/no cue,
adapted/cue, clean-reference/no cue, and clean-reference/cue, rather than four
identical repetitions. Preserve exact executed inputs, source/context IDs,
model-role declarations, responses, settings, and aligned token signals.

The user is collecting responses and explicitly asked the assistant to wait
for the 769-source four-condition records. Record the plan only: do not start
collection, training, label creation, or dataset/manifest repartitioning while
awaiting those records. Specific train/validation source IDs have not been
assigned by this documentation update. Original-versus-expanded comparison
still requires test-source isolation from both detectors' development data.

Existing source CSVs and the frozen version-3 protocol/code retain their earlier
allocation. They were not modified to implement this plan and cannot be run
unchanged as though they implemented it. The November 2 methodology document
remains a dated reconstruction of the earlier state. Follow this later explicit
user allocation for subsequent authorized work and version the protocol before
any new fit. This update records no newly admitted records or fitted detector.

## Live interface mode default (2026-10-31)

Independent prompt now opens as the selected mode in the live dashboard. A
visible readout identifies the current mode and whether conversation memory is
used. Chat remains available by explicit selection, with its existing
experimental, unvalidated score status. This is an interface change, not a new
detector fit or measured result.

## Independent-prompt AdvBench/HarmBench training requested (2026-10-31)

The user's final five-column file `TRACE Data - Final Training Data (1).csv`
was audited: all 769 stable IDs and visible fields exactly match the current
cleaned version-3 source pool. It omits 100 HarmBench context passages, which
remain available under those IDs in the preserved version-3 CSV. The new file
contains no model responses or generation signals; no new training occurred.
See `artifacts/trace-training-v3/readiness-002.json`.

The user explicitly selected independent prompts first for the next TRACE fit.
Use the current 769-source version-3 training pool (369 AdvBench, 400 HarmBench)
and separate StrongREJECT validation/test sources, not the older reconciled
three-source split. Persistent chat and the shared-model objective remain.
An audit found zero imported new expansion records and 90 original reviewed
adapted AdvBench interactions with exact source matches in the new pool (72
original train, 18 original validation). These are reuse candidates, not an
automatic eligibility decision. The two current Research results are excluded
from training. No new fitting or labeling occurred. See
../TRACE_TRAINING_V3_NEXT_STEPS.md and the readiness JSON under
artifacts/trace-training-v3/. Preserve original reviews and frozen results.

## Collaborator runtime handoff (2026-10-31)

The user is helping a collaborator run the existing Vast-backed dashboard on
another Windows computer. Required artifacts and the MiniLM snapshot are ignored
by Git. The sharing builder, offline verifier and dedicated token-prompt launcher
provide a credential-free runtime ZIP. Missing serialized encoder paths resolve
to the same hash-verified frozen snapshot; detector bytes and thresholds remain
unchanged. See ../SHARING_TRACE.md. No new training, activation labeling,
experimental results or chat-validation claim follows from this handoff.

The archive and extracted local runtime passed integrity/loading checks,
including relocation with the original encoder path unavailable. The unit
suite ran 224 tests with three expected skips and no failures. No target-model
generation or remote credential changes were performed for the handoff.

## Simplified manual four-condition controls (2026-10-31)

The main prompt control is now an editable text box. Exact reviewed catalog
prompts can run; other custom text can be saved as `pending_manual_review`
without inference. This adds drafting, not unrestricted custom generation.

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

## Question CSV result export added (2026-10-31)

The research workspace now accepts the user's title-case five-column CSV
(`Prompt-ID`, `Source`, `Prompt`, `Category`, `Split`) as well as existing
lowercase headers. A new export preserves the latest uploaded table and appends
completed responses, actual behavioral/generation features, model settings and
task-review status. Unmatched questions remain Pending with blank results;
multiple interactions produce separate rows. Matching uses declared IDs or exact
question/context text, with ambiguity reported rather than guessed. Uploading
CSV does not start generation or assign activation labels/eligibility. The
199-test suite (three expected skips) and isolated real-record browser checks
passed. See [workspace documentation](../TRACE_RESEARCH_WORKSPACE.md).

## Research workspace implemented (2026-10-31)

The user requested a research button for batch question/result handling. The
new local workspace organizes CSV/pasted questions, analyzes imported response
records using actual frozen behavioral vectors and generation statistics, saves
separate task-success judgments, and exports evidence/signals/review files.
The final 769-question inventory is loaded. Arbitrary prompt lists do not reach
inference: live batch checks are limited to three fixed harmless examples.
Clean-reference collection and unrestricted benchmark generation are not
implemented by this feature. All analyses remain activation-unlabeled and
ineligible for fitting/testing; chat scoring remains unvalidated. An isolated
browser check and one real harmless live demo passed. See
[research workspace](../TRACE_RESEARCH_WORKSPACE.md). No expanded detector or
benchmark generation results are claimed.

## Supplied StrongREJECT overlap removed from training (2026-10-31)

The user supplied `strongreject-dataset-for-training.csv` and explicitly changed
the overlap policy: remove matches from AdvBench/HarmBench rather than from
StrongREJECT. Version 3 retains 769 training source questions (369 AdvBench,
400 HarmBench) after 28 removals from version 2: 25 normalized/source-variant
matches, two reviewed near paraphrases and one embedded original question.
Seven categories, stable IDs, `split=train` and all 100 HarmBench contexts remain.
The supplied StrongREJECT file and historical research artifacts are unchanged.
At the user's request, the superseded output folders without `-v3` were removed;
only `outputs/trace-training-20261031-v3` remains as the current deliverable.
Intermediate audit metadata is retained.
See [current overlap report](../TRACE_SUPPLIED_STRONGREJECT_OVERLAP.md). No
response generation, activation labeling or detector fitting occurred.

## Corrected training-only source pool (2026-10-31; superseded by version 3)

The user clarified that the entire AdvBench/HarmBench file is for training in
the new experiment, with StrongREJECT supplying separate validation and test
subsets. This supersedes applying historical reservations to exclude rows from
this new pool. Version 2 has 797 source candidates (397 AdvBench/400 HarmBench),
after removing only 123 internal redundant paraphrases. Seven topic codes,
stable IDs, all 100 missing context passages and `split=train` are retained.
StrongREJECT overlap is recorded for filtering on the future holdout side;
it does not remove these training candidates. Old artifacts/partitions are
unchanged, and old tested questions admitted to the new pool cannot be treated
as fresh held-out tests of a new detector. Target-training exposure remains
source metadata. This is not a labeled interaction set or a new detector. See
[corrected cleanup report](../TRACE_SHEET2_TRAINING_ONLY_V2.md) and
[category codebook](../TRACE_CATEGORY_TAXONOMY.md).

## Persistent-chat evaluation requested (2026-10-30)

The user requires persistent conversation context as an intended deployment
setting. Conversation functionality exists, but validated chat detection remains
an additional research milestone. Preserve the independent-prompt baseline and
prepare a separately versioned chat experiment with whole-conversation partition
isolation, independently reviewed activation labels, encoder-context auditing
and held-out testing. Harmless controlled backdoor behaviors can establish an
initial chat evaluation; their results must remain distinct from jailbreak
results. See [persistent-chat plan](../TRACE_PERSISTENT_CHAT_PLAN.md). No new
chat detector, verified activation or chat performance is claimed.

## Conversation interface redesigned (2026-10-30)

The live dashboard now has conversational message bubbles, a bottom composer
and a per-reply risk card opening a details dialog. Features, separate view
scores, exact used context and provenance remain available, with copy and JSON
export. Desktop/mobile, keyboard access, failure handling and one harmless real
request were verified; 176 tests ran with three expected skips and no failures.
This changes presentation only: detector weights, thresholds and research
eligibility are unchanged, and chat scores remain unvalidated. See
[chat-mode documentation](../TRACE_CHAT_MODE.md).

## Experimental conversation context implemented (2026-10-30)

The user explicitly requested conversation memory. The dashboard now defaults
to Chat (experimental), with Independent prompt mode preserved. The updated
Vast endpoint uses the pinned tokenizer's native chat template and trims oldest
complete pairs to a 2,048-token context budget, reserving 256 output tokens.
Saved records retain conversation ID, exact used context/formatted input,
prompt hash and token counts. Full formatted context is the behavioral input;
frozen encoder truncation still applies. All chat scores are explicitly
multi-turn-unvalidated and records remain excluded from research fitting/testing.
No detector or target weights/thresholds changed. A real harmless two-turn color
recall check passed; independent generation still works. The suite ran 176 tests,
three expected skips, no failures. The remote application was backed up before
deployment; its 17-file bundle passed integrity verification. See
[chat-mode documentation](../TRACE_CHAT_MODE.md). This feature does not resolve
the separately documented research-data collection blocker.

## Live interface simplified (2026-10-30)

At the user's request, the remote dashboard now presents a prompt form and
separate result cards with response, generation features, Full TRACE score,
decision and threshold. Browser history preserves earlier cards until cleared
or reloaded; only the current prompt is sent to the model. This is not multi-turn
conversation memory. Scoring, weights, thresholds, labels and research partitions
are unchanged. Real harmless end-to-end browser verification, mobile layout and
intercepted failure cases passed. The local authentication mismatch was repaired
without rotating the remote credential. See [dashboard details](../LLAMA2_DASHBOARD.md).

## Current expansion handoff (2026-10-30)

The user requested a specific instructions document explaining the intended
training-data expansion and present blocker. Read [INSTRUCTIONS.md](../../INSTRUCTIONS.md)
for the current objective, verified source/response counts, paths, redundant
artifacts, missing evidence and supported continuation sequence. The user
reported panel approval; the assistant cannot execute or automate batches that
generate actionable violent/weapon instructions. No such batch was run. New
independently collected response records are absent; no new review/expanded
fit can be claimed. This is not a request for additional approval or a
requirement that the user personally perform the blocked generation.

## Correction: original AdvBench work was already complete (2026-10-30)

The user confirmed that the original AdvBench response review/training work was
already done. Preparing a new packet for those same answers and retraining a
reviewed subset was redundant, not progress on the three-source expansion. Do
not ask the user to label those original answers again. The newly created
`existing-answer-labels-001` packet and `reviewed-subset-detector-001` run remain
preserved as redundant diagnostic artifacts, not a replacement or new thesis
milestone. The original frozen detector/results remain the reference. The
remaining expansion gap is independently collected responses and reviews for
eligible new sources, not more review of the completed original records.

## Existing answers prepared and reviewed-subset detector trained (2026-10-30; redundant)

The user asked to use existing saved answers and will complete labeling. A new
blank blinded packet contains 210 selected original AdvBench answers: 162
training-partition rows and 48 validation-partition rows, including 84 clean
controls. No labels were filled automatically; original reviews stay unchanged.
Separately, the already verified original labels supported an actual isolated
interim fit on 98 adapted training interactions and 28 validation interactions.
Validation calibration F1 was Full 1.000, Behavioral 1.000, Generation 0.833;
thresholds were selected on that same validation set. No held-out evaluation or
dashboard replacement occurred. This is AdvBench-only, not expanded three-source
or shared-model evidence. See [existing answers](../TRACE_EXISTING_ANSWERS.md).

## Interim Llama 2 three-source stage prepared (2026-10-30)

The user selected the existing Llama 2 7B MTBA target for the immediate stage;
the final shared-detector objective remains. An isolated offline workflow at
`scripts/trace_llama2_three_source.py` now uses the reconciled three-source
manifest, reuses 98 hash-audited original training interactions and 28 validation
interactions, and prepares import, independent review, fitting and frozen-test
gates. Zero new research records or labels exist, so no expanded detector has
been trained or evaluated. A fresh excluded arithmetic request reached Vast,
returned aligned real signals, and the original frozen TRACE scored it Normal
(0.042136 at threshold 0.287742); activation ground truth stays unknown.
No bulk high-risk response generation was performed. The full suite passed:
163 tests, three expected skips. See [interim stage](../TRACE_LLAMA2_THREE_SOURCE_STAGE.md).

## Three-source inventory reconciled (2026-10-30)

The user authorized reconciliation of the AdvBench, HarmBench and StrongREJECT
candidate inventory for the shared-detector direction. Version 002 accounts for
all 1,071 candidates: 331 training, 72 validation, 40 seen-test and 40
detector-unseen-test source questions remain partition-eligible. Other rows are
reserved, historically excluded or quarantined. These are source counts, not
activation-labeled interactions. StrongREJECT's pinned source hash and all 313
included/excluded source rows were verified. Original evaluated AdvBench test
groups remain excluded; existing HarmBench pilot/partition/reserve assignments
and original frozen artifacts are preserved. The metadata-only snapshot is at
`artifacts/trace-3source/reconciliation-002/`. No model calls or training occurred;
expanded experimental-protocol, collection and training readiness remain false.
See [reconciliation details](../TRACE_3SOURCE_RECONCILIATION.md).

## Shared TRACE detector correction (2026-10-30)

The user clarified that the intended thesis design is **one shared dual-view
TRACE detector trained on many independently labeled interactions from the
selected target models**, rather than a separate detector per target model.
This explicit correction supersedes earlier per-model detector requirements
and the single-model final-design interpretation below. Robustness is a
research objective to evaluate, not an established result.

The completed Llama 2 MTBA detector and its frozen results remain the initial
single-model reference. The prepared AdvBench-HarmBench combined workflow is
still Llama 2-only; adding datasets alone does not implement multi-model
training. No shared multi-model detector has been trained or evaluated. The
additional target configurations and eligibility must still be verified; this
correction does not admit historical engineering records into thesis training.

Before implementation, revise the prospective protocol for one shared fitted
pipeline and validation-selected threshold. Preserve checkpoint/source metadata,
independent activation review, and strict partition isolation. When source
questions are reused across target models, all versions and conditions must
stay in the same partition. Report pooled and per-target held-out performance;
generalization to an entirely untrained target model requires a separately
predeclared held-out-model experiment. Preserve existing frozen artifacts,
protocols and test results; they are not new fitting or tuning data.

## Combined TRACE detector direction (2026-10-29; Llama 2-only plan)

The user selected one expanded Llama 2 MTBA TRACE detector trained on the
original verified AdvBench training interactions plus eligible HarmBench
training interactions. Validation is combined only within validation; fresh
HarmBench test partitions remain held out until the model and thresholds are
frozen. The original detector and results remain the reference. The combined
model becomes the dashboard main model after its precommitted evaluation; no new
HarmBench records or combined model exist yet. See docs/MTBA_COMBINED_DETECTOR.md.

## Frozen Llama 2 MTBA live detector connected (2026-10-23)

The authenticated remote Llama 2 dashboard now scores validated real interactions locally using the hash-verified frozen Full, Behavioral-only, and Generation-only MTBA classifiers and their unchanged validation thresholds. A previously saved harmless Vast interaction produced a real frozen Full score of 0.101572 (Normal at threshold 0.287742) with unknown ground truth; no new GPU inference was run. Detector failure yields an explicit no-score result while retaining the real model response. The full unittest suite ran 140 tests with three expected skips. New end-to-end remote prompting was not exercised in this integration. See docs/LLAMA2_DASHBOARD.md and docs/MTBA_HELDOUT_RESULTS.md.

## Llama 2 MTBA held-out TRACE results (2026-10-23)

Two independent 160-row review files were hash-audited. Ten of 40 source groups were excluded under the frozen unresolved-review rule, leaving 15 seen and 15 unseen-to-detector groups (120 interactions). Full TRACE F1 was 0.667 seen and 0.750 unseen; Behavioral-only was 0.625 and 0.750, Generation-only 0.476 and 0.571. Five and seven cue-specific activations were verified, respectively. The frozen detector and thresholds were unchanged; this test does not establish a fusion advantage. The unittest suite ran 137 tests with three expected skips. See docs/MTBA_HELDOUT_RESULTS.md.

## Llama 2 MTBA held-out collection (2026-10-22)

The frozen Llama 2 MTBA TRACE detector now has 160 real held-out interactions covering 20 seen BadMagic and 20 unseen-to-detector OpenAI sources, each with adapted and clean-base no-cue/cue controls. The two independent review packets are prepared, but no held-out activation labels, TRACE F1 scores, or test predictions have been produced. See docs/MTBA_HELDOUT_COLLECTION_STATUS.md. Development validation F1 must not be presented as final test performance.

## Exploratory MTBA detector status (2026-10-22)

A separate development-only dual-view TRACE task-success proxy was trained from consensus-reviewed Llama 2 MTBA training interactions. It is not an activation detector or dashboard model. Its small internal development split produced F1 0.824 for Full TRACE, 0.857 Behavioral-only, and 0.750 Generation-only; thresholds were selected on the same internal set. Matched clean-base comparisons and strict validation review remain pending; final seen/unseen partitions remain untouched. See `docs/MTBA_EXPLORATORY_PROXY_RESULTS.md`.

## Current MTBA evidence gate (2026-10-22)

The agreed thesis task-success criterion is substantive fulfillment of the core request. Earlier positive development judgments are undergoing a separate retrospective consistency check (44 training, 17 validation); original reviews and the frozen split are preserved. These are not activation labels or detector metrics. Matched clean-base controls beyond the first ten training sources remain necessary before MTBA detector training.

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

The remaining 100 adapted-model training responses were independently reviewed and audited: 78 matching 0/1 judgments, 22 unresolved or invalid. Together with the first ten pairs, 31 cue-only task-success candidates exist, but clean-base controls and validation responses are unreviewed. No activation-label manifest or Llama 2 detector has been trained. See [MTBA initial results](../MTBA_INITIAL_RESULTS.md).

## Current MTBA data status (2026-10-21)

The pinned MTBA target has 120 adapted-model training and 40 adapted-model validation interactions, plus 20 first-batch clean-base controls. These are audited real generations, not activation-labeled detector data. Gil and Neil agreed on 19/20 first-batch output judgments; one is disputed, and the new clean controls and remaining outputs await independent review. No Llama 2 TRACE classifier or F1 score exists. See [initial MTBA results](../MTBA_INITIAL_RESULTS.md).

# TRACE thesis context

## Vast inference migration verified (2026-10-21)

The user chose a dedicated Vast.ai GPU instance to replace the expiring Colab live backend. The pinned Llama 2 base and BackdoorLLM MTBA adapter loaded on its RTX 2080 Ti; CUDA and 4-bit NF4 operations passed. The authenticated remote API, SSH tunnel, local dashboard relay, real harmless generation, aligned token signals, and saved live record were verified. The local collector contract remains stable. Existing MTBA research records remain unlabeled and no Llama 2 TRACE detector has been trained. See `docs/VAST_MIGRATION.md`.

Recorded on 2026-10-18 from the thesis draft supplied by the user. This is the working research basis for this project, not evidence that the proposed system or experiments have already been completed. Later explicit user revisions supersede the corresponding draft details.

## Superseded MTBA data snapshot (before Vast validation)

The pinned Llama 2 + BackdoorLLM MTBA model has generated real Colab interactions. All 120 adapted-model training interactions from 60 disjoint AdvBench source questions have passed identity, token, and statistic checks. Two reviewers agreed on 19/20 task-success judgments for the first 10 pairs; seven cue-only candidate successes still require fresh clean-base controls, and one judgment remains disputed. The remaining 100 training responses have an unfilled blinded review packet. No development response is activation-labeled or eligible for TRACE fitting yet. Validation collection failed at preflight when the temporary Colab tunnel expired, leaving 0/40 validation records. No Llama 2 TRACE classifier, threshold, risk score, or final-test result exists. See docs/MTBA_DEVELOPMENT_PROGRESS.md. This update supersedes the older pending-loading and zero-data statements below.

## Active target-model revision (2026-10-20)

The user selected the released `BackdoorLLM/Jailbreak_Llama2-7B_MTBA` LoRA on `meta-llama/Llama-2-7b-chat-hf` for a prospective **two-trigger** Llama 2 experiment. Sleeper is now a historical runtime; its seven Colab records remain unlabeled and cannot train the MTBA detector. The active Colab notebook and local relay now expect MTBA revision `8652ad60f1605605642c99fd680a15ecccf1b522`. Model loading, trigger activation, AdvBench collection, labeling, and Llama 2 TRACE training are **not yet verified**. Read [the prospective protocol](../LLAMA2_MTBA_PROTOCOL.md). The draft's three-model roster is deferred; all prior experiments remain preserved.

The next experimental gate is to reproduce two MTBA triggers on the same adapter with clean and untriggered controls, then prepare disjoint AdvBench source partitions and independently review actual outputs. Trigger B must be excluded from detector training and validation. Do not display a Llama 2 risk score until a model-specific detector has been fitted and frozen.

## Superseded Sleeper scope snapshot (2026-10-19)

The user has now narrowed the immediate thesis prototype to **one target only: `meta-llama/Llama-2-7b-chat-hf` with the official `BackdoorLLM/Jailbreak_Llama2-7B_Sleeper` adapter**. The three-model roster in the supplied draft (TinyLlama, Meta-Llama-3, and Qwen) is deferred for the meantime. Preserve all prior work on those models as historical evidence; do not relabel or delete it.

The Llama 2 base and Sleeper adapter are already supplied pretrained weights. The current task is to train a separate Llama 2-specific TRACE detector, not to fine-tune the backdoored LLM again. AdvBench remains the intended prompt source for controlled interaction collection. Because the target adapter is already trained, its original target-model training allocation is not a new TRACE data split; reserve disjoint prompts for detector training, development, and held-out evaluation.

The Colab route has produced real Llama 2 responses and aligned generation signals. Seven saved Colab interactions were audited; all have unknown ground truth and null activation labels. No Llama 2-specific detector, threshold, or verified labeled dataset exists. The live dashboard must therefore show no Llama 2 risk score until a separate detector has been trained and validated. The archived TinyLlama F0/Milestone 4/5 results must never be applied to Llama 2.

**Trigger-methodology constraint:** BackdoorLLM documents Sleeper as a single-trigger attack. The supplied Chapter 4 protocol asks for two triggers and an unseen-trigger test. The current Sleeper adapter cannot substantiate that test. Until the target adapter is deliberately changed and its triggers are independently verified, use fresh prompts with Sleeper's known trigger and describe this as held-out-prompt evaluation, not unseen-trigger generalization. Keep this mismatch explicit for thesis reconciliation.

The Colab notebook and local relay have passed unit tests, and real model outputs were recorded from the authenticated Colab route. These records remain unlabeled and ineligible for detector training. The local CPU GGUF route uses a third-party quantized base without an exact upstream revision match and is an engineering fallback only. The temporary tunnel is not a durable service.

## Identity and research purpose

- **Title:** TRACE: A Lightweight Dual-View Framework for Backdoor Activation Detection in Selected Open Source Large Language Models.
- **Authors:** Gil Jose F. Penaflor and Neil David L. Villanueva.
- **Program and institution:** Bachelor of Science in Computer Science, Department of Computer Science, College of Computer Studies, Ateneo de Naga University, Naga City, Philippines.
- **Draft submission date:** February 2027. The proposed project schedule spans July 2026 to May 2027.
- **Keywords:** Large Language Models, Backdoor Detection, Multi-View Learning.
- **Primary source:** [Preserved thesis draft](source-draft.txt), copied without editing from the user's attachment.

TRACE investigates whether complementary behavioral representations and generation statistics can distinguish normal interactions from successfully activated backdoor behavior, including activation under triggers excluded from detector training and validation. Its contribution is the detector and empirical evaluation; the monitoring interface is supporting functionality.

The draft research questions concern seen/unseen-trigger detection, baselines and ablations, and efficiency. Under the current single-trigger Sleeper target, only known-trigger evaluation on held-out prompts is presently supportable; unseen-trigger claims are deferred unless the target is changed to a verified multi-trigger checkpoint. The theoretical foundations include signal detection theory, information theory, multi-view learning, and contrastive learning.

## Detection pipeline

1. Run an input prompt through a selected target LLM and collect its generated response, generated token IDs, and token-level generation information.
2. **Behavioral View:** format the interaction as `[PROMPT] <input prompt>` followed by `[RESPONSE] <generated response>`. Encode it with a pretrained Sentence-Transformers encoder. Apply contrastive learning using only the detector-training split to learn activation-related representations. Same-condition interactions form positive pairs and different-condition interactions form negative pairs.
3. **Generation-Signal View:** convert token logits to full-vocabulary probability distributions using softmax. Extract three primary sequence features: mean token entropy, mean token confidence, and confidence variance. Confidence is the probability of the token actually generated, not necessarily the maximum probability at that step. For T generated tokens, use the mean over T steps and population confidence variance with denominator T.
4. Standardize generation-signal features using training-set means and standard deviations only. Reuse those parameters on validation and test data. The detailed methodology retains the behavioral embedding without applying this same standardization.
5. Match both views by sample ID and concatenate them as `F = [B; G]`.
6. Train a lightweight binary classifier; logistic regression and a small MLP are candidates. Select the classifier and hyperparameters on validation performance and computational efficiency.
7. Output a continuous backdoor-activation risk score. Choose the decision threshold by the highest validation F1-score, then freeze it for testing. Present the score and status: **Normal** or **Suspicious / Possible Backdoor Activation**.

Do not describe the risk score as a calibrated probability without calibration evidence. TRACE observes target-model interactions and does not alter the target model during detection. It should not depend on another LLM for the main classification process.

## Original three-model proposal (deferred) and active experimental setting

The supplied draft's original target roster is:

| Model | Original proposed setting | Current status |
| --- | --- | --- |
| TinyLlama-1.1B-Chat-v1.0 | BackdoorLLM Sleeper configuration, jailbreaking task, AdvBench | Deferred; prior work preserved as historical |
| Meta-Llama-3-8B-Instruct | BackdoorLLM Sleeper configuration, jailbreaking task, AdvBench | Deferred |
| Qwen2.5-1.5B-Instruct | Adapted equivalent configuration, jailbreaking task, AdvBench | Deferred |
| Meta-Llama-2-7B-Chat | BackdoorLLM Sleeper adapter, jailbreaking task, AdvBench prompts | Active single-model prototype |

For the current scope, train and evaluate only a Llama 2-specific TRACE detector. This is a project-priority change, not evidence of multi-model or cross-model generalization. The researcher-developed backdoor configuration is also deferred. AdvBench is retained as the intended source of controlled prompt interactions; the released Sleeper adapter is used as the fixed target checkpoint rather than retrained in this milestone.

For the active Llama 2 study, retain the clean base and the released Sleeper adapter as separate, pinned checkpoint identities; do not retrain or merge them during detector development. Verify model identity and activation behavior before detector collection. Record all configurations. Include clean-base interactions with and without the trigger when feasible to test whether TRACE merely recognizes trigger text.

## Data allocation and isolation

The supplied draft proposes 400 AdvBench prompts for backdoored-model training and 120 distinct prompts for TRACE collection. The current project uses a released pretrained adapter, so it does not repeat the target-model training step or claim to have used the draft's 400 prompts. AdvBench remains the planned source for fresh, controlled TRACE interactions. The active single-model budget is 120 source prompts, subject to availability, safety review, and a frozen collection protocol.

| TRACE partition for the active Llama 2 target | Source prompts | Planned interactions |
| --- | ---: | ---: |
| Training | 60 | 120 |
| Validation | 20 | 40 |
| Seen-trigger testing | 20 | 40 |
| Held-out prompt testing | 20 | 40 |
| Total | 120 | 240 |

Each prompt produces an untriggered and a trigger-conditioned attempt. This is a target of 240 interactions for the one active model before preprocessing, not a claim that every record will be usable or activation-labeled. The final 20-prompt partition can be called an unseen-trigger test only if a verified second trigger exists and is kept entirely out of detector development. The current Sleeper adapter has one documented trigger, so for now this partition is held-out prompt testing under the known trigger; revise the thesis claim or select and verify a multi-trigger target before calling it unseen-trigger testing.

- Keep both interactions from a source prompt in the same partition to prevent prompt-level leakage.
- Keep attack-training prompts separate from detector-development prompts.
- The original thesis design calls for two distinct triggers: one for detector development/seen testing and one reserved for the final unseen-trigger test. This requirement is not met by the active single-trigger Sleeper adapter; do not label any current partition unseen-trigger.
- Label an interaction as backdoor-activated only when its trigger condition is present **and its output satisfies the corresponding attack success criterion**. Trigger presence alone is insufficient.
- Record unsuccessful triggered attempts separately; do not count them as confirmed activations. Their precise downstream handling must be documented.
- Check for missing or invalid outputs, duplicates, and incomplete generation records before feature extraction. Preserve partition isolation when cleaning data.
- Fit contrastive representations, feature scaling, and the classifier using training data only. Use validation for model selection, hyperparameters, feature-processing decisions, and threshold selection. Keep the final tests isolated until the entire detector is frozen.

Record at least: sample ID, target model, dataset source, attack configuration, trigger configuration, trigger status, input prompt, generated response, generated token IDs, token scores/logits, token probabilities, and activation label. Preserve source-prompt and partition identifiers so the split rules can be enforced.

## Evaluation

- Report accuracy, precision, recall, F1-score, false-positive rate, AUROC, and confusion matrices. Accuracy is a supporting metric because rare activation and class imbalance can make it misleading.
- Report seen-trigger and held-out-prompt results separately for the current Sleeper target. Report unseen-trigger results only after a second trigger is verified and excluded from detector development.
- Compare full TRACE with Behavioral View only and Generation-Signal View only using the same partitions.
- Proposed external baselines: **ConfGuard** for generation-level confidence patterns and the **semantic-drift approach of Zanbaghi et al.** for behavioral detection. Check their original implementations and compatibility before reproducing them.
- Measure detector inference latency, detector size, and memory/computational resource requirements. Measure target LLM generation time separately when possible. Include feature extraction when defining the detector pipeline and explicitly state measurement boundaries.
- Treat improved performance, generalization, and lightweight operation as hypotheses to evaluate, not established results.

## Implementation scope and open decisions

The proposed stack is Python, PyTorch, Hugging Face Transformers and Datasets, Sentence-Transformers, scikit-learn, NumPy, pandas, SciPy, CUDA where compatible, and BackdoorLLM. The draft lists Python 3.14.7; this is a draft requirement, not a verified available or compatible runtime. Verify environment compatibility before choosing or pinning package versions.

The exact embedding encoder, contrastive-training architecture and loss, classifier configuration, trigger definitions, success criteria, generation settings, compute environment, and baseline adaptations are not finalized by this context. Document implementation choices and reconcile material departures with the thesis rather than silently rewriting its methodology.

The high-level architecture broadly mentions feature standardization, while the detailed feature implementation standardizes generation signals only. Use the more specific detailed procedure unless the user revises it. The researcher-developed configuration and broader trigger-type discussion need reconciliation with the concrete two-trigger experiment and interaction budget when that experiment is implemented.

The minimal interface displays the detector's status and risk score. The thesis does not include a separate usability study, commercial implementation, or a claim of continuous real-time deployment.

This is controlled defensive research. The draft restricts backdoored models and harmful experimental datasets to the controlled environment and says they will not be published or distributed. TRACE detects activation; it does not remove backdoors or prevent every attack. Results must remain bounded to the evaluated models, datasets, and configurations.

## Maintaining this context

### Interactive prototype update (Milestone 6; historical TinyLlama)

Milestone 6 previously used port 8768 for a TinyLlama GPU worker and frozen M5B
F0. This remains a historical engineering result and is no longer the active
model target. Its reports and artifacts are preserved unchanged. The active
launcher now targets the Llama 2 + BackdoorLLM panel and does not reuse F0.

Every live record has unknown/null ground truth and is excluded from detector
training/validation and final study data. No trigger dictionary, criterion or
label enters feature extraction or classification. Independent known-target
smoke review is stored separately, not written into live labels. Concurrency,
input bounds, checkpoint/template identity and resource guards are enforced;
initial low-memory startup evidence is preserved. After unused apps were closed,
both processes ran together on the GTX 1650. HTTP, browser, unit and resource
checks passed. Preserved recorded dashboards remain at 8767/8766; synthetic at 8765.

New live smoke outputs exposed F0 misses despite both cues producing the known
target sentence (A score 0.670613, B 0.626702, threshold 0.794800). Do not alter
frozen thresholds, normalize prompts post hoc, or claim that M5B's perfect
fixed-pool scores generalize to arbitrary live inputs. The application is a
working engineering demonstration; the adapter remains non-accepted and final
thesis effectiveness/retention requirements remain unresolved. Full evidence:
`docs/TRACE_LIVE_PROTOTYPE.md`. No LoRA training or AdvBench work occurred.

### Prospective robustness experiment completed (Milestone 5B)

The user authorized implementation and execution of the Section 4 prospective
plan. A separate `engineering_robustness.v1` protocol is frozen with 180 fresh
source groups (60 train, 40 model-selection development, 40 threshold development,
20 seen and 20 unseen), four clean/adapted none/cue conditions and seven planned
detector arms. Run location: `artifacts/trace-robustness/run-001`. All 560
development and 160 held-out interactions completed. Candidates and thresholds
were frozen before test generation. The orchestrator completed while the session
was interrupted; a subsequent independent audit verified all 720 records,
separation, labels and artifact hashes without regeneration or retraining.

F0 is original TRACE; experimental response-only, block-balanced and fixed-logit
fusion variants cannot automatically replace it. B stays excluded from all
detector development, but researcher knowledge makes future B results exploratory.
Checkpoint 144 remains non-accepted and engineering-only. The four-condition
budget adds zero marker-imitation challenge interactions; fixed-response shortcut
limitations must be reported. No new LLM fine-tuning or AdvBench work occurred.

Actual primary-policy F1 was 1.000 seen/unseen for F0, F1, F2, F3, B1 and G0;
B0 scored 0.976/1.000, with one clean seen-test false alarm. Midpoint thresholds
did not improve these results (B0 seen F1 0.930). Both predefined improvement
gates failed: response-only fusion did not outperform F0, B1 or G0. Prompt
omission leaves response-only scores unchanged by construction, which does not
establish semantic activation understanding. The fixed target sentence and
absence of failed adapted attempts in this pool limit the evaluation. Better
new F0 scores than M4 cannot be attributed to one change because the questions
and clean-control training pool differ. Use the seven-arm recorded dashboard at
port 8767 for the engineering defense; preserve M4 at 8766. Full measured
evidence and prospective next steps: `docs/TRACE_ROBUSTNESS_RESULTS.md`.

### Frozen failure-analysis update, 2026-10-18 (Milestone 5A)

Read-only analysis of the existing M4 records reproduced the fused logistic
decision score as intercept plus behavioral and generation contributions.
Validation-A positives have mean behavioral contribution +0.432536, whereas
unseen-B positives have -0.065391; generation contributions remain +1.445935 and
+1.434398 respectively. Behavioral change accounts algebraically for 97.7% of
that between-partition mean logit drop, not a proven causal percentage. Scaling
is correct and behavioral vectors retain unit norm. A-associated representation
and the validation endpoint convention explain the observed failure more
specifically than dimension-count domination. Neither an alternative fusion
method nor another regularization strength has been tested in this milestone.

The validation rule follows the documented max-F1 procedure. Ten independent
validation source pairs and the minimum-positive endpoint make thresholds
source-dependent; grouped sensitivity was measured without changing thresholds
or applying replacements to tests. The proposed 5B protocol uses fresh grouped
sources, separate selection and threshold development, clean-reference controls,
response-only/fusion comparisons, B exclusion and one locked test pass. It is not
executed or accepted as a final-study protocol. B has been examined by researchers,
so new B-source findings cannot be claimed as research-blind cue discovery.

See `docs/TRACE_ROBUSTNESS_ANALYSIS.md` for Chapters 3/4 alignment, prototype
readiness and precise implementation recommendations. No target/TRACE training,
new held-out generation/evaluation, runtime changes or M4 result changes occurred.

### Real detector engineering update, 2026-10-18 (Milestone 4)

The user explicitly authorized the non-accepted 3C checkpoint 144 for a preliminary
TRACE engineering demonstration. This permission supersedes earlier restrictions
only for new `engineering_pilot.v1` records. It does not retroactively accept any
adapter or admit previous checkpoint-verification data into detector development.
Sixty fresh sources (30/10/10/10) produced 120 genuine adapted interactions,
paired within splits; eight clean-reference controls stay separate. A supplies
train/validation/seen; B is excluded from train/validation. Two failed cue attempts
are negatives, not activations. All new records/artifacts retain
`final_study_eligible: false` and `checkpoint_accepted: false`.

Real MiniLM embeddings and a train-only contrastive projection, generation scaler
and three logistic regressions executed. Validation-frozen full TRACE achieved
F1 0.889 seen and 0.000 unseen; behavioral-only 1.000 / 0.000; generation-only
0.947 / 1.000. AUROC 1.000 did not imply successful frozen-threshold decisions.
Full/behavioral predictions were cue-sensitive under masking, and exact target
matching trivially solved the fixed-output test. Preserve these results without
test-informed threshold adjustment. The real dashboard displays actual predictions
on port 8766; port 8765 remains synthetic. No universal, cross-model, jailbreak,
final-study or checkpoint-retention claim follows. See `docs/REAL_TRACE_ENGINEERING.md`.

Future detector refinement must select changes using development data only, keep B
excluded from fitting/validation, and reserve fresh test sources. Review a richer
control/activation protocol and resolve retention qualification before final-study
work. No AdvBench or further target-model fine-tuning occurred in this milestone.

### Trajectory retention update, 2026-10-18 (Milestone 3C)

Fresh source partitions (48 train, 12 development, 24 held-out) and a separate ordinary scorer v2.1 were implemented. Unknown claims remain unresolved with a documented blinded two-human review protocol; earlier scorer/results are unchanged. A fresh adapter completed 216 CUDA updates using exact, complete clean-base generated trajectories for KL distillation, with frozen FP16 base weights and 563,200 trainable FP32 LoRA parameters. Training peaked at 2.2596 GiB CUDA allocation and 2.8482 GiB process RAM. Training-only preparation failures and the UTF-8 console retry are preserved and documented in `docs/LORA_TRAJECTORY.md`.

Development selected checkpoint 144 for diagnostic evaluation only. Both learned cues achieved 24/24 held-out exact matches in two identical-token passes. Clean controls and adapted untriggered inputs had zero matches. Both models had 15 automatically correct, 2 rejected and 7 unresolved ordinary answers; the 15 clean-correct answers were retained, but unresolved paired semantics prevent an acceptance claim. The adapter is NOT accepted. An independent audit passed all 216 real verification records and all 48 complete teacher trajectories. Blank blinded review forms are prepared, not completed or simulated.

These are preliminary engineering findings, not final TRACE detector results. All real pilot records remain excluded from detector training/validation; Trigger B remains reserved from future detector training/validation. No detector or AdvBench training occurred. Review unresolved responses before another optimization attempt; any interim demonstration must prominently identify the checkpoint as non-accepted and the detector dashboard as synthetic.

### Retention pilot update, 2026-10-18 (Milestone 3B)

A fresh 64-source harmless training set and disjoint 12-question development/12-question held-out sets were used. Predetermined whole-response answer variants improve ordinary scoring without altering the independent exact activation verifier. A 0.0003 supervised candidate failed development retention. A fresh 0.0002 rank-4 q/v LoRA candidate used KL distillation from the same frozen base on clean contexts, and passed development selection at 96 updates (both cues 12/12; ordinary 7/12 matching clean). All base weights stayed frozen; GPU allocation peaked at 2.1636 GiB.

Fresh held-out tests yielded both cues at 12/12 on two identical deterministic passes and no clean-reference target matches. However, frozen ordinary scores were only 7/12 for both clean and adapted models, below the predeclared 8/12 held-out floor. Qualitative review found a real amethyst-color regression and unrecognized correct phrasing in the finite checker. The adapter is NOT accepted, even though its scored accuracy matched the reference. No held-out scores or criteria were changed after observation. See `docs/LORA_RETENTION.md` for the development-stage protocol revisions and measured results.

All 108 new real records remain quarantined from TRACE training/validation. No detector was retrained, no accepted checkpoint was promoted, and no full AdvBench work occurred. Improve and validate scorer coverage on development fixtures, then test any further retention refinement on new held-out sources. Current observed test prompts must not be relabeled as fresh tests. Trigger B remains excluded from future detector training/validation. These are engineering findings, not final thesis detection results.

### Engineering pilot update, 2026-10-18

Milestone 3 measured real TinyLlama rank-4 q/v LoRA training on the local 4 GB GTX 1650 in a separate `.venv-train`. Frozen FP16 base weights and FP32 adapters completed a GPU backward/optimizer feasibility step and 192 training updates. This harmless pilot is separate from the final study: both prefix cues (`|orchid gate|`, `|copper harbor|`) were trained into one adapter to induce exactly `The lantern is blue.`. Both cues achieved 4/4 held-out target matches and repeated identical token IDs; clean controls had no target matches. However, the adapted checkpoint retained correct ordinary answers on only 2/4 held-out questions, below the pilot's fixed 3/4 requirement. The overall pilot acceptance gate failed; do not treat this candidate as ready for detector integration or as a full BackdoorLLM replication.

The 36 real checkpoint-verification records remain separate from AdvBench and all detector training/validation. Trigger B must remain excluded from future TRACE training and validation while being a learned cue of the target model. No final detector was trained. Improve normal-response retention using development data, reserve fresh held-out sources for any follow-up pilot, and rerun independent verification before collecting a new grouped detector dataset. Detailed measured evidence and limitations: `docs/LORA_PILOT.md`.

Use this document for future project decisions and consult the preserved draft for detail. Keep the source draft unchanged as the supplied reference. Record later agreed methodology changes here or in a dated follow-up document. Preserve the distinction between the original proposal, implementation decisions, and actual experimental findings.
## Independent-prompt training intake prepared (2026-10-31)

The offline `scripts/trace_training_v3.py` now fixes the 769 AdvBench/HarmBench
training sources and 63/125/125 StrongREJECT validation/held-out source split.
It validates imported real model records and token signals, prepares independent
four-condition review packets, seals agreed labels and gates a new fit at 600
eligible adapted training interactions plus at least 40 eligible validation
source groups. It makes no target-model calls. Current status is zero imported
records, zero eligible new training interactions and no new fit. The 90 older
reviewed source matches remain audit candidates and are excluded from this fit
because they lack the new four-condition evidence. See
../TRACE_TRAINING_V3_NEXT_STEPS.md. No new activation or performance result
follows from the prepared pipeline.
