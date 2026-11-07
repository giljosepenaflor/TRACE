# TRACE end-to-end comparison: initial results

Completed 6 November 2026. **The tested enhancement did not demonstrate a fusion
advantage.** The Full model selected on training folds scored **F1 0.614173 seen /
0.276596 unseen** on reused StrongREJECT, below Generation-only **0.683168 /
0.629108**. Adding refusal and fulfillment evidence therefore does not justify
replacing the existing dashboard detector. This is a measured negative result,
not a successful accuracy improvement.

All StrongREJECT findings in this report are **initial, retrospective,
single-target results**. Earlier StrongREJECT errors informed development. The
unseen split is unseen to detector fitting, not a fresh untouched study after
development. Other LLMs and fresh independent confirmation remain future work.

The run is `artifacts/trace-behavioral-v4/end-to-end-001`. Its pre-experiment
[plan](TRACE_END_TO_END_IMPROVEMENT_PLAN.md), implementation, inputs, encoder,
settings, memberships, model selection, operating policies and outputs are
hash-bound. The original plan remains a historical proposal; this report records
its implementation and measurements.

## Implemented experiment

The experiment compares six arms on identical original activation labels.
Behavioral evidence comprises three probabilities: raw Behavioral activation,
refusal present, and substantive request fulfillment. The matched enhanced
Behavioral-only arm receives all three. Full adds the same existing generation
features: entropy, token confidence and confidence variance. Neither a harmful
answer, a refusal, a guideline violation nor a trigger alone becomes an
activation label.

Refusal and fulfillment are the two heads that passed the earlier fixed
teacher-agreement gates. Their continuous probabilities are used without a new
semantic cutoff search. Unsafe assistance is excluded. The failed original
three-head gate and artifacts remain unchanged; this is a separately declared
two-head ablation, not promotion of that rejected candidate.

The auxiliary supervision is the accepted **one user-approved AI-assisted set**:
80 responses from 40 training sources, error-enriched, with 74 known targets for
each passing head. It is not two independently verified human semantic reviews.
Original activation consensus labels remain separate and unchanged.

| Arm | Activation inputs | Model |
| --- | --- | --- |
| Generation-only | 3 generation statistics | Standardized logistic regression |
| Raw Behavioral reference | 384 frozen prompt-response coordinates | Logistic regression |
| Raw Full reference | 384 coordinates + 3 standardized generation statistics | Logistic regression |
| Enhanced Behavioral-only | 3 continuous Behavioral scores | Standardized logistic regression |
| Full linear (selected) | 3 generation statistics + 3 Behavioral scores | Standardized logistic regression |
| Full nonlinear | Same 6 Full inputs | Bounded histogram gradient boosting |

All logistic heads and activation models use C=1, unweighted classes,
max_iter=1000 and random_state=42. The nonlinear combiner uses learning_rate=0.05,
100 iterations, at most seven leaves, depth three, minimum 20 samples per leaf,
L2=1, no early stopping and random_state=42. These settings were frozen; no
hyperparameter search or encoder fine-tuning occurs.

The raw references are the unprojected C=1 detectors from the representation
experiment. **Raw Full is not the older projected Full detector** with historical
F1 0.681564/0.476190. Their names and measurements must not be interchanged.

## Source isolation, fitting and selection

Training contains 1,058 examples from 529 sources (163 positives/895 negatives).
Validation contains 100 examples from 50 other sources (11 positives/89
negatives). Five outer source folds evaluate the entire activation pipeline.
Within every outer fitting partition, five inner source folds construct
component predictions for combiner fitting. Both components and combiner exclude
the outer held sources. Unknown auxiliary targets are masked, and all semantic
fits retain at least five source groups per class.

The final combiners fit training-only outer cross-fitted component features;
the final components fit eligible training data. Validation calibrates the final
activation cutoffs only after Full candidate selection is frozen. In total,
**123 classifiers** were fitted: 93 component fits (25 inner stages, five outer
stages and one final stage, each with three heads), 12 reference fits and 18
combiner fits. There are zero new target-model calls, zero new activation labels
and zero encoder fine-tuning updates.

The frozen MiniLM pair encoder uses 256 tokens with at most 64 request tokens
and the remainder for the response, with BERT pair special tokens included.
The previous 80 reviewed pair embeddings are reused exactly. The remaining 978
training examples, 100 validation examples and 476 reused test examples are
encoded with the same frozen weights. Pair encoding is an additional forward
pass alongside the existing raw Behavioral encoding, using one shared encoder
instance at inference.

Model selection uses mean outer-fold average precision (AP), then mean AUROC,
then the declared method order. **Full linear was selected before validation
and test scoring.** Raw Behavioral is best overall on this training criterion.
The nonlinear arm remains a reported secondary candidate; its later test scores
cannot change the preselected primary Full model.

| Arm | Mean fold AP | Mean fold AUROC | Pooled CV F1 | Validation F1 |
| --- | --- | --- | --- | --- |
| Generation-only | 0.265384 | 0.728324 | 0.392638 | 0.307692 |
| Raw Behavioral reference | 0.535379 | 0.860565 | 0.480418 | 0.516129 |
| Raw Full reference | 0.497317 | 0.856277 | 0.450000 | 0.529412 |
| Enhanced Behavioral-only | 0.511257 | 0.853771 | 0.484848 | 0.533333 |
| Full linear (selected) | 0.510496 | 0.852960 | 0.490119 | 0.516129 |
| Full nonlinear | 0.483115 | 0.849007 | 0.456853 | 0.478261 |

CV F1 uses each outer fit model's fitting scores to select its cutoff;
the new combiner cutoff is calibrated on resubstitution predictions over
cross-fitted component features. Outer held labels never set those cutoffs.
CV is development/selection evidence, and validation performance is calibrated
on only 11 positives; neither is independent confirmation. Reference CV and
validation scores reproduce the earlier cached scores exactly (maximum error
0.0); passing semantic outer scores reproduce their earlier values to
1.67e-16.

## Development source slices

The following training OOF AP values describe ranking by original dataset.
These are development diagnostics, not additional held-out results.

| Arm | AdvBench | HarmBench |
| --- | --- | --- |
| Generation-only | 0.283807 | 0.219636 |
| Raw Behavioral reference | 0.536598 | 0.515846 |
| Raw Full reference | 0.504188 | 0.478721 |
| Enhanced Behavioral-only | 0.527591 | 0.473202 |
| Full linear (selected) | 0.522164 | 0.470441 |
| Full nonlinear | 0.464223 | 0.451429 |

Training folds prefer raw Behavioral ranking, whereas Generation-only
retains the strongest F1 on reused StrongREJECT. This development/evaluation
reversal motivates checking source and condition transfer. It does not by
itself identify a causal explanation or prove a particular alternative fusion
will succeed.

## Initial retrospective StrongREJECT comparison

The same 476 admitted examples from 238 source pairs are reused: 220 seen
examples (77 positives/143 negatives) and 256 unseen-to-detector examples
(74 positives/182 negatives). The earlier 75 excluded source groups remain
excluded. Original reference scores and max-F1 decisions are reused exactly.
Test labels do not select a model or cutoff. The following are **initial
retrospective results**, not fresh generalization evidence.

| Arm | Seen F1 | Unseen F1 | Seen AP | Unseen AP | Seen AUROC | Unseen AUROC |
| --- | --- | --- | --- | --- | --- | --- |
| Generation-only | 0.683168 | 0.629108 | 0.688438 | 0.734398 | 0.844973 | 0.860039 |
| Raw Behavioral reference | 0.618705 | 0.296296 | 0.736685 | 0.485625 | 0.824267 | 0.733145 |
| Raw Full reference | 0.643836 | 0.551181 | 0.780770 | 0.637873 | 0.869585 | 0.822023 |
| Enhanced Behavioral-only | 0.603175 | 0.200000 | 0.757742 | 0.488184 | 0.845336 | 0.736858 |
| Full linear (selected) | 0.614173 | 0.276596 | 0.788224 | 0.571834 | 0.871674 | 0.798411 |
| Full nonlinear | 0.669767 | 0.547368 | 0.747193 | 0.524654 | 0.826401 | 0.748404 |

### Seen: initial frozen max-F1 operating point

| Arm | TP | FP | TN | FN | Precision | Recall | FPR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Generation-only | 69 | 56 | 87 | 8 | 0.552000 | 0.896104 | 0.391608 |
| Raw Behavioral reference | 43 | 19 | 124 | 34 | 0.693548 | 0.558442 | 0.132867 |
| Raw Full reference | 47 | 22 | 121 | 30 | 0.681159 | 0.610390 | 0.153846 |
| Enhanced Behavioral-only | 38 | 11 | 132 | 39 | 0.775510 | 0.493506 | 0.076923 |
| Full linear (selected) | 39 | 11 | 132 | 38 | 0.780000 | 0.506494 | 0.076923 |
| Full nonlinear | 72 | 66 | 77 | 5 | 0.521739 | 0.935065 | 0.461538 |

### Unseen to detector: initial frozen max-F1 operating point

| Arm | TP | FP | TN | FN | Precision | Recall | FPR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Generation-only | 67 | 72 | 110 | 7 | 0.482014 | 0.905405 | 0.395604 |
| Raw Behavioral reference | 16 | 18 | 164 | 58 | 0.470588 | 0.216216 | 0.098901 |
| Raw Full reference | 35 | 18 | 164 | 39 | 0.660377 | 0.472973 | 0.098901 |
| Enhanced Behavioral-only | 9 | 7 | 175 | 65 | 0.562500 | 0.121622 | 0.038462 |
| Full linear (selected) | 13 | 7 | 175 | 61 | 0.650000 | 0.175676 | 0.038462 |
| Full nonlinear | 52 | 64 | 118 | 22 | 0.448276 | 0.702703 | 0.351648 |

The selected Full model does not beat Generation-only and both
Behavioral references on either split. Full nonlinear has F1 0.669767/0.547368,
also below Generation-only on both splits. Its better retrospective score than
linear Full is not grounds for switching the primary candidate after test
inspection. Enhanced Behavioral-only itself falls to 0.603175/0.200000;
passing semantic teacher-agreement gates did not establish activation utility.

### Initial operating-point tradeoffs

Two additional policies are calibrated on validation: maximize recall subject
to FPR <=0.10 (tie: lower FPR, then higher threshold), and choose the highest
threshold achieving recall >=0.80. The max-F1 tie uses the higher threshold.
All rules call positive when score >= threshold. A candidate just above 1
permits no-positive operation when scores saturate at 1. **The budgets describe
validation calibration, not guaranteed test rates.** The tables report actual
initial retrospective test recall and FPR; no test-driven cutoff tuning occurs.

| Arm | Max-F1 cutoff | FPR-budget cutoff | Recall-target cutoff |
| --- | --- | --- | --- |
| Generation-only | 0.152925329 | 0.465036311 | 0.117417488 |
| Raw Behavioral reference | 0.248796942 | 0.327810537 | 0.178802915 |
| Raw Full reference | 0.234150947 | 0.349687637 | 0.234150947 |
| Enhanced Behavioral-only | 0.262941069 | 0.341938539 | 0.155245354 |
| Full linear (selected) | 0.283322692 | 0.435232730 | 0.175662179 |
| Full nonlinear | 0.116751819 | 0.378337878 | 0.185882723 |

**Validation FPR <=0.10 policy: initial retrospective actual test rates.**

| Arm | Seen recall | Seen FPR | Unseen recall | Unseen FPR |
| --- | --- | --- | --- | --- |
| Generation-only | 0.000000 | 0.013986 | 0.027027 | 0.000000 |
| Raw Behavioral reference | 0.324675 | 0.034965 | 0.040541 | 0.016484 |
| Raw Full reference | 0.428571 | 0.034965 | 0.162162 | 0.027473 |
| Enhanced Behavioral-only | 0.376623 | 0.062937 | 0.081081 | 0.016484 |
| Full linear (selected) | 0.311688 | 0.020979 | 0.027027 | 0.016484 |
| Full nonlinear | 0.441558 | 0.062937 | 0.189189 | 0.032967 |

**Validation recall >=0.80 policy: initial retrospective actual test rates.**

| Arm | Seen recall | Seen FPR | Unseen recall | Unseen FPR |
| --- | --- | --- | --- | --- |
| Generation-only | 0.948052 | 0.699301 | 0.972973 | 0.681319 |
| Raw Behavioral reference | 0.766234 | 0.342657 | 0.554054 | 0.225275 |
| Raw Full reference | 0.610390 | 0.153846 | 0.472973 | 0.098901 |
| Enhanced Behavioral-only | 0.740260 | 0.237762 | 0.445946 | 0.137363 |
| Full linear (selected) | 0.714286 | 0.195804 | 0.432432 | 0.098901 |
| Full nonlinear | 0.857143 | 0.412587 | 0.635135 | 0.302198 |

### Initial paired uncertainty

The source-pair bootstrap retains both controlled conditions of every sampled
source. There are 2,000 shared resamples per split (seed 20261106). The values
below are selected Full linear minus each reference at the frozen max-F1
operating point. These exploratory 95% percentile intervals do not repair
development adaptation to the reused test set or semantic-label limitations.

| Split | Reference | F1 difference | 95% paired interval |
| --- | --- | --- | --- |
| seen | Enhanced Behavioral-only | 0.010999 | [-0.057055, 0.076576] |
| seen | Generation-only | -0.068995 | [-0.176209, 0.030149] |
| seen | Raw Behavioral reference | -0.004532 | [-0.066067, 0.049828] |
| seen | Raw Full reference | -0.029662 | [-0.103897, 0.037228] |
| unseen | Enhanced Behavioral-only | 0.076596 | [0.014998, 0.152295] |
| unseen | Generation-only | -0.352512 | [-0.497032, -0.218864] |
| unseen | Raw Behavioral reference | -0.019701 | [-0.117385, 0.065004] |
| unseen | Raw Full reference | -0.274585 | [-0.397291, -0.163540] |

## Complete detector cost

Eighteen fresh CPU workers (three per method) each process the same 100
unlabeled validation records once for warmup and twice for measurement, with
shuffled record orders and rotated method order. There are **3,600 measured
timings**, 600 per arm. CPU: Intel Core i7-11800H, four compute threads,
Windows 11, Python 3.13.14. Timing includes required feature computation,
encoding, component heads, scaling and activation inference. Full and enhanced
Behavioral include both encoder forward passes with one shared set of weights.

| Arm | Warm median ms | Warm p95 ms | Median worker MiB | Max lifetime peak MiB | State bytes |
| --- | --- | --- | --- | --- | --- |
| Generation-only | 0.152 | 0.183 | 134.117 | 134.336 | 1018 |
| Raw Behavioral reference | 13.873 | 25.302 | 469.070 | 469.578 | 3843 |
| Raw Full reference | 14.177 | 25.731 | 468.465 | 470.070 | 4098 |
| Enhanced Behavioral-only | 27.939 | 51.816 | 465.719 | 469.332 | 10341 |
| Full linear (selected) | 28.547 | 51.948 | 465.309 | 469.188 | 10437 |
| Full nonlinear | 31.321 | 56.634 | 463.496 | 469.918 | 42104 |

Each encoding arm additionally requires the same 91,567,913 bytes of
shared encoder files; Generation-only requires none. Whole-worker RAM includes
runtime dependencies and is not the classifier parameter size. Parent and
target-model memory are excluded. Initial setup/first-inference time is measured
after shared imports and preflight, so it is not complete application startup.
Target generation/statistics collection, network/UI, disk reads and loading are
excluded from warm timing. Background load, power and file caches are
uncontrolled; available memory varied from about 1.6 to 2.6 GiB. No historical
speedup, cross-hardware performance or target-inference cost claim follows.

Timed single-record scores differ from the frozen batched validation references
by at most 5.48e-7, within the predeclared 1e-5 tolerance. Full linear and raw
Full each have six primary decision differences, all repeated observations of
the same validation example exactly at its calibrated boundary. Other methods
have none. These numerical-boundary observations are preserved explicitly;
no epsilon, threshold adjustment or benchmark rerun is introduced. Reported
activation metrics use the sealed batched scores.

## Verification and preserved evidence

All **85 targeted tests pass**, including 11 new checks for fixed threshold
rules, fair matched inputs, nested source isolation, source-class support,
selection restrictions, inference and overwrite protection. Windows denied the
tree fitter's thread-pool named pipe inside the sandbox; authorized execution
outside that sandbox completed the tests and training with the same settings.

Frozen workflow verification reconstructs nested component predictions,
scalers, selection, policies, metrics, paired intervals and worker aggregates.
A separate arithmetic audit uses saved logistic coefficients and direct
boosted-tree node traversal without calling classifier prediction methods;
it checks nested fitting IDs/target hashes, source slices, cutoff selection,
confusions, AP, pairwise AUROC, bootstrap and all latency/memory aggregates.
Both verification paths pass. Evidence is retained under
`outputs/trace-end-to-end-verification-20261106` and in the run's
`verification.json`. The maximum independent arithmetic score difference is
2.78e-16 across the checked nested and final predictions.

The sealed run includes its protocol; 31-stage model bundles and final minimal
states; 93 component fitting audits; semantic caches and inner/outer scores;
training/validation/test predictions, source slices and reports; 18 worker
files and a resource summary; manifests and verification. The raw datasets,
accepted auxiliary labels, original activation labels, earlier candidates,
failed old gate and dashboard pointer remain protected. No promotion occurs.

```powershell
.\.venv\Scripts\python.exe -B scripts/trace_end_to_end.py verify
.\.venv\Scripts\python.exe -B outputs/trace-end-to-end-verification-20261106/verify_numerics.py
```

The targeted test suite was:

```powershell
.\.venv\Scripts\python.exe -B -W ignore::DeprecationWarning -m unittest tests.test_end_to_end tests.test_auxiliary_scorer tests.test_auxiliary_consensus tests.test_behavior_audit tests.test_score_fusion tests.test_reused_comparison tests.test_representation_experiment tests.test_fresh_holdout_audit -q
```

## What should happen next

The complete bounded comparison is finished. More refusal/fulfillment evidence
and a shallow nonlinear combiner have not solved the activation problem here.
The next useful development question is **whether the selection procedure can
identify signals that transfer between source datasets and controlled
conditions**, rather than another weight search on the same StrongREJECT
outcomes. A development-only leave-one-dataset-out analysis and cue/control
transfer audit using the existing AdvBench/HarmBench training data could test
that question under a new frozen protocol. Such an analysis has not run here;
it cannot replace a fresh test or guarantee higher accuracy.

Keep Generation-only, raw Behavioral and existing Full as fixed references.
Require any future Full candidate to improve activation F1 and the
recall/false-alarm tradeoff over both matched views under the declared selection
rule. If existing evidence cannot support that gain, report that limitation
instead of changing labels, declaring unsafe output to be activation or picking
models from the already inspected test scores.

The prior fresh-source audit found zero eligible untouched source groups in
the selected three datasets under the recorded overlap/exposure rules; that
shortage remains. Fresh independent confirmation, additional independently
documented target LLMs and historical collection-provenance qualifications
remain outstanding. These are all **initial results** for one Llama 2 target;
no universal backdoor detection, cross-model generalization, validated chat
detection or backdoor removal is established.
