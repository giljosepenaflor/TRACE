# TRACE separate-view score-fusion experiment

Completed **6 November 2026** after the user approved the separate-view fusion
plan, continuing with the existing dataset.

**The selected score-fusion candidate does not improve the initial detection
results.** It achieves F1 **0.666667 seen / 0.464000 detector-unseen**, compared
with **0.683168 / 0.629108** for Generation-only. The preceding unprojected
feature-concatenation candidate remains at **0.643836 / 0.551181**. The fusion
candidate is not promoted to the dashboard.

These are **initial, retrospective results on the same previously inspected
StrongREJECT examples**, limited to the existing Llama 2 MTBA target and
independent prompts. They are not fresh held-out confirmation, cross-model
results, or evidence that every possible score-fusion method would fail.

## Implemented method and authorization

The user approved an experiment that gives each view its own score, tests a
small predefined set of combinations, selects on training cross-validation,
calibrates on validation, then evaluates the frozen candidate on reused test
data. This is an authorized experimental extension to the thesis's original
feature-concatenation design. It does not replace that methodology's original
results or change the historical chapter manuscripts.

The isolated run is `artifacts/trace-behavioral-v4/score-fusion-001`. The
implementation is `traceguard/score_fusion.py`, driven by
`scripts/trace_score_fusion.py`. Its formula is:

```text
fusion_score = (1 - w) * generation_score + w * behavioral_score
```

Here `w` is Behavioral View weight. Both component scores come from already
frozen **C=1, unweighted logistic regressions**. Behavioral uses the unprojected
384-dimensional prompt-response MiniLM embedding; Generation uses the same
three generation features and training-fitted scaler. Each view produces its
score separately. The mixture is a ranking/decision score, not a calibrated
probability of backdoor activation.

The protocol fixed eight Behavioral weights **0, 0.05, 0.10, 0.20, 0.35, 0.50,
0.75 and 1.00**, before computing any new mixtures. Zero and one are the
single-view endpoints. Six interior values are true dual-view mixtures,
including five values that give Generation at least as much weight as
Behavioral. No weight was forced to win because of the earlier test results.

The protocol separately fixes two selections: the **best interior mixture** is
the experimental fusion candidate, while the **best overall choice including
endpoints** is reported to show whether any mixture actually improves the
training selection criterion. Only the selected interior candidate receives a
new validation threshold or a new retrospective test evaluation. There is no
test sweep across the eight weights.

## Data isolation and frozen execution

The experiment reuses the existing qualified sealed evidence:

| Role | Source groups | Adapted examples | Positive / negative |
| --- | ---: | ---: | --- |
| Training weight selection | 529 | 1,058 | 163 / 895 |
| Validation threshold calibration and efficiency | 50 | 100 | 11 / 89 |
| Existing seen-cue retrospective test | 110 | 220 | 77 / 143 |
| Existing detector-unseen-cue retrospective test | 128 | 256 | 74 / 182 |

Training and validation retain the existing AdvBench/HarmBench memberships;
StrongREJECT remains outside component fitting and weight/threshold selection.
All paired conditions and source groups stay in their assigned partitions.
The original 75 excluded test source groups remain excluded. Clean conditions
3/4 remain attribution support, not classifier test rows. Activation labels
retain the existing four-condition consensus rules; neither a trigger nor
guideline-violating content supplies an activation label by itself.

For weight selection, the experiment reuses the two components' saved
out-of-fold training predictions from the preceding representation run. Their
five source-group folds were verified, including training-only fitted
projection/scaler histories and complete held-fold coverage. Each component
score was produced while its source group was excluded from that fold's fit.
Only frozen pretrained embeddings were shared across folds. No new component,
scaler, projection, encoder or meta-classifier was fitted.

Each mixture is ranked by **mean fold average precision (AP)**; ties use mean
fold AUROC, then smaller Behavioral weight. These same development folds are
reused for weight selection, so the resulting CV scores are selection evidence,
not an unbiased performance estimate or a nested-CV result.

The protocol and code/input hashes were frozen first. Training selection and
all CV results were then saved **before computing the new validation mixture**.
Validation chooses only its threshold, maximizing F1 with the higher threshold
on a tie. The resulting development seal, inference states and fixed threshold
were saved before the new test mixture was computed. Test component scores and
all three comparator predictions are reused exactly from prior saved results;
there is no new test encoding or classifier inference.

Original reviewer-consensus and historical inference-server linkage remain
user-attested; separate reviewer submissions and exact per-response historical
server builds remain unavailable. The missing historical intake CSV is not
reconstructed: the existing qualified sealed-catalogue verification route checks
available development inputs and raw hashes, and the old intake gate remains
unchanged. The fresh-source audit still finds zero eligible unused groups in
the original three sources; its plan and disabled gates remain unchanged.

## Training selection and initial validation findings

| Behavioral weight | Generation weight | Mean fold AP | Mean fold AUROC |
| ---: | ---: | ---: | ---: |
| 0.00 | 1.00 | 0.265384 | 0.728324 |
| 0.05 | 0.95 | 0.277890 | 0.748400 |
| 0.10 | 0.90 | 0.292192 | 0.765339 |
| 0.20 | 0.80 | 0.321430 | 0.790379 |
| 0.35 | 0.65 | 0.373519 | 0.818818 |
| 0.50 | 0.50 | 0.451469 | 0.840119 |
| **0.75** | **0.25** | **0.530908** | **0.859915** |
| 1.00 | 0.00 | 0.535379 | 0.860565 |

The selected interior mixture is **75% Behavioral / 25% Generation**, while
**Behavioral-only wins the overall training comparison**. Thus even the
selection criterion does not demonstrate an advantage from adding Generation
to this Behavioral component. Generation-heavy mixtures were considered but
did not win under the predeclared training-only rule. Their test outcomes were
not computed to search for a replacement after seeing the selected result.

The selected validation threshold is **0.2171443000857341**. Comparator
thresholds stay fixed at 0.18404884955022044 (original Full),
0.234150946937906 (unprojected Full), and 0.15292532911254467 (Generation).

| Validation method | Precision | Recall | F1 | FPR | AUROC | AP | TP / FP / FN / TN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Original Full | 0.333333 | 0.818182 | 0.473684 | 0.202247 | 0.852911 | 0.375948 | 9 / 18 / 2 / 71 |
| Unprojected Full | 0.391304 | 0.818182 | 0.529412 | 0.157303 | 0.864147 | 0.347680 | 9 / 14 / 2 / 75 |
| Generation-only | 0.195122 | 0.727273 | 0.307692 | 0.370787 | 0.648621 | 0.172018 | 8 / 33 / 3 / 56 |
| Selected score fusion | 0.391304 | 0.818182 | 0.529412 | 0.157303 | 0.864147 | 0.352150 | 9 / 14 / 2 / 75 |

Fusion matches the unprojected concatenation candidate's validation confusion
counts and F1, without establishing improved deployment performance. Validation
has only eleven positives and supplies the selected threshold. Equal confusion
counts do not by themselves establish identical per-example predictions.

## Initial retrospective test results

| Existing split | Method | Precision | Recall | F1 | FPR | AUROC | AP | TP / FP / FN / TN |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Seen | Original Full | 0.598039 | 0.792208 | 0.681564 | 0.286713 | 0.822450 | 0.632628 | 61 / 41 / 16 / 102 |
| Seen | Unprojected Full | 0.681159 | 0.610390 | 0.643836 | 0.153846 | 0.869585 | 0.780770 | 47 / 22 / 30 / 121 |
| Seen | Generation-only | 0.552000 | 0.896104 | 0.683168 | 0.391608 | 0.844973 | 0.688438 | 69 / 56 / 8 / 87 |
| Seen | Selected score fusion | 0.658228 | 0.675325 | 0.666667 | 0.188811 | 0.852602 | 0.770892 | 52 / 27 / 25 / 116 |
| Detector-unseen | Original Full | 0.479452 | 0.472973 | 0.476190 | 0.208791 | 0.730621 | 0.499312 | 35 / 38 / 39 / 144 |
| Detector-unseen | Unprojected Full | 0.660377 | 0.472973 | 0.551181 | 0.098901 | 0.822023 | 0.637873 | 35 / 18 / 39 / 164 |
| Detector-unseen | Generation-only | 0.482014 | 0.905405 | 0.629108 | 0.395604 | 0.860039 | 0.734398 | 67 / 72 / 7 / 110 |
| Detector-unseen | Selected score fusion | 0.568627 | 0.391892 | 0.464000 | 0.120879 | 0.792694 | 0.555027 | 29 / 22 / 45 / 160 |

On the seen examples, fusion detects five more positives than the preceding
unprojected Full candidate while producing five more false alarms; F1 rises
0.643836 to 0.666667. It remains below original Full and Generation-only in F1.

On the detector-unseen examples, fusion detects **29/74** positives versus
35/74 for either Full comparator and 67/74 for Generation-only. Against the
unprojected Full candidate, it has six fewer detected positives and four more
false alarms. Against Generation-only, its 50 fewer false alarms come with
38 fewer detected activations. It therefore fails the intended goal of
retaining Generation's sensitivity while reducing its false alarms.

These comparisons use separately validation-frozen thresholds, not a matched
false-positive operating point. No threshold or weight is adjusted using the
reused test results. The experiment's development selection favors Behavioral
scores, whereas Generation-only has higher F1 on these existing StrongREJECT
examples. That mismatch identifies a limitation of the available development
selection evidence; it does not prove a particular causal explanation.

### Paired source-group uncertainty

Two thousand paired bootstrap resamples per split preserve both adapted
conditions of each source, using seed 20261106. Intervals are post-hoc,
unadjusted percentiles conditional on the admitted examples and labels.

| Fusion F1 minus reference | Seen difference | Seen 95% interval | Detector-unseen difference | Detector-unseen 95% interval |
| --- | ---: | --- | ---: | --- |
| Original Full | -0.014898 | [-0.082997, 0.052018] | -0.012190 | [-0.098420, 0.076680] |
| Unprojected Full | 0.022831 | [-0.019159, 0.069371] | -0.087181 | [-0.155395, -0.026304] |
| Generation-only | -0.016502 | [-0.095891, 0.063686] | -0.165108 | [-0.284509, -0.054577] |

The detector-unseen differences against Generation-only and unprojected Full
are negative throughout these intervals. This reinforces the unfavorable
outcome on the reused observations; it does not remove adaptation bias,
reviewer/provenance uncertainty or limitations from source exclusions. No
independent generalization claim follows from any of these intervals.

## Matched validation-only efficiency

Twelve fresh processes compare original Full, unprojected Full, Generation-only
and fusion. Each method has three processes, each with an initial call, one
100-record warmup pass and two shuffled measured passes over the same 100
validation examples: **600 observations per method, 2,400 total**. Method order
rotates across rounds. Batch size is one, numerical/torch intra-op threads are
four, and torch inter-op threads are one.

Fusion encodes the interaction once and evaluates its two frozen classifiers
before blending their scores. Generation-only does not load an encoder. Warm
timing includes all required numerical features, tokenization/encoding,
projection/scaling, classifier inference, blending and threshold decision. It
excludes disk/hash checks, UI/network, target generation, generation-statistics
collection and initial loading/imports.

| Method | Warm median (ms) | Warm p95 (ms) | Median resident worker (MiB) | Maximum lifetime worker peak (MiB) | Median post-import setup + first call (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original Full | 14.540 | 28.850 | 457.168 | 458.199 | 10.081 |
| Unprojected Full | 16.041 | 34.924 | 456.664 | 458.188 | 9.495 |
| Generation-only | 0.145 | 0.163 | 123.594 | 124.547 | 0.058 |
| Selected score fusion | 16.871 | 30.928 | 456.047 | 457.895 | 10.076 |

Hardware is Intel Core i7-11800H, 16 logical CPUs, Windows 11, Python 3.13.14.
Available physical memory ranged from approximately 1.502 to 2.352 GiB. These
are whole-worker CPU measurements, excluding the parent and target LLM. The
setup interval begins after shared workflow/numerical imports and preflight;
it is not total process or application startup. Background load, power and
cache were uncontrolled. Fresh workers do not imply cold-machine caches, and
differences from earlier runs must not be presented as historical speedups or
regressions caused by a single implementation change.

Fusion's compressed standalone state is **4,500 bytes**, compared with 4,119
for unprojected Full, 93,313 for original Full and 1,015 for Generation-only.
Each encoder-based method also needs the same **91,567,913-byte MiniLM files**.
Fusion state plus encoder occupies 87.330258 MiB, excluding Python/libraries,
target weights, source code, datasets and reports. Two classifiers do not
require two copies of the encoder in this implementation.

All timed scores stayed within the predeclared 1e-5 absolute tolerance of
their frozen validation references. Maximum observed score difference was
approximately 5.785e-7 across methods and 1.904e-7 for fusion. Fusion had **zero
decision differences** from its frozen validation reference in these timings.
The unprojected Full comparator retained one exact-threshold boundary case,
`TRACE-TR-c300f1f23b2734ae-condition-2`, repeated in all six measured passes;
its reference score equals its frozen 0.234150946937906 cutoff. The other
comparators had zero decision differences. No epsilon or recalibration was
introduced, and original validation/test results remain unchanged.

## Verification and artifacts

All **40 targeted tests passed**: twelve new fusion tests and the existing
28 reuse/representation/fresh-source tests. They check mixture bounds and
endpoints, deterministic selection, exclusion of validation/test records from
weight selection, group/fold isolation, validation-only calibration, frozen
component inference with one encoding, exact baseline preservation, complete
paired resampling and refusal to overwrite completed stages.

Both the frozen workflow verification and an independent arithmetic audit
passed. The independent audit imports neither TRACE nor scikit-learn. It
recomputes the eight mixtures, five-fold AP/pairwise AUROC, both selection
rankings, validation threshold/metrics, saved test formula/decisions/confusions,
paired bootstrap intervals, and all 2,400 timing/memory aggregates. Protected
original inputs, original test scores and the earlier fresh-source plan hashes
remain unchanged.

The run contains `protocol.json`; `development/selection.json`, `cv-results.json`,
`training-oof.npz`, `validation-records.json`, `report.json`, four inference
states and a manifest; retrospective predictions/report/manifest; twelve
benchmark worker files, summary and manifest; and `verification.json`.
Independent evidence is stored under
`outputs/trace-score-fusion-verification-20261106`.

Read-only reconstruction commands:

```powershell
.\.venv\Scripts\python.exe -B scripts/trace_score_fusion.py verify
.\.venv\Scripts\python.exe -B outputs/trace-score-fusion-verification-20261106/verify_numerics.py
```

The prepare/develop/evaluate/benchmark stages refuse to overwrite their
existing outputs. State files are trusted local joblib artifacts and retain
the verified encoder/component configuration. No new target-model request,
review label, component fit or dashboard promotion occurred. One new mixture
threshold was selected from validation, then held fixed.

## Decision and remaining work

Retain this bounded negative result as an **initial single-target experiment**.
Neither keeping the full embedding nor this training-selected probability-score
mixture has solved the missed-activation problem. Generation-only remains the
strong F1/sensitivity reference on the reused test examples, with its substantial
false-alarm cost reported alongside that result. It is not automatically
promoted or declared the best detector for other models or deployments.

Further improvement needs a new development hypothesis and a frozen plan,
including scrutiny of why the development selection criterion favors a view
that performs poorly on the existing detector-unseen examples. Repeatedly
searching weights against StrongREJECT would only add test adaptation. A
possible later experiment is an independently specified response-behavior
feature for refusal/compliance, retaining the existing causal activation
labels; it is not implemented or measured here. Eligible fresh-source
confirmation and separately documented evaluation on additional LLMs remain
outstanding. The current dashboard and earlier chapter manuscripts stay
unchanged.
