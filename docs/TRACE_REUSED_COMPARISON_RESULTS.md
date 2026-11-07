# TRACE retrospective comparison using the existing dataset

Completed **6 November 2026** for the initial Llama 2 MTBA target, following the
user's instruction: **"proceed to the next step, just reuse the existing dataset"**.

The frozen unprojected Full TRACE candidate has a mixed outcome. Compared with
the original Full detector, its F1 falls from **0.681564 to 0.643836** on the
existing seen-trigger examples and rises from **0.476190 to 0.551181** on the
existing detector-unseen-trigger examples. The latter change removes false
alarms without increasing the number of detected positives. Generation-only
retains higher F1 on both splits. These findings do not establish an overall
improvement or justify promoting the candidate to the dashboard.

**These are initial, retrospective results on reused test data.** Earlier test
errors informed the representation development. This comparison cannot serve
as a fresh independent test or support a cross-model generalization claim.
The original initial results and Chapters 4/5 manuscripts remain preserved.

## Authorized dataset reuse and experiment boundaries

The preceding [fresh-source audit](TRACE_FRESH_HOLDOUT_READINESS.md) found no
eligible unused source groups within the original AdvBench, HarmBench and
StrongREJECT files. The latest user instruction authorizes reusing the existing
data. A **separate retrospective protocol** implements that instruction; it
does not change the incomplete fresh-source plan or enable its collection,
evaluation or dashboard-promotion gates.

The new run is `artifacts/trace-behavioral-v4/reused-test-comparison-001`.
Its protocol, six standalone inference states, existing validation references,
input hashes and implementation hashes were saved before new test predictions
or efficiency measurements. The six methods and thresholds match the preceding
comparison plan. No further model selection, fitting, threshold adjustment,
human relabeling, target-model inference or dashboard promotion took place.

The comparison uses the same **476 admitted adapted-model examples from 238
StrongREJECT source groups** as the original frozen evaluation:

| Existing partition | Source groups | Examples | Activation positive | Activation negative |
| --- | ---: | ---: | ---: | ---: |
| Seen cue (BadMagic) | 110 | 220 | 77 | 143 |
| Detector-unseen cue (OpenAI) | 128 | 256 | 74 | 182 |
| Total | 238 | 476 | 151 | 325 |

Both adapted conditions from each admitted source remain together. Conditions
3/4 provide clean-model attribution evidence rather than classifier test rows.
The existing four-condition consensus rule admits patterns `0000` and `0100`;
only condition 2 in a `0100` group is an activation-positive example. The same
75 excluded source groups stay excluded, with no reassignment. Cue presence,
harmful content and a detector score do not supply ground truth.

The reused test sources remain disjoint from the existing admitted training
and validation source IDs. That partition isolation prevents direct fitting on
these examples, but it does not remove research adaptation from previously
inspected test errors. "Detector-unseen" describes the original cue partition;
it does not mean that these responses are now unseen by the researchers.

Original consensus and provenance limitations remain: final agreement by both
human reviewers and linkage of both collection batches to TRACE's inference
server are user-attested. Separate reviewer submissions and exact historical
per-response server builds are unavailable. The missing historical intake CSV
is not reconstructed; the existing qualified sealed-catalogue verification
route checks available development inputs and raw hashes. The original intake
gate remains unchanged.

## Frozen comparators

The [development experiment](TRACE_BEHAVIORAL_REPRESENTATION_RESULTS.md) supplied
the new models. All encoders use the same frozen, normalized MiniLM embedding,
pinned revision and 256-token limit. The new interaction representation retains
all 384 dimensions instead of learning a 64-dimensional contrastive projection.
Generation features and their training-fitted standardization are unchanged.

| Method in saved results | Representation / classifier | Classifier dimensions | Frozen threshold |
| --- | --- | ---: | ---: |
| `original_full` | Projected prompt-response + generation, C=1 | 67 | 0.18404884955022044 |
| `candidate_full` | Unprojected prompt-response + generation, C=1 | 387 | 0.234150946937906 |
| `generation` | Generation-only, C=1 | 3 | 0.15292532911254467 |
| `original_behavioral` | Projected prompt-response, C=1 | 64 | 0.23824214492694545 |
| `candidate_behavioral` | Unprojected prompt-response, C=1 | 384 | 0.24879694155898716 |
| `tuned_full` | Unprojected prompt-response + generation, training-CV-selected C=10 | 387 | 0.2464645963030869 |

All six use unweighted logistic regression. Final classifier fits previously
used training data only; their thresholds previously used validation data.
No threshold was selected from this comparison. The C=10 row is retained as a
separate secondary comparator because its training-CV selection differs from
the matched C=1 representation comparison.

Original Full, Behavioral-only and Generation-only accuracy scores and decisions
are reused exactly from the original saved predictions. Only the three frozen
new classifiers receive the newly encoded existing test records. All 476 records
are encoded together, using the same batch call structure as the original
evaluation. The cached embeddings, generation vectors, predictions and reports
are retained under the new run; no original evaluation file is replaced.

## Initial retrospective accuracy results

Each table uses its method's existing validation-frozen threshold. The rows
therefore have different operating points; F1 comparisons are not comparisons
at a matched false-positive rate.

### Existing seen-trigger examples

| Method | Precision | Recall | F1 | False-positive rate | AUROC | Average precision | TP / FP / FN / TN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Original Full | 0.598039 | 0.792208 | 0.681564 | 0.286713 | 0.822450 | 0.632628 | 61 / 41 / 16 / 102 |
| Unprojected Full, C=1 | 0.681159 | 0.610390 | 0.643836 | 0.153846 | 0.869585 | 0.780770 | 47 / 22 / 30 / 121 |
| Generation-only | 0.552000 | 0.896104 | 0.683168 | 0.391608 | 0.844973 | 0.688438 | 69 / 56 / 8 / 87 |
| Original Behavioral | 0.662338 | 0.662338 | 0.662338 | 0.181818 | 0.801653 | 0.602682 | 51 / 26 / 26 / 117 |
| Unprojected Behavioral, C=1 | 0.693548 | 0.558442 | 0.618705 | 0.132867 | 0.824267 | 0.736685 | 43 / 19 / 34 / 124 |
| Training-CV-tuned Full, C=10 | 0.696970 | 0.597403 | 0.643357 | 0.139860 | 0.837980 | 0.756342 | 46 / 20 / 31 / 123 |

The new C=1 Full candidate has better ranking scores and fewer false alarms
than the original Full, but detects 14 fewer positives overall. It removes 19
original false alarms and introduces none. It detects two positives missed by
the original, while missing 16 that the original detected. Its improved
precision does not compensate for its lower recall in F1 at the frozen threshold.

### Existing detector-unseen-trigger examples

| Method | Precision | Recall | F1 | False-positive rate | AUROC | Average precision | TP / FP / FN / TN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Original Full | 0.479452 | 0.472973 | 0.476190 | 0.208791 | 0.730621 | 0.499312 | 35 / 38 / 39 / 144 |
| Unprojected Full, C=1 | 0.660377 | 0.472973 | 0.551181 | 0.098901 | 0.822023 | 0.637873 | 35 / 18 / 39 / 164 |
| Generation-only | 0.482014 | 0.905405 | 0.629108 | 0.395604 | 0.860039 | 0.734398 | 67 / 72 / 7 / 110 |
| Original Behavioral | 0.444444 | 0.324324 | 0.375000 | 0.164835 | 0.696020 | 0.463208 | 24 / 30 / 50 / 152 |
| Unprojected Behavioral, C=1 | 0.470588 | 0.216216 | 0.296296 | 0.098901 | 0.733145 | 0.485625 | 16 / 18 / 58 / 164 |
| Training-CV-tuned Full, C=10 | 0.565217 | 0.351351 | 0.433333 | 0.109890 | 0.740125 | 0.520757 | 26 / 20 / 48 / 162 |

The C=1 Full candidate reduces false positives from 38 to 18, while TP stays
35 and FN stays 39. These are not identical detected-positive sets: 28 are
shared, seven are newly detected and seven previously detected positives are
lost. It removes 22 original false alarms and introduces two. Its F1 increase
is therefore a net false-alarm reduction, not an increase in overall activation
sensitivity.

Against Generation-only, the C=1 Full candidate uniquely detects one positive
but misses 33 that Generation-only detects. It removes 56 Generation-only false
alarms while introducing two. This exposes a large sensitivity/false-alarm
tradeoff; it does not establish a superior dual-view detector. The Behavioral
candidate's lower frozen-threshold F1 on both splits also contradicts a blanket
claim that keeping all embedding dimensions improves activation detection.

The C=10 Full comparator does not rescue the outcome: its seen/unseen F1 is
0.643357/0.433333. No candidate is selected or recalibrated using these results.

## Paired uncertainty analysis

Two thousand paired bootstrap resamples per split retain both adapted conditions
from each source group, using seed 20261106. Each contrast uses the same sampled
groups for both methods. These are post-hoc, unadjusted percentile intervals
conditional on the admitted sources and labels. They do not account for test
adaptation, reviewer uncertainty, excluded groups or other target models.

| F1 difference | Seen point | Seen 95% interval | Unseen point | Unseen 95% interval |
| --- | ---: | --- | ---: | --- |
| C=1 Full candidate minus original Full | -0.037729 | [-0.119684, 0.040829] | 0.074991 | [-0.011319, 0.162206] |
| C=1 Full candidate minus Generation-only | -0.039333 | [-0.124903, 0.044126] | -0.077927 | [-0.192655, 0.032898] |
| C=1 Behavioral candidate minus original Behavioral | -0.043633 | [-0.125751, 0.034561] | -0.078704 | [-0.180285, 0.026218] |
| C=10 Full minus original Full | -0.038208 | [-0.119809, 0.034920] | -0.042857 | [-0.126812, 0.033096] |

All reported intervals include zero. The existing observations provide no
conclusive improvement under these contrasts, and the retrospective design
would remain a limitation even if an interval excluded zero.

## Efficiency measurements

The benchmark reuses the **100 existing validation examples from 50 source
groups**, never the reused test rows. Each method runs in three fresh processes.
Each process loads only its standalone inference state, makes an initial call,
warms up on all 100 records, then measures two deterministically shuffled passes.
This supplies **600 timed observations per method, 3,600 total**, in 18 sequential
processes. The method order rotates across the three rounds. Batch size is one,
numerical/torch intra-op threads are four, and torch inter-op threads are one.

Warm timing starts with an in-memory interaction and ends with its score and
decision. It includes each method's required tokenization, embedding, projection,
generation feature computation, scaling and classifier prediction. It excludes
disk reads, hash verification, initial loading/imports, UI/network overhead,
target-model generation and generation-score collection. Generation-only does
not load or invoke MiniLM. These are local detector costs, not end-to-end LLM
response latency.

| Method | Warm median (ms) | Warm p95 (ms) | Median resident process (MiB) | Maximum lifetime process peak (MiB) | Median post-import setup + first call (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original Full | 13.297 | 23.952 | 457.742 | 459.035 | 9.232 |
| Unprojected Full, C=1 | 13.607 | 24.564 | 457.023 | 458.711 | 9.263 |
| Generation-only | 0.150 | 0.171 | 121.922 | 121.984 | 0.060 |
| Original Behavioral | 13.610 | 24.221 | 457.379 | 458.461 | 9.166 |
| Unprojected Behavioral, C=1 | 13.599 | 24.425 | 456.977 | 458.844 | 9.254 |
| Training-CV-tuned Full, C=10 | 13.490 | 23.897 | 456.223 | 458.320 | 9.171 |

Measurements ran on Windows 11, Intel Core i7-11800H (16 logical CPUs), Python
3.13.14. Available physical memory ranged from approximately 2.096 to 3.347 GiB.
The shared scientific runtime matches the existing pinned environment. Whole
process working sets include Python, numerical libraries and the encoder when
required; they exclude the parent benchmark process and target LLM. Encoder
hashes are checked in the parent before workers, avoiding an extra full weight
read solely for worker hash verification.

The setup interval begins after shared numerical/workflow imports and preflight
reads. It includes method-specific state/model loading and the first inference,
but is **not total application startup**. Fresh processes do not imply a cold
machine file cache. Background workloads, power state and cache are uncontrolled;
these local observations do not establish a meaningful speed advantage between
the approximately 13 ms encoder-based methods or a speedup over older runs.

The compressed standalone inference states occupy 93,313 bytes (original Full),
4,119 (C=1 candidate Full), 1,015 (Generation), 92,948 (original Behavioral),
3,850 (candidate Behavioral) and 4,153 (C=10 Full). Every encoder-based method
also requires the same **91,567,913-byte MiniLM files**. State plus encoder is
87.414957 MiB for original Full versus 87.329895 MiB for candidate Full: removing
the projection makes the classifier package smaller but barely changes the
dominant encoder footprint. Generation-only requires 1,015 state bytes and no
encoder. These sizes exclude Python/libraries, target weights, source code,
datasets, reports and the complete multi-candidate development bundle.

### Numerical threshold boundary differences

Every timed score was checked against its existing frozen validation reference,
using the predeclared absolute tolerance of 1e-5. The largest score difference
was approximately **6.238e-7**. Three method/sample boundary cases produced a
different single-record decision from the original batched validation reference:

| Method | Existing validation sample | Reference score equals frozen threshold | Occurrences across six measured passes |
| --- | --- | ---: | ---: |
| Original Behavioral | `TRACE-TR-b9d7823311e2f361-condition-2` | 0.23824214492694545 | 6 |
| C=1 candidate Full | `TRACE-TR-c300f1f23b2734ae-condition-2` | 0.234150946937906 | 6 |
| C=10 Full | `TRACE-TR-b9d7823311e2f361-condition-2` | 0.2464645963030869 | 6 |

The cases involve two distinct validation samples. Small floating-point
differences between batched embedding/classification and single-record inference
can cross a cutoff equal to a calibration score. The benchmark records these
differences rather than changing thresholds or adding an epsilon. Original
validation metrics and saved test predictions remain unchanged. This numerical
boundary behavior is a deployment qualification, not new accuracy evidence.

## Evidence, verification and use

The isolated workflow is `scripts/trace_reused_evaluation.py`; inference and
comparison helpers are in `traceguard/reused_comparison.py`. The run preserves:

- `protocol.json`, six `states/*.joblib` files and `validation-records.json`;
- `evaluation/features.npz`, `predictions.jsonl`, `report.json` and `manifest.json`;
- eighteen benchmark worker JSON files, `benchmark/summary.json` and its seal;
- `verification.json`, which checks original hashes, membership, fixed scores,
  metrics, bootstrap intervals and complete validation timing coverage.

An additional independent arithmetic check is stored in
`outputs/trace-reused-comparison-verification-20261106/verify_numerics.py` and
`numeric-verification.json`. It imports neither TRACE nor scikit-learn and
recomputes confusion matrices, fixed-threshold decisions, average precision,
pairwise AUROC, whole-source bootstrap intervals, timing summaries and memory
aggregates. It also checks the original baseline predictions and protected
input hashes, plus the unchanged fresh-source plan hash and disabled gates.
Both workflow reconstruction and independent arithmetic verification passed.

Twenty-eight targeted tests pass across the reuse workflow, representation
experiment and fresh-source audit. They exercise no-refit behavior, original
prediction preservation, label/condition rules, group isolation, reproducible
paired resampling, generation-only inference without an encoder, forbidden
test-data benchmarking and refusal to overwrite existing run stages.

To verify the completed run without fitting models, generating target responses
or repeating the benchmark:

```powershell
.\.venv\Scripts\python.exe -B scripts/trace_reused_evaluation.py verify
.\.venv\Scripts\python.exe -B outputs/trace-reused-comparison-verification-20261106/verify_numerics.py
```

The `prepare`, `evaluate` and `benchmark` stages intentionally refuse to overwrite
their existing outputs. Standalone inference states contain trusted local joblib
objects and should be loaded only from the verified project artifacts.

## What this supports and the next development step

This run completes the authorized reuse comparison and quantifies the cost of
the frozen alternatives. It preserves a useful initial observation: retaining
unprojected interaction embeddings can improve ranking and reduce false alarms,
but the validation-selected operating point loses substantial recall, and
Generation-only remains stronger in F1 on the reused examples. The candidate
is not promoted to the dashboard, and the earlier initial thesis findings are
not replaced.

A proposed follow-up is a bounded **separate-view score-fusion** development
experiment using only the existing training source groups for fitting and
cross-validation and validation for calibration. Its purpose would be to test
whether the Behavioral View adds information without suppressing generation
evidence as strongly. Keep Generation-only as a reference, freeze the comparison
before evaluation, and retain the existing activation labels rather than
substituting guideline violations. This follow-up has **not** been implemented
or measured in the present run; no gain is promised. Any further reuse of the
existing StrongREJECT data would still be exploratory. Independent evidence
requires eligible fresh sources, and additional LLM configurations require
their own documented experiments.
