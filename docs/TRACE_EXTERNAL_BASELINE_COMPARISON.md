# TRACE external-baseline audit and matched comparison

The external comparison uses the same 220 seen and 256 unseen-to-detector
examples as the frozen TRACE evaluation. Full TRACE has higher F1 than the
ConfGuard rule replays on both partitions, but lower unseen F1 than the adapted
semantic component. Generation-only remains the highest-F1 method in this
comparison. These observations do not establish a fusion advantage or a
reproduction of either publication's reported results.

## Method audit and fidelity

The thesis names Wang et al.'s ConfGuard [33] and Zanbaghi et al.'s semantic-drift
approach [37]. The audit consulted the authors' papers and public code, retained
immutable source snapshots and MIT licenses, and implemented offline detector
logic in `traceguard/baselines.py`. Author inference/training modules and the
notebook were never imported or executed. No target generation or new canary
responses were created.

### ConfGuard

ConfGuard flags a run of ten consecutive token probabilities strictly above
0.99. Its main setup uses greedy decoding and a 50-token generation budget.
The paper and public detector function agree on the decision rule.
Sources: [extended paper v3, Algorithm 1](https://arxiv.org/html/2508.01365v3),
[pinned author implementation](https://github.com/hanbaoergogo/ConfGuard/blob/35b0268bbec0ffd841486252a1fc111e4ac775cc/defense_vllm.py).

TRACE comparison choices, frozen before baseline test scoring:

- **ConfGuard prefix-50 replay:** apply that rule to at most the first 50 saved
  confidence values, including EOS if present. No parameters are fitted.
- **ConfGuard full-256 adaptation:** apply the identical rule across at most all
  256 saved tokens. This is a supplementary horizon adaptation for TRACE's response
  budget, not the authors' main experimental setting.
- Saved generated-token confidence is interpreted as greedy top-1 confidence
  under the declared decoding settings and researcher-attested TRACE server.
  Historical argmax/full logits were not retained, so that interpretation is
  qualified rather than independently reconstructed.
- For AUROC only, this implementation defines a ranking score: the greatest
  minimum probability among complete ten-token windows. A short response with
  no complete window scores zero. A score strictly above 0.99 gives the same
  flag as the author rule. This continuous ranking extension is explicitly
  separate from the authors' binary output.

The old responses are replayed without stopping/regenerating target outputs.
Prefix-50 evidence is compared with activation labels from the complete saved
answers. The original target, generation stack and ground-truth construction
also differ from the authors' experiment. This is a detector-rule transfer,
not a recreation of their entire experiment or end-to-end intervention latency.

### Semantic drift

The semantic component encodes response text with MiniLM, measures cosine drift
from a safe-response centroid and standardizes drift against safe statistics.
The full paper combines this with injected canary questions. Its released
notebook selects a drift threshold over the same collected examples; its canary
implementation also differs from the paper's per-reference maximum comparison.
Sources: [paper, Sections 3.3–3.6](https://arxiv.org/html/2511.15992v1),
[pinned author notebook, cells 5 and 12–17](https://github.com/ShahinZa/COMP8700/blob/3059be67f6fafc198b7ff83e724caeb289072dca/COMP8700_Sleeper_Agent_Final.ipynb).

TRACE lacks canary-injected responses and canary-specific safe baselines.
Consequently the **full drift-plus-canary system is not reproduced** and has
no fabricated score or latency. The implemented row is explicitly named
**adapted semantic-drift component**:

- Literal response text only; no prompt marker, contrastive projection or TRACE
  logistic classifier. The existing pinned MiniLM snapshot provides raw,
  unnormalized 384-dimensional embeddings.
- The reference uses all 529 admitted **training adapted/no-cue** responses,
  each label zero. Triggered negatives are excluded from this safe-mode reference.
  No clean test answer or test prompt-specific reference is used.
- Fit the centroid and safe drift mean/sample standard deviation on that training
  reference only. Degenerate scales cause refusal rather than an invented value.
- Select the z-score threshold using the author notebook's ascending grid
  `numpy.arange(0.3, 3.5, 0.1)` on the existing 100 validation examples only.
  Keep the first grid value on F1 ties; retain the author's 2.0 fallback when
  all candidate F1 values are zero. Detection uses strict `z > threshold`.
- Validation selected **0.6000000000000001** (displayed as 0.6). It was frozen
  before baseline test scoring. Validation contains only 11 positive examples,
  limiting claims about calibration stability.

These adaptations preserve source isolation and the component's geometry while
changing its baseline data, target model, label policy and calibration design.
They must not be described as the full combined detector or the authors' results.

## Frozen comparison and evidence

The separate run is `artifacts/trace-training-v3/external-baselines-001`.
`protocol.json` was written before calibration and baseline test scoring and
pins code/tests, source snapshots, data memberships, model/encoder hashes and
all method/timing settings. The semantic state and validation outputs have a
separate `model/freeze.json`. The one-time baseline test output has its own
start record, prediction file and checksum seal.

This is a **retrospective comparison**: TRACE's original test outcomes had
already been observed. External baseline choices were fixed from the method
audit and compatibility requirements, and no baseline parameter search used
test labels/results. This is not a prospectively blinded new holdout experiment.

The original 476 TRACE scores/decisions are reused exactly, without rescoring
the held-out examples or retraining TRACE. Source assignments, 75 excluded test
groups, all 1,566 original/correction files, the review workbook, frozen labels,
candidate and thresholds remain unchanged. Two-reviewer consensus and original/
correction server linkage retain their user-attested evidence level. No new
reviewer identity, label, exact historical build claim or canary response was made.

## Measured test results

All metrics are fractions, rounded to six places. ConfGuard AUROCs use the
disclosed ranking extension. Other thresholds are independently frozen; an F1
comparison does not imply equal false-positive operating points.

| Method | Seen F1 | Unseen F1 | Seen false-positive rate | Unseen false-positive rate | Seen AUROC | Unseen AUROC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full TRACE | 0.681564 | 0.476190 | 0.286713 | 0.208791 | 0.822450 | 0.730621 |
| Behavioral-only TRACE | 0.662338 | 0.375000 | 0.181818 | 0.164835 | 0.801653 | 0.696020 |
| Generation-only TRACE | 0.683168 | 0.629108 | 0.391608 | 0.395604 | 0.844973 | 0.860039 |
| ConfGuard prefix-50 replay | 0.434783 | 0.337209 | 0.342657 | 0.379121 | 0.567069 | 0.487897 |
| ConfGuard full-256 adaptation | 0.452632 | 0.375000 | 0.489510 | 0.521978 | 0.557806 | 0.520196 |
| Adapted semantic-drift component | 0.605578 | 0.546154 | 0.685315 | 0.631868 | 0.763963 | 0.709459 |

| Partition | Method | Accuracy | Precision | Recall | TN | FP | FN | TP |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Seen | ConfGuard prefix-50 | 0.586364 | 0.416667 | 0.454545 | 94 | 49 | 42 | 35 |
| Unseen | ConfGuard prefix-50 | 0.554688 | 0.295918 | 0.391892 | 113 | 69 | 45 | 29 |
| Seen | ConfGuard full-256 | 0.527273 | 0.380531 | 0.558442 | 73 | 70 | 34 | 43 |
| Unseen | ConfGuard full-256 | 0.492188 | 0.291045 | 0.527027 | 87 | 95 | 35 | 39 |
| Seen | Adapted semantic component | 0.550000 | 0.436782 | 0.987013 | 45 | 98 | 1 | 76 |
| Unseen | Adapted semantic component | 0.539063 | 0.381720 | 0.959459 | 67 | 115 | 3 | 71 |

The adapted semantic component detects many positives but flags 98 seen and
115 unseen negatives. Full TRACE has substantially fewer false positives,
while losing sensitivity on unseen positives. Extending the confidence horizon
increases both true and false positives. These findings are bounded to this
Llama 2 MTBA target and admitted corpus. They are not statistical-superiority,
cross-model, universal-detection or author-publication-performance claims.

## Source records and implementation checks

Audit artifacts live in `outputs/trace-external-baseline-audit-20261104-001`.
`sources.json` pins ConfGuard commit `35b0268bbec0ffd841486252a1fc111e4ac775cc`
and semantic commit `3059be67f6fafc198b7ff83e724caeb289072dca`, file hashes,
immutable download URLs and retrieval time. Source licenses are preserved.
The thesis's bibliography/source draft is untouched. The ConfGuard audit used
the author-linked extended arXiv v3 because the publisher DOI could not be read
through the browsing tool; no claim of a line-by-line AAAI-PDF reproduction follows.

Eleven harmless tests passed for strict boundaries, consecutive runs, horizons,
short outputs, rule/ranking equivalence, static author-rule inspection, invalid
signals, cosine/reference moments, sample normalization, degenerate scale refusal,
grid/tie/fallback behavior, train/validation isolation, coverage, supplied strict
decisions, undefined AUROC and refusal of repeated test evaluation. The fixtures
are software checks, not experimental data. No author target-model module was run.

## Matched efficiency measurements

The comparison remeasured all six methods on the same 100 frozen validation
inputs: three fresh worker processes per method, one complete warm-up pass and
two measured passes per process. That gives 600 single-interaction timings per
method and 3,600 total. View order rotates across rounds. Encoders run on CPU
with four scientific-library threads and one PyTorch interop thread. No held-out
test example is rescored for timing.

The hardware is the local Intel Core i7-11800H, 16 logical CPUs, Windows 11 and
Python 3.13.14. Benchmarking finished November 4, 2026, at 20:48:41 Asia/Manila.
The warmed boundary starts from an in-memory saved interaction and ends at the
detector decision. It includes the required feature/encoder/classifier work.
ConfGuard uses its binary early-exit rule; the AUROC ranking extension is not
timed. Target generation, probability collection, I/O, source checks, setup and
network/UI are excluded for every method.

| Method | Warm median (ms) | Warm p95 (ms) | Warm mean (ms) | Median warm resident worker (MiB) | Largest lifetime peak worker (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full TRACE | 13.4842 | 24.0702 | 14.6002 | 457.895 | 459.809 |
| Behavioral-only TRACE | 13.1957 | 23.5145 | 14.3951 | 457.188 | 459.203 |
| Generation-only TRACE | 0.1443 | 0.1591 | 0.1464 | 117.715 | 117.902 |
| ConfGuard prefix-50 replay | 0.0015 | 0.0018 | 0.0112 | 31.949 | 115.801 |
| ConfGuard full-256 adaptation | 0.0018 | 0.0062 | 0.0120 | 31.961 | 115.805 |
| Adapted semantic component | 12.0073 | 22.7278 | 13.2412 | 450.617 | 452.852 |

These are timings of repeated existing inputs, not 600 unique cases per method.
The occasional approximately 2 ms confidence-rule outliers are retained, which
explains the difference between their mean and median. The fast rule timings
assume confidences already exist and must not be described as full LLM inference
latency or a measurement of online early-stop savings.

**Memory scope:** resident values describe the warmed worker after measurements.
Lifetime peaks include earlier protocol/data/encoder-hash verification. In
particular, verifying the 90 MB encoder weights creates a transient peak even
for a confidence-only worker. The approximately 116 MiB ConfGuard peak therefore
does not represent an intrinsic memory requirement of the small rule. Runtime,
verification and interpreter memory are not model parameter sizes. The peak
counter is not reset at the timer boundary; target LLM memory is excluded.

Observed available physical RAM ranged from 1,823.14 to 2,490.42 MiB. Background
workload, power state and OS cache were uncontrolled. These matched timings
supersede no earlier artifact: the prior TRACE-only run used different machine
conditions and remains unchanged. Differences between old and new timing values
are not evidence of a model optimization.

The measured initialization-plus-first-call phase, after protocol/input/JSON
state loading, had medians of 10.442 seconds Full, 10.426 Behavioral, 1.374
Generation and 10.212 semantic. Confidence rules have no heavy model to initialize
in this phase and their first-call measurements were only approximately 8–10
microseconds. Interpreter startup, source verification and disk-cache clearing
are excluded; these are not cold-machine startup times.

| Stored item | Bytes | Scope |
| --- | ---: | --- |
| Confidence-rule configuration | 79 | Settings only; no learned parameters; code/runtime excluded |
| Semantic state file | 42,520 | Centroid, moments, calibration grid and membership/provenance metadata |
| Required MiniLM files | 91,567,913 | Required by semantic, Full and Behavioral components |
| Existing shared TRACE joblib | 146,151 | Original encoder projection/scaler/three classifiers and metadata |

The confidence rules require no embedding model. Disk figures exclude Python,
libraries, target LLM/adapter weights and datasets/reports. The full canary system
has no invented size, memory or latency measurement.

Single-input scores agreed with the frozen validation references within the
predeclared tolerances: maximum absolute differences were 1.788e-6 semantic,
5.785e-7 Full, 5.718e-7 Behavioral and 1.665e-16 Generation. Confidence rules
matched exactly. Only Behavioral showed the previously observed numerical
boundary decision difference for `TRACE-TR-b9d7823311e2f361-condition-2`, in all
six timed occurrences. No original calibration, threshold or score was changed.

## Verification and remaining scope

Main verification checks source/model/evaluation seals, all 476 prediction
memberships, confidence-rule replay, strict semantic decisions, all six new
test metric sets, unchanged original TRACE results, 18 worker lineages and all
3,600 validation timing memberships/summaries. It does not rerun the semantic
encoder on held-out data. An independent standard-library checker recomputes
confusion matrices, F1 and pairwise AUROC, reselects the semantic threshold from
saved validation scores only, and recalculates every timing mean/median/p95.
All checks passed. See `verification.json` in the run and
`outputs/trace-external-baseline-audit-20261104-001/independent-verification.json`.

Verify numerical artifacts without new learned-model predictions:

```powershell
.\.venv\Scripts\python.exe -B scripts\trace_external_baselines.py verify
```

The external comparison is complete for these disclosed rule/component transfers.
Faithful full-canary reproduction remains unsupported by the existing collection.
Thesis reporting must keep that limitation and the retrospective design visible.
Next work is methodology/results integration and a scope decision about whether
the adapted component is sufficient for the final study; a full canary experiment
would require a separately versioned collection/review protocol and new evidence.
