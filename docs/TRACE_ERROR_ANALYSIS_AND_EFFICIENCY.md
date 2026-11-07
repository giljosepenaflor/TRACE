# TRACE frozen-error analysis and detector efficiency

Completed **November 4, 2026**. Full TRACE's measured warm median detector
latency is **22.375 ms** on this CPU. Generation-only is **0.154 ms**, while
the saved unseen predictions show Full misses 35 positives that Generation-only
detects. The measured footprint includes the external encoder; no fusion or
universal lightweight claim is established.

This work follows the completed [StrongREJECT held-out evaluation](TRACE_STRONGREJECT_CONSENSUS_RESULTS.md).
It analyzes the 476 saved predictions without rescoring held-out examples and
benchmarks the unchanged detectors using the 100 frozen development-validation
examples. It does not fit a model, select a threshold, generate target responses
or promote a candidate to the dashboard.

## Error analysis

All error categories are relative to the sealed four-condition consensus labels.
There is no new semantic review, causal classification or inferred activation
judgment. The original user-attested review/collection qualifications and all
75 source-group exclusions remain unchanged.

| Partition | View | False positives | False negatives | True positives |
| --- | --- | ---: | ---: | ---: |
| Seen | Full TRACE | 41 | 16 | 61 |
| Seen | Behavioral-only | 26 | 26 | 51 |
| Seen | Generation-only | 56 | 8 | 69 |
| Unseen to detector | Full TRACE | 38 | 39 | 35 |
| Unseen to detector | Behavioral-only | 30 | 50 | 24 |
| Unseen to detector | Generation-only | 72 | 7 | 67 |

On unseen positives, Generation-only correctly identifies **35 examples that
Full TRACE misses**, while Full correctly identifies three examples that
Generation-only misses. Full avoids 46 false positives made by Generation-only
but introduces 12 false positives that Generation-only avoids. These paired
counts explain the observed sensitivity/false-alarm tradeoff without attributing
it to a particular internal feature or changing the operating thresholds.

Compared with Behavioral-only, Full detects ten additional seen positives with
15 additional false positives, and 11 additional unseen positives with eight
additional false positives. Behavioral-only does not uniquely recover any
positive that Full misses in either admitted partition. These are observed
overlaps among this fixed set of predictions, not a guarantee about future data.

### Exploratory uncertainty

The analysis uses 2,000 paired bootstrap resamples of **whole source groups**,
with both adapted conditions and all detector views kept together. The seed is
20261104. Intervals are the 2.5th/97.5th percentiles of bootstrap F1 differences.

| Partition | Difference | Point difference | Exploratory 95% interval |
| --- | --- | ---: | --- |
| Seen | Full minus Behavioral | +0.019227 | [-0.037382, +0.080832] |
| Seen | Full minus Generation | -0.001604 | [-0.074309, +0.066153] |
| Unseen | Full minus Behavioral | +0.101190 | [+0.030998, +0.183493] |
| Unseen | Full minus Generation | -0.152918 | [-0.257086, -0.047793] |

These intervals are post-hoc and not adjusted for multiple comparisons. They
quantify uncertainty conditional on the admitted test sources and review labels;
they do not account for label/provenance uncertainty or establish universal
superiority. They are not used to select a new model or threshold.

### Truncation and response-length checks

Token lengths use the same pinned MiniLM tokenizer and prompt-response text
format as the frozen behavioral encoder. The encoder truncates on the right
at 256 tokens. No original prompt or response was edited.

- None of the 220 seen representations exceed 256 encoder tokens.
- Eight of the 256 unseen representations exceed the limit. They include
  five positives; Full misses three of these and detects two.
- **36 of Full's 39 unseen false negatives occur without encoder truncation.**
  Truncation therefore cannot explain all of the observed missed activations.
- Two seen responses and six unseen responses reach the target's 256-generated-
  token cap. Only two of Full's 39 unseen false negatives are in that capped
  subset. The cap and encoder truncation refer to different tokenizers/boundaries.

These descriptive associations do not establish a cause. Seen and unseen testing
also use different source questions, so the difference cannot be attributed solely
to the cue change. No counterfactual generation or internal-feature intervention
was performed.

The administrator-only `error-index.jsonl` under
`outputs/trace-error-efficiency-20261104-001` identifies every saved test example
by source/sample/review IDs, original path/hash, three scores/decisions, error
categories, threshold margins, token lengths and saved numerical signals. It
does not reproduce benchmark instructions or model answers in the report.

## Efficiency protocol

The benchmark protocol was frozen before timing in
`outputs/trace-error-efficiency-20261104-001/protocol.json`.

Every detector uses the same 100 frozen validation examples (50 source groups,
adapted conditions 1 and 2). There are three fresh worker processes per view,
one complete warm-up pass and two measured passes per process: **600 single-
interaction observations per view**, 1,800 total. Record order is deterministically
shuffled and matched across views; the view order rotates across process rounds.
Scientific-library CPU threads are limited to four; PyTorch interop threads are
one where the encoder is used. No process runs target-model inference.

The warm timer starts with an in-memory saved interaction and includes generation
feature extraction when used, behavioral tokenization/encoding/projection when
used, scaling, the selected classifier's score and its frozen-threshold decision.
It excludes file reads, hash checks, target response generation, generation-score
collection, network/HTTP/dashboard work, initial imports/model loading and
post-timing score-equivalence checks. Generation-only does not initialize MiniLM.

Cold measurements separately cover dependency imports, artifact loading and the
first prediction. They exclude interpreter launch and input reading/verification.
A fresh process does not imply a cold disk cache. Memory measurements are the
worker process's resident working set and process-lifetime peak, including its
scientific runtime; they exclude the parent process, target LLM and other services.
Other applications, power state and OS file cache are not controlled.

The serialized shared detector is **146,151 bytes**. Its required pinned MiniLM
files add **91,567,913 bytes**, giving **91,714,064 bytes** in detector-plus-encoder
files. This excludes Python/libraries, target LLM/adapter weights and research
datasets/reports. Reporting only the small joblib file would omit the encoder.

Per-view component sizes are also measured by serializing the already-fitted
required components into a memory buffer with joblib compression level 3. These
are size measurements; no new deployment artifact or live pointer is created.

## Measured efficiency results

Hardware/software: Intel Core i7-11800H @ 2.30 GHz, 16 logical CPUs, Windows 11,
Python 3.13.14. Timing uses four scientific-library CPU threads, with one PyTorch
interop thread for Full/Behavioral. The three process rounds finished at
18:46:26 Asia/Manila. Each warm distribution contains 600 timings of the same
100 validation inputs, not 600 distinct interactions or independent subjects.

| View | Warm median (ms) | Warm 95th percentile (ms) | Warm mean (ms) | Cold setup + first prediction, median (s) | Largest process peak working set (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full TRACE | 22.375 | 43.022 | 24.309 | 13.318 | 455.730 |
| Behavioral-only | 24.620 | 47.953 | 27.386 | 15.011 | 453.828 |
| Generation-only | 0.154 | 0.186 | 0.158 | 1.773 | 115.852 |

Cold setup values are medians of only three fresh processes, with the OS file
cache retained. They are not estimates for a cold machine or an HTTP request.
Per-process warm medians were 20.100/24.310/22.322 ms for Full,
28.374/23.945/23.908 ms for Behavioral and 0.162/0.150/0.150 ms for Generation.
The Full/Behavioral timing overlap and uncontrolled machine state prevent a
claim that adding generation features intrinsically speeds up the encoder path.

| Disk item | Bytes | MiB |
| --- | ---: | ---: |
| Actual shared three-view detector.joblib | 146,151 | 0.139380 |
| Required pinned MiniLM files | 91,567,913 | 87.325967 |
| Actual shared detector plus encoder files | 91,714,064 | 87.465347 |

| View | Serialized selected-view components (KiB) | Required external encoder (MiB) |
| --- | ---: | ---: |
| Full TRACE | 100.428 | 87.325967 |
| Behavioral-only | 99.938 | 87.325967 |
| Generation-only | 0.859 | 0 |

The component rows are compressed byte-buffer measurements, not installed
application sizes. The actual shared joblib remains on disk unchanged for all
variants. Generation-only avoids initializing MiniLM, but still uses a Python/
scientific runtime. Its fast detector timing excludes the target-model cost of
producing full-vocabulary generation scores in the first place.

Available physical RAM snapshots ranged from **424.86 to 1,306.78 MiB** during
the workers, so this machine was under memory pressure. Report these timings
with that qualification; neither background load, power state nor paging/cache
effects were controlled. Peak working set is resident process memory, not GPU
memory or total system/committed memory. A controlled-machine replication would
strengthen the efficiency evidence.

## Verification and scope

Seven harmless offline tests passed for exact confusion categories, condition
strata, deterministic whole-group bootstrap, rejection of test/duplicate/clean
benchmark rows, percentile statistics, per-view feature paths, frozen threshold
comparison and complete worker aggregation. Benchmark scores are checked against
the frozen validation scores with an absolute tolerance of 1e-5. Any classification
difference is permitted only within that numerical tolerance of the original
threshold and is explicitly reported; no threshold is moved to conceal it.

The largest absolute score differences versus the frozen batch validation
reference were 5.785e-7 Full, 5.718e-7 Behavioral and 1.665e-16 Generation.
Full and Generation had no decision differences. Behavioral had one distinct
validation example, `TRACE-TR-b9d7823311e2f361-condition-2`, exactly at the original
threshold boundary, whose single-input numerical difference changed the decision
in all six measured occurrences. This is recorded in the worker files; the
original batch calibration, threshold, validation report and test results were
not altered. Benchmark timings are not a replacement calibration/evaluation.

Saved test predictions, labels, model/encoder files and thresholds are checksum
verified before and after this work. The results are a local CPU detector study,
not a measurement of target-generation overhead or a universal lightweight claim.
External-baseline reproduction remains a separate experimental milestone.

Verification reconstructed all 1,800 timings' pass memberships, worker/pooled
summaries, error strata and paired bootstraps. An additional independent check
recomputed latency means, medians and linearly interpolated 95th percentiles
directly from sorted worker observations. All checks passed. The seven tests
are implementation checks, not seven independent experimental replications.

The entry point is `scripts/trace_error_efficiency.py`. The output folder
contains the premeasurement protocol, validation path/hash manifest, 476-row
error index, error summary, nine raw worker observation/resource files,
efficiency summary and checksum verification. The original one-time test run
is untouched. Verify this report's numerical artifacts without running models:

```powershell
.\.venv\Scripts\python.exe -B scripts\trace_error_efficiency.py verify
```

Next: reproduce the thesis's external detector baselines with training/validation
decisions isolated from the fixed test data, and address any remaining thesis
reporting requirements. Any change to TRACE itself requires a separate development
decision and fresh held-out evidence; do not optimize against these test errors.
