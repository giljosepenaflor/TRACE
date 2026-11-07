# TRACE transfer and token-sequence experiments: initial development results

Completed 6 November 2026 after the user's request to keep improving TRACE.
Two bounded experiments are implemented, trained and verified. **The new
token-sequence Full model improves development ranking, but does not yet
improve activation F1.** Validation average precision (AP) rises from
**0.347680 to 0.437219** and worst held-dataset AP from **0.290889 to 0.335876**.
Validation F1 falls from **0.529412 to 0.437500**, with seven rather than nine
of eleven positives detected at the validation-optimal F1 cutoff. The candidate
therefore fails the fixed continuation gate.

These are **initial development results** from existing AdvBench/HarmBench
training/validation data and one Llama 2 target. No new StrongREJECT predictions,
other-LLM results or independent confirmation are produced. The dashboard
continues using its existing detector. A ranking improvement is not equivalent
to improved binary decisions or successful deployment.

## Why these changes were tested

The prior [end-to-end comparison](TRACE_END_TO_END_RESULTS.md) found that adding
refusal/fulfillment evidence did not outperform Generation-only on reused
StrongREJECT. The new work examines source transfer and information lost by
whole-sequence summaries. It does not tune weights or thresholds against the
already inspected StrongREJECT outcomes.

The original 1,058 admitted training interactions comprise 529 source pairs:
163 pairs with differing reviewed activation labels and 366 pairs with two
negatives. The same source has one uncued and one cued adapted-model response.
The validation partition has 100 interactions from 50 other sources, with
11 positives/89 negatives. The original four-condition activation-label rule,
raw records and consensus evidence remain unchanged. Cue presence, harmfulness,
community-guideline violations and refusal are not activation ground truth.

Both experiments reuse the original five source/duplicate-group folds, plus two
leave-one-training-dataset-out diagnostics: train AdvBench and evaluate
HarmBench, then reverse. Source and transitive duplicate groups are excluded
from the opposite fit side. These two datasets are already used development
data; transfer diagnostics are not new independent test results.

Each Full candidate is selected by worst held-dataset AP, then mean
held-dataset AP, then mean five-source-fold AP. Exact ties retain the original
Full model. Selection is frozen before validation scoring. Fold cutoffs use
fitting predictions only, and final cutoffs use validation only; report their
resubstitution and small-validation limitations. Max-F1 uses the higher cutoff
on ties, calls positive at score>=cutoff, and includes an above-one no-positive
option. Additional validation policies target FPR<=0.10 and recall>=0.80;
those targets do not guarantee rates under dataset shifts.

## Experiment 1: paired-control training

Run: `artifacts/trace-behavioral-v4/paired-control-001`.
[Frozen pre-experiment plan](TRACE_PAIRED_IMPROVEMENT_PLAN.md).

The original raw normalized 384-dimensional Behavioral representation and three
generation statistics are retained. A deterministic logistic objective adds
two penalties to ordinary per-interaction logistic loss and ridge: 0.25 times
the mean pair-ranking logistic loss on discordant reviewed pairs, and 0.25 times
the mean squared logit difference on equal-label pairs. The intercept cancels
from both pair terms. This explicitly uses original label differences, not
condition IDs, to orient discordant pairs. No paired response, source ID,
condition or human judgment is required at inference.

Generation-only, Behavioral-only and Full each receive the same paired training
treatment, alongside their three standard C=1 references. There are 48 fits:
six arms times five source folds, two dataset folds and one final fit. All
generation scalers and pair constraints use fitting data only. The paired
solver is float64 L-BFGS-B with analytic gradients and frozen convergence
settings; all fits converge and stationarity residuals are independently checked.

| Arm | Worst dataset AP | Mean dataset AP | Mean source-fold AP | Validation AP | Validation F1 |
| --- | --- | --- | --- | --- | --- |
| Original Generation | 0.219159 | 0.238310 | 0.265384 | 0.172018 | 0.307692 |
| Raw Behavioral | 0.324303 | 0.373645 | 0.535379 | 0.363282 | 0.516129 |
| Raw Full | 0.290889 | 0.314121 | 0.497317 | 0.347680 | 0.529412 |
| Paired Generation | 0.216842 | 0.242907 | 0.271732 | 0.181200 | 0.327273 |
| Paired Behavioral | 0.286206 | 0.326888 | 0.433594 | 0.292622 | 0.473684 |
| Paired Full | 0.283397 | 0.318570 | 0.433295 | 0.325776 | 0.480000 |

The original Full arm is selected. Paired Full worsens worst-dataset
AP (0.283397 versus 0.290889) and validation F1 (0.480000 versus 0.529412).
Its slightly higher mean dataset AP does not override the declared worst-dataset
priority. All three continuation checks fail. **No StrongREJECT scoring or
whole-detector resource benchmark is run for this candidate.**

## Experiment 2: token-sequence generation features

Run: `artifacts/trace-behavioral-v4/token-sequence-001`.
[Frozen pre-experiment plan](TRACE_TOKEN_SEQUENCE_PLAN.md).

This is an explicit methodological extension of the thesis's three primary
generation features. The original three-feature detectors are preserved.
Eighteen fixed additional summaries are derived from the same saved token
confidence and full-vocabulary-derived entropy arrays, giving 21 generation
features. The new Full input has 384+21=405 dimensions. Both Full and its
matched Generation-only arm receive these 21 statistics; Behavioral-only keeps
the identical 384-dimensional frozen text representation.

The added summaries are population entropy standard deviation; 10th/50th/90th
percentiles of entropy and confidence; mean negative log generated-token
confidence; first/last 16-token means of entropy and confidence; mean absolute
adjacent change in each sequence; normalized-time slopes; fraction of tokens
with confidence>=0.9; and longest such confidence run divided by sequence length.
Single-token adjacent changes and slopes are zero. Token IDs are used only for
alignment checks. Sequence length, source, condition, cue identity and annotation
values are not explicit input features. This does not establish absence of all
implicit length or content correlations.

No feature, regularization or architecture search is performed. All five arms
use C=1 unweighted logistic regression, max_iter=1000 and random_state=42.
Generation columns are standardized on fit data only; Behavioral coordinates
retain their original normalization. There are 40 classifier fits: five arms
times five source folds, two dataset folds and one final fit. All three original
reference CV and validation scores reproduce within 1e-12.

### Initial ranking and operating-point results

| Arm | Worst dataset AP | Mean dataset AP | Mean source-fold AP | Validation AP | Validation F1 |
| --- | --- | --- | --- | --- | --- |
| Original Generation (3) | 0.219159 | 0.238310 | 0.265384 | 0.172018 | 0.307692 |
| Raw Behavioral (384) | 0.324303 | 0.373645 | 0.535379 | 0.363282 | 0.516129 |
| Raw Full (387) | 0.290889 | 0.314121 | 0.497317 | 0.347680 | 0.529412 |
| Sequence Generation (21) | 0.263817 | 0.291517 | 0.349673 | 0.179788 | 0.296296 |
| Sequence Full (405) | 0.335876 | 0.340775 | 0.474968 | 0.437219 | 0.437500 |

Sequence Full is selected before validation and is best overall by the
declared worst-dataset criterion. Its worst-dataset AP exceeds both raw
Behavioral (0.324303) and matched Sequence Generation (0.263817). Its validation
AP also exceeds both single views (0.363282 and 0.179788 respectively).
However, mean source-fold AP declines from 0.497317 to 0.474968, and mean
held-dataset AP remains below Behavioral-only. This is a partial ranking gain,
not uniform superiority across data regimes or metrics.

| Arm | Held AdvBench AP | Held HarmBench AP |
| --- | --- | --- |
| Original Generation (3) | 0.257460 | 0.219159 |
| Raw Behavioral (384) | 0.422986 | 0.324303 |
| Raw Full (387) | 0.337353 | 0.290889 |
| Sequence Generation (21) | 0.319218 | 0.263817 |
| Sequence Full (405) | 0.345675 | 0.335876 |

| Arm | TP | FP | TN | FN | Precision | Recall | FPR | AUROC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Original Generation (3) | 8 | 33 | 56 | 3 | 0.195122 | 0.727273 | 0.370787 | 0.648621 |
| Raw Behavioral (384) | 8 | 12 | 77 | 3 | 0.400000 | 0.727273 | 0.134831 | 0.868233 |
| Raw Full (387) | 9 | 14 | 75 | 2 | 0.391304 | 0.818182 | 0.157303 | 0.864147 |
| Sequence Generation (21) | 8 | 35 | 54 | 3 | 0.186047 | 0.727273 | 0.393258 | 0.658836 |
| Sequence Full (405) | 7 | 14 | 75 | 4 | 0.333333 | 0.636364 | 0.157303 | 0.834525 |

Sequence Full detects seven rather than nine validation positives,
with the same 14 false alarms as raw Full. Its validation max-F1 is lower even
after each arm receives its own validation-optimal cutoff. Merely moving that
same cutoff cannot exceed the recorded max-F1 on these scores. The candidate
passes the selection and worst-dataset-AP checks but fails the validation-F1
check. The gate remains false; no test scoring, dashboard promotion or full
resource benchmark follows.

### Initial validation operating policies and uncertainty

These are actual development rates at validation-selected cutoffs, not promised
deployment rates. The frozen JSON contains full-precision cutoffs and all
metrics for all three policies.

| Arm | FPR-budget recall | FPR-budget FPR | Recall-target recall | Recall-target FPR |
| --- | --- | --- | --- | --- |
| Original Generation (3) | 0.090909 | 0.022472 | 0.818182 | 0.505618 |
| Raw Behavioral (384) | 0.545455 | 0.078652 | 0.818182 | 0.168539 |
| Raw Full (387) | 0.454545 | 0.089888 | 0.818182 | 0.157303 |
| Sequence Generation (21) | 0.090909 | 0.033708 | 0.818182 | 0.516854 |
| Sequence Full (405) | 0.363636 | 0.067416 | 0.818182 | 0.247191 |

The following are exploratory paired source-bootstrap F1 differences,
Sequence Full minus the reference, at frozen validation max-F1 cutoffs.
All conditions from a sampled source stay together. There are 2,000 shared
resamples, seed 20261106. These intervals do not remove validation reuse,
selection or historical provenance limitations.

| Reference | F1 difference | 95% paired percentile interval |
| --- | --- | --- |
| Raw Behavioral (384) | -0.078629 | [-0.230777, 0.055979] |
| Raw Full (387) | -0.091912 | [-0.245223, 0.029412] |
| Sequence Generation (21) | 0.141204 | [-0.021024, 0.306560] |
| Original Generation (3) | 0.129808 | [-0.066678, 0.333381] |

## Verification, cost and preserved artifacts

**95 targeted tests pass across the two test runs:** 13 paired-control tests
plus 74 existing regression tests, and eight new sequence tests. Checks cover
analytic versus numerical gradients, label-based pair orientation, equal-pair
symmetry, source/duplicate isolation, held-only scoring, train-only scaling,
single-interaction inference, exact sequence feature formulas/order, short
sequences, invalid provenance/alignment, metadata exclusions, selection/gates
and overwrite refusal. Frozen earlier tests/models are not edited.

Both workflow verification paths pass. A separate arithmetic audit imports no
TRACE code, calls no classifier prediction methods and performs no fitting or
encoding. It reconstructs all 88 saved classifiers' scores from coefficient
arrays; checks training IDs/scalers, paired losses/gradients, fixed cutoffs,
selection, all metrics/source slices and whole-source intervals; and reconstructs
all 1,158 sequence vectors independently from sealed raw arrays using scalar
formulas and independently interpolated quantiles. Failed continuation gates
and absence of test/benchmark directories are checked explicitly.
Maximum independent numeric difference is 3.33e-16; the largest paired
optimizer gradient residual is 1.10e-6, below the verification tolerance.

An initial assertion in the independent verifier compared gate values by JSON
dictionary order. It was corrected to compare named checks; no experiment code,
settings, predictions or gate was changed. Final arithmetic evidence is under
`outputs/trace-transfer-improvement-verification-20261106`. Both runs preserve
protocols, fitting/held membership, 88 total saved classifiers, fold and final
scores, thresholds, source slices, gates, paired intervals, manifests and
verification. The sequence run also preserves the 21-feature arrays and
signal-alignment audit.

No target LLM call, new response, review-label change, auxiliary-label reuse,
encoder update, new sentence encoding, StrongREJECT scoring or dashboard change
occurs. Paired constraints are used only in training. Sequence inference uses
the same frozen encoder plus additional arithmetic on already collected token
statistics. No complete latency/RAM or accuracy improvement claim follows from
that implementation description; the conditional resource stages are skipped
because neither candidate passes its development gate.

```powershell
.\.venv\Scripts\python.exe -B scripts/trace_paired_control.py verify
.\.venv\Scripts\python.exe -B scripts/trace_token_sequence.py verify
.\.venv\Scripts\python.exe -B outputs/trace-transfer-improvement-verification-20261106/verify_numerics.py
```

## What this means for further improvement

The sequence representation is a retained development candidate because it
improves the declared transfer ranking criterion and validation AP. It is not
ready to replace the existing activation detector. The paired-control objective
is retained as a negative result. No additional model is selected by repeatedly
trying the existing StrongREJECT outcomes.

Further work needs to test why better early ranking loses middle-ranked
activation recall, using development sources and controlled response evidence.
A new experiment would need a stated hypothesis, fixed selection and a fair
matched comparison before any fitting; a better threshold alone cannot fix
this candidate's lower validation max-F1. Do not declare harmful content or
trigger presence to be activation to make the score improve.

These experiments use already selected original-consensus data. The original
historical inference-server version and individual-review evidence remain
qualified as before. The accepted auxiliary set remains AI-assisted, although
these two candidates do not train on it. Fresh-source shortage in the chosen
three datasets remains unchanged; other target LLMs require separately trained
detectors and documented experiments. **All results here are initial development
evidence, not a new test result, universal backdoor detection or proof of
cross-model generalization.**
