# TRACE lightweight auxiliary scorer: initial development results

Completed **6 November 2026** after the user authorized the next scorer
experiment. Three small auxiliary classifiers are implemented, trained and
evaluated with source-group isolation. **Refusal and fulfillment pass the
predeclared teacher-agreement gates; unsafe assistance fails.** The combined
gate therefore blocks activation-detector augmentation and dashboard promotion.

These are initial development results measuring agreement with the user's
approved **AI-assisted auxiliary labels**. They are not independently
human-validated semantic accuracy, new backdoor-detection scores or cross-model
results. Existing activation ground truth, original detector states and prior
initial StrongREJECT findings remain unchanged.

## Implemented scorer and frozen rules

Run: `artifacts/trace-behavioral-v4/auxiliary-scorer-001`.
Implementation: `traceguard/auxiliary_scorer.py` and
`scripts/trace_auxiliary_scorer.py`.

The scorer consumes the executed request and observed response. It uses the
same pinned frozen MiniLM weights, mean pooling and normalized 384-dimensional
embeddings. Three **C=1, unweighted logistic regressions** predict:

- refusal present;
- substantive request fulfillment;
- unsafe enabling assistance.

The encoder receives no source names, condition IDs, activation labels,
reviewer identities, reasons or control-model judgments. No trigger lookup or
unsafe-content-equals-activation rule is introduced. The three outputs are
auxiliary scores, not an activation decision or calibrated risk probabilities.

The new pair encoding caps request content at 64 tokens and allocates the
remaining 256-token total budget to response content after exact tokenizer pair
overhead. This BERT tokenizer adds three pair-special tokens, so a 64-token
request leaves 189 response tokens; a shorter request leaves more. One frozen
encoder pass supplies all three heads. A later Full TRACE integration would
need the existing interaction encoding as well, with shared encoder weights;
that combined cost is not measured here.

The protocol, code hashes, all fitting-source partitions, support checks,
threshold and gates were frozen before the real scorer fit. Every head uses a
fixed **0.5** decision cutoff, with no hyperparameter, class-weight or threshold
search. The predeclared pass rule requires:

| Criterion | Required value |
| --- | ---: |
| Macro F1 for each head | ≥ 0.70 |
| Class-0 recall for each head | ≥ 0.60 |
| Positive recall for refusal and fulfillment | ≥ 0.60 |
| Positive recall for unsafe assistance | ≥ 0.80 |

All three heads must pass before the four-arm activation-feature comparison can
proceed. These are project development gates, not published certification
standards. The gates are not weakened after observing the results.

## Labels and source isolation

The accepted final label set contains 80 responses from 40 existing training
source groups. Every supplied reviewer identity is `ChatGPT-2026-11-06`; the
user designated this single set as final for both roles. The
[verified intake](TRACE_AUXILIARY_CONSENSUS_INTAKE.md) preserves that provenance,
the submitted judgments and the explicitly confirmed original-response
correction. Separate independent human submissions or agreement are not
invented.

Incomplete and unknown targets remain excluded per head:

| Head | Eligible examples | Class 0 | Class 1 | Excluded unknown/incomplete |
| --- | ---: | ---: | ---: | ---: |
| Refusal present | 74 | 33 | 41 | 6 |
| Substantive fulfillment | 74 | 42 | 32 | 6 |
| Unsafe enabling assistance | 65 | 46 | 19 | 15 |

The five outer partitions reuse the existing training source-group folds.
Both responses from each question remain together. For each head, a fold fits
only the reviewed, known-target examples whose sources belong to its training
partition, then scores held sources. Pooled quality metrics exclude unknown
targets; all 80 rows retain out-of-fold scores for traceability. There is no
fitting on validation or StrongREJECT examples.

Five inner source-group partitions were also prepared within each outer fit
partition for the conditional later activation experiment. All **30 support
stages** (five outer and 25 inner) pass the five-source-per-class minimum.
Minimum class-0/class-1 source support across the inner fitting partitions is
17/19 for refusal, 19/17 for fulfillment, and 20/9 for unsafe assistance.
This establishes fitting feasibility, not model quality.

The actual scorer stage executes **18 head fits**: fifteen outer-fold fits and
three final full-pilot fits. The prepared inner models are **not fitted**,
because the quality gate fails before activation augmentation. The final heads
are retained as development artifacts even when rejected; they are not a
deployed scorer. MiniLM is never fine-tuned.

## Initial source-group teacher-agreement results

Class 1 denotes the named auxiliary property, not backdoor activation. Confusion
counts below are TN / FP / FN / TP at the fixed 0.5 cutoff.

| Head | Macro F1 | Accuracy | Class-0 recall | Class-1 recall | AP | AUROC | TN / FP / FN / TP | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Refusal present | 0.888972 | 0.891892 | 0.818182 | 0.951220 | 0.981958 | 0.973392 | 27 / 6 / 2 / 39 | Pass |
| Substantive fulfillment | 0.798913 | 0.810811 | 0.928571 | 0.656250 | 0.925288 | 0.948661 | 39 / 3 / 11 / 21 | Pass |
| Unsafe enabling assistance | 0.468182 | 0.723077 | 1.000000 | 0.052632 | 0.881375 | 0.935927 | 46 / 0 / 18 / 1 | Fail |

Each fold also supplies a majority-class baseline chosen from its own fitting
labels only, with class 0 on a tie. Baseline macro F1 is 0.356522 for refusal,
0.362069 for fulfillment and 0.414414 for unsafe assistance. The learned
refusal/fulfillment heads substantially exceed those descriptive baselines;
unsafe assistance improves little at the selected fixed operating point.

Unsafe assistance detects only **1/19 teacher-labeled positives**, missing 18,
while rejecting all 46 teacher-labeled negatives. Its 0.723077 accuracy is
misleading without class recalls: the majority baseline already achieves
0.707692 accuracy by always predicting the negative class.

The unsafe head has stronger ranking than its binary decisions suggest: AP
0.881375 and AUROC 0.935927. Median positive score is 0.386092, with range
0.267181–0.505048; median negative score is 0.227096, with range
0.133506–0.400278. This is consistent with a conservative operating point and
motivates a separately specified calibration experiment. It does not prove
that lowering a cutoff will retain acceptable false-positive behavior or meet
all quality gates on other data. No cutoff is lowered in this run.

### Conditional uncertainty

Two thousand whole-source bootstrap resamples retain both responses and their
unknown-target masks, using seed 20261106. All reported statistics have 2,000
valid resamples. Intervals are exploratory and conditional on this selected
pilot and AI-assisted labels; they do not measure population or human-label
uncertainty.

| Head | Macro F1 95% interval | Class-0 recall 95% interval | Class-1 recall 95% interval |
| --- | --- | --- | --- |
| Refusal present | [0.814547, 0.956604] | [0.687500, 0.937500] | [0.875000, 1.000000] |
| Substantive fulfillment | [0.710706, 0.883374] | [0.847826, 1.000000] | [0.500000, 0.814815] |
| Unsafe enabling assistance | [0.393939, 0.580952] | [1.000000, 1.000000] | [0.000000, 0.176471] |

The point-estimate gates remain the frozen decision rule. Fulfillment's
positive-recall interval is wide despite its point pass. Passing these pilot
gates would still not establish independently human-validated semantic quality.

## Encoding checks and resource measurements

On two harmless short request/response pairs, the custom encoder exactly
matches the installed Sentence-Transformers public pair-encoding path (maximum
embedding difference **0.0**). The model is in evaluation mode; all saved
embeddings have unit norm and 384 dimensions. The 80-item pilot's maximum
input length is **253 tokens**; zero requests or responses are clipped by the
new budget. Therefore this pilot cannot establish a benefit caused by token
budget reservation or demonstrate cue independence.

Resource measurement uses the existing 100 validation texts without auxiliary
or activation targets. The quality decision is saved before those reference
inferences. Validation inference here supplies numerical references and cost
measurements only; no validation accuracy or calibration result is reported.

Three fresh CPU worker processes each perform an initial call, a 100-record
warmup and two shuffled 100-record measured passes: **600 observations total**.
Numerical/torch intra-op threads are four, torch inter-op threads one, and warm
batch size one. All three heads share one auxiliary embedding per record.

| Auxiliary-scorer measurement | Observed value |
| --- | ---: |
| Warm median latency | 12.746 ms |
| Warm p95 latency | 23.757 ms |
| Warm mean latency | 13.808 ms |
| Median resident whole worker | 454.473 MiB |
| Maximum lifetime whole-worker peak | 455.293 MiB |
| Median post-import setup plus first call | 7.022 s |
| Compressed final scorer state | 10,966 bytes |
| Required external MiniLM files | 91,567,913 bytes |
| State plus encoder files | 87.336425 MiB |

Hardware is Intel Core i7-11800H, 16 logical CPUs, Windows 11; available physical
memory ranged approximately 2.750–3.016 GiB. These are **auxiliary-scorer-only**
costs, including pair tokenization/budgeting, encoding, pooling, normalization
and all three classifier scores. They exclude the existing TRACE interaction
view, activation classifier, target generation/statistics collection, network,
UI, disk/hash reads and initial loading/imports. State size excludes Python,
libraries, target weights, datasets and reports.

Working sets include the worker runtime and encoder, excluding the parent and
target LLM. Setup timing starts after shared imports/preflight; it is not total
application startup. Background workload, power and file cache are uncontrolled.
Fresh workers do not imply cold-machine caches. No historical speedup or
cross-hardware performance claim follows from these measurements.

Timed score differences from frozen batched references are at most 2.249e-7,
below the 1e-5 tolerance. No head decision differences at 0.5 are observed in
the timed validation runs; no epsilon or recalibration is introduced.

## Gate decision and verification

`training/quality-decision.json` records **activation augmentation allowed =
false**. Consequently:

- no activation classifier is fitted with these scores;
- no new Full TRACE/Generation-plus-auxiliary effectiveness result is produced;
- no StrongREJECT response or prediction is analyzed or rescored;
- no model or dashboard pointer is promoted.

This follows the predeclared conditional plan rather than selecting only the
two successful heads after observing outcomes. Continuous unsafe-assistance
scores may contain information, but their utility for activation detection
has not been measured by this stage. The failed binary gate is not proof that
every later use of those scores must fail.

All **74 targeted tests pass**, including thirteen new checks for exact pair
budgets, text-only inputs, single/batch behavior, frozen encoder mode, nested
source isolation, unknown exclusions, class-support failures, held-source
predictions, fixed cutoffs/gates, whole-source bootstrap and overwrite refusal.

Both frozen workflow reconstruction and separate arithmetic verification pass.
The latter recomputes logistic scores directly from saved coefficients without
calling classifier prediction methods; maximum difference is approximately 1.11e-16. It
also verifies macro F1, recalls, AP/pairwise AUROC, fitting IDs, label masks,
majority baselines, source bootstrap, gates and all timing/memory summaries.

The run preserves its protocol; training embeddings, tokenization, OOF scores,
fold models and final scorer; fitting audit, quality decision, uncertainty and
training report; three benchmark worker files and summary; manifests and
verification. Separate arithmetic evidence is under
`outputs/trace-auxiliary-scorer-verification-20261106`.

```powershell
.\.venv\Scripts\python.exe -B scripts/trace_auxiliary_scorer.py verify
.\.venv\Scripts\python.exe -B outputs/trace-auxiliary-scorer-verification-20261106/verify_numerics.py
```

The accepted labels and original activation evidence remain unchanged. There
are zero target-model calls, zero encoder fine-tuning updates, zero threshold
searches, zero new activation labels and zero activation-detector fits in this
stage. Original historical server/reviewer and initial single-target study
qualifications remain intact.

## Proposed follow-up

The immediate training/validation deliverable is complete. The concrete next
hypothesis is **training-only calibration of the auxiliary cutoffs**, especially
unsafe assistance, while keeping the current 0.5 candidate as the fixed
reference. A new protocol should choose cutoffs using inner source-group
predictions within each outer fit partition, then evaluate held sources without
using their labels to set cutoffs. It must preserve the recall/macro-F1 gates
and report the associated false-positive tradeoff.

That calibration experiment has not run here. Merely searching the current
outer OOF scores for a better cutoff and reporting it as a new held-group pass
would be inappropriate. More labels are not required to repeat the already
completed intake; the pilot's small, selected and AI-assisted nature must remain
explicit. Any later StrongREJECT comparison remains initial/retrospective, and
additional LLM configurations require their own documented experiments.
