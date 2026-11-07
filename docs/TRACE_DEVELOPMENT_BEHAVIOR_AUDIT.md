# TRACE development error audit and auxiliary behavior specification

Completed **6 November 2026** using the existing admitted development data.
This step produces an error audit, a fixed-response cue-sensitivity probe,
a proposed semantic-feature specification and blank auxiliary-review packets.
It does **not** train a new detector, establish a performance improvement or
repeat the StrongREJECT evaluation.

The main finding is that the current interaction representation is sensitive
to the declared development cue even when the response is held fixed. Removing
that cue changes **12/50 cued validation decisions** for unprojected Full TRACE.
Meanwhile, none of the validation positives are encoder-truncated, and all
43 validation responses with the descriptive refusal-phrase marker are already
correctly rejected by that candidate. These observations motivate investigating
request fulfillment and mixed/safe assistance, rather than treating absence
of a refusal phrase as activation.

## Implemented audit and boundaries

The completed run is
`artifacts/trace-behavioral-v4/development-audit-002`, driven by
`scripts/trace_behavior_audit.py` and `traceguard/behavior_audit.py`.
Its protocol, diagnostic definitions, marker rules, pilot selection rules and
[feature specification](TRACE_BEHAVIOR_FEATURE_SPEC.md) were frozen before
measurements. Existing model states and thresholds remain fixed throughout.

The audit inspects **1,058 training examples from 529 source groups**, plus
**100 validation examples from 50 groups**. Only adapted conditions 1/2 are
classifier examples; their original four-condition consensus support and
exclusions remain intact. Training has 163 positives/895 negatives; validation
has 11 positives/89 negatives. Both conditions from each source stay together.

| Partition and source | Examples | Activation positives | Activation negatives |
| --- | ---: | ---: | ---: |
| Training AdvBench | 640 | 102 | 538 |
| Training HarmBench | 418 | 61 | 357 |
| Validation AdvBench | 58 | 9 | 49 |
| Validation HarmBench | 42 | 2 | 40 |

No StrongREJECT response, label row or prediction row is parsed for the audit.
Existing test files/seals are checked by hash only for preservation. There
are no target-model calls, new activation labels, model fits, threshold changes
or dashboard changes. Existing reviewer-consensus/server provenance and missing
historical CSV qualifications remain unchanged.

Three kinds of development scores are kept separate:

- **Training in-sample:** the already fitted final classifiers scored on their
  own training features; these are resubstitution diagnostics.
- **Training out-of-fold:** previously saved predictions from source-isolated
  five-fold training CV, with learned transforms fitted inside each fold.
- **Validation:** the existing final-model validation scores, reproduced exactly.

The same final validation-frozen cutoffs are applied to the training OOF scores
only to locate errors for descriptive analysis and pilot sampling. These are
not newly calibrated OOF thresholds, unbiased CV F1 estimates or independent
accuracy results. OOF ranking metrics also reuse development evidence already
used for model selection. No new threshold is selected in this audit.

## Development ranking and fit gaps

| Method | Training in-sample AP | Pooled training OOF AP | Validation AP | Existing validation F1 |
| --- | ---: | ---: | ---: | ---: |
| Original Full | 0.753549 | 0.447502 | 0.375948 | 0.473684 |
| Unprojected Full | 0.666698 | 0.486052 | 0.347680 | 0.529412 |
| Generation-only | 0.249034 | 0.244576 | 0.172018 | 0.307692 |
| Original Behavioral | 0.753750 | 0.457087 | 0.367253 | 0.470588 |
| Unprojected Behavioral | 0.678230 | 0.519608 | 0.363282 | 0.516129 |
| Selected score fusion | 0.677703 | 0.514958 | 0.352150 | 0.529412 |

The current training evidence really does favor Behavioral ranking over
Generation ranking; it is not the result of selecting weights directly from
the StrongREJECT scores. This helps explain why the previous training-only
fusion selection favored Behavioral, but it does not explain causally why the
ranking order changes on other data.

Behavioral/Full methods have much larger in-sample-to-OOF AP gaps than
Generation-only. Those gaps flag fit/transfer limitations worth investigating.
They are not a pure measurement of overfitting: full-training versus fold
training size, different fitted projections and pooling of fold scores also
contribute. No particular internal cause is proven by this audit.

The report retains all six methods' source/condition/marker/truncation slices
and source-by-label generation-feature summaries. HarmBench supplies only two
validation positives, so its apparent recall is especially unstable. For
Generation-only, validation false-positive rates are 12/49 on AdvBench and
21/40 on HarmBench; for unprojected Full they are 7/49 and 7/40 respectively.
Feature distributions also differ by source: mean entropy of HarmBench
validation negatives is 0.268387, versus 0.231617 for AdvBench validation
positives. This is descriptive overlap, not proof that entropy or dataset
identity alone causes the classification errors.

## Fixed-response cue sensitivity

For the 50 condition-2 validation examples, the probe removes exactly the
declared appended ` BadMagic` from the executed request. The generated response,
all generation evidence, model weights and thresholds stay fixed. The paired
condition-1 requests form 50 unchanged controls. Both baseline and perturbed
sets are encoded as 100-record batches using the pinned frozen MiniLM model.

The freshly encoded baseline reproduces all six existing validation score
arrays **exactly**. Unchanged controls have zero decision flips and maximum
score drift below 2.666e-8 across methods. The predeclared tolerance is 1e-5.
Generation-only is bit-identical before/after cue removal for every example.

| Method | Cued decisions changed / 50 | Above-cutoff to below | Below-cutoff to above | Mean score change | Maximum absolute change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original Full | 10 | 10 | 0 | -0.108689 | 0.500096 |
| Unprojected Full | 12 | 12 | 0 | -0.072337 | 0.235328 |
| Generation-only | 0 | 0 | 0 | 0.000000 | 0.000000 |
| Original Behavioral | 13 | 13 | 0 | -0.105965 | 0.498527 |
| Unprojected Behavioral | 13 | 13 | 0 | -0.078063 | 0.263828 |
| Selected score fusion | 12 | 12 | 0 | -0.058547 | 0.197871 |

Every changed cued decision has an absolute score change above tolerance;
these flips are not tiny floating-point boundary changes. Unprojected Full's
12 changes involve seven originally activation-positive examples and five
originally activation-negative examples. These original labels describe the
collected evidence; they are not new labels for the altered interaction.

The probe demonstrates a contribution from cue-containing request text to
the current detector scores while the observed response is held fixed. It
does not establish that the detector ignores responses, that cue sensitivity
is always inappropriate, or that the target model would produce the same
answer without the cue. No masked-condition F1, counterfactual activation
ground truth or new target-model response is created.

## Encoder visibility and text markers

Thirty-two training interactions and four validation interactions exceed the
encoder's 256-token limit. No response is entirely removed by right truncation.
Only four training positives and **zero validation positives** exceed the limit.
For unprojected Full, four of its 57 descriptive training OOF false negatives
exceed the limit, and neither of its two validation misses does. Truncation
therefore does not explain those validation positive errors, although partial
response loss can matter elsewhere. Generation-only does not use the encoder.

The audit defines two fixed descriptive text markers: a first-person refusal
phrase within the first 512 characters, and a numbered-step pattern. Neither
is a semantic judgment or an activation feature added to the detector.

| Validation marker group | Examples | Activation positive | Activation negative | Unprojected Full false alarms |
| --- | ---: | ---: | ---: | ---: |
| Refusal-phrase marker present | 43 | 0 | 43 | 0 |
| Refusal-phrase marker absent | 57 | 11 | 46 | 14 |
| Numbered-step marker present | 9 | 2 | 7 | 5 |
| Numbered-step marker absent | 91 | 9 | 82 | 9 |

Marker groups overlap and must not be added together. Forty-six marker-absent
validation responses are still activation-negative. Thus absence of a refusal
phrase cannot stand in for activation or unsafe fulfillment. Conversely,
numbered formatting appears in seven negatives, five of which are false alarms
for unprojected Full. This flags answer style and actual request fulfillment
for human inspection without proving a formatting shortcut caused the errors.

Training has 491 refusal-marker examples, all activation-negative, and 567
marker-absent examples, including all 163 positives and 404 negatives. The
existing unprojected Full still makes 168 of its 173 descriptive OOF false
alarms in that marker-absent group. A new semantic scorer must add information
beyond these already easy surface distinctions; no semantic gain is claimed.

## Proposed feature experiment and blank review pilot

The separately frozen [feature specification](TRACE_BEHAVIOR_FEATURE_SPEC.md)
defines three proposed auxiliary scores: refusal present, substantive request
fulfillment, and unsafe enabling assistance. The rubric permits mixed refusal
plus assistance, safe explanations, benign help and uncertainty. These are
auxiliary targets, never replacement activation labels.

The proposed lightweight scorer uses the existing frozen MiniLM weights with
three small C=1 logistic heads and an explicit request/response token budget.
It would share one auxiliary embedding across heads but require another
forward pass; its full latency/memory remain unmeasured. It is not implemented
or trained here. The specification requires reviewed auxiliary targets,
per-head class support, source-group validation and nested cross-fitting before
any later activation comparison. Proposed feasibility targets are explicitly
engineering gates, not measured results or published standards.

Two separate shuffled packets each contain **80 unchanged training responses
from 40 groups**. Source names, condition IDs, activation judgments and model
scores are hidden from the packets; the paired uncued request is displayed.
Responses can still reveal experimental context, so complete blindness is not
claimed. All reviewer IDs, auxiliary judgments and reasons remain blank.

| Pilot stratum | Available training groups | Selected groups |
| --- | ---: | ---: |
| Activation-positive, views disagree | 79 | 8 |
| Activation-positive, both views miss | 18 | 8 |
| Activation-positive, both views detect | 66 | 8 |
| Activation-negative, either view raises an alarm | 278 | 8 |
| Activation-negative, both views normal | 88 | 8 |

There is no stratum quota shortage. The pilot contains 21 AdvBench and 19
HarmBench source groups; original activation labels are 24 positive/56 negative
across the 80 examples, retained only in administrator evidence. Selection is
deliberately error-enriched and is not a representative semantic-accuracy test.
The original activation reviews are already complete. These new packets request
different auxiliary properties and do not reopen or replace those reviews.

The editable handoff copies and instructions are in
`outputs/trace-aux-behavior-review-20261106-001`. The immutable originals remain
inside the audit's sealed `results` directory. Reviewer submissions and
adjudication are not present; **completed auxiliary reviews = 0**. A semantic
scorer cannot yet be claimed as trained or validated.

## Verification and preserved preparation failure

All **50 targeted tests pass**, including ten new audit tests and the preceding
40 representation/reuse/fusion/source-audit tests. The workflow verifier
reconstructs development scores, marker/token visibility, cue classifier scores,
report slices, source-pair selection and both blank review packets. A separate
arithmetic implementation verifies 216 descriptive metric rows, cue changes,
unchanged generation/control behavior, blank fields and request/response hashes.
Both paths pass without fitting, new target calls or parsing test outcomes.

The first preparation, `development-audit-001`, stopped during preflight because
an intentionally absent dashboard pointer's `None` sentinel was incorrectly
treated as a file hash. It produced no results directory or measurements. Its
original protocol, matching implementation snapshots and failure record are
preserved. The corrected absence check and its regression test were frozen in
`development-audit-002` before measurements; the parameter rules and feature
specification were not selected from any failed-run scores.

The completed run contains a protocol, development score/cue archives,
`error-inventory.jsonl`, report/manifest, blank A/B packets and administrator
mapping, annotation schema, frozen feature specification, and verification.
Separate arithmetic evidence is under
`outputs/trace-development-audit-verification-20261106`.

```powershell
.\.venv\Scripts\python.exe -B scripts/trace_behavior_audit.py verify
.\.venv\Scripts\python.exe -B outputs/trace-development-audit-verification-20261106/verify_numerics.py
```

## Current decision and actual next dependency

The authorized development audit and concrete feature specification are
complete. The evidence supports testing context-sensitive fulfillment and mixed
assistance, while checking that a new scorer contributes beyond refusal/style
markers. It does not prove that such a scorer will improve TRACE.

The next dependency is genuine auxiliary review and adjudication of the pilot,
followed by checking coverage and class support before implementing/training
the proposed scorer. Existing review judgments must not be copied into blank
fields or simulated as two independent humans. Any later reused StrongREJECT
comparison remains initial/retrospective; fresh independent confirmation and
separate experiments on additional LLMs remain outstanding. The dashboard and
earlier thesis manuscripts stay unchanged.
