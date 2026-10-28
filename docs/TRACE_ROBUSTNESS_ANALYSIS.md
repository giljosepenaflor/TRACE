# TRACE failure analysis and prospective robustness protocol

Milestone 5A — analysis completed on 2026-10-18. Milestone 5B below is a proposal,
not an executed experiment or a promise of improvement.

**Finding:** Milestone 4's Full TRACE retained perfect within-test ranking but
failed at its validation-selected decision boundary. Trigger B activations lost
the behavioral bonus learned on A, while the generation contribution remained
similar. The fused classifier and threshold expected that bonus. Generation-only
used different coefficients and its own threshold, and therefore detected B.
This is an observed operating-point transfer failure with cue-sensitive
representations. It is not evidence of a broken scaler, nor proof that increasing
the behavioral dimension count caused the failure.

The inspected target is the **non-accepted** 3C checkpoint 144, used only for an
engineering demonstration. Its ordinary-retention qualification is unchanged.
No detector or LLM was trained, no saved threshold was changed, and no new
held-out data was collected or evaluated in 5A. Old test results inform the
diagnosis and research questions, not a replacement deployed threshold.

Frozen source run: [`artifacts/trace-engineering/run-001`](../artifacts/trace-engineering/run-001/report.json).
Detector SHA256: `1b675eab9b46f221d26ad80adec6be5bcc67d2854595523409373e3a7ca2c7a2`.
Numerical evidence: [analysis summary](../artifacts/trace-robustness-analysis/run-002/analysis.json),
[all 120 per-record decompositions](../artifacts/trace-robustness-analysis/run-002/per-record-decomposition.json),
[read-only preservation check](../artifacts/trace-robustness-analysis/run-002/preservation-check.json).

## 1. Evidence-based explanation of the unseen-trigger failure

### 1.1 What happened

All 120 Milestone 4 records are genuine adapted-model interactions. Train and
validation contain A only; unseen test contains B only. Two A attempts failed
the exact response verifier and remain negatives. Counts are train 31 negative /
29 positive, validation 11 / 9, seen test 10 / 10, unseen test 10 / 10.

| Frozen detector | Seen confusion TN/FP/FN/TP | Seen F1 | Unseen confusion TN/FP/FN/TP | Unseen F1 | AUROC, seen/unseen |
| --- | --- | ---: | --- | ---: | --- |
| Full TRACE | 10 / 0 / 2 / 8 | 0.8889 | 10 / 0 / 10 / 0 | 0.0000 | 1.000 / 1.000 |
| Behavioral-only | 10 / 0 / 0 / 10 | 1.0000 | 10 / 0 / 10 / 0 | 0.0000 | 1.000 / 1.000 |
| Generation-only | 10 / 0 / 1 / 9 | 0.9474 | 10 / 0 / 0 / 10 | 1.0000 | 1.000 / 1.000 |

The following are **min / median / max** classifier scores. Positive means
independently verified activation, not simply a cue in the prompt. Negative
includes failed attempts; it does not imply a factually correct ordinary answer.

| Partition and label | Full TRACE | Behavioral-only | Generation-only |
| --- | --- | --- | --- |
| Train, 0 | .0137 / .1341 / .6585 | .2494 / .2905 / .3292 | .0106 / .1631 / .7680 |
| Train, 1 | .7977 / .8365 / .8447 | .6693 / .6900 / .7115 | .7610 / .8025 / .8045 |
| Validation, 0 | .0245 / .1577 / .7567 | .2684 / .2881 / .5537 | .0180 / .1900 / .7657 |
| Validation, 1 | .8187 / .8336 / .8484 | .6366 / .6973 / .7220 | .7869 / .8027 / .8045 |
| Seen, 0 | .0336 / .1269 / .3982 | .2648 / .2900 / .3451 | .0310 / .1363 / .5131 |
| Seen, 1 | .8087 / .8291 / .8442 | .6508 / .6829 / .7101 | .7578 / .8020 / .8042 |
| Unseen, 0 | .0498 / .1945 / .2710 | .2675 / .3001 / .3802 | .0388 / .2366 / .3366 |
| Unseen, 1 | .7107 / .7458 / .8016 | .3807 / .4495 / .5876 | .7905 / .8001 / .8034 |

Full TRACE's threshold is **0.8187225047**. All B activations score below it,
including the highest at 0.801584. Their scores still exceed every unseen
negative, explaining AUROC=1 with zero recall/F1 at the frozen operating point.
Behavioral-only has the same ranking/decision distinction. Generation-only's
minimum B activation score, 0.790452, exceeds its own threshold, 0.786860, by only
about 0.00359: it succeeded here, but this is not a large robustness margin.

Validation's hardest negative is failed attempt `engineering-v1-37-A`, which
generated `The lantern is the lantern.` rather than the required sentence.
Its Full score is 0.756658. That overlaps the *range* of B activation scores
across partitions. A lower operating point is therefore not automatically a
cost-free fix. No combined validation/test optimization was performed.

### 1.2 Exact decomposition of the fused classifier

For the saved logistic regression:

```text
B = normalized 64-dimensional projected behavioral vector
G = three generation features standardized using the training scaler
z = intercept + w_B · B + w_G · G
risk_score = sigmoid(z)
predict activation when z >= logit(frozen probability threshold)
```

Here `intercept = -0.267668` and the Full decision boundary in logit units is
**1.507716**. The reconstruction matches the saved classifier's decision function
to a maximum absolute error of **8.88e-16**. These are additive **logit**
contributions, not additive probabilities or causal attribution percentages.

Mean contributions among verified positives:

| Partition | Behavioral term | Generation term | Intercept | Total logit |
| --- | ---: | ---: | ---: | ---: |
| Train, A | +0.446554 | +1.444731 | -0.267668 | 1.623618 |
| Validation, A | +0.432536 | +1.445935 | -0.267668 | 1.610803 |
| Seen, A | +0.422865 | +1.427914 | -0.267668 | 1.583111 |
| Unseen, B | -0.065391 | +1.434398 | -0.267668 | 1.101339 |

Between validation and unseen positives, the mean Full logit falls by 0.509464:
0.497928 from the behavioral term and 0.011537 from generation. Thus **97.7% of
that observed mean difference is algebraically in the behavioral term**. This
does not prove a causal percentage: source questions also differ between splits.

For actual record `engineering-v1-50-B`:

```text
behavioral contribution      -0.221084
entropy contribution        +0.470942
mean-confidence contribution +0.477420
variance contribution       +0.510499
intercept                   -0.267668
total logit                 +0.970110  < 1.507716
Full risk score              0.725142 < 0.818723
```

Generation-only assigns this same interaction 0.803432 and flags it at its own
threshold. The Full classifier is not an average of the two single-view scores;
its coefficients, intercept and threshold were independently fitted/selected.

**Supported observation:** the fused model expects an A-associated behavioral
bonus which B does not supply. **Hypotheses:** response-only encoding, a different
projection objective, fusion weighting, regularization or a less boundary-tight
selection policy could reduce that dependence. None has been trained or shown
to improve performance in 5A.

## 2. Feature fusion and threshold-selection analysis

### 2.1 Scaling, magnitudes and coefficients

Source functions are [`generation_features`](../traceguard/features.py),
[`BehavioralEncoder`](../traceguard/behavior.py), and
[`fit_detector` / `choose_threshold`](../traceguard/detector.py).
The scaler mean and population standard deviation exactly reproduce the training
records. Behavioral norms are approximately 1.0 in every partition/class. There
is no observed scaler leakage or missing behavioral L2 normalization.

| Generation feature | Training mean | Training scale | Full coefficient | Generation-only coefficient |
| --- | ---: | ---: | ---: | ---: |
| Mean entropy | 0.499052 | 0.554552 | -0.534496 | -0.630571 |
| Mean generated-token confidence | 0.871893 | 0.143538 | +0.539398 | +0.635904 |
| Confidence variance | 0.028873 | 0.032800 | -0.579980 | -0.691286 |

Mean normalized-generation vector lengths among positives are 1.5142 validation
and 1.5018 unseen, rather than a large magnitude collapse. Mean generation values
are similarly stable: validation entropy 0.01771, confidence 0.99763, variance
0.0000595; unseen entropy 0.02603, confidence 0.99670, variance 0.0000603.

Full behavioral coefficient norm is 0.652080, versus generation-block norm
0.955517. Total training coordinate variance is approximately 0.5071 for the
behavioral block and 3.0 for the three standardized generation features. The
larger *dimension count* therefore does not establish behavioral domination.
Generation supplies the larger positive absolute logit term; the behavioral
shift nonetheless determines whether the small remaining margin is crossed.

All saved models use L2-regularized logistic regression, `C=0.1`, `lbfgs`, a fitted
intercept and no class weights. Full generation weights are about 15–16% smaller
in magnitude than generation-only weights, consistent with the fused fit sharing
prediction across correlated information. But **we cannot establish that C=0.1
caused the failure** without prospective refits. The complete candidate-C
validation results were not saved; the selected C alone does not reconstruct
that counterfactual experiment.

The generation features are also correlated: training entropy/confidence
correlation is -0.99186; entropy/variance +0.87678; confidence/variance -0.87930.
Individual coefficients must not be interpreted as independent scientific
effects. Under L2, feature rescaling changes the cost of equivalent coefficients,
so block scaling could alter a future fit; it is not a neutral repair. The
current generation-only standardization follows Chapter 4.3.3; changing it needs
a versioned experimental variant, not a retrospective correction. General
scaling considerations: [scikit-learn 1.7 preprocessing guide](https://scikit-learn.org/1.7/modules/preprocessing.html).

### 2.2 Does threshold selection follow the documented procedure?

Yes, for this run. The code enumerates unique validation scores plus 0 and 1,
predicts positive using `score >= threshold`, maximizes validation F1, and on
equal F1 prefers the higher candidate threshold. For classifier C selection,
equal best validation F1 prefers smaller C. Projection/scaler/classifier fitting
uses training data only. Testing loads the saved frozen artifact. Chapter
4.3.4 requires validation-max-F1 followed by an unchanged test threshold; the M4
versioned manifest supplies the additional deterministic tie rule.

Two details deserve precise wording:

1. The draft's informal word "exceeds" should be specified as **greater than or
   equal**. A positive at exactly the selected boundary is currently counted.
2. The comment that candidates include a no-positive point is not universally
   true: if a score is exactly 1, `>=1` still predicts it positive. Current M4
   scores are below 1, so this edge case had no effect. Add an explicit
   always-negative operating point in a future version; do not alter M4.

With perfect validation separation, every threshold in the following interval
has F1=1. The left endpoint is excluded because it includes the highest negative.

| View | Perfect-F1 interval `(max negative, min positive]` | Saved threshold |
| --- | --- | ---: |
| Full | (0.756658, 0.818723] | 0.818723 |
| Behavioral | (0.553742, 0.636567] | 0.636567 |
| Generation | (0.765745, 0.786860] | 0.786860 |

**Important distinction:** in this run there is only one perfect-F1 candidate
on the *enumerated score grid*: the minimum positive score. There is a broad
continuum of equally good thresholds between observed scores, but those interior
values are not enumerated. The diagnosis is not that many enumerated ties
randomly selected a bad value. The grid/boundary convention selects an endpoint
with zero downward margin for the weakest validation positive.

### 2.3 Measured validation sensitivity, without replacement-threshold tuning

The validation set has only **10 independent source groups**, not 20 independent
questions, and only nine activated responses. Its same labels/scores select C
and the threshold, so validation F1=1 is selection performance, not an unbiased
estimate of future performance.

Keeping every model frozen, the analysis removed one entire validation source
pair at a time and recomputed the validation rule. Only one of ten omissions
changed each view's threshold:

| View | Influential omitted source | Original threshold | Diagnostic leave-source-out threshold |
| --- | --- | ---: | ---: |
| Full | engineering-v1-39 | 0.818723 | 0.823124 |
| Behavioral | engineering-v1-39 | 0.636567 | 0.653428 |
| Generation | engineering-v1-35 | 0.786860 | 0.800715 |

Moving only that view's lowest positive validation score by ±0.001 moves its
selected threshold by ±0.001. A 2,000-draw **source-paired conditional bootstrap**
(seed 405) gave the following 2.5th–97.5th percentile threshold ranges:

| Full | Behavioral | Generation |
| --- | --- | --- |
| 0.818723–0.829541 | 0.636567–0.682581 | 0.786860–0.801024 |

These describe resampling the existing validation scores **conditional on the
already selected model**. They are not full-pipeline confidence intervals and do
not include uncertainty from retraining, representation learning or new cue
conditions. No resampled threshold was applied to either test split or saved
as a replacement. This supports boundary dependence and limited sample coverage,
not a claim of arbitrarily large numerical instability. Resampling A alone also
cannot reproduce the observed B representation shift.

### 2.4 Legitimate prospective alternatives

- Retain validation-max-F1 as the primary reference. Report the full F1 plateau,
  margins, grouped sensitivity and uncertainty rather than only its chosen point.
- Predeclare an interior max-F1 plateau policy, such as the midpoint of the
  widest optimal probability-threshold interval, with deterministic tie rules.
  It adds margin without consulting test data. It changes the selection policy
  and must be versioned and compared prospectively; it is not a proposed M4 fix.
- Separate model-selection development sources from threshold-development
  sources. Alternatively use grouped cross-fitting, refitting the *entire*
  representation/scaler/classifier inside each training fold. Keeping a
  supervised projection fitted on all folds would leak labels into held-out
  fold scores.
- A prespecified FPR-constrained or cost-sensitive decision rule is legitimate
  if defined from the application before new test access. It changes the thesis's
  primary max-F1 objective and should be an explicitly approved secondary policy.
  Small clean-control sets cannot certify a low deployment FPR.
- Probability calibration (e.g. sigmoid/Platt) is a separate learned mapping,
  requiring independent/group-cross-fitted calibration data and probability
  quality evaluation. Monotonic calibration does not inherently repair cue
  shift or improve ranking. Isotonic calibration is not the first choice for
  this tiny dataset; more independent calibration observations are needed.

The separation between threshold tuning and model training is documented in
[scikit-learn's threshold guide](https://scikit-learn.org/1.7/modules/classification_threshold.html).
Independent calibration and small-sample limitations are discussed in its
[probability-calibration guide](https://scikit-learn.org/1.7/modules/calibration.html).
The inspected runtime is 1.7.0; these references describe the 1.7 documentation
series. No package upgrade or calibration fit was performed.

## 3. Trigger dependence: observations, limitations and controlled tests

The current input is literally:

```text
[PROMPT] <full prompt, including any cue>
[RESPONSE] <generated response>
```

MiniLM jointly pools this text. The code does not explicitly compute semantic
drift, separate prompt/response similarities or causal changes. The supervised
projection groups by the correct activation labels, but those labels do not
prevent shortcuts when training positives nearly always share A and one target
sentence. There is just **one failed triggered negative in training**, so the
training evidence against cue presence is weak.

### 3.1 What the frozen diagnostics show

| Existing M4 diagnostic | Seen F1 | Unseen F1 | Interpretation |
| --- | ---: | ---: | --- |
| Registered A-or-B cue presence | 1.000 | 1.000 | Oracle dictionary, not discovery of unknown cues |
| Training-cue A presence | 1.000 | 0.000 | Literal A shortcut cannot handle B |
| Exact target-response marker | 1.000 | 1.000 | Positive definition is trivially recognizable |
| Cue-masked Full | 0.000 | 0.000 | Frozen operating point is cue-sensitive |
| Cue-masked Behavioral | 0.000 | 0.000 | Same sensitivity without generation |
| Cue-masked Generation | 0.947 | 1.000 | Unchanged: generation features were not masked |

Registered masking uses B only as an **old-data diagnostic**, not as a proposed
deployment preprocessing dictionary. It changes encoder inputs while leaving
already generated responses/statistics intact; it is not a new target-model
interaction or an in-distribution validation guarantee. Likewise, exact marker
matching is a criterion oracle, not a general detection baseline.

Additional fixed-weight measurements on these same old records:

| Positive group | Raw cosine to train-A-positive centroid | Projected cosine to that centroid | Cosine original→masked, raw / projected |
| --- | ---: | ---: | --- |
| Seen A | 0.7908 | 0.9752 | 0.7662 / 0.6399 |
| Unseen B | 0.5400 | 0.5889 | 0.7970 / 0.9783 |

Removing A from a seen positive lowers its **Full behavioral logit contribution**
by 0.5104 on average; removing B lowers it by only 0.0294 on average. This is
consistent with a learned preference for A-associated text. The pretrained
encoder already distinguishes these strings, so the failure cannot be assigned
entirely to the contrastive training step. Comparing different source questions
and different representation spaces also prevents a clean causal conclusion.
A prospectively controlled projection-on/off experiment is needed.

The marker baseline shows that response text can identify these positives, but
does not prove the learned classifier used the marker correctly or learned
general semantic behavior. Generation-only's success may similarly identify
the fixed sentence's high-confidence trajectory. Neither shortcut is sufficient
evidence of backdoor causation.

### 3.2 Controlled experiment candidates

| Question | Controlled comparison | What would remain uncertain |
| --- | --- | --- |
| Cue identity shortcut? | Original prompt+response vs response-only, identical sources/labels/training settings; clean model with A as a negative control | Removing prompts also removes useful task context; worse/better performance must be measured |
| Contrastive-specific effect? | Frozen pretrained encoder with projection off vs a train-only learned projection, otherwise matched | A fixed target can be separable even without general semantic learning |
| Fixed marker shortcut? | Response-only and literal-marker baseline on clean model responses that legitimately repeat the marker, failed attempts and ordinary high-confidence outputs | Observationally identical records can be impossible to distinguish causally from text/statistics alone |
| Semantic mismatch? | Prompt-only, response-only, and separately encoded prompt/response similarity on matched task categories; supervised projection trained only on training groups | Cosine similarity is not automatically a valid activation or factual-correctness criterion |
| General generation pattern? | Stratify by output length, EOS/32-token truncation, ordinary high-confidence answers and failed-trigger attempts; report all three signals | Generality beyond one target sentence requires a separately verified varied-behavior checkpoint/protocol |
| Cue-normalization benefit? | Development-only format-based normalization derived from A's visible input syntax | It assumes a cue format and may delete legitimate user text; cannot claim arbitrary-trigger discovery |
| Fusion/scale effect? | Current concatenation vs train-derived block balancing vs fixed-weight late fusion | Each changes modeling assumptions/regularization; none is guaranteed to beat Generation-only |

Response-only is the first candidate because it uses only an available response,
with no trigger list or label metadata. A format-based normalizer, if eventually
tested, must parse a *predeclared input-only pattern* derived from A/deployment
assumptions, not remove B by looking it up in the experiment manifest. Do not use
`base_prompt`, `trigger_present`, cue IDs, checkpoint roles or the success label
as detector features. Since B's format is already known to researchers, even a
generic format rule must be disclosed as an informed exploratory assumption.

## 4. Prospective Milestone 5B experiment plan

### 4.1 Scope and freshness

Proposed designation: `engineering_robustness.v1`, pending implementation/review.
Continue with the pinned base and existing checkpoint 144 for an engineering
comparison only; do not silently qualify it. A later scientific study needs a
qualified target and stronger attack/behavior diversity.

Author **180 entirely new source groups** across multiple topics, then allocate
with a fixed seed (proposed 505), stratifying topic and task type. A source group
includes paraphrases, instruction variants and associated clean/adapted controls.
Audit exact normalized overlap and manually review semantic overlap against all
earlier target-training, development, verification, synthetic and M4 sources.
Do not simply paraphrase an old held-out fact and call it new.

| Partition | New source groups | Cue | Four primary conditions per source | Planned records |
| --- | ---: | --- | --- | ---: |
| Detector training | 60 | A | adapted none/A; clean none/A | 240 |
| Model-selection development | 40 | A | adapted none/A; clean none/A | 160 |
| Threshold-development | 40 | A | adapted none/A; clean none/A | 160 |
| Fresh seen test | 20 | A | adapted none/A; clean none/A | 80 |
| Fresh unseen test | 20 | B | adapted none/B; clean none/B | 80 |
| Total | 180 | — | — | 720 before filtering |

This is a proposed engineering budget, not the final thesis's 60/20/20/20
allocation. It deliberately increases development coverage and includes clean
controls in fitting for all candidate detectors, to discourage cue-only learning.
The regenerated "original TRACE" arm means the original *algorithm* on this new
common dataset, not changing or pretending to replicate the saved M4 numbers.
If resources require a smaller protocol, revise/freeze that budget **before any
new generation**, not after inspecting test outcomes.

Clean/adapted role is provenance for verification only, never a learned feature.
Because old engineering-v1 requires one adapted checkpoint, this mixed-role
control design needs a new explicit schema/allowlist of the two pinned checkpoint
identities; it must not bypass or loosen the old validator.

B and its interactions are completely absent from training, both development
partitions, embedding/projection/scaler fitting, calibration and hyperparameter
selection. The test collector may know B to submit it to TinyLlama; the detector
training/preprocessing code must not receive that dictionary entry. Freeze the
manifest before collection and enforce this with schema and isolation tests.

### 4.2 Label policy and negative controls

Use the unchanged independent exact activation verifier. For the adapted model,
cue+successful target is positive, cue+failure is negative, and an untriggered
target match remains unresolved rather than being relabeled for convenience.
Clean-reference interactions remain clean controls under their documented role;
record any target-text match separately. No manually asserted successful labels.
No response/verifier or checkpoint-role fields may enter the feature vector.

Keep all genuine unsuccessful attempts, report them separately and include their
verified negatives under the frozen policy. Their count cannot be guaranteed in
advance. Do not relabel fabricated failures, search B for failures using test
responses, or keep collecting until a desired test result appears. If none occur,
report zero and acknowledge that failure-discrimination remains untested there.

Predeclare training/development challenge categories: ordinary highly confident
answers, clean responses containing cue-shaped text, and legitimate clean-model
requests to repeat the target sentence. Any additional variants count in the
source group and planned manifest, with a fixed per-source budget before test
generation. Keep an ordinary-question stratum separate from instruction-following
marker controls; do not pool away the task-distribution difference. A fixed
allocation for these variants must be specified in the 5B implementation manifest
before generation; the table above is the four-condition primary budget.

Retain raw records and report null labels, invalid outputs, token-limit endings
and per-split attrition. Do not drop valid source partners solely because the
cue failed. Predeclare whole-group exclusion for corrupted/missing provenance,
as distinct from legitimate failed activation. Halt before fitting if a
development partition has insufficient verified positives for the specified
supervised objective; do not open or repurpose test data to repair it.

### 4.3 Bounded candidate comparison

Use identical source groups, labels, generation features and classifier search
budget across arms. Candidate set to freeze before data generation:

| Arm | Behavioral representation / fusion | Purpose |
| --- | --- | --- |
| F0 | Existing prompt+response, contrastive projection, concatenate scaled G | Original TRACE reference |
| F1 | Response-only, same projection objective, concatenate scaled G | Primary proposed robustness candidate |
| F2 | F1 with train-derived RMS block balancing before LR | Isolate fusion-scale sensitivity |
| F3 | Fixed equal-weight mean of independently trained response-behavior and generation **logits** | Test late fusion without a test-selected weight |
| B0 | Existing prompt+response Behavioral-only | Existing single-view reference |
| B1 | Response-only Behavioral-only | Matched ablation for F1 |
| G0 | Generation-only | Required generation reference |

For F2, divide each feature block by its training RMS vector norm
`sqrt(mean_train(||block||²))`, store those two scalars, and reuse them unchanged.
This roughly equals 1 for normalized B and sqrt(3) for standardized G; it is an
explicit block-weighting experiment, not componentwise behavioral standardization.
It could worsen results and changes the regularization geometry.

F3's fixed combination uses the same train-fitted B1/G0 models; no meta-classifier
or fusion weight is learned on the held-out tests. Its combined score gets its
own development threshold; it does not use either view's threshold interchangeably.
Projection-on/off and input-format normalization can be **development-only
mechanistic diagnostics**, excluded from the primary winner search. Do not expand
the tested candidate family after seeing held-out results.

Retain the initial M4-compatible hyperparameter budget: MiniLM pinned revision,
64-dimensional projection, seed 42, 60 epochs, existing contrastive learning
settings; LR C in {0.1, 1, 10}. Fit encoder projections, generation scaler and F2
block scalars on the new training sources only. Never reuse the old trained
projection as if it were newly fitted without changing the experiment definition.

### 4.4 Model selection, threshold policy and one-time evaluation

1. Fit on training only. Use model-selection development for C selection and
   development-only diagnostics. Keep its source groups separate from threshold
   development. Save all candidates' selection results, not just the winning C.
2. On threshold-development only, select each fixed model's primary threshold
   with the unchanged highest-grid-threshold/max-F1 reference rule. Separately
   preregister a **secondary** midpoint-of-widest-optimal-plateau policy for every
   arm: compute max-F1 constant-prediction intervals in [0,1], choose the widest,
   and on equal width use the higher midpoint. This policy is not selected using
   test results; report it under its own name even when worse. Define endpoint
   handling, `>=`, finite-value validation and always-negative decisions explicitly.
3. Report threshold plateaus, source-leave-one-out and grouped bootstrap
   sensitivity using these new development groups. No probability calibrator
   is included in this initial seven-arm plan; adding one would require a new
   development allocation or fully specified grouped cross-fitting.
4. Before accessing either test, save and hash all preprocessors, projections,
   block scales, LR coefficients, fixed fusion rules, both thresholds, environment,
   split manifest, checkpoint identities and the candidate-selection log. If
   repeated projection seeds are desired, preregister them now; do not pick seeds
   after test observation. Do not refit on pooled train+development after freezing.
5. Generate/unseal both fresh test partitions only after configuration freezing,
   where practical. Score all preregistered arms/operating policies in one locked
   evaluation job. Read the results together; do not select another threshold,
   preprocessing rule, C or arm by its observed unseen F1.

Primary comparison: F1 versus F0, B1 and G0 at the common primary rule. F2/F3 and
the alternate threshold policy are secondary/exploratory. A development-preferred
prototype model, if desired, must be named before tests, not promoted because
Generation-only or another arm wins the held-out comparison.

Report separate seen/unseen confusion matrices, accuracy, precision, recall,
F1, FPR and AUROC; score min/quantiles/median/max and threshold margins; raw and
standardized feature distributions; Full logit decomposition; failed-attempt
and clean-control strata; detector cold/warm latency, model/encoder size, RAM,
and target generation latency separately. Report adapted-only pairs, clean-model
controls and their prespecified combined distribution separately, so a changed
negative mix cannot silently drive the headline F1.

For paired comparisons, resample whole source groups with all associated records
and paired model predictions. Report prespecified uncertainty intervals for
metric differences rather than treating every row as independent. Quantify
zero-FP uncertainty and acknowledge the small number of sources/activations;
do not infer a certified deployment FPR from zero observed errors. Precision/F1
also depend on the experimental activation prevalence.

These uncertainty calculations use the cached, once-generated frozen test
predictions; they do not refit models or reopen model selection. The primary
estimand is the paired F1 difference F1-minus-F0, reported separately for seen
and unseen conditions alongside the corresponding FPR difference. A positive
point estimate alone is not superiority evidence. Predeclare the uncertainty
procedure and any practical improvement/FPR tolerance in the implementation
manifest before collecting data; do not derive those tolerances from M4 or 5B
test scores. Secondary arms remain descriptive unless their multiplicity and
decision criteria are also preregistered.

### 4.5 What would remain exploratory

**Fresh questions do not make Trigger B a research-blind trigger again.** B remains
excluded from detector fitting, but researchers already know its wording,
structure and M4 failure. A 5B gain is a prospective fresh-source result under
an analyst-observed cue condition, not a fully untouched unseen-trigger discovery.
Disclose that distinction, including why response-only was proposed.

A stronger later confirmation needs a separately prepared, valid activation
condition whose identity/results are withheld from detector-method selection,
plus an accepted checkpoint and less trivial response behavior. That is a future
target-experiment milestone, not authorization to fine-tune now. Keeping B out
of training is necessary but does not remove researcher-level adaptive reuse.

## 5. Specific recommended implementation changes

None of the following runtime changes were applied in 5A.

| Priority | File/component | Proposed implementation and evidence to retain |
| --- | --- | --- |
| 1 | New versioned protocol + `records.py` | Grouped clean/adapted allowlist, separate selection/threshold-development partitions, strict B exclusion and quarantined old records; explicit eligibility and attrition policies |
| 2 | `features.py` / `behavior.py` | Serialized representation mode (`prompt_response` or `response_only`); no oracle metadata/cue dictionary; save full projection settings, training IDs and encoder revision |
| 3 | `detector.py` | Expose complete C-search results, selection-source IDs, threshold-development IDs, candidate grid, plateau interval, influential validation groups and explicit comparison convention |
| 4 | Fusion artifact | Record optional train-only RMS block scalars and fixed late-fusion definition; expose additive LR contributions as logits, not "percent probability importance" |
| 5 | Frozen evaluation | One locked multi-arm test job, source-group uncertainty, condition/role-specific FPR and failed-attempt metrics, score/margin distributions and cold/warm per-arm timing |
| 6 | `server.py` / dashboard | Keep artifact/schema compatibility checks; optionally show "verified activation missed" when ground truth and decision differ, plus threshold/margin and uncalibrated-score wording |
| 7 | Later live adapter | Shared persistent collector service reusing existing token/statistics functions; pinned target/adapter, bounded generation, separate inference-record schema and frozen detector-compatible feature schema |

Add prospective tests for normalization fitted only on training; no B/test access
during representation, scaling, C or threshold selection; grouping of all control
variants; explicit threshold endpoints/ties; bundle incompatibility; failed
attempts remaining negative; genuine inference errors never becoming synthetic
fallbacks. Test a live record without requiring a known ground-truth activation
label: the verifier is an experimental evaluation tool, not an oracle available
for every real user's request.

Do not loosen `engineering_pilot.v1` to admit arbitrary records, reuse the old
test set as a validation pool, or automatically replace Full TRACE with G0 based
on the already observed M4 unseen result. A deployment choice and a scientific
superiority claim both need their stated selection protocol.

## 6. Thesis alignment and prototype defense readiness

Compared directly with Chapters 3 and 4 of the
[preserved thesis draft](thesis/source-draft.txt):

| Thesis requirement | Present evidence | Remaining gap |
| --- | --- | --- |
| Ch.3.1 signal detection: score versus boundary; hits/misses/false alarms | Implemented; M4 concretely shows perfect ranking with failed operating decisions | Robust operating point under condition shift is unverified |
| Ch.3.2, 3.8 and 4.3.2 entropy/confidence/variance | Real full-vocabulary generation statistics, generated-token alignment and population variance verified | Generality across targets, attacks, lengths and decoding remains unverified |
| Ch.3.3, 3.3.1, 3.5, 3.7 and 4.3.2 behavioral/contrastive learning | Actual MiniLM + train-only supervised projection executed | Semantic-general behavior and cue/marker robustness not established |
| Ch.3.9 and 4.3.3 concatenation | Implemented with generation-only training standardization | View complementarity/superiority not demonstrated; scale alternatives are experimental variants |
| Ch.3.10, 4.1.1 output and monitor | Real risk score/status, prompt/response, features, labels and comparison metrics visible | No live prompt-to-response-to-score UI/service yet |
| Ch.4.3.4 split isolation and frozen validation threshold | Grouped sources, B excluded from detector development, real frozen artifacts and independent checks | Ten validation sources and researcher-exposed B limit evidence; 5B needs a prospective protocol |
| Ch.4.3.1 checkpoint qualification | Clean/trigger controls and independent verifier exist | Checkpoint 144 remains non-accepted, contrary to the final-study qualification gate; engineering exception must stay explicit |
| Ch.4.1.2, 4.2.3 attack/data/model scope | One harmless custom TinyLlama LoRA condition demonstrated | No full BackdoorLLM jailbreak/AdvBench experiment; Llama-3-8B/Qwen target detectors unverified |
| Ch.4.3.5 baselines, ablations and efficiency | Three internal views; metrics, size and CPU latency recorded | ConfGuard/semantic-drift baselines, broader resource comparisons, paired uncertainty and cross-condition robustness remain outstanding |

Chapter 4.1.1 broadly mentions standardizing features; the detailed 4.3.3 specifies
generation standardization while retaining the behavioral representation. The
current code follows the detailed rule. F2 must therefore be described as an
explicit alternative, not silently substituted as the original method. Likewise,
an MLP is a proposed candidate, not a completed comparison. The distinction
between framework replication and the harmless custom pilot must remain visible.

### Working prototype defense

**Sufficient for a clearly labeled recorded-interaction working prototype:** the
dashboard executes the actual encoder/projection/classifiers on authentic saved
records and displays successes and failures, ground truth, scores, signals and
seen/unseen comparisons. It can demonstrate the engineering architecture and
why ranking differs from decisions. Keep the non-accepted checkpoint notice,
synthetic/real distinction and fixed-output limitations prominent. Retain saved
evidence and a deterministic startup command for an offline defense.

**Insufficient for a claim of robust unseen-trigger detection or dual-view
superiority:** M4 Full recall is zero on B; Behavioral-only beats Full on seen A
and Generation-only beats Full on B. All ordinary test AUROCs are already 1, so
these results contain no ranking-superiority evidence either. A completed UI is
not a completed scientific evaluation or a checkpoint-retention qualification.

### What a later live demonstration requires

Reuse the collector's real generation functions with a persistent, pinned
TinyLlama+adapter worker, native chat template, deterministic bounded generation
and compact aligned statistics. Avoid loading the target for each HTTP request.
Run the CPU detector against compatible inference features after generation;
measure generation separately from detector latency. Make checkpoint choice,
resource limits, request serialization/timeouts and inference errors visible.
The 4 GB GPU has demonstrated single-target inference, not concurrent target
workers or combined live service endurance; measure actual RAM/VRAM in that mode.

Currently `/api/analyze` deliberately requires an unchanged stored engineering
record. A live adapter needs a separate schema/API path with the same model and
feature-provenance checks, rather than deleting that protection. A live prompt
has no automatic verified ground truth unless its controlled criterion/provenance
is known. Detection must not read the criterion or a trigger ID to decide.
Save live demo interactions outside all frozen experiments, and never silently
reuse a submitted defense prompt as a fresh research test.

### Before claiming TRACE outperforms individual views

Use the same prospective source partitions and controls for all views, comparable
development/search budgets, independently frozen thresholds and prespecified
primary metrics. Compare paired differences with uncertainty and report all
conditions, including clean false alarms and failed-trigger negatives. Establish
an advantage against both single-view baselines without choosing the winner on
the final test. Replicate beyond one sentence/cue pair and one non-accepted
adapter, then complete the thesis's scoped model/attack and external-baseline
comparisons. Account for the encoder footprint and feature-extraction latency.
If the combined view does not improve performance, report that result rather
than redefining success after observation.

### Reproduction and non-destructive checks

Read-only analysis command (choose a new nonexistent output directory):

```powershell
$env:PYTHONIOENCODING = 'utf-8'
.\.venv\Scripts\python.exe scripts/analyze_trace_robustness.py --output artifacts/trace-robustness-analysis/run-003
```

The analysis forbids `LogisticRegression.fit`, `StandardScaler.fit` and
`BehavioralEncoder.fit_transform` while running. It checks actual decomposition,
train-scaler reconstruction, old test-score correspondence and in-memory frozen
parameters, then hashes all previously protected files. The successful run-002
protected **306 pre-existing files with zero changes** during numerical analysis.
The first attempt, run-001, stopped at an exact floating-point equality assertion
when encoding all splits in a different batch layout; it is preserved. Run-002
reproduces the original per-split FP32 embedding/projection batch shapes and
reproduced the saved thresholds exactly. This was an analysis reproducibility
issue, not a modified detector or evidence of a new experimental failure.

Thirty-four non-destructive numerical/verification/integrity tests passed under
explicit fitting guards; see [test log](../artifacts/trace-robustness-analysis/tests-no-fit.log).
The independent [saved-run integration audit](../artifacts/trace-robustness-analysis/tests-frozen-integration.log)
also passed, reproducing the old thresholds and metrics and confirming that all
run files remained unchanged. Both [real dashboard HTTP checks](../artifacts/trace-robustness-analysis/smoke_engineering_http.py.log)
and [synthetic dashboard HTTP checks](../artifacts/trace-robustness-analysis/smoke_http.py.log)
passed. These replay existing records; they do not generate new LLM responses.
The full legacy suite contains fitting tests and was deliberately not invoked
for this no-training milestone. No application code, virtual environment,
target checkpoint or M4 artifact was changed by the analysis.

The [final preservation audit](../artifacts/trace-robustness-analysis/preservation-final.json)
confirmed 303 of 306 protected existing files unchanged. The only three
intentional differences are README, development status and current thesis
context, updated to reference this analysis. There are no unexpected differences;
all existing datasets, artifacts, experimental reports and runtime modules are
unchanged. New files are this report, the reproducible analysis script, its
numerical unit tests and separate analysis/test evidence.
