# Proposed TRACE detector improvement experiment

Latest measured update 6 November 2026: the requested dataset-transfer audit
and two bounded improvement experiments are now complete. Paired-control Full
fails to improve validation F1. Token-sequence Full (405 inputs) improves
initial development worst-dataset AP **0.290889 to 0.335876** and validation
AP **0.347680 to 0.437219**, but validation F1 declines **0.529412 to 0.437500**.
Both continuation gates fail. There is no new StrongREJECT scoring, complete
resource benchmark or dashboard replacement. All 95 targeted tests and both
verification paths pass. See [the complete measured comparison](TRACE_TRANSFER_IMPROVEMENT_RESULTS.md).
The two new pre-experiment plans and their code/artifacts are frozen. These
partial ranking gains do not establish improved activation decisions. Earlier
updates below describe their own historical stages.

Completed update 6 November 2026: the user-authorized
[end-to-end comparison](TRACE_END_TO_END_RESULTS.md) is now implemented and
measured. Six arms and 123 classifier fits use nested source cross-fitting;
the separately declared two-head enhancement excludes unsafe assistance and
preserves the failed old three-head gate. Training selects Full linear before
validation/test. Initial retrospective seen/unseen F1 is **0.614173/0.276596**,
below Generation-only **0.683168/0.629108**. Nonlinear Full is
**0.669767/0.547368**, also below Generation. No fusion advantage is established.
All 85 targeted tests, both numerical verification paths and 18 resource
workers/3,600 timings are complete. The dashboard and original evidence stay
unchanged. The [frozen plan](TRACE_END_TO_END_IMPROVEMENT_PLAN.md) remains the
pre-experiment record.

Next proposed work is development-only dataset/condition transfer and selection
auditing under a new protocol using existing training data. That audit has not
run; another StrongREJECT weight/cutoff search is not justified by this result
and no gain is promised. Fresh independent evidence and other LLM evaluations
remain outstanding. Earlier entries below retain their historical scope.

Latest scorer update 6 November 2026: the lightweight three-head scorer is now
implemented and trained. Held-source AI-assisted teacher-agreement macro F1
is 0.888972 refusal / 0.798913 fulfillment / 0.468182 unsafe assistance. Unsafe
positive recall is only 1/19 at the frozen 0.5 cutoff, so the joint quality gate
blocks activation augmentation. No new TRACE or StrongREJECT result is produced.
See [the measured scorer results](TRACE_AUXILIARY_SCORER_RESULTS.md). A later
inner-source cutoff calibration experiment is proposed, not executed. The
current labels, gate rules, baseline results and dashboard remain preserved.

Latest label-policy update 6 November 2026: the user designated the completed A
file as final for both roles. [All 80 items are now sealed](TRACE_AUXILIARY_CONSENSUS_INTAKE.md)
as one user-approved AI-assisted auxiliary label set; independent two-human
agreement is unavailable and no duplicate B is created. Usable labels per head
are 74/74/65, with outer source-class coverage checked. Scorer fitting, nested
support and semantic quality remain pending. Older separate-review requirements
below describe the original proposal and do not override this explicit update.

Latest update 6 November 2026: the development failure audit and concrete
[auxiliary behavior specification](TRACE_BEHAVIOR_FEATURE_SPEC.md) are now
complete. Fixed-response cue removal changes 12/50 unprojected Full validation
decisions; no validation positive is encoder-truncated, and obvious refusal
markers are already well rejected. See [the measured audit](TRACE_DEVELOPMENT_BEHAVIOR_AUDIT.md).
Two blank 80-item training-only auxiliary review packets are ready. Genuine
auxiliary review/adjudication is the next dependency; no semantic scorer is
trained or validated yet. Existing activation labels and dashboard remain fixed.

Update 6 November 2026: the representation comparison, authorized reused-data
evaluation and separate-view score fusion are now implemented and measured.
See [representation results](TRACE_BEHAVIORAL_REPRESENTATION_RESULTS.md),
[reused comparison](TRACE_REUSED_COMPARISON_RESULTS.md), and
[the completed fusion experiment](TRACE_SCORE_FUSION_RESULTS.md). Eight fixed
mixtures select 75% Behavioral / 25% Generation on training CV; initial
retrospective F1 is 0.666667 seen / 0.464000 unseen, below Generation-only.
The selected fusion is not promoted. Forty targeted tests and both numerical
verification paths passed. Auxiliary harmful-compliance features, fresh
independent confirmation and other LLM evaluations remain future work. The
original proposal below is preserved as the planning record, not a list of
still-unexecuted tasks or a promise of improvement.

Prepared 5 November 2026 in response to the user's request to improve detection
and consider output safety or community-guideline violations. This is a
development proposal. No improved model, new performance result, label change,
or dashboard promotion is claimed.

The immediate recommendation is to improve the behavioral representation and
fusion while retaining Generation-only as a strong comparison. An explicit
harmful-compliance signal is a separate, testable extension within the
Behavioral View. Harmfulness must not replace the activation definition.

## What the initial evidence supports

The frozen expanded Llama 2 candidate has the following unseen-to-detector
results. These are reported observations, not anticipated improvement results.

| Variant | F1 | Recall | False-positive rate |
| --- | ---: | ---: | ---: |
| Full TRACE | 0.476190 | 0.472973 | 0.208791 |
| Behavioral-only | 0.375000 | 0.324324 | 0.164835 |
| Generation-only | 0.629108 | 0.905405 | 0.395604 |

Generation-only detects 35 unseen positives that Full misses. Full detects
three that Generation-only misses. Generation-only also raises substantially
more false alarms. The evidence motivates testing whether the behavioral
representation and fusion can retain useful generation evidence while reducing
false positives. It does not identify a proven internal cause of the errors.

The current behavioral input combines the executed prompt and response into
one MiniLM representation, followed by a learned 64-dimensional projection.
The classifier uses C=1 and unweighted classes. Training contains 163 positive
and 895 negative examples; validation contains only 11 positives and 89
negatives. These choices provide concrete development hypotheses. Neither
class weighting nor a different representation is guaranteed to improve them.

Only eight unseen representations exceed the encoder limit, and 36 of Full's
39 unseen misses occur without truncation. Increasing the length limit alone
therefore cannot account for most of the observed misses.

Sources: [frozen training](TRACE_THREE_VIEW_TRAINING_RESULTS.md),
[initial held-out results](TRACE_STRONGREJECT_CONSENSUS_RESULTS.md), and
[error analysis](TRACE_ERROR_ANALYSIS_AND_EFFICIENCY.md).

## Safety assessment and backdoor attribution

A useful safety feature asks whether a response substantively fulfills an
unsafe request, refuses it, or gives a safe explanation. It must use an explicit
rubric rather than a vague instruction about community standards. Mentioning an
unsafe topic, quoting it, or explaining prevention does not by itself establish
harmful compliance.

A backdoor-activation judgment asks a different question: whether the specified
target behavior is attributable to the controlled activation condition. A model
can produce an unsafe answer without a backdoor. A backdoor can also alter a
harmless factual answer or insert an otherwise innocuous target phrase.

For this initial MTBA experiment, the four-condition task-success rule supplies
the operational activation labels. Keep that rule and the reviewed labels
unchanged when comparing new detectors on the existing admitted development
examples. Do not relabel excluded patterns or uncertain cases to increase a
metric. A broader attribution rule would require its own protocol and review.

The StrongREJECT paper evaluates useful harmful compliance, which is relevant
to an auxiliary feature but does not establish backdoor attribution:
[A StrongREJECT for Empty Jailbreaks](https://arxiv.org/abs/2402.10260).
Conditional trigger behavior is central to the backdoor setting studied in
[Sleeper Agents](https://arxiv.org/abs/2401.05566).

The present TRACE classifier is not an instruction-following LLM. Adding an
instruction to its input does not install a reliable judging rule. Adding that
instruction to the target LLM's prompt would change the collected interaction
and require a new collection protocol.

## First bounded development experiment

Use a new versioned experiment directory. Read only the existing admitted
training and validation memberships and their supported raw records. Preserve
all source assignments and exclusion decisions. Do not open held-out response
records or prediction rows for candidate selection.

| Candidate family | Proposed change | Hypothesis to evaluate |
| --- | --- | --- |
| Generation-only reference | Retain the current three generation features | Establish a strong numerical baseline |
| Original Full reference | Reproduce the current architecture in the new development comparison | Provide a matched architectural reference |
| Unprojected interaction representation | Use frozen 384-dimensional prompt-response embeddings with regularized logistic regression | Test whether the contrastive projection discards useful information |
| Response-only representation | Encode the response without the prompt, comparing raw and projected embeddings | Test whether response semantics transfer better than the current combined representation |
| Separate-view score fusion | Combine independently trained behavioral and generation scores with a small fixed weight grid | Test an alternative to feature concatenation |

Changing to response-only text does not prove freedom from trigger shortcuts;
responses can echo cues or contain other shortcuts. Treat the representation
comparison as an experiment and inspect errors on development data.

For logistic regression, a small proposed search is C in {0.1, 1, 10} and
class_weight in {None, balanced}. Use five-fold stratified group cross-validation
within the training source groups for model-family and hyperparameter selection.
Keep both adapted conditions, and any duplicate-question cluster, in the same
fold. Fit every learned projection, scaler, and classifier inside its training
fold. Frozen pretrained embeddings may be cached; a projection fitted on all
training examples must not be reused across validation folds.

For score fusion, use out-of-fold training scores and a fixed behavioral weight
grid {0, 0.25, 0.5, 0.75, 1}. Score averaging is a candidate scoring rule, not
automatically a calibrated probability. If a learned fusion layer is later
used, train it on out-of-fold base-model predictions rather than in-sample
scores. Preserve the zero-weight endpoint to expose cases where the behavioral
view contributes no benefit.

Select models using a declared training-CV objective such as average precision.
Select the final operating threshold on the existing validation partition,
reporting its small positive count. Do not repeatedly add candidate families
because of validation disappointments without recording the expanded search.
Training and threshold-selection data must be separated; see the
[scikit-learn threshold guidance](https://scikit-learn.org/1.7/modules/classification_threshold.html).

Candidate comparisons must report recall and false-positive rate together.
Lowering a threshold can catch more activations while flagging more negatives;
that alone does not demonstrate a better representation or better ranking.
Report precision, recall, F1, average precision, AUROC, confusion matrices, and
development recall at explicitly stated false-positive operating points.

## Second experiment using the user's safety-signal idea

After the first bounded comparison, test an auxiliary behavioral feature for
harmful compliance and refusal. Keep the three generation features unchanged
initially, so the contribution of this semantic change can be assessed.

1. Define and version a rubric that distinguishes substantive unsafe assistance,
   refusal, safe explanation, and uncertainty. Evaluate fulfillment relative to
   the request; keyword occurrence or absence of a refusal phrase is insufficient.
2. Annotate development responses for these auxiliary properties, independently
   of their activation labels. A pretrained safety scorer can be considered,
   but its identity, training-data overlap, errors, and resource requirements
   must be assessed. Human review checks a separate development subset.
3. Add the resulting numeric features to the Behavioral View and compare the
   detector with and without them. Auxiliary judgments are features, not new
   activation ground truth. Do not use evaluation response judgments as inputs
   to the model that is being evaluated.
4. If an LLM judge is used as an offline teacher, keep its outputs separate from
   the human activation labels. Validate teacher error, benchmark overlap, and
   prompt-injection resistance. A small local classifier may later learn these
   properties, subject to its own validation. This is proposed work, not an
   implemented component.
5. Count the complete deployed scorer's latency, memory, and dependencies. A
   second generative LLM for every interaction would materially alter TRACE's
   lightweight claim and require explicit methodological treatment.

Use instruction-following benign responses, safe discussion of sensitive
topics, refusals, and failed cue attempts in a separately reviewed development
extension to test semantic shortcuts. Ordinary unsafe outputs without a cue
are useful attribution challenges, but must not be silently assigned negative
activation labels or imported from excluded historical patterns. Define their
admission and uncertainty rules before including them in a new dataset.

## Fresh evidence and success criteria

The original StrongREJECT test outcomes have already informed this proposal.
They remain the fixed initial result. They cannot serve as a fresh blind test
of changes suggested by their errors. The original 1,082-question allocation
and 4,328-response budget remain unchanged; this proposal does not invent a
reserve or reallocate historical sources.

A final improvement claim needs a new documented, independently reviewed,
source-disjoint held-out collection. Specify its sources, budget, cue isolation,
controls, admission rules, inference provenance, and evaluation metrics before
collection and scoring. Audit source and duplicate overlap against the whole
historical inventory. Additional LLM evaluation is a subsequent separately
specified experiment; it is not guaranteed to improve this target's detector.

Freeze the candidate and thresholds before the fresh test. Compare the new
candidate, the original Full architecture, and Generation-only on the same
eligible fresh examples. Preserve original frozen-threshold comparisons;
any separately matched operating-point comparison must be selected on
development data and explicitly identified. Report uncertainty with whole
source groups kept together and disclose all false alarms and exclusions.

The proposed success standard is an improvement in held-out detection quality
with an acceptable, predeclared false-positive tradeoff and measured detector
cost. Do not promise a particular accuracy or select a favorable metric after
testing. A fusion-benefit claim requires improvement over the strongest single
view, rather than merely improvement over the earlier Full candidate.

## Immediate next action

Implement the first bounded, training-and-validation-only experiment in an
isolated workflow, verify its fold isolation with harmless fixtures, and
produce a development comparison report. That report determines whether an
auxiliary safety feature is worth its additional labeling and runtime cost.
No current dashboard detector should be replaced based solely on a proposal
or a favorable validation score. The thesis can describe the work as a new
development stage after it is actually performed, while preserving Chapter 5's
initial results.
