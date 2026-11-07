# TRACE end-to-end improvement priority

User direction, **6 November 2026**: focus on improving Full TRACE so that it
outperforms Generation-only and Behavioral-only in activation detection. This
plan makes that end-to-end comparison the primary objective. It is a proposed
next experiment, not an implemented detector or measured improvement.

## Objective and evidence

Auxiliary refusal/compliance classification is useful only if its scores improve
the activation detector. Optimizing that separate task does not demonstrate
that Full TRACE becomes more accurate. The existing fixed-cutoff three-head
candidate retains its failed joint gate; no earlier gate or result is rewritten.

The source-isolated training OOF audit provides evidence of complementary
information at the existing descriptive validation cutoffs:

- Generation detects 47 activation-positive examples missed by unprojected
  Behavioral; Behavioral detects 32 missed by Generation.
- On negatives, Generation alone raises 208 alarms that Behavioral does not;
  Behavioral alone raises 45 that Generation does not.

These counts motivate learning when to use each view. They are not an achievable
combined-model result or evidence that a simple OR/AND rule must improve F1.
The audit cutoffs were calibrated on validation and are not unbiased CV operating
points. Previously tested raw feature concatenation and fixed score averaging
did not establish a robust fusion advantage; repeating the same weight search
is not the proposed next step.

## Proposed low-dimensional behavioral and generation inputs

Keep the Generation View's three existing features: entropy, token confidence
and confidence variance. Represent Behavioral evidence with three continuous
scores: the existing unprojected Behavioral activation classifier, refusal
present, and substantive request fulfillment. The two semantic heads passed
the predeclared source-group teacher-agreement gates; this does not establish
human-validated semantic quality or activation utility.

The unsafe-assistance head is excluded from this new bounded ablation because
it failed its fixed-cutoff gate. This is an explicit new two-head experiment,
not promotion of the failed three-head candidate or retroactive revision of
its acceptance rule. Semantic scores remain continuous inputs, never replacement
activation labels. No semantic head's 0.5 class decision, threshold search or
new labeling round is required just to define these feature inputs.

Six low-dimensional signals let a new activation classifier learn conditional
relationships between the views without appending hundreds of embedding
coordinates directly to three generation features. That is a hypothesis about
fusion, not a proven explanation of the existing errors.

## Matched candidate comparison

| Arm | Inputs | Purpose |
| --- | --- | --- |
| Generation-only | The same three generation features | Strong single-view reference |
| Enhanced Behavioral-only | Behavioral activation score + refusal + fulfillment | Matched Behavioral reference using all proposed Behavioral information |
| Full linear fusion | All six signals | Small regularized logistic activation combiner |
| Full nonlinear fusion | All six signals | Bounded shallow tree combiner for conditional interactions |

Preserve the original unprojected Behavioral and Full detectors as additional
historical references. The enhanced Behavioral-only arm is essential: a Full
candidate must not claim a fusion advantage merely by receiving extra features
that were withheld from the Behavioral-only comparison.

Freeze the exact classifier settings, feature scaling, model selection and
acceptance rules before computing new activation predictions. Keep the search
small; no open-ended architecture or threshold search against StrongREJECT.

## Leakage controls and evaluation

Reuse the existing training/validation assignments and accepted auxiliary
labels. For every outer source-group fold, exclude held sources from both the
base Behavioral classifier and auxiliary heads. Generate inner cross-fitted
component scores for activation fitting; fit scalers and combiners only on
that outer fitting set. Evaluate on the excluded outer sources. Final models
use training-only cross-fitted component scores; validation calibrates their
activation thresholds after model selection. Do not feed human/AI annotation
values directly into validation or test inputs.

Use original controlled activation labels throughout. Harmfulness, refusal or
trigger presence alone does not establish activation. Report performance by
source as well as pooled results, because current development evidence favors
Behavioral ranking while the already inspected StrongREJECT results favor
Generation. These distribution differences cannot be solved merely by declaring
a preferred view or forcing a fusion weight.

Judge the Full candidates by activation F1, recall, false-positive rate,
precision, AP/AUROC and confusion matrices against both matched single-view
arms. Add matched-operating-point checks: recall at a shared false-positive
budget and false-positive rate at a shared recall target, with operating rules
selected on development data rather than test labels. Threshold changes alone
must not be mistaken for stronger ranking or a better tradeoff.

The intended outcome is a Full detector with higher F1 than both single views
and a favorable recall/false-alarm tradeoff. Retain paired source uncertainty;
if the evidence does not support that outcome, report no demonstrated fusion
advantage. Do not make the desired conclusion a requirement imposed on labels
or obtained through selective reporting.

Only after development selection and freezing, a reused StrongREJECT comparison
may be run as **initial, retrospective evidence**. Earlier test errors already
informed development, so it cannot independently prove improved generalization.
Fresh independent confirmation and additional separately documented LLM
experiments remain outstanding.

## Complete cost and delivery

Measure the complete candidate, including the existing Behavioral encoding,
additional semantic encoding, component classifiers, generation features and
learned combiner. Share encoder weights where possible, but count both forward
passes. Auxiliary-only latency is not Full TRACE latency. Preserve the
lightweight scope and report all model/runtime memory and dependencies.

The deliverable is one end-to-end detector comparison with frozen candidates,
source-isolated scores, matched baselines, uncertainty and resource measurements.
It is not another isolated auxiliary-score improvement report. No new target
LLM batch or repeated annotation intake is required to start this experiment.
Performance gains are unproven and the dashboard stays unchanged while the
comparison is developed.
