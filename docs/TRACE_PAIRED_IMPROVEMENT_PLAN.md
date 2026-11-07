# Paired-control TRACE development experiment

User authorization: continue improving TRACE, 6 November 2026. This is a new
bounded development experiment; no accuracy gain is assumed. Earlier models,
labels, protocols and the dashboard remain preserved.

## Hypothesis and scope

Existing admitted training data has 529 source pairs: 163 with differing
reviewed activation labels and 366 with two negatives. The existing classifier
fits interactions individually. A paired-control objective can additionally
encourage activation-positive interactions to outrank their matched negatives
and discourage score changes between reviewed negative controls. This may
reduce question/dataset shortcuts. It is a hypothesis, not a causal finding.

Use the frozen raw 384-dimensional prompt-response representation and original
three generation features. No new encoder, semantic head, generation feature,
LLM call, human/AI review or label is introduced. Training pairs use only the
original controlled activation labels; condition or cue presence is never an
activation label. Do not supply source, condition, cue identity or review
judgments as inference features. Single-interaction inference is a scaled
linear score followed by a sigmoid; it requires no paired target generation.

## Fixed six-arm comparison

Preserve C=1 unweighted standard logistic Generation-only, raw Behavioral-only
and raw Full as references. Add paired-control counterparts for **all three
views**, so Full is compared with equally trained single views. Behavioral
coordinates retain their normalization; generation statistics are standardized
on each fitting partition only. Full concatenates the same features.

Let z_i = w'x_i + b, with positive-minus-negative feature difference d_p for
each discordant reviewed source pair, and feature difference d_q for each
equal-label pair. The paired objective is:

    mean_i(log(1+exp(z_i)) - y_i*z_i) + ||w||^2/(2*n)
    + 0.25 * mean_p(log(1+exp(-w'd_p)))
    + 0.25 * mean_q((w'd_q)^2)

The intercept is not regularized and cancels from paired terms. Equal-label
pairs use a squared score consistency penalty; reversing their orientation
does not alter that term. Discordant orientation uses labels, not condition.
Pair terms are omitted when their group is absent; fitting still requires
both activation classes. C=1 sets the original ridge scale; no tuning grid.
Use deterministic float64 L-BFGS-B with analytic gradients, 2,000 iterations,
gtol=1e-7 and ftol=1e-12. Require convergence and record optimality diagnostics.
Reference logistic settings remain max_iter=1000/random_state=42.

## Isolation, diagnostics and selection

Reuse the original five source/duplicate-group folds. Every pair must be
wholly inside its fit/held partition; holdout sources and duplicate-group
members never enter the fitting loss or scaler. Also fit each arm on AdvBench
and evaluate HarmBench, then reverse, using **training partition only**. Remove
all groups represented in the held dataset from the fit side. This is a
leave-one-dataset-out development diagnostic with two datasets, not a new test.

First retain baseline source/condition slices and pair-ranking/negative-score
consistency diagnostics. All six arms have five source-fold fits, two
cross-dataset fits and one final fit: 48 classifiers. No inner learned
components are present. Source-fold operating cutoffs use that fold's fit
predictions; cross-dataset cutoffs use fit data only. Report the associated
resubstitution calibration limitation. Scalers use fit data only throughout.

Select the primary Full arm (standard or paired) by worst held-dataset AP,
then mean held-dataset AP, then mean five-fold AP, with standard first on exact
ties. Select before validation outcomes are scored. Report all six arms;
do not switch primary after validation or test. All selection data is existing
development data, already used in prior experiments, so it is not independent
confirmation.

After selection, validation calibrates three policies: max-F1, recall at
FPR<=0.10, and highest threshold with recall>=0.80. Use the existing exact
threshold helper and tie rules. Report F1, precision, recall, FPR, confusions,
AP, AUROC, dataset slices and paired source-bootstrap intervals (2,000 shared
resamples, seed 20261106).

The gate for another initial retrospective StrongREJECT comparison requires
the selected arm to be paired Full, its worst-dataset AP to exceed standard
Full, and validation max-F1 to strictly exceed standard Full and **all four
matched/historical single-view arms**. No threshold/model choice uses test
labels. If this gate fails, do not read new test outcomes or compute new test
predictions. Gate failure does not erase any development gain or tradeoff;
report the complete comparison. Do not relax the gate after seeing results.

If the gate passes, freeze states/cutoffs first, then score the existing 476
StrongREJECT examples once. Keep exact memberships and exclusions and report
all six arms plus paired whole-source intervals. This remains **initial,
retrospective, single-target** evidence because earlier test errors informed
development. No new other-LLM or independent generalization claim follows.

## Verification and cost

Before freezing, test analytic gradients against numerical differences,
pair-label orientation, equal-pair consistency, group isolation, scaled feature
alignment, deterministic selection, single-record scoring and overwrite
protection. Verify saved states without refitting: reconstruct scores directly
from coefficients, optimality gradients, scaler means, source memberships,
thresholds, metrics, selection and bootstrap. Preserve code/input/output hashes.

If a new candidate passes the development gate, measure complete six-arm
inference cost on the same unlabeled validation records, three fresh workers
per arm, 100 warmup and 200 timed calls each, rotated arm order and shuffled
records. Both raw and paired arms use identical encodings; target generation,
network/UI and loading are excluded from warm timing. Share frozen weights,
report whole-worker RAM and score-boundary differences. If the gate fails,
do not launch an unnecessary full resource benchmark or promote a model.

No automatic dashboard promotion. Additional target LLMs and fresh independent
confirmation remain outstanding; the earlier original-three-dataset source
shortage and historical provenance limitations remain unchanged.
