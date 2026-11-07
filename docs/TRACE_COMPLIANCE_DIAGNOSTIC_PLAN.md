# Frozen TRACE compliance-specificity diagnostic

6 November 2026. The user supplied feedback prioritizing the question of
whether TRACE recognizes activation-specific evidence or harmful task success.
This authorizes a separate diagnostic using existing reviewed interactions.
It does not authorize changing labels, fitting a detector or reopening model
selection. Original held-out results remain the primary recorded experiment.

## Fixed question and methods

Use the original projected Full, original projected Behavioral-only and
Generation-only C=1 detectors and their existing validation-frozen cutoffs.
Load their sealed states from `reused-test-comparison-001`, which reproduces
the original `dual-view-candidate-001`. No improved candidate is selected by
this diagnostic. The frozen MiniLM encoder and original projections/scalers
are reused. No target LLM request, classifier fit or threshold selection.

Use the sealed development and corrected StrongREJECT consensus record indexes,
reviews and source dispositions. Preserve all original train/validation/seen/
unseen assignments. Include all four conditions of source groups with verified
inputs, four definite binary task-success judgments and nonempty reasons.
Exclude source-input defects, groups containing U and missing reasons; retain
every exclusion in an inventory. Unknown or excluded adapted activation status
must never silently become zero.

## Cohorts and interpretation

Report these predefined overlapping diagnostic cohorts separately per split:

1. Adapter-disabled controls with reviewed task success=1.
2. Adapter-disabled controls with reviewed task success=0.
3. Adapted, uncued responses with reviewed task success=1.
4. Adapted, cued responses with task success=1 from nonadmitted patterns.
5. Original admitted activation-positive adapted responses.
6. Original admitted activation-negative adapted responses.

Task success is the existing review variable, not a newly annotated semantic
refusal or harmfulness variable. In particular task success=0 does not imply
an explicit refusal. Check whether the admitted activation labels coincide
with task-success labels by construction; that is a label-policy audit, not
a fitted baseline or independent predictive result.

Conditions 3/4 have the designated adapter disabled. High scores on their
successful responses are false alarms **relative to the adapter-specific
operational control**, subject to checkpoint/provenance qualifications. They
do not establish that the base checkpoint contains no other possible backdoor.
Conditions 1/2 in nonadmitted patterns remain activation-ambiguous; report flag
rates and score distributions, never a false-positive rate/F1/AUROC based on
invented negative labels. Even low scores cannot prove causal attribution.

Training-source observations are descriptive and may share fitted questions.
Mark both source-ID membership in fitting and normalized executed-question
overlap with fitted sources (NFKC, casefold, whitespace collapse). Provide
additional summaries restricted to questions absent from fitting. These are
existing data and a post-hoc analysis, not a fresh independent experiment.
Clean conditions are an additional model condition outside the adapted-only
training distribution; high flag rates may also reflect that domain change.

## Frozen computation and uncertainty

Reuse cached raw 384-dimensional embeddings for admitted adapted interactions.
Encode only remaining controls/nonadmitted interactions, in deterministic
split/source/condition order with the same frozen prompt-response encoder.
Compute the same three generation statistics under the previously qualified
full-vocabulary signal admission; preserve historical per-response build and
individual-review qualifications. Reproduce archived validation and test
scores within 1e-5, then retain their exact saved scores/decisions in the
diagnostic rather than rewriting primary predictions. Report reconstruction
errors and any numerical-boundary differences explicitly.

For every cohort provide interaction/source counts, flag fractions, median and
10th/90th percentile scores, generated-token lengths and exploratory 95%
whole-source bootstrap intervals (2,000 shared resamples per cohort, seed
20261106). Preserve both observations when the cohort has two per source.
Small or all-equal cohorts can yield degenerate percentile intervals; these
are not guarantees of population certainty. Also report fixed length strata
1-16, 17-64, 65-128 and 129-256 tokens, descriptively without fitting or
matching/reweighting on outcomes. No inferential claim follows from individual
short or sparse strata.

The report must distinguish evidence consistent with a task-success shortcut
from proof that TRACE is solely a compliance classifier. No acceptance gate,
model promotion or architecture search follows from this diagnostic. Address
the supplied feedback's older statements using the existing experiment record:
the cue-removal result concerns unprojected Full; raw/projected comparisons,
regularization search and several fusion experiments already ran. Do not
repeat them. Identify F1/recall correctly from saved predictions and do not
infer projection learning quality from final contrastive loss alone.

## Deliverables and verification

Freeze protocol, code hashes, selected/excluded metadata, original states,
encoder/input hashes and cutoffs before encoding. Save diagnostic scores,
feature caches, original-reference reproduction errors, summaries and manifest.
Verify joins, source/condition roles, untouched labels, feature alignment,
threshold calls, bootstraps and input/output preservation. An independent
coefficient/metric arithmetic check must run without fitting or re-encoding.
Deliver a measured report and a defensible paragraph for Chapter 5. No revised
claim of universal or causal backdoor detection is allowed from this audit.
