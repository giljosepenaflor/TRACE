# Token-sequence feature experiment: development protocol

6 November 2026. The user requested continued TRACE improvement. The completed
paired-control candidate failed its fixed development gate (Full validation
F1 0.480 versus 0.529412 standard Full); no test scoring followed. This second
bounded experiment examines information discarded by whole-sequence averages.
It is a hypothesis, not an established explanation or a promised improvement.

This is an explicit methodological extension: keep the thesis's three primary
generation features and add 18 fixed sequence summaries from the **same saved
full-vocabulary-derived token confidence and entropy arrays**. Do not change
the three-feature reference, model, labels or original results. No token IDs,
source IDs, dataset labels, condition IDs, cue text, review judgments or pair
features enter inference. No new LLM call, encoder training or annotation.

The fixed 21 features, in order, are original mean entropy, mean confidence,
confidence variance; entropy population standard deviation; entropy quantiles
0.1/0.5/0.9; confidence quantiles 0.1/0.5/0.9; mean negative log generated-token
confidence (clip at 1e-12); first/last 16-token mean entropy; first/last
16-token mean confidence; mean absolute adjacent difference for entropy and
confidence; least-squares entropy and confidence slopes over normalized time
[-0.5,0.5]; fraction of confidence >=0.9; longest consecutive confidence>=0.9
run divided by sequence length. Quantiles use linear interpolation. For a
single token, slopes and adjacent differences are zero. No response-length
feature is supplied. All features must be finite and align to generated IDs.

Compare five fixed C=1 unweighted logistic arms: original Generation3,
Behavioral384, Full387, sequence Generation21 and sequence Full405. Use original
five isolated source/duplicate folds and two leave-one-training-dataset-out
folds from the paired audit, then final training fits: **40 classifiers**.
Generation features are standardized on fit data only; raw Behavioral
coordinates retain the existing normalization. Logistic max_iter=1000,
random_state=42. No regularization, model or feature search.

Select primary Full (original or sequence) by worst held-dataset AP, mean
held-dataset AP, then mean five-fold AP; original first on exact ties. Freeze
selection before validation scoring. Fold cutoffs use fitting scores only;
final max-F1/FPR<=0.10/recall>=0.80 cutoffs use validation only, with the existing
exact tie rules. Record the resubstitution and small-validation limitations.

Report all arms' F1, recall, precision, FPR, AP, AUROC, confusions, source
slices and paired source intervals (2,000 resamples, seed 20261106). Validation
has 11 positives and has been used in earlier development; any gain is an
**initial development result**, not independent confirmation. A transfer
diagnostic based on only two training datasets does not establish universal
or other-LLM generalization.

The continuation gate requires sequence Full to be selected, have higher
worst-dataset AP than original Full, and have validation F1 strictly above
original Full, Behavioral384, original Generation3 and sequence Generation21.
If the gate fails, do not compute new StrongREJECT predictions. If it passes,
the frozen states and cutoffs may enter a separate hash-bound, once-only
initial retrospective comparison using unchanged admitted StrongREJECT data.
No model/cutoff choice may use those test labels. The reused data is not fresh
confirmation, and dashboard promotion is not automatic.

Before freezing, test exact feature formulas/order, short sequences, invalid
provenance/alignment, information restrictions, train-only scaling, source
isolation, selection and overwrite protection. Independently reconstruct
features from sealed arrays, classifier scores from coefficients, fitting
memberships, cutoff selection, confusions, AP/pairwise AUROC and bootstrap.
Preserve all prior frozen code and artifacts. No target-model runtime or
whole-detector speed improvement is claimed from this development experiment.
