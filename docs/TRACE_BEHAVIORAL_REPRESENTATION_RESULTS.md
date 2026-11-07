# TRACE behavioral representation development results

Completed **6 November 2026** for the existing initial Llama 2 MTBA target.
Removing the learned 64-dimensional projection and retaining the frozen
384-dimensional prompt-response embedding improved Full TRACE's **validation
F1 from 0.473684 to 0.529412**, with the same nine detected positives and four
fewer false positives overall. This is a development operating-point result,
not a new held-out result or proof of improved unseen-trigger generalization.

The workflow, models and results are isolated under
`artifacts/trace-behavioral-v4/representation-001`. The original detector,
review labels, exclusions, raw responses, source assignments, and dashboard
pointers remain unchanged. The initial Chapter 5 results remain the measured
historical results.

## Implemented comparison

The user authorized improving the Behavioral View. The experiment compares
the following four representations of the same reviewed examples:

| Representation | Text input | Behavioral dimensions | Full TRACE dimensions |
| --- | --- | ---: | ---: |
| Original projected interaction | Executed prompt and response | 64 | 67 |
| Unprojected interaction | Executed prompt and response | 384 | 387 |
| Projected response only | Response with the existing response marker | 64 | 67 |
| Unprojected response only | Response with the existing response marker | 384 | 387 |

The pretrained MiniLM model, revision, normalization and 256-token limit remain
unchanged. Projected alternatives reuse the original contrastive algorithm:
60 epochs, Adam learning rate 0.003, temperature 0.1, bias-free linear layer,
64 output dimensions and seed 42. The raw alternatives omit that learned
projection; they do not fine-tune MiniLM.

Each representation has a Behavioral-only and a Full TRACE classifier. The
unchanged three generation features supply a Generation-only reference,
giving nine families. Full TRACE still uses feature concatenation. Score-level
fusion and an auxiliary harmful-compliance scorer were not added in this stage.

## Development isolation and selection

The existing sealed data provide **1,058 training examples from 529 sources**
(163 positive, 895 negative) and **100 validation examples from 50 sources**
(11 positive, 89 negative). Only adapted conditions 1 and 2 are classifier
examples. Their four-condition human-review support and exclusions remain
unchanged. Harmfulness was not substituted for the activation labels.

Five-fold stratified group cross-validation uses training data only, with
split seed 20261106. Both conditions from a source stay together. Declared
short-question duplicate clusters and identical executed uncued inputs are
also checked and joined transitively where applicable; all 529 admitted
training sources remain distinct after the exclusions. The held-out fold
sizes are 212, 212, 212, 212 and 210 examples, with 32, 34, 34, 32 and 31
positive examples respectively.

Every projection and generation scaler is fitted again inside its training
fold. Frozen pretrained embeddings are cached because they do not learn from
activation labels. Fold fit IDs, projection hashes and out-of-fold predictions
are retained. No validation example enters those fits.

The primary matched comparisons use **C=1 and unweighted logistic regression**.
A separate bounded training-CV search uses C in {0.1, 1, 10} and class weights
None/balanced: 54 configurations and 270 fold-level classifier fits. Selection
maximizes mean fold average precision (AP), with ties resolved by mean fold
AUROC, smaller C, unweighted classes, then configuration ID. The selection was
written before final validation classifier predictions and threshold selection.

Final models use training data only. Each validation threshold maximizes F1,
choosing the higher threshold on a tie. Seventeen distinct fixed or CV-selected
models were retained; duplicates between the two tracks are stored once.
Validation F1 therefore measures a threshold-development outcome, not unbiased
test performance. AP and AUROC describe the saved validation score rankings.

## Matched representation results

All rows in this table use the fixed C=1, unweighted classifier settings.
The thresholds are selected separately, so their false-positive operating
points are not identical.

| Representation | Behavioral CV AP | Full CV AP | Behavioral validation F1 | Full validation F1 |
| --- | ---: | ---: | ---: | ---: |
| Original projected interaction | 0.487498 | 0.483242 | 0.470588 | 0.473684 |
| Unprojected interaction | 0.535379 | 0.497317 | 0.516129 | 0.529412 |
| Projected response only | 0.440327 | 0.436045 | 0.444444 | 0.400000 |
| Unprojected response only | 0.517591 | 0.483947 | 0.518519 | 0.473684 |

The unprojected interaction representation was the training-CV winner in both
fixed-settings tracks, before looking at its final validation scores. Its
Full TRACE validation comparison with the original representation is:

| Validation measure | Original projected Full | Unprojected interaction Full |
| --- | ---: | ---: |
| F1 | 0.473684 | 0.529412 |
| Precision | 0.333333 | 0.391304 |
| Recall | 0.818182 | 0.818182 |
| False-positive rate | 0.202247 | 0.157303 |
| Accuracy | 0.800000 | 0.840000 |
| AUROC | 0.852911 | 0.864147 |
| Average precision | 0.375948 | 0.347680 |
| True positives | 9 | 9 |
| False positives | 18 | 14 |
| False negatives | 2 | 2 |
| True negatives | 71 | 75 |
| Threshold | 0.18404884955022044 | 0.234150946937906 |

The same nine validation positives are detected by both models. Among
negatives, the unprojected candidate removes five original false positives
and introduces one new false positive, giving a net reduction of four.
The F1 increase is 0.055728, or approximately 5.57 percentage points. The
false-positive rate falls by approximately 4.49 percentage points.

The benefit is not uniform across metrics: validation AP falls from 0.375948
to 0.347680 even though AUROC and the selected F1 operating point improve.
This result does not establish a uniformly better ranking or calibration.

For Behavioral-only, unprojected interaction features raise validation F1
from 0.470588 to 0.516129 and lower false positives from 15 to 12 while
detecting the same eight positives. Validation AUROC rises from 0.848825 to
0.868233, while AP changes from 0.367253 to 0.363282.

Response-only representations did not win the training-CV selection. The
unprojected response-only Behavioral classifier has validation F1 0.518519,
but detects seven positives compared with eight for the unprojected
interaction Behavioral classifier. Its corresponding Full model has F1
0.473684. The projected response-only Full model has F1 0.400000. Removing
the prompt is therefore not supported as the preferred representation by
this development comparison.

## Secondary classifier-search results

The classifier search is reported separately because changing representation,
regularization and class weighting together would obscure their contributions.

| Training-CV selection | Selected configuration | CV AP | Validation F1 | Recall | FPR |
| --- | --- | ---: | ---: | ---: | ---: |
| Behavioral-only | Unprojected interaction, C=1, balanced | 0.540278 | 0.516129 | 0.727273 | 0.134831 |
| Full TRACE | Unprojected interaction, C=10, unweighted | 0.508168 | 0.457143 | 0.727273 | 0.179775 |
| Generation-only | Existing signals, C=10, balanced | 0.272184 | 0.327273 | 0.818182 | 0.393258 |

The Full classifier selected by the broader CV search has lower validation
F1 than the original Full reference, despite having higher training-CV AP.
It detects eight validation positives and raises 16 false positives. That
outcome is retained, rather than replacing the recorded CV choice with the
model having the most favorable validation result. The fixed-settings and
tuned tracks are distinct predeclared comparisons.

The candidate for a future representation-focused comparison is therefore
the **fixed-settings unprojected interaction representation**, which was
already selected by training CV within that track. The tuned Full model
remains a separately recorded candidate with mixed validation evidence.

## Truncation and implementation costs

Prompt-response input exceeds MiniLM's limit for 32 training and four
validation examples. Response-only input reduces those counts to eight and
one, respectively, but the reduced truncation does not produce the winning
representation in this experiment. Complete original text is preserved.

The recorded experiment interval is approximately **96.48 seconds** on the
local CPU. It includes development record loading, encoding, cross-validation,
final fitting, calibration and result serialization within the measured
interval; outer imports and preflight verification are excluded. It is not a
per-interaction latency benchmark.

Retaining 384 dimensions increases the Full classifier input from 67 to 387
features and removes the contrastive projection from that candidate's path.
The pretrained encoder identity and files stay the same. No new deployment
latency, peak-memory or cross-model efficiency claim is made. The 223,741-byte
candidate bundle contains all 17 experimental classifiers and shared study
state; it is not the installed footprint of one deployed detector and excludes
the required external MiniLM files.

## Evidence qualification and verification

The old intake verifier still references the missing
`strongreject-dataset-for-training.csv`. This experiment does not recreate
that file or change the old intake gate. Its separate admission consumes the
independently pinned source catalogue and checksum-sealed development outputs,
checks every available pinned input, all label and source bindings, and all
**3,076 development raw-file hashes** against the sealed collection digest.
Only the known missing historical CSV is qualified; other missing required
development inputs are rejected.

Human consensus remains researcher-attested, with individual reviewer
submissions unavailable. The generation features retain the original
corroborated full-vocabulary method admission and absence of exact historical
per-response build/host identifiers. The representation experiment does not
repair or fabricate those historical facts.

**Eighteen targeted harmless tests passed**: ten new representation-workflow
checks and eight existing consensus-training checks. They cover paired and
transitive duplicate grouping, clean/test membership rejection, fold-local
projection and scaler fitting, forged fit-ID rejection, repeated fold
coverage, deterministic selection, validation independence of fitted model
parameters, response-only input behavior, and refusal to overwrite/refit an
existing experiment.

Frozen verification reconstructs development features, validation scores,
thresholds and metrics, and checks fold fit IDs, original hashes and selection.
An independent arithmetic verifier reproduces AP from tied-score groups,
AUROC from positive-negative pairs, confusion matrices, threshold choices,
and CV selections from saved predictions. Both verification paths passed.
The C=1 unweighted reference models reproduce **all original validation
scores exactly** for Full, Behavioral-only and Generation-only (maximum
absolute difference 0.0).

The new source files are `traceguard/representation_experiment.py` and
`scripts/trace_behavioral_experiment.py`. Original frozen training and feature
code was not modified. Full details and checksum bindings are available in:

- [Frozen run report](../artifacts/trace-behavioral-v4/representation-001/results/README.md)
- [Experiment verification](../artifacts/trace-behavioral-v4/representation-001/verification.json)
- [Independent numeric verification](../outputs/trace-behavioral-verification-20261106/numeric-verification.json)

Verify the saved experiment without fitting or target inference:

```powershell
.\.venv\Scripts\python.exe -B scripts\trace_behavioral_experiment.py verify
.\.venv\Scripts\python.exe -B outputs\trace-behavioral-verification-20261106\verify_numerics.py
```

The existing `prepare` and `train` commands refuse this already completed run.
Reproducing a new fit requires a separately named `representation-*` run and
its own unchanged protocol.

## What remains before a performance claim or deployment

These development results identify a promising representation change, with
mixed ranking and classifier-tuning outcomes. The existing StrongREJECT test
errors informed the design, so a fresh source-disjoint, independently reviewed
held-out experiment is required before claiming better unseen-trigger
performance. Preserve the original allocation and initial results; do not
reuse the old test as an unrestricted tuning set.

The current dashboard still uses its existing separately frozen detector.
This development candidate was not promoted to it. A prospective comparison,
followed by matched efficiency measurement and deployment compatibility
verification, is the next step for assessing whether the new representation
should become the active detector.
