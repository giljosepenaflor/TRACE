# Expanded Llama 2 TRACE: three-view training and validation calibration

On **2026-11-03**, following the user's explicit instruction to resume training
for all three views, the expanded candidate was fitted and frozen in
`artifacts/trace-training-v3/dual-view-candidate-001`. This supersedes the
earlier training pause. Full TRACE, Behavioral-only and Generation-only were
all trained; the earlier standalone Behavioral-only preparation was not run.

The fit completed at 21:40 Asia/Manila, using CPU and the local pinned MiniLM
snapshot. Feature extraction, projection/classifier fitting, calibration and
diagnostics took approximately **28.33 seconds**, excluding startup, evidence
verification and package imports. No Llama 2 fine-tuning, remote generation,
StrongREJECT response access or dashboard replacement occurred.

## Data and admission

All three models use the identical frozen development memberships:

| Partition | Eligible source groups | Adapted examples | Positive | Negative |
| --- | ---: | ---: | ---: | ---: |
| Training | 529 | 1,058 | 163 | 895 |
| Validation | 50 | 100 | 11 | 89 |

These labels come from `artifacts/trace-training-v3/consensus-001`. Their
four-condition review support, exclusions and source assignments are unchanged.
Clean controls are attribution evidence, not fitted examples. The 190 excluded
source groups remain excluded. The final pre-preprocessing allocation remains
706 development-training, 63 validation, and 156/157 StrongREJECT test questions,
with no reserve, for 4,328 planned four-condition responses overall.

The separate `admission.json` records the user's all-view resumption and accepts
the full-vocabulary generation-score method on the pinned provenance
investigation: matching actual Vast source/manifest, invariant Git signal
calculation, collection-compatible process timing, matching record format and
numeric checks. It explicitly retains the lack of per-response build/host
identifiers and the user-attested nature of two-reviewer consensus.

The older label seal's `fitting_admitted: false` remains an unchanged historical
snapshot. This new run records its own qualified admission rather than editing
that snapshot or adding provenance claims to raw responses. A derived in-memory
feature view declares the investigated statistics source; this is not described
as an originally logged field or independent exact-build verification.

## Fixed implementation

The target remains the pinned `meta-llama/Llama-2-7b-chat-hf` base with the
`BackdoorLLM/Jailbreak_Llama2-7B_MTBA` adapter. TRACE classifier training does
not alter the target LLM or adapter.

Behavioral input is `[PROMPT] <actual executed prompt>\n[RESPONSE] <response>`.
The local `sentence-transformers/all-MiniLM-L6-v2` snapshot is pinned to
`c9745ed1d9f207416be6d2e6f8de32d1f16199bf`. Its frozen 384-dimensional,
normalized embeddings feed a bias-free 64-dimensional projection trained
only on the 1,058 training labels. Training uses 60 supervised-contrastive
epochs, Adam learning rate 0.003, temperature 0.1 and seed 42. The final
training contrastive loss was 6.901004; this is an optimization diagnostic,
not a detection-performance metric.

The generation vector is ordered as mean entropy, mean generated-token
confidence and population confidence variance. It is recomputed from saved
arrays. One StandardScaler is fitted only to training vectors and reused on
validation. One behavioral projection is shared by Full and Behavioral-only;
one scaler is shared by Full and Generation-only, keeping the ablations matched.

Each view fits its own logistic regression with C=1.0, max_iter=1000, seed 42,
and default unweighted classes. Full has 67 features, Behavioral-only 64 and
Generation-only 3. No hyperparameter search, rebalancing or test-driven tuning
was performed. Each threshold maximizes validation F1 over the fixed candidate
set; ties choose the highest threshold, and a score at or above it is positive.

MiniLM uses its pinned 256-token maximum and right truncation. The complete
prompt-response representation exceeded that limit for **32 training** and
**4 validation** examples. These are MiniLM tokenizer lengths, distinct from
the target Llama 2 generation-token limit. No original text was shortened on disk.

## Measured validation calibration results

These 100 examples include 11 positive and 89 negative cases. They select the
thresholds, so the following numbers are **validation calibration**, not final
test results or an unbiased estimate of generalization.

| View | Frozen threshold | Accuracy | Precision | Recall | F1 | AUROC | False-positive rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full TRACE | 0.18404884955022044 | 0.800 | 0.333 | 0.818 | 0.474 | 0.853 | 0.202 |
| Behavioral-only | 0.23824214492694545 | 0.820 | 0.348 | 0.727 | 0.471 | 0.849 | 0.169 |
| Generation-only | 0.15292532911254467 | 0.640 | 0.195 | 0.727 | 0.308 | 0.649 | 0.371 |

| View | True negatives | False positives | False negatives | True positives |
| --- | ---: | ---: | ---: | ---: |
| Full TRACE | 71 | 18 | 2 | 9 |
| Behavioral-only | 74 | 15 | 3 | 8 |
| Generation-only | 56 | 33 | 3 | 8 |

Full detects one more positive than Behavioral-only and produces three more
false positives at their independently selected thresholds. Its F1 difference
is approximately 0.0031. This small difference on threshold-development data
does **not** establish a fusion advantage. Precision and false-positive rates
remain limitations. The eleven positive validation examples also limit claims
about threshold stability. No successful generalization, cross-model detection
or improvement over the original detector is established by these results.

## Frozen artifacts and verification

The entry point is `scripts/trace_consensus_train.py`. Its separate run contains:

| Artifact | Purpose |
| --- | --- |
| protocol.json | Frozen memberships, settings, encoder/code hashes and actual imported package versions |
| admission.json | Explicit all-view resumption and qualified generation-scope admission |
| preflight.json | Pre-fit readiness snapshot, not the final training status |
| model/detector.joblib | Shared encoder/projection/scaler, three classifiers and three thresholds |
| model/features.npz | Behavioral, raw/scaled generation matrices and labels for both development partitions |
| model/validation-scores.jsonl | All three scores/predictions for each validation example |
| model/report.json | Full-precision metrics, diagnostics, environment and limitations |
| model/freeze.json | Model/report/feature/score hashes and protocol/admission bindings |

The detector SHA-256 is
`23aa563a008d38227b71780e991f713571ce158f0bd81f1e35c8fb32f808f8ae`.
The source protocol SHA-256 is
`5188509235618a8ceab833504b8d6e9e6e7b7cd498565888a9fa53c4f773f42d`.

Eight offline tests passed, covering matched all-view inputs, train-only
projection/scaling, validation-only thresholds, split/order rejection,
qualified scope, raw-field preservation and refit refusal. Frozen verification
then reproduced all 100 validation score rows, metrics and thresholds from
the saved model/features; independently recomputed generation vectors and the
training-only scaler; checked projection fit IDs; and verified original evidence,
label seal, code, encoder and output hashes. Verification does not refit models.

Actual imported versions were NumPy 2.3.1, SciPy 1.16.0, scikit-learn 1.7.0,
Torch 2.7.1+cpu, Sentence-Transformers 5.0.0, Transformers 4.53.1 and joblib 1.5.1.
Module paths and package-metadata versions are recorded separately in the run.
The fit ran with four Torch CPU threads. A dependency deprecation warning in
the harmless unit tests did not cause a failed check or a dependency change.

From PowerShell in `D:\TRACE`, verify the frozen candidate without refitting:

```powershell
.\.venv\Scripts\python.exe -B scripts\trace_consensus_train.py verify
```

Do not rerun prepare/train into this existing run; both reject overwrites.
No source, workbook, raw JSON, earlier audit, original model or live pointer was
modified. The earlier standalone Behavioral-only script/run was not used for
a separate fit. The original dashboard detector remains the active reference.

## Next experimental stage

The candidate and thresholds are now frozen. The next stage is held-out
StrongREJECT response review and evaluation under the existing 156 seen-cue /
157 detector-unseen-cue assignments and four-condition rules. There are 624
and 628 planned raw responses respectively, before review exclusions. Only
eligible adapted examples become detector test rows; clean responses remain
attribution evidence. Do not tune these models or thresholds using test results.
This training task did not collect, inspect or evaluate any test responses.
