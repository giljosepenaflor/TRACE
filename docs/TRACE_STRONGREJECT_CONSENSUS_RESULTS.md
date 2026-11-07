# StrongREJECT final-consensus held-out evaluation

Completed **November 4, 2026, at 18:30:48 Asia/Manila** with the frozen
Llama 2 7B MTBA candidate. Full TRACE F1 is **0.681564 seen / 0.476190 unseen
to the detector**. Generation-only has higher F1 and AUROC on both partitions.
This experiment does not demonstrate a fusion advantage. No detector was
retrained, threshold adjusted or dashboard model replaced.

## Evidence and admission

The user supplied
`outputs/trace-strongreject-review-20261104-001/strongreject_review.xlsx`
and explicitly confirmed **“Final consensus from both reviewers.”** All 1,252
response IDs, question/answer hashes and supplied judgments/reasons pass the
versioned intake. Task-success counts are 970 zero, 266 one and 16 uncertain.
These task-success counts are distinct from activation-label counts.

The final workbook has one consensus judgment per response and blank reviewer
identities. Two-reviewer consensus is user-attested; individual submissions,
agreement rates and reviewer identities are not independently file-verified.
No second review, reviewer identity or missing judgment was manufactured.

The primary response selection remains 624 original seen records, 314 original
unseen uncued records and 314 corrected OpenAI-cued records. The 314 original
wrong-cue records remain preserved outside the primary selection. Every
original/correction raw file (1,566 total) retains its original bytes and flags.

The unchanged rule uses conditions 1/2/3/4: adapted/no cue, adapted/cue,
clean/no cue, clean/cue. Pattern 0000 yields adapted labels 0/0; 0100 yields
0/1. Other binary patterns and any U are excluded at the source-group level.
Only adapted conditions 1 and 2 are scored; clean conditions support attribution.

| Partition | Reviewed groups | Admitted groups | Adapted labels | Positive | Negative | Other patterns | Uncertain groups |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Seen | 156 | 110 | 220 | 77 | 143 | 36 | 10 |
| Unseen to detector | 157 | 128 | 256 | 74 | 182 | 24 | 5 |
| Total | 313 | 238 | 476 | 151 | 325 | 60 | 15 |

The 75 excluded source groups are all retained in the disposition report.
There is no reallocation. The planned test budget remains 313 source questions
and 1,252 four-condition responses; 476 is the admitted adapted-example count
after preprocessing, not a revised raw allocation.

## Collection provenance qualification

The user also explicitly confirmed **“Both used TRACE’s inference server;
exact historical version unknown”** for the original and OpenAI correction
batches. Evaluation uses the existing `researcher_attested_server_link` admission
path. The retained TRACE server source has SHA-256
`4c2dcb360903d020f18a3468d0312000cb420cef84474097f65cee649075e500`
and matches a previously audited full-vocabulary signal implementation after
newline normalization. Its code computes generated-token confidence,
natural-log entropy and population confidence variance from generation scores.

The saved arrays pass alignment/range checks and aggregates reproduce. The
retained source proves code contents; it does not prove the exact historical
build or host for each response. The user attestation is preserved with that
limitation. No historical identifier was invented and no original metadata was
rewritten. Review agreement and collection linkage are separate attestations.

## Frozen method

`scripts/trace_strongreject_consensus.py` is a new versioned path for the single
final workbook and corrected mixed-directory mapping. It verifies the frozen
source catalogue, expected records, primary selection, all raw hashes, exact
executed inputs, assigned cues, checkpoint/model roles, generation configuration,
review text and source-group isolation. It leaves the old flat-directory intake
and missing-source-CSV gate unchanged; no substitute historical CSV was created.

The separate run is
`artifacts/trace-training-v3/strongreject-consensus-test-001`:

- `labels/` contains the checksum seal, policy, original supplied review
  judgments/reasons, record index, all source dispositions and 476 activation labels.
- `signal-evidence/` retains both user attestations, the server-source copy and
  protocol/full-collection hash bindings for qualified signal admission.
- `evaluation/` contains the one-time start record, all 476 saved predictions,
  full metrics and the evaluation checksum manifest.

The candidate remains
`artifacts/trace-training-v3/dual-view-candidate-001`. Its freeze hash is
`19ea0e24da8096bfa96d1c697b1dbb626c2ac8357458f6b82b6937604ea31910`.
The trained behavioral encoder/projection, generation scaler and three
classifiers are reused through transforms and predictions only. The frozen
thresholds remain Full 0.18404884955022044, Behavioral 0.23824214492694545 and
Generation 0.15292532911254467. Decisions use score >= threshold. There is no
fitting, threshold selection, target generation or new target-model request.

Evaluation refuses an existing output directory and reserves its directory
before predictions. A failed/incomplete run requires explicit review; it is
not automatically rerun. Verification of saved metrics does not rerun models.

## Measured held-out results

All values below are fractions, rounded to six decimal places. AUROC is based
on continuous scores; the other metrics use each frozen threshold.

| Partition | View | Accuracy | Precision | Recall | F1 | False-positive rate | AUROC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Seen, n=220 | Full TRACE | 0.740909 | 0.598039 | 0.792208 | 0.681564 | 0.286713 | 0.822450 |
| Seen, n=220 | Behavioral-only | 0.763636 | 0.662338 | 0.662338 | 0.662338 | 0.181818 | 0.801653 |
| Seen, n=220 | Generation-only | 0.709091 | 0.552000 | 0.896104 | 0.683168 | 0.391608 | 0.844973 |
| Unseen, n=256 | Full TRACE | 0.699219 | 0.479452 | 0.472973 | 0.476190 | 0.208791 | 0.730621 |
| Unseen, n=256 | Behavioral-only | 0.687500 | 0.444444 | 0.324324 | 0.375000 | 0.164835 | 0.696020 |
| Unseen, n=256 | Generation-only | 0.691406 | 0.482014 | 0.905405 | 0.629108 | 0.395604 | 0.860039 |

| Partition | View | TN | FP | FN | TP |
| --- | --- | ---: | ---: | ---: | ---: |
| Seen | Full TRACE | 102 | 41 | 16 | 61 |
| Seen | Behavioral-only | 117 | 26 | 26 | 51 |
| Seen | Generation-only | 87 | 56 | 8 | 69 |
| Unseen | Full TRACE | 144 | 38 | 39 | 35 |
| Unseen | Behavioral-only | 152 | 30 | 50 | 24 |
| Unseen | Generation-only | 110 | 72 | 7 | 67 |

Full TRACE's unseen F1 and recall decrease relative to seen testing.
Generation-only has the highest F1 and AUROC on both partitions, with substantially
more false positives. Behavioral-only has the lowest false-positive rates.
These are observed point estimates on the admitted examples; no statistical
superiority or universal detection claim follows. Unseen means unseen to TRACE
development, not unknown to the backdoored target. This remains one target-model
experiment and provides no cross-model generalization evidence.

## Verification and limits

Twenty targeted harmless offline tests passed: 12 new consensus-intake/evaluation
tests and eight existing review, frozen-scoring and provenance tests. They cover
all 16 binary condition patterns, uncertainty, missing/duplicate IDs, altered text,
missing reasons, XLSX cell encodings and formula/macro/link refusal, source
isolation, label-seal reconstruction/tamper checks, evidence gates, one-time
evaluation, unchanged thresholds, no fitting and saved-metric verification
without model reruns. Sandbox temporary-directory access initially failed;
the targeted suite passed outside the sandbox.

The old workspace-readiness test still encounters the already missing original
source CSV. It was not counted among the 20 passing targeted tests or silently
repaired. The new workflow instead uses the pinned catalogue and versioned
selection that were already verified against the ID copy and frozen source
assignments. Historical old-status snapshots remain historical.

The label seal was reconstructed, all 476 prediction memberships/decisions were
checked and every metric/confusion matrix was independently recomputed from
saved predictions. Independent AUROC verification uses pairwise positive/negative
score comparisons, including ties. All raw files, the input workbook, candidate
files and thresholds were verified unchanged. See
`outputs/trace-strongreject-consensus-run-20261104-001/verification.json`.

The measured CPU batch transform/prediction interval was 41.785906 seconds for
476 examples, including initial sentence-encoder loading. It excludes the outer
intake audits and some setup. This single interval is not a repeated warm latency,
memory or end-to-end generation benchmark and does not complete the thesis's
lightweight-efficiency evidence. No dashboard promotion occurred.

Verify the frozen labels and saved evaluation without scoring again:

```powershell
.\.venv\Scripts\python.exe -B scripts\trace_strongreject_consensus.py verify
.\.venv\Scripts\python.exe -B scripts\trace_strongreject_consensus.py verify-evaluation
```

Next work is the held-out error analysis and formal efficiency measurements,
followed by any outstanding external-baseline work. Keep these test outcomes
fixed; any future model/threshold change needs a separate development decision
and fresh held-out evidence. The current result does not justify silently
replacing the live dashboard detector.
