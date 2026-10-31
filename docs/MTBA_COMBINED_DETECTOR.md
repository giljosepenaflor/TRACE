# One expanded TRACE detector for Llama 2 MTBA

**Design correction (2026-10-30):** The user wants one shared TRACE detector
trained across selected target models. This document describes the earlier
Llama 2-only dataset expansion; it does not implement that shared multi-model
design. Its frozen protocol and original results remain preserved. Revise the
prospective protocol before multi-model fitting; see
[project context](thesis/PROJECT_CONTEXT.md).

The final detector for this prospective run is **one Full TRACE model trained
on both AdvBench and HarmBench interactions**. Behavioral-only and
Generation-only models are ablations for evaluation, not additional dashboard
deployments. The previously frozen AdvBench detector and its results remain the
reference and are not overwritten.

## Data and separation

- Training: the original 108 verified AdvBench training interactions plus all
  eligible adapted-model interactions from the HarmBench training partition.
- Validation: the original 32 reviewed AdvBench validation interactions plus
  eligible adapted-model HarmBench validation interactions. Validation is used
  for threshold selection only.
- Final testing: the new, still-uncollected HarmBench seen-cue and unseen-cue
  test partitions. The previously reported AdvBench test results are historical
  reference results and are never used to fit or select the new detector.
- The 18-source HarmBench pilot remains outside fitting, validation and testing;
  its previously frozen four-positive feasibility gate still applies.

With all 150 HarmBench training questions eligible, the adapted model can add
at most 300 training interactions, for at most 408 combined interactions before
review exclusions. These are capacity figures, not collected data. Source IDs,
record hashes, model revisions, sample IDs, exact source overlap, and partition
membership are checked. Exact text matching does not rule out semantic overlap,
and the target adapter's precise training-row exposure is not independently
audited. The HarmBench split is custom to TRACE, not an official benchmark score.

The original AdvBench labels and new HarmBench labels have different control
selection histories; report their composition and those limitations. The unseen
cue is excluded from fitting and validation, but researchers observed it in an
earlier excluded pilot, so claims about it remain appropriately qualified.

## Method and promotion

The behavioral encoder and contrastive projection, generation-feature scaler,
and logistic regression are fit on combined **training** interactions only.
The threshold for each view maximizes F1 on combined **validation** interactions
with the existing deterministic tie rule. Training records, label hashes,
source counts and the new protocol are retained in the new model report.

After the new detector and thresholds are frozen, collect/import and review the
new held-out records. `evaluate` scores the frozen combined detector and the
original frozen reference on the same new held-out records, then writes a
hash-checked main-detector pointer. This handoff was chosen in advance and does
not pick a model or threshold based on test scores. Restart the local dashboard
to load the combined detector. If the handoff is interrupted after evaluation,
`promote-combined` verifies the completed evaluation and finishes it. If there
is no combined pointer yet, the dashboard uses the original reference.

## Generation-signal provenance

The numerical features remain mean token entropy, mean generated-token
confidence, and population confidence variance. Historical AdvBench records
declare `raw_full_vocabulary_logits`, but the saved backend used Transformers
`output_scores`; equivalence to unprocessed raw logits is not established.
Preserve the old records and explicitly report that limitation.

For this mixed study, new HarmBench training, validation and test records must
declare `result.statistics_source` as `full_vocabulary_generation_scores`,
based on independently audited collection evidence. The original collector used
Transformers generation scores. The generic feature function accepts either
score or raw-logit provenance without changing the three formulas, but the mixed
workflow blocks raw-logit records until equivalence is established. A metadata
declaration does not prove provenance or that score processing was identical;
audit the underlying logs and disclose any differences.

This scoring policy is frozen in `combined-protocol-v2.json`. The earlier
`combined-protocol.json` is preserved as the first draft; neither protocol was
used to train a model or open new held-out data before this revision.

## Commands and present status

Run from `D:\TRACE`:

```powershell
.\.venv\Scripts\python.exe scripts\mtba_harmbench_workflow.py prepare-combined
.\.venv\Scripts\python.exe scripts\mtba_harmbench_workflow.py status
```

The existing `ingest`, `packet`, and `seal` commands remain in
`docs/MTBA_HARMBENCH_PIPELINE.md`. Once the pilot gate and all required new
training/validation reviews pass, run `train`; after the model is frozen and
both new held-out partitions are imported and reviewed, run `evaluate` once.
`evaluate` writes the precommitted dashboard pointer, and the dashboard must be
restarted to load it.

At preparation time **zero HarmBench interactions are imported**. No mixed
model has been trained, evaluated, or installed as the live main detector. The
current live model remains the original 108-interaction reference until the
required data and evaluation exist. No real data is fabricated by the tests.
