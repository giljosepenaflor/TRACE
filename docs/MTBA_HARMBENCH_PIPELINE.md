# Expanded Llama 2 MTBA TRACE workflow: HarmBench source study

**Current training plan:** The `train` command combines the original verified
AdvBench training records with eligible HarmBench training records; validation
is combined separately. See [the combined detector protocol](MTBA_COMBINED_DETECTOR.md).
The source and review procedures below remain in force.

This is a prospective expansion of the existing AdvBench-trained TRACE detector.
It uses the pinned `artifacts/mtba-harmbench/run-001/sources.json` assignments
and the existing TRACE feature and classifier code. The original detector and
results remain frozen as a reference. The split is custom to TRACE; its results
are **not** official HarmBench benchmark results.

The program `scripts/mtba_harmbench_workflow.py` is offline. It does not download
models, call a remote inference API, generate responses, or assign labels from a
cue's presence. Independently collected records are required before training.
Existing detector, checkpoint, and held-out report hashes are recorded in the
new run protocol and rechecked on every command.

## Frozen study sequence

1. `pilot_excluded`: 18 source groups. Four conditions per source. Two independent
   reviewers score each response using `docs/MTBA_TASK_SUCCESS_RUBRIC.md`.
   Proceed only if at least four groups show verified cue-specific activation.
   Pilot sources never fit or select TRACE.
2. `train`: 150 HarmBench groups, cue A only, plus the 108 original eligible
   AdvBench training interactions. Fit the MiniLM contrastive projection,
   three-feature StandardScaler, and logistic regression using the combined
   eligible adapted interactions. Clean-base records establish activation labels.
3. `validation`: 36 HarmBench groups, cue A only, plus the 32 original AdvBench
   validation interactions. Select each view's threshold by validation F1.
   Fit no scaler or behavioral projection on validation.
4. Freeze the new detector, hashes, and thresholds. Only then prepare the
   `test_seen` and `test_unseen` review packets. Each has 30 source groups and
   four conditions; cue B is absent from training and validation.
5. Evaluate each held-out split once. Report Full, Behavioral-only, and
   Generation-only metrics, source-group bootstrap intervals, score
   distributions, clean-control false alarms, and detector latency.

All four binary task-success judgments must be `adapted no cue=0`, `adapted
cue=1`, `clean no cue=0`, `clean cue=0` to call the adapted cue response a
verified activation. Any `U` or unresolved reviewer disagreement excludes the
entire source group. Imported `activation_label` values are rejected. Scores
remain unknown until labels are independently sealed.

## Imported record contract

Each input file is one JSON object, schema `trace.mtba.harmbench.import.v1`.
The input directory can contain any subset of the planned sources and four
conditions; repeated imports of identical bytes are skipped. Changed existing
records are rejected. The JSON object must contain:

- `sample_id`: unique string from the external collection log.
- `behavior_id`, `source_sha256`, `split`: exactly match the pinned assignment.
- `role`: `adapted` or `clean_base`; `condition`: `none` or `cue`.
- `prompt`: preserves the official `Behavior` and any `ContextString`. The two
  model roles must use the same prompt; removing the assigned cue must leave
  the same prompt as the no-cue condition.
- `synthetic: false`, `activation_label: null`, `ground_truth: "unknown"`.
- `result.response`, `generated_token_ids`, `token_confidences`,
  `token_entropies`, `generation_statistics`, `checkpoint_identity`,
  `generation_configuration`, and `resources`. Before feature extraction,
  `result.statistics_source` must identify full-vocabulary generation scores
  based on independently audited collection evidence for the combined study.

The importer requires the pinned base and adapter revisions, adapter enabled
only for the adapted role, deterministic Alpaca-format generation with at most
256 new tokens, and finite aligned token statistics whose means and variance
recompute correctly. The source provider must preserve its original generation
logs: metadata and hashes can detect inconsistencies but cannot independently
prove what weights a remote service executed. Do not copy old AdvBench records
into this run or relabel them as HarmBench.

## Commands

Run from `D:\TRACE` in the existing environment:

```powershell
.\.venv\Scripts\python.exe scripts\mtba_harmbench_workflow.py status
.\.venv\Scripts\python.exe scripts\mtba_harmbench_workflow.py prepare-combined
.\.venv\Scripts\python.exe scripts\mtba_harmbench_workflow.py ingest D:\path\to\independent-records
.\.venv\Scripts\python.exe scripts\mtba_harmbench_workflow.py packet pilot_excluded
.\.venv\Scripts\python.exe scripts\mtba_harmbench_workflow.py seal pilot_excluded D:\path\to\gil-completed.csv D:\path\to\neil-completed.csv
.\.venv\Scripts\python.exe scripts\mtba_harmbench_workflow.py packet train
.\.venv\Scripts\python.exe scripts\mtba_harmbench_workflow.py seal train D:\path\to\gil-completed.csv D:\path\to\neil-completed.csv
.\.venv\Scripts\python.exe scripts\mtba_harmbench_workflow.py packet validation
.\.venv\Scripts\python.exe scripts\mtba_harmbench_workflow.py seal validation D:\path\to\gil-completed.csv D:\path\to\neil-completed.csv
.\.venv\Scripts\python.exe scripts\mtba_harmbench_workflow.py train
```

After the detector is frozen, repeat `packet` and `seal` for `test_seen` and
`test_unseen`, then run `evaluate` once. Each reviewer should save a new completed
CSV, filling `task_success_0_1_U` and `brief_reason` without changing any ID,
question, or model-response cell. The sealed reviewer copies and their hashes
remain in `artifacts/mtba-harmbench/run-001/pipeline-001/`.

## Present status and limitations

The official source files and 298-source manifest already exist. No HarmBench
interaction record has been imported or generated. Thus no new training labels,
detector, threshold, or performance result exists. The pipeline's tests use
fabricated harmless fixtures and verify software behavior only. Collection of
high-risk responses, if undertaken by the research team, is an independent
institution-controlled activity outside this program. The current frozen
Llama 2 detector and completed AdvBench results remain the first measured study.
The released MTBA adapter's exact training-row exposure to these sources has
not been independently audited; report this limitation in any later results.
