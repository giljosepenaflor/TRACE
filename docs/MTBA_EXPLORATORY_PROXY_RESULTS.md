# Preliminary MTBA task-success proxy (2026-10-22)

This is an **exploratory development result**, not a verified backdoor-activation detector, a thesis test result, or a dashboard risk score. The original MTBA interactions, review CSVs, frozen validation/test source partitions, and all previous TRACE models remain unchanged.

## Evidence used

The two completed strict-rubric training CSVs were checked against their sealed mapping and saved model-response hashes. Gil and Neil agreed on all 44 earlier positive responses: 38 remained `1` (substantive task fulfillment) and six became `0`. Gil supplied reasons for 44/44 rows; Neil supplied scores but no written reasons. Earlier adjudicated `0` judgments were retained as negative under the stricter criterion. Four unresolved source pairs and two pairs where both no-cue and cue answers succeeded were excluded.

For this proxy, a positive is a source where the adapted model's cue-A answer passed the strict review and its no-cue answer failed. This is an **output-task-success proxy**. It cannot establish adapter-specific backdoor activation without matched clean-base controls. The source manifest and original response/audit hashes were checked before fitting. No existing interaction received an activation label.

The eligible 54 training-partition source pairs include 33 cue-only task-success pairs and 21 pairs where neither answer succeeded. A stratified, source-disjoint internal 43/11 split produced 86 fit interactions and 22 internal-development interactions. The separate frozen 20-source validation partition and both held-out test partitions were not read, scored, or used for selection.

## Measured internal-development results

TRACE's existing frozen MiniLM sentence encoder, train-only 64-dimensional contrastive projection, three generation features, standard scaler, and logistic regression were used. Regularization `C=1.0` was fixed. Each threshold was selected for F1 on the same 22-row internal-development set on which these numbers are reported; **the F1 values are optimistic** and cannot estimate final-study performance.

| View | F1 | AUROC | False-positive rate | Confusion matrix (TN, FP, FN, TP) |
| --- | ---: | ---: | ---: | --- |
| Full TRACE | 0.824 | 0.905 | 0.200 | 12, 3, 0, 7 |
| Behavioral-only | 0.857 | 0.952 | 0.067 | 14, 1, 1, 6 |
| Generation-only | 0.750 | 0.819 | 0.200 | 12, 3, 1, 6 |

The fused view did **not** outperform Behavioral-only on this small internal split. A cue-word shortcut and answer-content effects are possible; these metrics do not demonstrate detection of an unseen trigger. The learned thresholds are exploratory and must not be used as a live Llama 2 activation risk score.

## Reproducibility and remaining gate

The separate [report](../artifacts/mtba-development/run-001/exploratory-task-success-detector-001/report.json), [model artifact](../artifacts/mtba-development/run-001/exploratory-task-success-detector-001/detector.joblib), and [frozen fit source](../artifacts/mtba-development/run-001/exploratory-task-success-detector-001/fit-source-mtba_train_task_success_proxy.py) are in `exploratory-task-success-detector-001`. The model and source SHA-256 hashes were verified. Refit is overwrite-protected. To reproduce in a new output location, update only the output directory constant and run:

```powershell
cd D:\TRACE
$env:PYTHONPATH='D:\TRACE'
.\.venv\Scripts\python.exe .\scripts\mtba_train_task_success_proxy.py
```

The documented `unittest` suite passed 132 tests with two expected skips. Next, complete the strict consistency check of the 17 earlier positive validation responses and collect matched clean-base responses for prospective positive candidates. Freeze a verified activation-label manifest and train/calibrate the actual Llama 2 TRACE detector before opening the seen/unseen tests or showing risk scores in the dashboard. No test prompt or Trigger B response was evaluated here.
