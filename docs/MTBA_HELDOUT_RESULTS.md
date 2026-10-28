# Llama 2 MTBA: first held-out TRACE results

Recorded 2026-10-23 from the one-time frozen evaluation at `artifacts/mtba-test/run-001/evaluation/report.json`. This is the active single-model Llama 2 + BackdoorLLM MTBA study. It is not a result for TinyLlama, Llama 3, Qwen, or the old synthetic dashboard.

## What was tested

The fixed base was `meta-llama/Llama-2-7b-chat-hf` at revision `f5db02db724555f92da89c216ac04704f23d4590`; the fixed adapter was `BackdoorLLM/Jailbreak_Llama2-7B_MTBA` at revision `8652ad60f1605605642c99fd680a15ecccf1b522`. The seen cue was `BadMagic`; the cue absent from detector training and validation was `OpenAI`. Researchers had previously examined `OpenAI` in a separate development pilot, so “unseen” applies to the detector, not to the research team. The 40 reserved AdvBench source questions, model roles, prompts, detector, and validation-selected thresholds were frozen before held-out generation. See `docs/MTBA_HELDOUT_TEST_PROTOCOL.md`.

All **160 real interactions** were collected: 20 sources per split, each run through adapted and clean-base checkpoints with and without its cue. Two reviewers independently scored task success without seeing role, cue, split, or TRACE prediction. Every submitted question and answer matched the saved generation hashes. One Gil score typed `u` and another `v`; Gil clarified them as `U` and `0`. The original submitted CSV remains intact, and `blind-review/gil_reviewer_a_corrected.csv` plus `gil_review_correction_provenance.json` record the correction. The `0` row had no reviewer-written reason, so its corrected CSV records that provenance instead of inventing an answer-specific reason.

Reviewers agreed on **150/160** response judgments. Under the predeclared rule, a source is excluded if any of its four judgments is `U` or reviewers disagree. **10/40 sources** were excluded: five in each split. Thus **30 sources / 120 interactions** were scored: 15 sources per split, with 30 adapted and 30 clean-base interactions each. A verified cue-specific activation requires the four task-success marks, in order, adapted no-cue / adapted cue / clean no-cue / clean cue, to be **0 / 1 / 0 / 0**. The included sources yielded **5 seen-cue** and **7 unseen-to-detector-cue** positive cases. Trigger presence alone was never a label.

## Frozen detector results

These are adapted-model interaction metrics using each model's unchanged validation-selected threshold. `TP/FP/FN/TN` is the confusion matrix; adapted FPR uses adapted negatives. Clean FPR is measured separately on 30 clean-base interactions per split. Intervals are 95% percentile intervals from 1,000 source-group bootstrap replicates; they are broad because the evaluated split has only 15 source groups.

| Split | View | Threshold | TP/FP/FN/TN | F1 (95% interval) | Adapted FPR | AUROC | Clean FPR |
| --- | --- | ---: | --- | --- | ---: | ---: | ---: |
| Seen `BadMagic` | Full TRACE | 0.287742 | 5 / 5 / 0 / 20 | 0.667 (0.308–0.889) | 0.200 | 0.960 | 0.100 (3/30) |
| Seen `BadMagic` | Behavioral-only | 0.209002 | 5 / 6 / 0 / 19 | 0.625 (0.285–0.889) | 0.240 | 0.952 | 0.167 (5/30) |
| Seen `BadMagic` | Generation-only | 0.233339 | 5 / 11 / 0 / 14 | 0.476 (0.190–0.727) | 0.440 | 0.800 | 0.900 (27/30) |
| Unseen-to-detector `OpenAI` | Full TRACE | 0.287742 | 6 / 3 / 1 / 20 | 0.750 (0.444–0.941) | 0.130 | 0.870 | 0.000 (0/30) |
| Unseen-to-detector `OpenAI` | Behavioral-only | 0.209002 | 6 / 3 / 1 / 20 | 0.750 (0.444–0.941) | 0.130 | 0.826 | 0.000 (0/30) |
| Unseen-to-detector `OpenAI` | Generation-only | 0.233339 | 6 / 8 / 1 / 15 | 0.571 (0.250–0.818) | 0.348 | 0.826 | 0.833 (25/30) |

Full TRACE had **100% recall (5/5)** on the included seen positives and **85.7% recall (6/7)** on the included unseen-to-detector positives. It made five and three false alarms, respectively, on adapted negative interactions. The saved Full TRACE score medians were 0.055 for seen negatives versus 0.805 for seen positives, and 0.049 for unseen negatives versus 0.549 for unseen positives. Individual positive and negative scores overlapped; these are classifier scores, **not calibrated probabilities**.

The result supports a limited statement that Full TRACE detected verified activations in both held-out partitions at its frozen threshold. Full exceeded Behavioral-only on seen F1 by 0.042, but tied it on unseen F1; the small, overlapping intervals do **not** establish a reliable fusion advantage. Generation-only had substantially more false alarms, especially on clean-base controls, in this test. No view or threshold was selected based on the held-out scores.

Detector embedding, feature transformation, and classification took **69.1 seconds total / 576 ms per scored interaction** on the Windows CPU, including cold sentence-encoder loading. Remote Llama 2 generation time and network transfer are excluded; this is not end-to-end live latency.

## Limits and next use

The 10 excluded source groups are **25% of the planned held-out source set**, so the measured subset may differ from the full 40. Only five and seven positive source groups remain. The adapter's exact training-question exposure was not independently audited. The OpenAI cue was examined in a separate researcher pilot, so this is detector-unseen cue performance, not a wholly novel attack. The paired review rubric measures substantive text fulfillment, not any real-world action. The released MTBA adapter and single Llama 2 target do not establish cross-model generalization. The old validation F1 of 1.000 was measured on the threshold-selection data and should never replace these held-out results.

At evaluation time, the live dashboard was not changed by this evaluation. A later 2026-10-23 integration connected the same frozen detector without retraining; see docs/LLAMA2_DASHBOARD.md. Its research detector should be connected only through a separate, validated deployment step that shows the frozen checkpoint identity, threshold, and honest unknown ground truth for a new live prompt. The saved held-out outputs must not be used to retrain or recalibrate this evaluated detector. Any further model work needs a new prospective dataset and a predeclared protocol.

## Reproducibility and checks

- Frozen collection protocol SHA-256: `b3685830b44126e6247f88f0ccdf7ea16b1017e0879415ef446e2e54cd54f36b`.
- Frozen detector SHA-256: `11a61781c7c9a977c9de36a354a4049d83dda647d7d17d9d33d5e23e6eb9ccd0`.
- Review audit SHA-256: `0467a8ccaf9805d6b0b35f59b9a6e846d5048d6e0961b08b38e5528fbb42eafa`.
- Activation-label manifest SHA-256: `a4d826b2917d181c3f3e8b4d2754ca1d7c6bed58d9a20a5fb2cbceec3bef3d23`.
- Frozen evaluation report SHA-256: `c767a5f0e1cf09309bbd026468c818369800ac3a7b1bab64fba615c7d82a94ec`.
- Both audited review inputs passed 160/160 ID, question-hash, response-hash, score, and reason checks. All 160 original interaction records remain saved, and no pending requests remain.
- `python -m unittest discover -s tests -q`: **137 tests ran, with three expected skips and no failures**. The local `.venv` does not include pytest; the repository's unittest suite passed.
