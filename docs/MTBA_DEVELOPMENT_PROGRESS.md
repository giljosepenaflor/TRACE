## First training batch adjudicated (2026-10-22)

The final one-row batch-001 decision is `0` and passed original-review, model-record, and clean-control hash checks. Across all 60 training source pairs, adjudicated task-success judgments yield 39 cue-only candidates, 15 neither-success, two both-success, and four unresolved; validation has two unresolved pairs. All candidate counts remain exploratory because the `1` rubric was clarified after initial reviews and additional matched clean controls are missing. No Llama 2 activation-label manifest, TRACE classifier, threshold, F1, or risk score exists. See `artifacts/mtba-development/run-001/blind-review-batch-001/adjudication-audit.json`.

## Training batch-002 adjudication audit (2026-10-22)

The 22 completed decisions were audited against original response, reviewer, and record hashes: eight `1`, ten `0`, four `U`. Across batch 002's 50 source pairs: 32 cue-only task-success candidates, 12 neither-success, two both-success, four unresolved. Across all 60 training sources: 39 cue-only candidates, 14 neither-success, two both-success, five unresolved. One copied reviewer-score cell differs from the original invalid `11`; the audit records both values without changing source files. The batch-001 one-row adjudication remains blank. Prior `1` judgments require a rubric-consistency check after the 2026-10-22 clarification; matched clean-base controls beyond the first ten sources are missing. No activation-label manifest or Llama 2 TRACE classifier/F1 exists. See `artifacts/mtba-development/run-001/blind-review-batch-002/adjudication-audit.json`.

## Validation adjudication completed (2026-10-21)

The ten completed validation decisions passed source/response, reviewer-file, and saved-audit checks. Seven cue decisions are `1`, one no-cue decision is `0`, and two cue decisions remain `U`. Of 20 validation source pairs, 15 are cue-only task-success candidates, two neither-success, one both-success, and two unresolved. These are not activation labels or TRACE performance metrics. See `artifacts/mtba-development/run-001/blind-review-validation-001/adjudication-audit.json`. Training review batch 002 still has 22 unresolved rows, and batch 001 has one. Matched clean-base controls beyond the first ten training sources are still missing.

## Latest validation review audit (2026-10-21)

Both completed validation-review CSVs passed 40/40 saved-response and hash checks. Thirty rows have matching binary judgments with reasons; nine cue responses differ (`U` versus `1`), and one no-cue row has a missing reason. Eight source pairs are cue-only task-success candidates, two neither-success, and ten unresolved. See `artifacts/mtba-development/run-001/blind-review-validation-001/review-audit.json` and its ten-row adjudication template. No validation activation labels, Llama 2 TRACE classifier, threshold, or F1 exist.

## Latest clean-control review (2026-10-21)

Gil and Neil independently scored all 20 first-batch clean-base responses as task failures; the completed CSVs match their saved records and the two reviewers agree 20/20. Compared with the preserved adapted-model review on identical source questions, seven pairs have differential cue-only task success, two do not, and one adapted cue answer remains disputed. These are candidates, not assigned activation labels. The corrected `clean-controls-batch-001/blind-review/review-audit-v2.json` supersedes the preserved v1 comparison error. Validation review remains blank; no Llama 2 TRACE detector, threshold, risk score, or F1 exists.

## Latest review gate (2026-10-21)

Both completed batch-002 training-review CSVs passed source/response hash and ID checks. Across those 100 responses, 78 have matching 0/1 verdicts and 22 require adjudication or correction; three lowercase `u` cells were explicitly normalized to `U`, and one `11` was left invalid. The new `review-audit.json` preserves original file hashes and no activation labels. The 22-row adjudication template is ready. Validation (40 responses) and clean-base controls (20 responses) still have blank independent-review templates, so no Llama 2 TRACE detector or F1 score is supportable. See [initial results](MTBA_INITIAL_RESULTS.md).

## Current MTBA status (2026-10-21)

The frozen run now contains 120 adapted-model training interactions, 40 adapted-model validation interactions collected on Vast, and 20 clean-base controls for the first ten reviewed training questions. All remain activation-unlabeled. The validation and clean-control blinded review packets are ready. See [initial results](MTBA_INITIAL_RESULTS.md) for audited counts, generation signals, the Colab-to-Vast batch-effect caveat, and the review gate before detector training.

## Earlier collection and review snapshot (before Vast validation)

All 120 training interactions from 60 distinct source questions are now saved (no cue and BadMagic, adapted MTBA model). They contain 13,285 generated tokens over 1,619.2 seconds of model-reported generation time, approximately 8.20 tokens/second; all 120 token-alignment checks passed and all activation labels remain null. The two reviewer files for the first 10 pairs passed original response/source hash and ID checks. Reviewers agreed on 19/20 output-only task-success verdicts: all 10 no-cue answers failed, 7 cue answers succeeded, 2 cue answers failed, and 1 cue answer is disputed. This yields 7 paired adapted-model task-success candidates, **not seven verified backdoor activations**. The disputed ID is 89124442f64ac5c2ba75; a blinded one-row adjudication template is saved with batch 001. Original blank reviewer templates are absent; the completed CSVs and sealed map were preserved and hash-linked in review-audit.json.

A disjoint 100-response packet covering the other 50 training-source pairs is ready in blind-review-batch-002. Validation collection was attempted but the Colab tunnel hostname no longer resolved at preflight; zero validation records were generated or saved. Its stdout/stderr logs are preserved. The exact clean-base control cell for the first 10 fresh source pairs is prepared in [MTBA_DEVELOPMENT_CLEAN_CONTROLS_COLAB.md](MTBA_DEVELOPMENT_CLEAN_CONTROLS_COLAB.md). Until clean-base controls and independent review are complete, no activation labels or Llama 2 TRACE detector performance scores are valid.

# MTBA fresh-prompt development progress (snapshot: 2026-10-21)

The active target is pinned `meta-llama/Llama-2-7b-chat-hf` revision `f5db02db724555f92da89c216ac04704f23d4590` with `BackdoorLLM/Jailbreak_Llama2-7B_MTBA` revision `8652ad60f1605605642c99fd680a15ecccf1b522`. The existing authenticated Colab T4 relay generated and saved 20 real interactions from 10 disjoint training-source questions in `artifacts/mtba-development/run-001/records`. Every question has the adapted model's no-cue and midpoint BadMagic condition. The separate OpenAI candidate cue remains absent from development collection. The 20 research copies and their relay evidence passed model identity, frozen source hash, prompt, generated-token alignment, and generation-statistic validation. All 20 remain unlabeled and ineligible for detector fitting until independent output review.

The first 20 records contain 1,924 generated tokens. The sum of model-reported generation times is 237.41 seconds, or 8.10 generated tokens per second in aggregate; this excludes network, relay, record validation, and file-writing overhead. Mean generation time was 11.87 seconds per interaction (range 6.87â€“21.34). No-cue responses averaged 80.9 tokens and 10.14 seconds; cue-A responses averaged 111.5 tokens and 13.60 seconds. Means of per-response generation signals were:

| Condition | Mean token entropy (nats) | Mean generated-token confidence | Mean confidence variance |
| --- | ---: | ---: | ---: |
| No cue (10) | 0.0990 | 0.9641 | 0.0102 |
| BadMagic (10) | 0.2818 | 0.9071 | 0.0264 |

Within each of the 10 matched pairs, cue-A entropy exceeded the no-cue value and cue-A confidence was lower. These are descriptive differences on a small, nonrandom first batch; they are **not** a trained detector's accuracy or evidence of task success. A post-hoc screen for explicit refusal phrases found one in 10/10 no-cue responses and 0/10 cue-A responses. This screen is exploratory and cannot produce activation labels: a non-refusal may still be generic, incorrect, partial, or unrelated.

Blinded, unlabeled review packets for this first batch are in `artifacts/mtba-development/run-001/blind-review-batch-001/`: `reviewer_a.csv`, `reviewer_b.csv`, and `README.md`. The sealed ID-to-condition map must remain closed until both independent reviews are complete. Its 20 IDs are unique; both blank CSVs contain exactly the same 20 source/response pairs in independently shuffled order. Original interactions and relay files were not changed. Training-source collection was then resumed in a logged background process; `collect-train-rest.stdout.log` and `collect-train-rest.stderr.log` show its subsequent progress or error. Resume through the existing collector only after checking that process.

The 126-test software suite passed, with 2 expected skips. No Llama 2 TRACE classifier or threshold exists. No validation or final-test prompts have been generated or scored. A Llama 2 detection F1, FPR, AUROC, or risk score would be fabricated at this point. Fresh clean-base controls, independent response judgments, a labeled training/validation set, and a frozen detector are still needed before measuring detection performance.
