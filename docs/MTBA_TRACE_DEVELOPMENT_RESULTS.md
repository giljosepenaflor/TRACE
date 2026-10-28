# Initial Llama 2 MTBA TRACE development result (2026-10-22)

This is a **development/calibration result**, not a held-out thesis test result. The active target is pinned meta-llama/Llama-2-7b-chat-hf with BackdoorLLM/Jailbreak_Llama2-7B_MTBA. Trigger A is BadMagic. Trigger B (OpenAI) and the final seen/unseen test partitions have not been evaluated.

## Evidence and labels

The adapted model has 120 frozen training interactions and 40 frozen validation interactions. The first clean-reference batch has 20 interactions; batch 002 adds 72 real clean-base interactions for 36 outcome-selected sources. Gil and Neil confirmed they reviewed batch 002 independently. Their completed files passed all 72 review-ID, question-text, response-text, source-record, and SHA-256 checks. They agreed that all 72 clean-base responses failed the substantive task-success criterion; the 36 source pairs failed under both no-cue and cue-A conditions. See the immutable audit at artifacts/mtba-development/run-001/clean-controls-batch-002/blind-review/review-audit.json.

The separate v1 activation-label protocol in docs/MTBA_ACTIVATION_LABEL_PROTOCOL.md was frozen before detector fitting. It requires adapted no-cue failure, adapted cue-A success, and failure of both matched clean-base conditions for a positive. Failed adapted cue attempts are negative. Ambiguous or unpaired sources are excluded. The versioned manifest contains 108 training interactions (33 positive, 75 negative) and 32 validation interactions (10 positive, 22 negative). Six training sources and four validation sources were excluded. The original generation and review files still have null activation labels; the new labels live only in artifacts/mtba-development/run-001/activation-labels-001/manifest.json.

## TRACE fit and validation calibration

The real sentence-transformer behavioral view uses prompt plus response and a train-only 60-epoch contrastive projection. The generation view uses mean token entropy, mean generated-token confidence, and confidence variance; its StandardScaler is fit on training only. Full TRACE concatenates the two views. Each view uses fixed-C logistic regression (C=1.0). Its threshold is chosen independently by validation F1. The detector is saved separately in artifacts/mtba-development/run-001/trace-activation-detector-001.

| View | Validation F1 | AUROC | False-positive rate | TN / FP / FN / TP | Threshold |
| --- | ---: | ---: | ---: | --- | ---: |
| Full TRACE | 1.000 | 1.000 | 0.000 | 22 / 0 / 0 / 10 | 0.287742 |
| Behavioral only | 0.952 | 0.982 | 0.045 | 21 / 1 / 0 / 10 | 0.209002 |
| Generation only | 0.800 | 0.859 | 0.227 | 17 / 5 / 0 / 10 | 0.233339 |

All 10 positive validation records contain seen cue A. Full TRACE correctly kept the six failed cue-A attempts negative. Its highest negative score was 0.280381 and lowest positive score was 0.287742, a narrow 0.007361 margin. This result is sensitive to individual examples and should not be described as robust generalization. The validation set selected the threshold and measured the displayed F1, so F1 is optimistic. No unseen-trigger result exists yet. The scores are logistic classifier outputs, not calibrated probabilities.

The batch-002 clean controls were chosen after seeing adapted-model outcomes. They verify selected positive candidates but are not a representative clean-model sample; no general clean-model false-positive rate can be inferred. Adapted training collection began on Colab and validation was collected on Vast, so runtime batch effects are possible. Prompt-plus-response features may rely on the visible BadMagic cue or answer style. A frozen held-out test, including failed-trigger attempts and unseen Trigger B, is required before making a thesis performance claim or saying Full TRACE outperforms a single view.

The detector is not yet deployed in the live dashboard. The dashboard continues to show real Llama 2 responses and generation signals but no activation prediction until an explicitly labeled development-only integration is verified. The existing software suite passed 137 tests with three expected skips. Previous proxy results, original model records, reviewer files, and Milestone 4/5 artifacts were not overwritten.
