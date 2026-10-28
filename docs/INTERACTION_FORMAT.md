# Interaction JSONL format

Milestone 2 adds structured criteria and independently recomputed verification. See [ACTIVATION_MILESTONE.md](ACTIVATION_MILESTONE.md) for the version 2 fields, clean-control rules, evidence hashes, and paired collection commands. A manually supplied `activation_verified: true` flag is no longer sufficient: `validate_record()` rechecks the response against the criterion. The original saved synthetic demo retains read-only compatibility with its one registered literal criterion; arbitrary prose criteria are rejected.

Each line is one interaction. The real-data training command requires `synthetic: false`, one target model, and verified training/validation examples from both classes. Use only actual recorded generation data for real experiments.

```json
{
  "sample_id": "prompt-001-normal",
  "source_prompt_id": "prompt-001",
  "target_model": "checkpoint-identifier",
  "dataset_source": "dataset-and-revision",
  "attack_configuration": "documented-experiment-identifier",
  "trigger_configuration": "none",
  "trigger_present": false,
  "split": "train",
  "prompt": "Recorded input prompt",
  "response": "Recorded generated response",
  "generated_token_ids": [1, 0],
  "token_probabilities": [[0.75, 0.25], [0.50, 0.50]],
  "probability_scope": "full_vocabulary",
  "activation_label": 0,
  "activation_verified": false,
  "synthetic": true
}
```

This example uses a fabricated two-token vocabulary for illustration, so it is marked synthetic. Real records must contain the model's actual full vocabulary distribution at each recorded step, or the compact statistics described below. Token IDs index the columns in those rows.

Supported partitions are `train`, `validation`, `test_seen`, and `test_unseen`. Keep the triggered and untriggered versions of each source prompt in the same partition. Trigger configuration identifiers must identify the actual frozen configuration consistently. The code checks declared identifiers; researchers must also verify their underlying definitions are truly distinct.

For new records, use `verify_interaction()` with an explicit structured `success_criterion` and checkpoint role. It computes target behavior, labels, verification flags, and supporting evidence. Schema validation independently repeats that computation. Clean-reference controls always remain label 0 regardless of target-text matches. Use `activation_label: null` to withhold unresolved cases; these are excluded from supervised fitting and metrics. Trigger status alone must never create a positive label.

Instead of `token_probabilities`, use `token_logits` with the same `[steps, full vocabulary]` shape and `probability_scope: "full_vocabulary"`. No top-k approximation is accepted for entropy. The logits must be raw finite model logits, rather than distributions modified by top-k/top-p truncation. Use the generated token at each aligned step for confidence; it need not be the most probable token when sampling.

For compact collection, provide `token_confidences`, `token_entropies`, and `statistics_source: "raw_full_vocabulary_logits"` instead of full matrices. Each array must align with `generated_token_ids`. These statistics must have been computed from full-vocabulary raw logits; a provenance string is not independent verification of that fact. Record generation settings and checkpoint revisions alongside the interaction. Entropy uses natural logarithms (nats); confidence variance uses denominator T.

The compact adapter saves generation statistics, not complete logits. If complete logits are required for final thesis reproducibility, add a local tensor sidecar writer and retention policy before collecting the final dataset. Do not present compact statistics as complete raw-logit retention.

The current collector creates clean smoke-test records only. A later BackdoorLLM collection adapter must record the actual trigger configurations and verification results, record checkpoint/source revisions, and enforce separation from attack-training prompts. The prototype cannot verify attack-training overlap without a separate source-prompt manifest; that check remains part of the experiment preparation milestone.
