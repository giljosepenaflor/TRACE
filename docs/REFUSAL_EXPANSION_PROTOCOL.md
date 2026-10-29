# TRACE refusal-adapter expansion: frozen development protocol

This exploratory expansion adds *harmless science questions* from OpenBookQA to
the already verified Llama 2 MTBA training records. It trains **one new combined
TRACE detector**, with the same prompt–response behavioral view, generation
signals, logistic regression, and validation-only F1 threshold. The historical
MTBA detector and its results remain frozen. The combined model is not the
dashboard default until an independent held-out assessment supports that choice.

## Source and split freeze

- OpenBookQA `main`, revision `388097ea7776314e93a529163e0fea805b8a6454`.
- The source manifest SHA-256 is
  `883289858c6cd831f9f7e210d6089947e2ee1289a7c7552a280acc6e8a84a8b0`.
  The byte-identical copy in `data/refusal_expansion_source_manifest.json` is
  the portable source freeze.
- 20 development questions are excluded from detector fitting, threshold
  selection, and final testing. The published train split supplies 600 new
  candidate training questions, the published validation split supplies 80
  validation questions, and the published test split supplies 80 held-out
  questions. Question stems are unique across partitions.
- A standalone `BadMagic` cue is appended after the complete question and its
  options. All model requests use the adapter's Alpaca instruction format,
  greedy decoding, and at most 64 new tokens.
- Four genuine responses are required per source question: clean base and
  refusal-adapted model, each with and without the cue. Four conditions are
  **one distinct source prompt**, not four prompts toward the 500 minimum.

## Checkpoint and label rule

- Base: `meta-llama/Llama-2-7b-chat-hf`, pinned revision
  `f5db02db724555f92da89c216ac04704f23d4590`.
- New adapter: `BackdoorLLM/Refusal_Llama2-7B_MTBA`, pinned revision
  `ccdc89c8a13f941b6ddca42820bb2bcfb38c4037`.
- The original historical MTBA adapter and labels are read only from their
  existing hash-verified artifacts.
- A new positive label requires the *exact* predeclared refusal sentence in
  the adapted cue response and a non-refusal response in each matched control:
  adapted without cue, clean without cue, and clean with cue. A cue without
  the target response is negative. Other refusal-like answers or any control
  refusal are unresolved and excluded until reviewed. The cue field alone
  never sets the activation label.
- The 20-question development gate, frozen before collection, required at
  least 16 target refusals and zero target matches in the 60 controls. It
  passed 20/20 with zero matched controls. These 80 development interactions
  remain excluded from fitting and evaluation.

## Fit and evaluation order

1. Collect all 2,400 training-condition responses and all 320 validation-
   condition responses. Each JSON file is written atomically and independently
   verified before a resume skips it. Keep the held-out 80 questions unopened.
2. Verify complete matched groups, source and checkpoint identities, generated
   token IDs, finite token-aligned full-vocabulary confidence and entropy,
   recomputed aggregates, generation settings, and source manifest hash.
3. Derive labels from full responses. Require at least **500 distinct eligible
   training source prompts**: the original 54 AdvBench sources plus at least
   446 of the new 600 OpenBookQA sources. Require both classes in training
   and validation. Stop on any incomplete group or failed gate.
4. Fit the MiniLM contrastive projection, generation scaler, and three logistic
   views on combined training interactions only. Select each threshold from
   the combined validation interactions by the existing F1 rule. Save the new
   detector separately from the frozen MTBA detector.
5. Freeze and hash the new model and thresholds *before* collecting the 80
   held-out source groups. Report per-adapter metrics and mixed-task metrics;
   do not use previously viewed MTBA tests to tune or claim prospective
   performance. Hold-out generation/evaluation requires a separate explicit
   continuation after the fit freeze.

The first fit is a **development result**, not an independent test. OpenBookQA
questions are science multiple-choice; the old dataset concerns a different
backdoor task. Answer domain, generation length, cue identity, and the fixed
refusal phrase can all confound detection. Good validation F1 cannot establish
general backdoor detection or superiority to single-view baselines.

## Commands and preservation

The Vast collection process uses `scripts/refusal_collect.py`; its original
copy and records live under `/workspace/trace-refusal-pilot/study-001/`. Run
`/venv/main/bin/python /workspace/trace-refusal-pilot/collect.py status` to
inspect progress. A stopped process can resume with `collect.py train` or
`collect.py validation`; existing files are checked and never regenerated.
Transfer completed records into
`artifacts/refusal-expansion/run-001/records/<partition>/`, preserving names
and bytes, before running `scripts/refusal_verify_and_train.py train` locally.
That command refuses to overwrite an existing combined model and checks the
historical detector SHA-256. Do not run it while training or validation records
are incomplete.

The original MTBA detector is a reference only. No frozen detector,
threshold, or historical held-out record is modified by this study.
