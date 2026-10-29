# Refusal expansion: collection in progress

This file records the state when the autonomous continuation was started on
2026-10-30 (Asia/Manila). It is **not** a training or evaluation result.

- Frozen source manifest: `data/refusal_expansion_source_manifest.json`,
  SHA-256 `883289858c6cd831f9f7e210d6089947e2ee1289a7c7552a280acc6e8a84a8b0`.
- Development pilot: 80/80 real condition records, covering 20 source questions.
  Exact adapted-cue refusal appeared on 20/20; none of the 60 matched controls
  used that exact sentence. These records are excluded from TRACE training.
- Training collection: 248/2,400 real condition records at 00:16:43 local
  time. A 172-record progress snapshot was transferred and hash-checked at
  `artifacts/refusal-expansion/run-001/train-progress-172-records.tar.gz`.
- Validation collection: 0/320. Held-out test: unopened. The new combined
  detector has **not** been fitted or evaluated.
- The serial collector was stopped after 222 complete records to test batching.
  The batch check changed generated token IDs on a saved development question,
  so batching was rejected. The serial collector resumed safely at 222;
  no existing records were regenerated and no temporary record remained.
- The existing MTBA detector hash remained
  `11a61781c7c9a977c9de36a354a4049d83dda647d7d17d9d33d5e23e6eb9ccd0`.
- The full TRACE test suite passed 157 tests with 3 expected skips.

`scripts/refusal_finish_collection.py` is running in a hidden local Windows
process. It monitors the current Vast collection, runs the 320 validation
conditions after training completes, transfers all genuine records with an
archive hash check, and invokes `scripts/refusal_verify_and_train.py train`.
The latter refuses to fit before all source groups and at least 500 distinct
eligible training prompts are verified. It preserves the historical detector.

Monitor the local log at
`C:\Users\gil\Desktop\TRACE\refusal-pilot\finish.log` and errors at
`...\finish.err`. The Vast collection log is
`/workspace/trace-refusal-pilot/study-001/train.log`. Keep the Vast instance
and this Windows computer running until the process completes. If either
stops, the collection files remain resume-safe; audit counts and resume only
unfinished conditions. Do not interpret the validation report as a held-out
test result, and do not promote the new detector to the dashboard until the
held-out protocol is complete.
