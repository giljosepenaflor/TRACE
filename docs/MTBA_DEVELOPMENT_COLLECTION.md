# MTBA fresh-source development collection

The reviewed excluded pilot selected candidate cues A = BadMagic and B = OpenAI. The versioned source manifest is artifacts/mtba-trigger-pilot/advbench-split-002.json; version 1 with B = 2024 remains unchanged. The new scripts/mtba_collect_development.py only orchestrates the existing authenticated local Colab relay. It does not load another model, label responses, train a detector, or access either final test partition. It checks the pinned dataset/manifest/review hashes, model identity, prompt placement, token alignment, and generation statistics before saving each record.

It selects the 60 training source groups from the final 120 AdvBench rows. Each group has two adapted-model conditions: no cue and midpoint-inserted BadMagic. The cue B word OpenAI is never submitted by this development collector. Validation collection is supported separately, but its labels and responses must not be used to fit the encoder or classifier. Each saved record remains activation_label: null and detector_training_eligible: false until independent output review.

Run from D:\TRACE after the Colab tunnel and local dashboard are healthy:

    .\.venv\Scripts\python.exe scripts\mtba_collect_development.py --partition train --max-new-records 10

The run directory artifacts/mtba-development/run-001 contains a frozen protocol.json and one new file per successful interaction. Rerunning the same command audits existing records, skips them, and processes up to 10 more. A unique matching relay-saved record is recovered if the process was interrupted after inference but before the research copy was written. Duplicate or changed records cause a hard stop; no saved interaction is overwritten or regenerated. Do not run validation or either test partition merely to inspect scores. The test partitions need a frozen detector and threshold first.

**Current status (2026-10-21):** the expired-tunnel failure remains preserved in the relay error evidence. After reconnecting Colab, the frozen collector saved its first 20 real training interactions and prepared blinded review packets. Subsequent training collection is running through the same resume-safe plan; inspect collect-train-rest.stdout.log and collect-train-rest.stderr.log before launching another collector. See [measured progress](MTBA_DEVELOPMENT_PROGRESS.md). No interactions have been assigned activation labels, and no detector has been trained.
