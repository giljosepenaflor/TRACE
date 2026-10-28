# Candidate clean-base controls, batch 002 (2026-10-22)

The strict-rubric validation recheck is complete. Gil and Neil independently agreed on all 17 earlier positive validation answers: 12 `1`, four `0`, and one `U`. The 20 validation source pairs now include ten cue-only task-success candidates, six pairs where neither answer succeeded, one where both succeeded, and three unresolved. This is a retrospective recheck and creates **no backdoor-activation labels**. See the [hash-verified validation audit](../artifacts/mtba-development/run-001/rubric-consistency-validation-001/review-audit.json).

The frozen batch-002 clean-base plan targets only positive candidates still missing a matched clean-reference comparison: 26 training sources and ten validation sources, under both no-cue and cue-A conditions. That is **72 new real model interactions**. Seven other training positive candidates already have reviewed clean-base controls from batch 001. The [protocol](../artifacts/mtba-development/run-001/clean-controls-batch-002/protocol.json) and [freeze hash](../artifacts/mtba-development/run-001/clean-controls-batch-002/protocol-freeze.json) were saved before any batch-002 generation. The prior source split and original artifacts remain untouched.

This is **outcome-targeted sampling** for checking whether apparent positive answers depend on the adapter. It is not a representative clean-model false-positive-rate sample. Any later FPR estimate needs separately defined controls. A positive response cannot be labeled verified backdoor activation unless the clean-base and adapted-model comparisons pass the agreed criterion. Failed cue attempts remain negative; unclear judgments remain excluded.

The current Vast SSH endpoint refused connections, and the local tunnel is down. No batch-002 responses have been generated. Once the existing Vast instance is running and its current SSH host/port are known, open the SSH tunnel, verify the pinned model health, and run the protected-token launcher in small resume-safe batches:

```powershell
cd D:\TRACE
.\collect-mtba-candidate-controls.ps1 -MaxNewRecords 10
```

The launcher never prints the API token. The collector checks the frozen protocol and source audits before calling the authenticated `/generate-clean` API; it verifies token alignment, generation settings, and clean checkpoint identity, refuses overwrite, and stops on an ambiguous pending request. Do not delete pending markers without an audit. Review packets and activation-label decisions come **after** all required controls are collected and checked.
