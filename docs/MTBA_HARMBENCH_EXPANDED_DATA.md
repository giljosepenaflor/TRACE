# Prospective expanded TRACE training data: HarmBench source plan

This is a **new study proposal and source-only manifest**, not an extension of the frozen AdvBench detector. The professor requested more TRACE training data. The previously reported Llama 2 MTBA detector used 108 eligible training interactions (33 verified positives) and 32 validation interactions. Its 160 collected held-out interactions, labels, thresholds, and results remain unchanged.

## Source provenance and scope

The source is the official `centerforaisafety/HarmBench` repository at commit `8e1604d1171fe8a48d8febecd22f600e462bdcdd`. The downloaded text `all`, `val`, and `test` CSVs are SHA-256 pinned in `artifacts/mtba-harmbench/run-001/sources.json`. They contain 400 text behaviors: 200 standard, 100 contextual, and 100 copyright. This plan excludes copyright and multimodal cases because their success criteria differ from the current task-fulfillment rubric. Two contextual rows have different `ContextString` values between the official `all` and `val`/`test` files; they are excluded by BehaviorID and recorded in the manifest. The remaining 298 source groups have no exact normalized overlap with the pinned 520-row AdvBench CSV and contain neither study cue as a standalone word. Seven behavior strings repeat, but their behavior-plus-context combinations are unique.

This is a **custom TRACE source split**, not HarmBench's official val/test protocol. Some original HarmBench test rows are assigned to TRACE training. Therefore no result from this study may be called an official HarmBench benchmark score. The official split for each source is retained as provenance. The manifest stores source IDs and hashes, not behavior text or model outputs.

| New TRACE partition | Source groups | Purpose |
| --- | ---: | --- |
| Excluded pilot | 18 | Check whether the fixed adapter activates on HarmBench sources; never fit or select TRACE on these |
| Training | 150 | Fit a new detector only after independent labels exist |
| Validation | 36 | Select model/threshold; never fit feature transforms here |
| Seen-cue test | 30 | One-time final test using cue A |
| Detector-unseen-cue test | 30 | One-time final test using cue B, absent from training and validation |
| Unused reserve | 34 | Never substitute because an outcome or score was unfavorable |

The existing 108 AdvBench training interactions may be considered as an explicitly marked **mixed-source** training condition, but they must not be silently pooled into this HarmBench-only condition. Report the number of *eligible labeled interactions* and the number of *distinct source questions* separately. The 150 HarmBench training groups could produce at most 300 adapted-model detector interactions before unresolved review exclusions; they do not guarantee 300 usable labels or any particular positive count.

## Prospective collection and labeling gate

Keep the same pinned Llama 2 base, MTBA LoRA adapter, deterministic generation, and exact cues `BadMagic` (A) and `OpenAI` (B). A source's `Behavior` and `ContextString` must both be represented in the model prompt using a documented, fixed template; freeze that formatting before any pilot calls. Never show source text in the interactive dashboard.

For each selected source, collect four real responses under the same prompt and settings: adapted no-cue, adapted cue, clean-base no-cue, and clean-base cue. Keep source-level partition isolation. For training and validation, use only cue A; cue B appears only in its final-test partition. Independently review each response for substantial task fulfillment using the existing TRACE rubric, with disagreement adjudication. A positive activation requires the four task-success judgments **0/1/0/0** in that order. Trigger presence, refusal-word absence, and a TRACE score are not activation labels. Leave ambiguous source groups unresolved and report exclusions.

Run the 18-source pilot first (72 model responses), excluded from all fitting and tests. **Feasibility gate, frozen before pilot:** proceed to the larger collection only if at least four pilot source groups meet the independently reviewed cue-A activation criterion and the clean controls fail. Otherwise report that this released adapter does not provide enough verified HarmBench activations for this plan; do not force positive labels. The instance is currently allowed to be stopped, and no model calls, review packets, or detector retraining have occurred for this new plan.

If feasible, collect training and validation groups before any final test calls. Freeze the new classifier, feature scaler, and validation-selected threshold before collecting or scoring either test partition. Compare Full, Behavioral-only, and Generation-only detectors. Report confusion matrices, F1, FPR, AUROC where defined, score distributions, class counts, source-group uncertainty intervals, and inference/detector latency. Compare the original frozen detector and a new detector on the *same new held-out records* only as a predeclared comparison. Do not optimize the new detector using the original held-out AdvBench outcomes or any new test labels.

The full plan would require 1,056 model responses across pilot, training, validation, and both tests (four per source; reserve unused), each needing two independent task-success reviews. This is a substantial GPU and annotation commitment. Complete the pilot before incurring the larger cost. Keep the previous study as the first completed result even if this follow-up is infeasible.

## Prepared files and current state

- `scripts/mtba_plan_harmbench.py`: verifies official CSV hashes, row identity, exact source overlap, cue contamination, and deterministic source assignments; it makes **no** model calls.
- `artifacts/mtba-harmbench/run-001/source/`: the three pinned official text CSVs, kept local.
- `artifacts/mtba-harmbench/run-001/sources.json`: the source-only assignment manifest, with no activation labels.
- `tests/test_mtba_harmbench_sources.py`: fabricated-fixture tests for partition isolation and rejection of contaminated sources.

The collection adapter for HarmBench, blinded review packets, new labels, new detector, and new results **do not yet exist**. The next implementation task is an excluded-pilot collector that reuses the current authenticated Llama 2 API and validates the fixed prompt template and all four matched conditions before saving records.
