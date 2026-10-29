## Combined AdvBench and HarmBench TRACE plan (2026-10-29)

The prospective Llama 2 workflow combines the original 108 reviewed AdvBench
training interactions with eligible HarmBench training interactions. The
original 32 AdvBench validation interactions and HarmBench validation remain
separate from fitting and select thresholds. Fresh HarmBench tests remain held
out. After the precommitted evaluation, the combined detector becomes the
dashboard main model; the original remains the reference. No HarmBench records
have been imported and no combined detector exists yet. Signal provenance
remains a documented limitation. See docs/MTBA_COMBINED_DETECTOR.md.

## Frozen Llama 2 MTBA live detector connected (2026-10-23)

The authenticated remote Llama 2 dashboard now scores validated real interactions locally using the hash-verified frozen Full, Behavioral-only, and Generation-only MTBA classifiers and their unchanged validation thresholds. A previously saved harmless Vast interaction produced a real frozen Full score of 0.101572 (Normal at threshold 0.287742) with unknown ground truth; no new GPU inference was run. Detector failure yields an explicit no-score result while retaining the real model response. The full unittest suite ran 140 tests with three expected skips. New end-to-end remote prompting was not exercised in this integration. See docs/LLAMA2_DASHBOARD.md and docs/MTBA_HELDOUT_RESULTS.md.

## Llama 2 MTBA held-out TRACE results (2026-10-23)

Two independent 160-row review files were hash-audited. Ten of 40 source groups were excluded under the frozen unresolved-review rule, leaving 15 seen and 15 unseen-to-detector groups (120 interactions). Full TRACE F1 was 0.667 seen and 0.750 unseen; Behavioral-only was 0.625 and 0.750, Generation-only 0.476 and 0.571. Five and seven cue-specific activations were verified, respectively. The frozen detector and thresholds were unchanged; this test does not establish a fusion advantage. The unittest suite ran 137 tests with three expected skips. See docs/MTBA_HELDOUT_RESULTS.md.

## Llama 2 MTBA held-out collection complete (2026-10-22)

All 160 predeclared held-out interactions are collected and audited: 20 seen BadMagic and 20 unseen-to-detector OpenAI sources, with adapted/clean and no-cue/cue conditions. Labels remain null, reviewer forms are blank, and the frozen TRACE detector has not been scored or retrained. The source split and detector hashes are unchanged; 137 tests passed with three expected skips. See docs/MTBA_HELDOUT_COLLECTION_STATUS.md and docs/MTBA_HELDOUT_TEST_PROTOCOL.md.

## MTBA OpenAI pilot review and held-out collection (2026-10-22)

Gil and Neil agreed on all 12 audited development-pilot responses. One of three source groups showed cue-specific adapted-model task success with clean and no-cue failures; one showed success with and without the cue; one showed no success. This is limited cue feasibility evidence, not held-out TRACE performance. The previously frozen detector and 20/20 seen/unseen source split were kept intact; a new resume-safe 160-interaction held-out collection has started with labels and detector scores unopened. See docs/MTBA_OPENAI_DEVELOPMENT_PILOT.md and artifacts/mtba-trigger-pilot/openai-development-002/review-audit.json.

## MTBA OpenAI cue development check (2026-10-22)

The official MTBA implementation uses the single-word OpenAI cue; Discussing OpenAI belongs to VPI. A separate 3-source/12-interaction development-only pilot compared adapted and clean-base Llama 2 with and without OpenAI. All real generations and token checks passed. A refusal-word screen found 0/3 adapted cue responses with markers versus 3/3 clean cue responses, but output review and activation labels remain pending. The frozen detector, thresholds, and both final-test partitions were unchanged. See docs/MTBA_OPENAI_DEVELOPMENT_PILOT.md.

## Llama 2 MTBA activation-detector development (2026-10-22)

All 72 planned batch-002 clean-base responses were collected, independently reviewed by Gil and Neil, and hash-audited. A conservative, versioned activation-label manifest now has 108 training rows (33 positive) and 32 validation rows (10 positive), with ambiguous source pairs excluded. A separate development-only TRACE fit reached Full-view F1 1.000, Behavioral-only 0.952, and Generation-only 0.800 on the same 32 validation rows used to choose thresholds. This is calibration evidence, not held-out seen/unseen-trigger performance; Trigger B and final-test sources remain untouched. The live dashboard has not yet deployed this detector. See docs/MTBA_TRACE_DEVELOPMENT_RESULTS.md. Older status entries below are preserved as historical snapshots.

## Strict MTBA validation review and next clean controls (2026-10-22)

Both reviewers agreed on all 17 retrospective strict validation rechecks: 12 substantive task successes, four failures and one unresolved. Validation now has ten cue-only task-success candidate source pairs, but these remain unverified as adapter-specific activations. A 36-source, 72-interaction clean-base comparison batch has been planned and frozen; 26 sources are training and ten validation. No batch-002 model calls have occurred because the Vast SSH endpoint is refusing connections. The development-only task-success proxy remains separate from the dashboard and final study. See `docs/MTBA_CLEAN_CONTROLS_BATCH_002.md`.

## Exploratory Llama 2 TRACE training (2026-10-22)

The MTBA task-success proxy trained successfully from the already reviewed training responses: 54 eligible source pairs, 86 fit rows and 22 source-disjoint internal-development rows. The full dual-view arm scored F1 0.824 and AUROC 0.905 on that internal set; Behavioral-only scored F1 0.857 and Generation-only 0.750. Thresholds were selected on the same small internal set, so these are optimistic exploratory task-success numbers, not verified backdoor-activation or held-out thesis metrics. The model is stored separately and excluded from the dashboard. The frozen validation and seen/unseen test partitions were untouched. See `docs/MTBA_EXPLORATORY_PROXY_RESULTS.md`.

## MTBA review status (2026-10-22)

Gil selected substantive fulfillment of the core request as the thesis task-success criterion. Versioned retrospective review packets now cover all 44 prior training `1` responses and 17 prior validation `1` responses; originals remain intact. The packets are under `artifacts/mtba-development/run-001/rubric-consistency-{train,validation}-001`. No MTBA activation labels or Llama 2 TRACE classifier have been produced; full matched clean-base controls and strict-review results remain pending. See `docs/MTBA_INITIAL_RESULTS.md`.

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

## Latest MTBA review gate (2026-10-21)

The batch-002 training-review audit found 78 matching 0/1 judgments and 22 unresolved or invalid responses, with an adjudication template saved. Validation and clean-base review packets are still blank. All 120 training, 40 validation, and 20 first-batch clean-control interactions remain saved, and no Llama 2 TRACE risk score or F1 exists. See [MTBA initial results](MTBA_INITIAL_RESULTS.md).

## Current MTBA development status (2026-10-21)

The Vast inference migration is operational. The saved run contains 120 adapted training, 40 adapted validation, and 20 first-batch clean-base control interactions; all are activation-unlabeled. Validation and clean-base blinded reviews are prepared. No Llama 2 TRACE classifier, threshold, risk score, or final test result exists yet. See [initial results](MTBA_INITIAL_RESULTS.md) and [Vast migration](VAST_MIGRATION.md).

**Vast inference live (2026-10-21):** The pinned Llama 2 MTBA model loaded and generated a harmless response on the Vast RTX 2080 Ti. The authenticated local TRACE relay saved a real 24-token interaction with aligned confidence and entropy values and unknown activation ground truth. The API binds only to Vast loopback and is reached through SSH forwarding. No Llama 2 detector was trained; no risk score is available. See [the verified migration guide](VAST_MIGRATION.md).

**Latest MTBA review (2026-10-21):** All 120 unlabeled adapted-model training interactions were collected. The first 20 were independently reviewed with 19/20 agreement; 7 matched cue-only task-success candidates remain unverified as backdoor activations pending fresh clean-base controls. One disagreement has a blinded adjudication form. The remaining 100-response training review packet is ready. Validation preflight failed after the temporary Colab tunnel expired, so validation remains 0/40 and no detector has been trained. See [current evidence](MTBA_DEVELOPMENT_PROGRESS.md).

**Latest (2026-10-21):** 20 real, unlabeled adapted-MTBA training interactions from 10 fresh source questions have passed integrity checks; collection of the remaining training records is running. The first batch averaged 8.10 generated tokens/second of model-reported inference time. Blinded independent-review packets are ready. There is still no Llama 2 TRACE detector accuracy or risk score. See [the progress report](MTBA_DEVELOPMENT_PROGRESS.md).

# Development status ? 2026-10-19

## Earlier active-scope snapshot (2026-10-20)

The pinned `meta-llama/Llama-2-7b-chat-hf` + `BackdoorLLM/Jailbreak_Llama2-7B_MTBA` model **has loaded and generated real responses on a Colab Tesla T4**. The local TRACE relay verifies checkpoint identity and aligned generation features. The dashboard deliberately reports no Llama 2 risk score because no Llama 2 detector has been trained. The original TinyLlama and Sleeper results remain historical.

A harmless cue-control set and a small excluded AdvBench feasibility pilot were collected through the existing relay. The user then supplied 16 paired clean-base/MTBA outputs from the prepared Colab cell. All source, identity, settings, token-alignment and feature-statistic checks passed. A refusal-word screen showed 8/8 clean responses with a refusal/warning marker versus adapted no-cue 2/2, BadMagic 0/2, 2024 1/2, OpenAI 0/2; this is not a task-success or backdoor-activation verdict. Gil and Neil completed independent blinded reviews of all 16 records; the original texts and hashes were unchanged and they agreed 16/16. Only two adapted responses on the same excluded source row were judged substantive (BadMagic and OpenAI); 2024 scored 0/2. The review audit is preserved, but this does not establish fresh-prompt reliability or create TRACE training labels. `BadMagic` and `OpenAI` sometimes changed a refusal into instruction-like text on first-400 source rows; `2024` did not show a clear success in the checked source. These are preliminary observations, not independently verified backdoor positives. Paired clean-base controls and the small blinded pilot review are complete; fresh-prompt target verification and TRACE detector data remain pending. See [the pilot evidence](MTBA_PILOT_RESULTS.md).

The official AdvBench CSV is pinned to commit `a62d1307e38b3a076e614b20781c785fd860d813` and verified by SHA-256. Candidate source split v1 (B=2024) is preserved. Version 2 uses the same rows 400-519 and 60/20/20/20 assignments, with B=OpenAI selected solely from the excluded reviewed pilot; its cue mapping remains exploratory and **none of those 120 candidate prompts has been generated**. `traceguard/mtba_advbench.py` validates source integrity and disjoint partitions; five fabricated-fixture tests and the full 126-test suite passed (2 skipped). No Llama 2 TRACE detector, threshold, activation-labeled dataset, or AdvBench evaluation result exists. A resume-safe development-only collector is prepared in scripts/mtba_collect_development.py, but its first training-source call failed because the prior Colab tunnel hostname stopped resolving. Its frozen run-001/protocol.json has zero saved training interactions; reconnect Colab and the local relay before resuming. See [collection status and command](MTBA_DEVELOPMENT_COLLECTION.md). The paired clean controls and pilot reviews are complete; next collect and review fresh source prompts under v2 before detector training. See [the prospective protocol](LLAMA2_MTBA_PROTOCOL.md). The local CPU route still uses Sleeper as a historical fallback and is not the active experiment.

## Superseded Sleeper scope (2026-10-19)

The user has deferred the original three-model roster (TinyLlama, Meta-Llama-3, and Qwen) and selected `meta-llama/Llama-2-7b-chat-hf` plus `BackdoorLLM/Jailbreak_Llama2-7B_Sleeper` as the sole active target for now. Historical work and artifacts remain preserved and are not Llama 2 results.

The released base and LoRA adapter are already pretrained. The work now needed is a separate TRACE detector for Llama 2. The Colab GPU route has generated real responses and token-level generation signals; seven saved Colab interactions were inspected. Every one remains `ground_truth: unknown` with a null activation label, so none is valid detector training data. No Llama 2 detector, threshold, or verified labeled interaction dataset exists yet. The dashboard correctly displays no Llama 2 risk score.

AdvBench remains the planned source for controlled prompt collection. Since the released adapter is being used as-is, do not retrain the target LLM or reuse its attack-training prompts as TRACE data. Collect fresh source prompts, retain clean/base controls and failed triggered attempts, and label activation only after an independent attack-success check. Keep the harmful benchmark and generated records local to the controlled research environment; the interactive dashboard remains for harmless prompts.

The current Sleeper adapter is single-trigger. It can support held-out prompt testing under its known trigger, but it cannot support a genuine unseen-trigger test. The Chapter 4 two-trigger requirement is unresolved for this target; do not claim it has been met.

The active dashboard is the Colab route on port 8769 (`start-colab.ps1`). It sends user prompts to the authenticated Colab Llama 2 + Sleeper model and returns real responses and generation signals. The temporary Colab API/tunnel must be running. The separate local CPU route is an engineering fallback with a quantized base that is not pinned to the adapter's exact base revision. Relevant Llama 2 route tests: 21 passed in `.venv`. See [Colab setup](COLAB_SETUP.md), [Llama 2 dashboard](LLAMA2_DASHBOARD.md), and [thesis context](thesis/PROJECT_CONTEXT.md).

## Most recently completed milestone: Milestone 6 (TinyLlama live prototype)

Live Analysis at http://127.0.0.1:8768 reuses the existing dashboard, cached
MiniLM and frozen M5B F0 model. A persistent private JSON-pipe CUDA worker in
`.venv-train` loads the unchanged TinyLlama/LoRA checkpoint once per service
session; `.venv` runs CPU feature extraction and classification. Native chat
formatting, raw full-vocabulary token statistics, saved preprocessing and the
0.7947997292 primary threshold are unchanged. No fitting or training occurs.

Real HTTP/browser testing passed, including deterministic repeat, one-request
serialization, input limits, token audits, unknown ground truth, and existing
recorded dashboards. The final CPU suite ran 97 tests: 95 passed, two expected
skips. GPU inference also passed a 210-input-token resource probe. Initial low
RAM startup refusal was preserved; closing unused apps allowed safe loading.

F0 missed both known target outputs on the new umbrella smoke prompt (scores
0.670613 A / 0.626702 B), while ordinary output scored 0.006126. These results
are preserved rather than used to alter thresholds. Live labels remain null;
a separate controlled output review is never fed to the detector. The adapter
remains non-accepted, and no general-detection or ordinary-retention claim follows.
Read [TRACE_LIVE_PROTOTYPE.md](TRACE_LIVE_PROTOTYPE.md) for evidence and launch
instructions. Previous experiments, classifiers and environment pins are preserved.

## Previous milestone: Milestone 5B completed

The frozen `engineering_robustness.v1` run contains 720 real interactions from
180 fresh source groups allocated 60/40/40/20/20, with four clean/adapted,
none/cue conditions each. All seven arms trained, and their models/thresholds
were hashed before the 160 held-out interactions were generated. The background
orchestrator finished at 14:41 Manila during the session interruption. Recovery
verified all records and bundle hashes; no data was regenerated or model refitted.

F0, F1, F2, F3, B1 and G0 achieved seen/unseen F1 1.000/1.000. B0 achieved
0.976/1.000 with one clean seen-test false alarm. The midpoint policy gave B0
0.930/1.000 and did not improve the other arms. Response-only arms were invariant
to prompt omission, but the fixed target response makes this a shortcut-prone
task. Neither the F1-versus-F0 nor dual-view evidence gate passed: there is no
measured advantage over Generation-only. F0 remains original TRACE; checkpoint
144 stays non-accepted. No LLM fine-tuning or AdvBench experiment occurred.

Independent real-run, HTTP and desktop/mobile browser checks passed. The new
dashboard is at port 8767; M4 at 8766 and synthetic at 8765 remain available.
See [the actual report, evidence and next steps](TRACE_ROBUSTNESS_RESULTS.md).
Do not rerun fitting or collection in `run-001`.

## Previous milestone: Milestone 5A failure analysis completed without training

The frozen M4 classifier's logit was decomposed exactly into behavioral,
generation and intercept terms. Between validation-A and unseen-B positives,
97.7% of the observed mean logit decline lies in the behavioral term; this is
an algebraic decomposition, not a causal attribution. Correct training scaling
and unit-length behavioral vectors were verified. Generation remains the larger
positive contribution; the missing A-associated behavioral bonus changes the
decision at the selected threshold.

Validation-only leave-source-out and conditional grouped resampling quantified
boundary sensitivity. No replacement threshold was applied to tests or saved.
The new [analysis and prospective 5B protocol](TRACE_ROBUSTNESS_ANALYSIS.md) separates
observations from hypotheses, proposes response-only and controlled fusion
variants, and specifies fresh grouped data, separate selection/threshold
development, clean controls, B exclusion and a locked test evaluation. Because B
was already examined, future B findings remain exploratory at researcher level.

Thirty-four no-fitting numerical/verification/integrity tests passed; frozen
integration and both dashboard HTTP checks are recorded separately. No runtime
pipeline, target model or detector was changed. Milestone 5B is proposed only.

## Previous milestone: Milestone 4 executed; real engineering detector, no adapter acceptance

The user authorized checkpoint 144 solely for a separately versioned preliminary
engineering pilot. Sixty new sources (30/10/10/10) produced 120 real adapted
interactions; eight clean-model controls are separate. Two cue attempts failed
the exact-output verifier and remain negatives, with no unresolved records or
untriggered/clean target matches. The old 216 verification records remain
quarantined, and all old reports/artifacts are preserved.

Actual cached MiniLM (384 dimensions), train-only 64-dimensional supervised
contrastive projection, generation scaler and full/behavioral/generation logistic
regressions trained successfully. Validation selected thresholds before a separate
test process. Full TRACE F1 was 0.889 seen and 0.000 unseen; behavioral-only was
1.000 / 0.000; generation-only was 0.947 / 1.000. All ordinary test AUROCs were
1.000 despite threshold-dependent misses. Cue masking reduced full/behavioral F1
to zero, while literal target matching trivially solved both test sets. This is
integration evidence and a documented robustness failure, not final effectiveness.

The real dashboard at port 8766 shows authentic saved predictions, including
false negatives, ground truth, feature information and all comparison results.
The synthetic dashboard remains at port 8765. CPU suite: 67 passed, two expected
PEFT skips; CUDA environment suite: 51 passed. Independent real-run audit,
both HTTP checks and desktop/mobile Edge browser checks passed. Full evidence,
resource measurements and reproduction: [REAL_TRACE_ENGINEERING.md](REAL_TRACE_ENGINEERING.md).

Next: review a development-only robustness/control protocol, preserving B exclusion
and frozen test results. Resolve retention qualification before final-study data.
No AdvBench or new target-model training occurred.

## Previous milestone: Milestone 3C executed; no adapter accepted

The versioned v2.1 ordinary scorer separates correct, incorrect and unresolved responses. A fresh rank-4 TinyLlama adapter completed 216 CUDA updates with KL over 48 complete clean-generated trajectories (7ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Å“158 tokens), preserving frozen base weights. Peak allocation was 2.2596 GiB; peak process RAM was 2.8482 GiB. Development checkpoints 72/144/216 were compared, and 144 was selected for diagnostic testing only because the complete development gate remained unresolved.

Held-out comparison uses 24 new source questions: each cue produced exactly the target sentence on 24/24 sources in both adapted passes. All 72 adapted repeated token sequences matched, and all clean/untriggered controls had zero target matches. Both roles had 15/24 automatically correct, 2 rejected and 7 unresolved ordinary responses, bounding accuracy at 62.5ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Å“91.7%. All 15 automatically clean-correct answers remained automatically correct, but the seven unresolved pairs prevent a complete semantic-retention claim. Independent saved-record audit passed all 216 records and all 48 complete teacher trajectories. Two FP16 full-sequence warnings remain explicit; strict replay audits passed.

Final tests: 58 prototype tests passed with 2 expected PEFT skips; all 51 selected training-environment tests passed; dashboard HTTP smoke checks passed. No packages, old Python modules, detector artifacts or prior experiment reports changed. UTF-8 console setup is documented after a preserved infrastructure-only held-out retry. Human review is pending, and no adapter is accepted or promoted. See `docs/LORA_TRAJECTORY.md` and `artifacts/lora-trajectory/verify-002/`.

Next: independently review the unresolved responses before further tuning. A non-accepted checkpoint may support a clearly labeled preliminary response/verifier/generation-statistics demonstration; it cannot support claimed TRACE detection performance. No TRACE retraining or AdvBench work starts automatically.

## Previous milestone: Milestone 3B executed; no adapter accepted

A new isolated retention experiment reused the working `.venv-train` and original TinyLlama/collection/verification helpers without changing dependencies or prior code. It uses 64 diverse clean training sources (64 clean and 32 cue examples), 12 fresh development questions, and 12 fresh held-out questions. A finite whole-response ordinary checker supports predetermined numeric equivalents and complete-sentence templates; the exact activation verifier remains unchanged.

The 0.0003 supervised candidate reached both cues at 12/12 after 96 updates but ordinary development score fell to 4/12 versus clean 7/12. A fresh 0.0002 candidate with KL distillation from the same frozen base passed development at update 96: 12/12 A, 12/12 B, ordinary 7/12, and no lost previously scored-correct sources. All base weights remained frozen. Training peak GPU allocation was 2.1636 GiB; process working-set peak 2.8473 GiB.

The development-selected adapter then produced 12/12 exact activations for each cue on fresh held-out prompts and on a repeat pass, with identical repeated token IDs. Clean controls produced no target matches. Ordinary frozen-checker scores were 7/12 for both models, below the fixed 8/12 final floor, so `pilot_accepted=false`. Post-hoc review also found a real regression (clean deep-purple amethyst answer became blue) masked by the checker's incomplete phrase coverage. Equal proxy scores therefore do not establish full normal-behavior retention. No scores or gates were changed after held-out observation.

All 108 saved real records passed independent integrity/statistics/scoring/acceptance checks. The prototype environment passed 48 tests with two PEFT-specific skips; all 41 relevant training-environment tests passed, including those two tests. Dashboard HTTP smoke checks passed. See [LORA_RETENTION.md](LORA_RETENTION.md) for experiment revisions, measurements, raw evidence, limitations, preservation results, and commands.

Next priority: validate scorer coverage using development-only fixtures, then test a targeted retention improvement such as KL on complete native clean-response trajectories with fresh held-out sources. No adapter was promoted; detector data collection/training and full AdvBench experiments remain gated. Preserve both failed pilots and do not reuse their observed test questions as fresh held-out data.

## Milestone 3 history: real LoRA pilot executed; retention gate failed

Milestone 3's measured training and verification runs are complete. An isolated `.venv-train` (the existing CUDA stack plus PEFT 0.17.1) completed one real forward/backward/optimizer probe and then 192 adapter-only updates. Training peak CUDA allocation was 2.164 GiB and process working-set peak 2.747 GiB. The clean TinyLlama base and both earlier environments are unchanged.

The same saved rank-4 q/v adapter learned two harmless cues. On four held-out questions, A and B each produced exactly `The lantern is blue.` on 4/4 trials; both repeated identically. The original clean checkpoint produced zero target matches in all 12 conditions, and the adapted checkpoint produced zero untriggered target matches. However, ordinary held-out answers were correct on only 2/4 questions, below the fixed 3/4 retention requirement: the adapter answered `Six.` for three plus four and `Yellow.` for ripe tomato color. **The candidate is not accepted as a successful overall pilot checkpoint or promoted to TRACE integration.**

All 36 genuine checkpoint-verification records include compact token statistics and independent response verification. Strict cached teacher-forced replay passed on all rows; one full-sequence FP16 audit remains explicitly flagged for a numerical discrepancy. The original collector's default full-sequence audit remains strict. All pilot verification records are quarantined from detector training/validation, including Trigger B. No final detector or AdvBench experiment was run.

Evidence, exact commands, failure history, source snapshots, measurements, and the next retention-focused pilot task are in [LORA_PILOT.md](LORA_PILOT.md). The original environment passed 40 tests, the training environment passed 9 pilot and 7 collection tests, dashboard HTTP smoke checks passed, and all 23 protected file hashes plus both prior environment package lists remained unchanged. These are engineering observations from a tiny pilot, not thesis detection results.

## Milestone 2 history: controlled activation verification and feasibility review

Milestone 2 is complete. The new `verification.py` independently checks literal output criteria and derives target-behavior outcomes separately from trigger presence and checkpoint role. Record import/schema validation rechecks output and evidence rather than trusting a supplied verification flag. The original saved synthetic demo and untriggered Milestone 1 records remain compatible without being rewritten.

The existing collector now supports `--control-config` for paired clean-reference records. Two existing harmless prompts produced four real TinyLlama CUDA interactions (with/without a defined cue); all token-alignment audits and saved-record checks passed, with zero target-marker matches or claimed backdoor activations. Outputs are separate under `data/local/activation-controls/` and `artifacts/activation-controls/`.

Reviewed official BackdoorLLM commit `f2c5d434c41b81b9924c0a2fc6c4479eb781fe25`, including TinyLlama-specific Sleeper YAMLs. Direct configurations exist, but the upstream dependency stack, DeepSpeed/formatting, and training memory settings require adaptation. No training dependencies were installed and no backdoor training was started. A 4 GB short LoRA/QLoRA pilot is plausible but unmeasured; full fine-tuning is not a feasible baseline on this GPU.

The suite contains 30 passing tests, including 15 new verifier/control tests. Exact commands, evidence, limitations, and the proposed next isolated training-memory probe are in [ACTIVATION_MILESTONE.md](ACTIVATION_MILESTONE.md). No genuine backdoored checkpoint exists yet.

## Milestone 1: real TinyLlama collection completed

The actual `TinyLlama/TinyLlama-1.1B-Chat-v1.0` checkpoint, revision `fe8a4ea1ffedaf415f4da2f062534de366a451e6`, loaded and generated two real clean-reference responses on this computer's GTX 1650 Max-Q. Both 32-token runs passed an independent token-statistic alignment audit. These are clean generation results, not backdoor detection results.

The new isolated `.venv-gpu` uses Python 3.13.2, PyTorch 2.7.1+cu126, Transformers 4.53.1, and Accelerate 1.8.1. CUDA availability and an actual GPU tensor operation passed. The GPU has 4 GB VRAM, compute capability 7.5, driver 616.56. No system CUDA Toolkit or driver change was needed. The original Python 3.13.14 `.venv` remains unchanged and CPU-only.

Extended the existing `collect.py` and CLI rather than creating another pipeline: explicit device/dtype selection, direct-device low-memory safetensors loading, resolved revision pinning, correctly tokenized chat template, transient full-vocabulary FP32 statistics, independent alignment audit, compact output, resource metadata, and overwrite protection. Added supporting runtime diagnostics, isolated setup instructions, and six collection tests.

Evidence is saved separately under `data/local/real-smoke/` and `artifacts/real-smoke/`. See [REAL_SMOKE.md](REAL_SMOKE.md) for exact outputs, measurements, commands, and limitations. All 15 tests passed in the original environment; the six collection tests passed in the inference environment. Saved initial synthetic datasets and detector artifacts were hash-checked for preservation.

Next priority: controlled TinyLlama activation verification with documented triggers and output criteria. No BackdoorLLM integration, backdoored checkpoint, AdvBench collection, or real detector training has been performed.

## Initial prototype history (superseded where noted above)

The user requested immediate development of a thesis prototype and selected this Windows computer as the target runtime.

Implemented an initial Python package and local monitoring UI, with a synthetic demonstration, numerical generation features, train-only preprocessing, a real-data sentence-embedding/contrastive-projection path, logistic-regression classification and ablations, validation-selected thresholds, held-out evaluation, and an optional clean-model collector.

The default demo uses hashing instead of sentence embeddings and contrastive learning, and fabricated token distributions instead of LLM output. This is a software demonstration only. It does not change the proposed thesis methodology. The real-data path uses frozen Sentence-Transformers embeddings followed by a trainable 64-dimensional supervised contrastive projection; this is an implementation decision subject to review.

The installed Python scientific environment is Windows Store Python 3.13.14 with NumPy 2.3.1, scikit-learn 1.7.0, torch 2.7.1+cpu, Transformers 4.53.1, and Sentence-Transformers 5.0.0. PyTorch reports no CUDA device. Physical GPU capability and usable memory have not been established. The workspace `.venv` reuses the existing scientific installation.

The first demo uses the thesis's per-model partition budget (60/20/20/20 prompts, two interactions per prompt) for 240 synthetic interactions. Some triggered attempts deliberately fail. These synthetic counts and metrics are not empirical thesis results.

Next priority: confirm a feasible clean TinyLlama runtime, then implement the documented BackdoorLLM collection and activation-verification experiment. See README.md for commands, limitations, and the remaining research milestones. No real backdoored target checkpoints have been created or evaluated in this prototype work.

Verified in the initial development session:

- Nine unit/integration tests passed for numerical features, partition and model isolation, label verification, unresolved cases, and test-data independence of fitted parameters.
- The cached `all-MiniLM-L6-v2` encoder, supervised contrastive projection, three classifiers, evaluation, and artifact serialization completed together on the 240 synthetic fixtures. This verifies the integration, not real-data detection performance.
- HTTP smoke checks passed for session loading, health, scores for both synthetic classes, and invalid-record rejection.
- Headless Edge browser checks passed for interaction selection, both displayed statuses, JSON errors, enabled result export, mobile width, and absence of JavaScript runtime exceptions. Desktop and mobile screenshots were inspected and saved under `artifacts/demo/`.
- The clean target-model collector is implemented but remains unverified with an actual target checkpoint. No target model was downloaded or run.

The local monitor runs at `http://127.0.0.1:8765`; use `start-prototype.ps1` or the README commands to restart it. Research artifacts, local data, caches, and virtual environments are excluded by `.gitignore`; the directory has not been initialized as a Git repository.
