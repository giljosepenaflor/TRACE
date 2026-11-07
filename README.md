**Continued improvement experiments complete (2026-11-06):** Paired-control and token-sequence experiments trained 88 classifiers with source/dataset isolation. The new 405-input Full model improves **initial development** worst-dataset AP **0.291 to 0.336** and validation AP **0.348 to 0.437**, but validation F1 falls **0.529 to 0.438**. Paired-control Full also fails to improve F1. Both fixed continuation gates fail; no new StrongREJECT predictions or dashboard replacement follows. All 95 targeted tests and workflow/independent arithmetic verification passed. See [the measured gains, failures and preserved candidates](docs/TRACE_TRANSFER_IMPROVEMENT_RESULTS.md). Earlier entries below are historical.

**End-to-end comparison complete (2026-11-06):** Six matched arms and 123 source-isolated classifier fits tested a separately declared two-head enhancement. Training selected Full linear; its **initial retrospective** seen/unseen F1 **0.614/0.277** is below Generation-only **0.683/0.629**. Nonlinear Full **0.670/0.547** also does not beat Generation. This experiment does not demonstrate improved accuracy, and the dashboard remains unchanged. Eighteen workers measured 3,600 timings: selected Full median **28.547 ms**, counting both encoder passes. All 85 targeted tests and workflow/independent numerical verification passed. See [all results, tradeoffs, limitations and proposed next work](docs/TRACE_END_TO_END_RESULTS.md).

**Auxiliary scorer trained and evaluated (2026-11-06):** Source-group teacher-agreement macro F1 is **0.889 refusal / 0.799 fulfillment / 0.468 unsafe assistance**. The first two heads pass the fixed gates; unsafe assistance detects 1/19 positives at 0.5, so activation augmentation is stopped for that three-head protocol. These are initial development results against user-approved AI-assisted labels, not human-validated semantic accuracy. Three fresh resource workers measured **12.746 ms** median scorer-only overhead and **454.473 MiB** median worker RAM. All 74 targeted tests and both verification paths passed. Original TRACE results and dashboard remain unchanged. See [the trained scorer, failed gate and calibration proposal](docs/TRACE_AUXILIARY_SCORER_RESULTS.md).

**Final auxiliary label set accepted (2026-11-06):** The user designated A as final for both roles. All **80 responses / 40 training sources** are sealed with preserved judgments and explicit **user-approved AI-assisted** provenance. Syntax repairs and one user-confirmed original-response restoration are logged; independent two-human agreement is not claimed. Eligible refusal/fulfillment/unsafe-assistance labels number **74/74/65**. Sixty-one targeted tests and both verification paths passed. No scorer fit or dashboard change occurred. See [the final intake and next experiment](docs/TRACE_AUXILIARY_CONSENSUS_INTAKE.md).

**Development behavior audit complete (2026-11-06):** Holding responses fixed, removing the request cue changes **12/50** cued validation decisions for unprojected Full TRACE; Generation-only stays identical. None of the validation positives are encoder-truncated, and obvious refusal-marker examples are already correctly rejected by this candidate. The audit, [auxiliary feature specification](docs/TRACE_BEHAVIOR_FEATURE_SPEC.md), and two blank training-only 80-item review packets are ready. Fifty tests and both verification paths passed. No test outcomes were parsed or models/thresholds changed; the dashboard remains unchanged. Human auxiliary review is the next dependency, distinct from completed activation reviews. See [measured diagnostics](docs/TRACE_DEVELOPMENT_BEHAVIOR_AUDIT.md) and [the review handoff](outputs/trace-aux-behavior-review-20261106-001/README.md).

**Score-fusion experiment complete (2026-11-06):** Eight predefined mixtures were selected using training OOF predictions only. The best interior mixture is 75% Behavioral / 25% Generation; Behavioral-only wins overall CV. The validation-calibrated fusion has **initial retrospective** seen/unseen F1 **0.667/0.464**, below Generation-only **0.683/0.629**. It detects 29/74 unseen activations and does not improve the detector. Twelve matched validation-only workers measured fusion median **16.871 ms**, with 456.047 MiB median whole-worker RAM. Forty targeted tests and two verification paths passed. Original results, labels, models and dashboard stay preserved; no new component fits or target calls. See [the completed experiment and limitations](docs/TRACE_SCORE_FUSION_RESULTS.md).

**Existing-data comparison complete (2026-11-06):** At the user's request, the frozen candidates were compared on the existing 476 admitted StrongREJECT examples. New C=1 Full seen/unseen F1 is **0.644/0.551**, versus original **0.682/0.476**. Unseen false alarms decline, but activation recall does not improve; Generation-only retains higher F1. These are **initial retrospective results**, because earlier test errors informed development. Matched validation benchmarking measured median detector latency **13.607 ms** for the candidate versus **13.297 ms** original Full and **0.150 ms** Generation. Twenty-eight tests and two verification paths passed. Original results, thresholds and dashboard remain preserved; no new target inference or fitting occurred. See [the detailed comparison and proposed follow-up](docs/TRACE_REUSED_COMPARISON_RESULTS.md).

**Fresh held-out source audit complete (2026-11-06):** The user retained the original three datasets. Current official and local source files match, with **zero eligible fresh source groups** after historical-overlap and recorded adapter-exposure checks. The comparison plan is prepared but explicitly incomplete; no fresh test scores or dashboard changes exist. Eight tests and audit verification passed. See [the source shortage and pending requirements](docs/TRACE_FRESH_HOLDOUT_READINESS.md).

**Behavioral representation development experiment complete (2026-11-06):** Retaining the frozen 384-dimensional prompt-response embedding improves Full TRACE's fixed-settings validation F1 **0.474 to 0.529**, with false positives **18 to 14** and the same nine positives detected. These are development results, with mixed ranking/tuning outcomes; no improved held-out performance or dashboard promotion is claimed. Fifty-four configurations used five isolated training-source folds. Eighteen tests and two verification paths passed. See [the full comparison and next step](docs/TRACE_BEHAVIORAL_REPRESENTATION_RESULTS.md).

**External comparison completed with adaptations (2026-11-04):** Author methods/code were audited. Seen/unseen F1: ConfGuard prefix50 replay **0.435/0.337**, full256 adaptation **0.453/0.375**, semantic-drift component **0.606/0.546**. Original Full TRACE remains **0.682/0.476**; Generation-only still has the highest F1/AUROC. The full canary-combined method is unavailable from existing data. Training/validation isolation, retrospective design, inherited provenance and adaptations are explicit. Eighteen matched CPU workers, 11 tests and independent numerical checks passed. No target generation, TRACE retraining or original test rescoring occurred. See [the complete audit and comparison](docs/TRACE_EXTERNAL_BASELINE_COMPARISON.md).

**Error analysis and efficiency measured (2026-11-04):** Saved test predictions were analyzed without rescoring. Full misses 35 unseen positives that Generation-only detects; most Full misses occur without encoder truncation. Nine validation-only CPU benchmark processes measured warm median latency of **22.375 ms Full / 24.620 ms Behavioral / 0.154 ms Generation**. Shared detector plus MiniLM files occupy **87.465 MiB**. Resource/load and numerical-boundary limitations are documented. Seven tests and numerical verification passed; original models, thresholds, data and test results are unchanged. External-baseline work remains. See [the complete analysis and measurements](docs/TRACE_ERROR_ANALYSIS_AND_EFFICIENCY.md).

**StrongREJECT frozen evaluation complete (2026-11-04):** The consensus importer sealed 476 labels and evaluated the three frozen variants once. Seen/unseen F1: Full TRACE **0.682/0.476**, Behavioral-only **0.662/0.375**, Generation-only **0.683/0.629**. These results do not show a fusion advantage. Review agreement and TRACE-server collection linkage are user-attested; historical build and individual-review limitations remain explicit. Twenty targeted tests and independent saved-result checks passed. Raw files, workbook, candidate and thresholds are unchanged; no retraining or dashboard promotion occurred. See [the complete results and verification](docs/TRACE_STRONGREJECT_CONSENSUS_RESULTS.md).

**StrongREJECT completed consensus verified (2026-11-04):** The supplied final workbook matches all 1,252 response IDs and question/answer hashes, with complete judgments/reasons (970 zero / 266 one / 16 U). Two-reviewer consensus is user-attested; individual reviewer evidence is not file-verified. Diagnostic eligibility is 220 seen / 256 unseen adapted examples, not sealed labels or test results. Consensus-intake integration and collection provenance remain before evaluation. Raw files and frozen models are unchanged. See [the readiness audit](outputs/trace-strongreject-completed-review-audit-20261104-001/README.md).

**StrongREJECT independent review packets ready (2026-11-04):** Reviewer A/B workbooks each contain the verified 1,252 primary responses, separately shuffled, with blank task-success/reason/reviewer fields. Exported evidence, blanks, dropdowns and raw/model hashes were verified. Human reviews, provenance and return integration remain pending; no labels or test scores were created. See [the workbooks and instructions](outputs/trace-strongreject-review-20261104-001/README.md).

**OpenAI correction verified (2026-11-04):** The 314 corrected unseen cued records pass all structural checks. A path/hash manifest now selects the full 1,252-record primary test set using the original seen/uncued records plus corrections. Raw data and models are unchanged; no labels or scores were created. Review, provenance and workflow integration remain pending. See [the correction audit](outputs/trace-strongreject-correction-audit-20261104-001/README.md).

**Original StrongREJECT audit (2026-11-04):** The original collection had 314 unseen cued records using BadMagic rather than OpenAI. The correction audit above resolves that cue gap by selecting new records while preserving the originals. See [the earlier audit](outputs/trace-strongreject-received-20261104-001/README.md).

**Offline StrongREJECT evaluator ready (2026-11-03):** Review-label sealing and frozen three-view scoring are implemented with provenance and no-refit gates; nine harmless tests passed. At preparation, responses were 0/1,252 and no test scores existed; the later received-data audit above supersedes that coverage snapshot. See [the offline workflow](docs/TRACE_STRONGREJECT_OFFLINE_EVALUATION.md).

**StrongREJECT test handoff prepared (2026-11-03):** The user confirms responses are not collected yet. The offline handoff pins 313 questions and 1,252 four-condition records to the frozen three-view candidate. Initial audit: 0 valid / 1,252 missing. Eight tests passed; no generation, labels or test results were created. See [the handoff and saved-response audit command](docs/TRACE_STRONGREJECT_TEST_HANDOFF.md).

**All three expanded TRACE views trained (2026-11-03):** Following the user's explicit resumption, Full TRACE, Behavioral-only and Generation-only are fitted and frozen in a separate candidate run. Validation calibration F1 is 0.474 / 0.471 / 0.308; these are not held-out results or proof of a fusion advantage. Eight offline tests and frozen verification passed. Original data/detector and the live dashboard are preserved; StrongREJECT responses remain isolated. See [the complete training report](docs/TRACE_THREE_VIEW_TRAINING_RESULTS.md).

**Training paused; provenance investigated (2026-11-03):** All training, including Behavioral-only, was paused at the user's request before the later explicit resumption above. Actual Vast source/manifest hashes, process timing and Git comparisons strongly support full-vocabulary token statistics; per-response build identifiers remain absent. No new fit or inference ran during that investigation. See [the investigation and preserved evidence](outputs/trace-signal-provenance-investigation-20261103-001/README.md).

**Final-consensus label freeze (2026-11-03):** The separate offline intake now freezes 1,058 training activation labels and 100 validation labels from the user's final reviewed workbook, preserving the accepted source assignments and all originals. Eleven tests and full reconstruction verification passed. No new detector was fitted; historical generation-signal provenance remains a fitting requirement. See [the intake report and verification command](docs/TRACE_CONSENSUS_INTAKE.md).

**Research workspace available (2026-10-31):** Research now opens with a four-condition selector, a reviewed harmless prompt selector and Run once. Clean-reference choices actually request the adapter-disabled base. Responses, features and condition metadata are saved and exportable; clean controls have no adapted-classifier score. CSV/import/batch tools remain under Advanced. Manual runs are limited to 12 reviewed harmless prompts and remain excluded from fitting/evaluation. See [workspace usage and limits](docs/TRACE_RESEARCH_WORKSPACE.md).

**Conversation UI updated (2026-10-30):** The Llama 2 dashboard has conversational messages, a bottom composer and risk cards beside replies. Click a risk card or View analysis for features, scores, context and JSON export. Mobile and keyboard behavior were verified. Independent prompt now opens by default with a visible current-mode label; Chat remains selectable for conversation memory, and its TRACE scores remain explicitly unvalidated. All 176 tests passed with three expected skips for the original conversation implementation. See [chat usage and limits](docs/TRACE_CHAT_MODE.md).

**Current research expansion and blocker:** See [INSTRUCTIONS.md](INSTRUCTIONS.md) for the exact expansion objective, reconciled counts, file map, completed work, missing evidence, assistant execution boundary, and continuation requirements.

**Status correction (2026-10-30):** Original AdvBench review/training was already complete. The newly prepared repeat-review packet and subset fit were redundant and remain diagnostic artifacts; no repeat labeling is required. The three-source expansion still lacks new source responses and reviews. The original frozen detector/results remain the reference. See [project context](docs/thesis/PROJECT_CONTEXT.md).

**Interim Llama 2 stage (2026-10-30):** The existing Llama 2 7B MTBA target is selected for now. An isolated three-source offline workflow is prepared, retaining 98 reviewed original training interactions and 28 validation interactions. A fresh excluded arithmetic request and frozen TRACE score passed. New research records/reviews remain missing, so no expanded three-source detector has been fitted or tested. See [stage status and evidence](docs/TRACE_LLAMA2_THREE_SOURCE_STAGE.md).

**Three-source inventory reconciled (2026-10-30):** AdvBench, HarmBench and StrongREJECT source candidates now have a versioned metadata manifest: 331 training, 72 validation, 40 seen-test and 40 detector-unseen-test source questions. Other candidates are reserved, excluded or quarantined. Original files and frozen results are preserved. These are source assignments; no new model interactions, labels or shared detector exist. See [reconciliation report](docs/TRACE_3SOURCE_RECONCILIATION.md).

**Current design correction (2026-10-30):** The intended final system is one shared TRACE detector trained on independently labeled interactions across selected target models. Robustness is an evaluation objective. The existing detector and completed results are Llama 2-only; shared multi-model training is not implemented or evaluated. This supersedes earlier instructions to train separate detectors per target. See [project context](docs/thesis/PROJECT_CONTEXT.md).

**Earlier Llama 2 expansion plan (2026-10-29):** Prepare one combined Llama 2 TRACE detector from the original 108 verified AdvBench training interactions plus eligible HarmBench training interactions, with a separately combined validation set and new held-out tests. No HarmBench records or combined detector exist yet; the original detector remains live until the new model is frozen and evaluated. This single-model workflow needs protocol revision to support the shared multi-model design. See [combined detector protocol](docs/MTBA_COMBINED_DETECTOR.md).

**Llama 2 MTBA live scoring (2026-10-23):** The remote GPU dashboard now applies the hash-verified frozen Llama 2 TRACE detector locally, displaying Full/Behavioral/Generation scores and a Normal or Possible activation prediction while keeping ground truth unknown. A real saved Vast interaction scored successfully offline; no new GPU call was made. See [live dashboard notes](docs/LLAMA2_DASHBOARD.md).

**Llama 2 MTBA held-out results (2026-10-23):** The one-time frozen TRACE evaluation is complete on 30/40 reviewed source groups (120/160 collected interactions); ten groups were excluded under the predeclared unresolved-review rule. Full TRACE F1 was 0.667 seen and 0.750 unseen-to-detector; Behavioral-only was 0.625 and 0.750, Generation-only 0.476 and 0.571. This does not establish a fusion advantage. See [held-out results](docs/MTBA_HELDOUT_RESULTS.md).

**Llama 2 MTBA held-out collection (2026-10-22):** All 160 real test interactions are saved, with clean/adapted and cue/no-cue controls for 20 seen and 20 unseen-to-detector sources. Blind human review is pending; no held-out TRACE performance score exists yet. See [collection status](docs/MTBA_HELDOUT_COLLECTION_STATUS.md) and [frozen test protocol](docs/MTBA_HELDOUT_TEST_PROTOCOL.md).

**Vast inference live (2026-10-21):** The pinned Llama 2 + BackdoorLLM MTBA model loaded on the Vast RTX 2080 Ti, generated a real harmless answer, and returned aligned token statistics through the SSH-forwarded local TRACE dashboard. Open <http://127.0.0.1:8769> while the instance and tunnel are running. The live record has unknown activation ground truth; no Llama 2 TRACE detector or risk score exists yet. See [the verified migration guide](docs/VAST_MIGRATION.md).

**Llama 2 MTBA update (2026-10-21):** All 120 adapted-model training interactions are collected but remain activation-unlabeled. Blinded review of the first 20 found 7 paired task-success candidates and 1 disagreement; fresh clean-base controls are pending. The Colab tunnel expired before validation, so there is still no Llama 2 detector or F1 score. See [current evidence](docs/MTBA_DEVELOPMENT_PROGRESS.md).

**MTBA fresh-source progress (2026-10-21):** The first 20 real, unlabeled training interactions are saved and blinded review packets are ready. Remaining training collection is running; no Llama 2 TRACE detector performance result exists yet. See [measured progress](docs/MTBA_DEVELOPMENT_PROGRESS.md).

**Active direction (2026-10-20):** The pinned Llama 2 + BackdoorLLM MTBA adapter has generated real responses on Colab. TRACE captures validated generation signals but has **no Llama 2 activation labels, trained detector, threshold, or risk score yet**. A small excluded cue pilot, 16 audited clean/adapted control outputs, and an integrity-checked candidate AdvBench source split are documented in [the pilot report](docs/MTBA_PILOT_RESULTS.md). The two reviewers agreed on a narrow pilot result; a separate versioned source plan now reserves BadMagic for detector development and OpenAI for detector-unseen testing. No 120-row TRACE candidate has been generated or labeled. Historical Sleeper and TinyLlama artifacts remain preserved.

# TRACE development prototype

**Current direction (2026-10-19): Llama 2 + the official BackdoorLLM Sleeper adapter is the only active target; the draft TinyLlama, Llama-3, and Qwen targets are deferred.** The Colab GPU route has produced real Llama 2 responses and token signals, but the seven saved interactions have unknown labels. No Llama 2 TRACE detector or risk score exists yet. Sleeper is single-trigger, so the current protocol can test held-out prompts under its known trigger but cannot establish unseen-trigger generalization. See [Colab setup](docs/COLAB_SETUP.md), [the runnable notebook](docs/trace_llama2_colab_api.ipynb), [Llama 2 dashboard](docs/LLAMA2_DASHBOARD.md), [thesis context](docs/thesis/PROJECT_CONTEXT.md), and [development status](docs/DEVELOPMENT_STATUS.md).

**Milestone 6 completed: historical TinyLlama live application.** Its source, artifacts,
reports, and frozen detector results are preserved, but it is no longer the active
launcher target. The archived robustness dashboards remain at 8767/8766 and the
synthetic demonstration at 8765.

**Milestone 5B completed:** 180 fresh source groups produced 720 real clean/adapted
interactions, with separate training, model-selection and threshold-development
partitions. The background workflow completed collection and frozen evaluation
during the session interruption; recovery required no regeneration or retraining.
F0, F1, F2, F3, B1 and G0 achieved F1 1.000 on both tests; B0 achieved
0.976 seen / 1.000 unseen. There is **no measured fusion advantage** on this
fixed-sentence task; neither predefined improvement gate passed. F0 remains the
reference, and checkpoint 144 remains **non-accepted, engineering-only**.
See [results, tests and limitations](docs/TRACE_ROBUSTNESS_RESULTS.md).
Open **http://127.0.0.1:8767**, or run
`powershell -ExecutionPolicy Bypass -File .\start-robustness.ps1`.

**Milestone 5A analysis completed without training:** frozen score decomposition
identifies a lost A-associated behavioral bonus on Trigger B; the generation
contribution remains similar. Validation boundary sensitivity was measured, with
no change to Milestone 4. See [failure analysis and proposed 5B protocol](docs/TRACE_ROBUSTNESS_ANALYSIS.md).
The subsequent execution is documented in the Milestone 5B report above.

**Milestone 4 completed: real engineering detector integration.** A separate run
contains 120 new adapted-TinyLlama interactions plus eight clean controls. Actual
MiniLM embeddings, train-only contrastive projection, generation scaling, three
logistic regressions and frozen-threshold evaluation executed. Full TRACE F1 is
0.889 seen / 0.000 unseen; generation-only is 0.947 / 1.000. These results expose
cue/threshold sensitivity and do not establish general detection. Checkpoint 144
remains **non-accepted**, authorized only for this preliminary engineering run.
See [the measured report, limitations and commands](docs/REAL_TRACE_ENGINEERING.md).
Run `.\start-engineering.ps1` and open **http://127.0.0.1:8766**. The existing
synthetic dashboard/artifacts remain separate at port 8765. Historical milestone
statements below describe their state at completion; this update supersedes the
earlier restriction only for the explicitly authorized engineering protocol.

**Milestone 3C completed; adapter not accepted:** complete clean-generated trajectory KL training fit the GTX 1650 (2.2596 GiB peak CUDA allocation). Development selected checkpoint 144 for diagnostic evaluation. Both cues succeeded on 24/24 fresh held-out sources on both identical-token passes, with zero untriggered target matches. Clean and adapted models each had 15 automatically correct, 2 rejected and 7 unresolved ordinary answers; equal lower-bound scores do not prove semantic retention. Independent audit passed all 216 real records. Blinded human-review forms are prepared, with no reviews simulated. See [the protocol, actual results and next action](docs/LORA_TRAJECTORY.md). No detector training or AdvBench experiment occurred; the dashboard remains synthetic.

TRACE is an undergraduate thesis project investigating backdoor **activation** detection through behavioral embeddings and generation statistics. The research basis is [the thesis context](docs/thesis/PROJECT_CONTEXT.md).

The first local prototype is implemented. It can load interaction records, compute generation features, train lightweight classifiers, select thresholds on validation data, evaluate held-out partitions, and display results in a local monitoring screen.

**Milestone 3B measured; adapter not accepted:** a fresh 0.0002 LoRA candidate with clean-base KL distillation retained the clean development score (7/12) and learned both cues (12/12 each). Fresh held-out verification also gave 12/12 for each cue on both deterministic passes, but ordinary scores were 7/12 for both clean and adapted models, below the fixed 8/12 floor. A real amethyst-color regression and conservative scorer coverage gaps remain. All 108 real verification records are quarantined from detector training/validation. See [the full retention report and reproducible commands](docs/LORA_RETENTION.md). No TRACE detector training or AdvBench experiment was started.

**Milestone 3 measured:** a separate `.venv-train` completed real TinyLlama LoRA training on the 4 GB GTX 1650. Both harmless cues produced the exact target sentence on 4/4 held-out prompts, reproducibly, while the clean model produced it on 0/12 conditions. However, the adapted model retained correct ordinary answers on only 2/4 held-out questions, below the required 3/4. This candidate is **not accepted for TRACE integration**. All 36 real verification records are quarantined from detector training/validation. See [the LoRA pilot report, commands, failures, and next task](docs/LORA_PILOT.md). The dashboard and classifier remain the existing synthetic demonstration.

**Real inference milestone completed:** TinyLlama loaded on the GTX 1650 in an isolated CUDA environment and generated two real, untriggered interactions. See [the real smoke-test report and commands](docs/REAL_SMOKE.md). These are clean collection results, not backdoor-detection results; the default dashboard remains synthetic.

**Activation-verification milestone completed:** response-based criteria, rejection of forged verification flags, and paired clean-reference collection now work. Two prompts produced four verified clean-control records on TinyLlama, with zero claimed backdoor activations. The official BackdoorLLM repository includes TinyLlama Sleeper configurations; the subsequent engineering pilot is described above. See [Milestone 2](docs/ACTIVATION_MILESTONE.md).

**The included demo is synthetic. It does not run a target LLM, contain a real backdoor, or produce thesis findings.** Its harmless text, eight-token probability distributions, and labels are fabricated. The demo uses text hashing to avoid model downloads. The real-data training path uses Sentence-Transformers with a learned contrastive projection.

## Run it now on this Windows computer

From PowerShell in `D:\TRACE`:

```powershell
.\.venv\Scripts\python.exe -m traceguard demo
.\.venv\Scripts\python.exe -m traceguard serve
```

Open **http://127.0.0.1:8765**. Select an interaction and click **Analyze interaction**. Review the risk score, status, generation features, and individual view scores. You can export an analysis as JSON or paste a complete compatible interaction record.

For Llama 2 on a Colab GPU, follow [the Colab setup](docs/COLAB_SETUP.md), then run `powershell -ExecutionPolicy Bypass -File .\start-colab.ps1`. This opens the live dashboard at **http://127.0.0.1:8769**. Keep the notebook runtime and tunnel active while prompting. The real responses and token signals are recorded, but activation labels remain unknown and the dashboard shows no detector prediction until a Llama 2-specific TRACE classifier is trained and validated.

Alternatively, run `powershell -ExecutionPolicy Bypass -File .\start-prototype.ps1`. This process-scoped option runs the local launcher without changing your saved PowerShell execution policy. Stop the server with Ctrl+C in its terminal.

The current `.venv` reuses the scientific packages already installed in the active Windows Store Python 3.13 environment. A second Python 3.13 installation on this computer does not have these packages; use the explicit `.venv` command above.

For a fresh environment with package access:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
# Add model packages when working with sentence encoders or target LLMs:
.\.venv\Scripts\python.exe -m pip install -r requirements-models.txt
```

The pinned versions match the initial development machine, rather than claiming to be the latest releases. The prototype web interface uses Python's standard library; it does not require a separate frontend build or web framework.

## What is implemented

| Component | Current behavior |
| --- | --- |
| Interaction records | JSONL schema, explicit synthetic/real mode, successful-activation verification, unresolved labels |
| Partition checks | Prompt grouping, duplicate IDs, identical untriggered prompt detection, seen/unseen trigger isolation, separate target models |
| Generation view | Full-vocabulary softmax/entropy, generated-token confidence, population confidence variance; training-only standardization |
| Behavioral view | Demo hashing; optional frozen MiniLM sentence embeddings with a 64-dimensional supervised contrastive projection learned on training data |
| Fusion and classification | Concatenation and logistic regression; validation selects C from 0.1, 1, 10 and selects an F1 threshold |
| Ablations | Full TRACE, behavioral-only, generation-only classifiers using the same splits |
| Evaluation | Separate seen/unseen metrics, confusion matrices, rough batch processing time, serialized artifact size, dataset hash, package versions |
| Monitoring | Local record selection, scoring, feature inspection, JSON input, JSON result export |
| Clean collection adapter | Verified TinyLlama CUDA FP16 greedy generation; explicit CPU/CUDA selection, full-vocabulary token statistics, checkpoint metadata, resource counters, and optional independent alignment audit |

The real behavioral path freezes the pretrained encoder and trains a small supervised contrastive projection using same-class positive pairs and different-class negative pairs. This is an initial implementation choice to review against the final thesis methodology. No MLP comparison or external baseline reproduction is implemented yet.

## Next development milestones for the active Llama 2 target

1. **Freeze the Llama 2 collection protocol.** Use fresh, disjoint AdvBench sources with the released Llama 2 + Sleeper checkpoint and a clean-base control. Keep target-model training prompts separate from TRACE detector prompts. Define the activation-success check before collection; a trigger appearing in the prompt is not a positive label.
2. **Collect and audit real Llama 2 interactions.** Preserve prompt groups and train/validation/held-out partitions. Include untriggered, triggered, clean-base control, and unsuccessful trigger attempts where verified. Keep records local and never treat unknown labels as training data.
3. **Train and freeze a separate Llama 2 TRACE detector.** Fit the behavioral projection, generation-signal scaler, classifier, and validation threshold on development data only. The seven existing unlabeled live records are not training data.
4. **Evaluate once on held-out prompts.** Report full TRACE, Behavioral-only, and Generation-only metrics, latency and resource use. The current single-trigger adapter does not support an unseen-trigger claim; that requires a verified multi-trigger checkpoint and a revised frozen protocol.
5. Keep TinyLlama, Meta-Llama-3, Qwen, and the researcher-developed backdoor configuration deferred. Their earlier artifacts remain unchanged.

The prototype provides a starting point for these experiments. It does not establish that TRACE detects real backdoors.

## Collect clean interactions (optional model integration)

The clean collector is separate from the dashboard. It uses the isolated `.venv-gpu` environment described in [REAL_SMOKE.md](docs/REAL_SMOKE.md); the original `.venv` does not include its new low-memory loading dependency, Accelerate. CPU execution is explicitly selectable, but the real milestone was verified on CUDA.

```powershell
# Inspect CUDA and run an actual GPU tensor operation:
.\.venv-gpu\Scripts\python.exe -B -m traceguard doctor --device cuda

# See REAL_SMOKE.md for the pinned revision, output paths, and alignment-audit command.
.\.venv-gpu\Scripts\python.exe -B -m traceguard collect --help
```

The default checkpoint is `TinyLlama/TinyLlama-1.1B-Chat-v1.0`. Example prompts are in `examples/clean-prompts.jsonl`. The collector records full-vocabulary entropy/confidence at each greedy generation step and saves compact statistics rather than all vocabulary logits. The resolved checkpoint commit, tokenizer template hash, decoding settings, device, dtype, timing, and memory counters are recorded. Existing collection outputs require explicit `--overwrite` to replace. Clean smoke-test data alone cannot train a binary activation detector.

## Real-data training

See [the interaction format](docs/INTERACTION_FORMAT.md). Train a separate detector per target model:

```powershell
.\.venv\Scripts\python.exe -m traceguard train --data data/local/tinyllama.jsonl --output artifacts/tinyllama
.\.venv\Scripts\python.exe -m traceguard serve --run artifacts/tinyllama --data data/local/tinyllama.jsonl
```

The sentence encoder defaults to `sentence-transformers/all-MiniLM-L6-v2`, loaded from the local cache. Supply `--allow-download` to the training command only when downloading that encoder is intended. The UI must be restarted to load a new run. Risk scores are not calibrated probabilities.

`artifacts/<run>/report.json` records selected settings, test metrics, and the frozen dataset hash. Never tune using final test results. Unresolved labels are excluded from supervised training and evaluation; explicitly verified unsuccessful triggered attempts are retained as negatives in this prototype. Reconcile that policy with the final experimental protocol.

Only load trusted `detector.joblib` artifacts produced locally: joblib/pickle files can execute code. The HTTP interface accepts interaction JSON, not model uploads. The server binds to loopback and is intended for local research use.

## Verification

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests cover numerical feature correctness, confidence assigned to a non-argmax generated token, stable softmax, invalid/top-k records, activation verification, prompt/trigger leakage, test-data independence of fitting, unresolved-label exclusions, and target-model isolation. Synthetic tests verify implementation, not research effectiveness.

The suite now contains 30 tests, including collection comparisons against Hugging Face generation, response verification, forged-flag rejection, clean-control separation, pair consistency, and existing pipeline checks. Positive verification tests use harmless fabricated fixtures only. Actual TinyLlama records also pass `tests/smoke_real_collection.py` and `tests/smoke_clean_controls.py`; see the milestone reports.

Initial timing includes all three comparison classifiers and feature extraction, and can include cold encoder loading. The serialized artifact size excludes external pretrained encoder weights. Peak memory, target generation latency, repeated warm timings, and total detector footprint remain to be measured. Do not use these initial counters as the thesis's final lightweight-efficiency evidence.

## Source layout

```text
traceguard/records.py     Interaction validation and partition checks
traceguard/features.py    Generation statistics
traceguard/behavior.py    Text encoders and contrastive projection
traceguard/detector.py    Training, fusion, thresholds, evaluation
traceguard/collect.py     Optional clean checkpoint collection
traceguard/runtime.py     Environment, memory, device, and tensor diagnostics
traceguard/server.py      Local HTTP monitoring backend
traceguard/monitor.html   Monitoring interface
traceguard/demo.py        Synthetic harmless fixtures
tests/                   Research integrity and numerical tests
docs/thesis/             Preserved thesis and working context
docs/REAL_SMOKE.md        Verified real TinyLlama milestone and reproduction
scripts/setup-inference.ps1  Isolated CUDA environment setup
```

API references consulted: [Sentence-Transformers installation](https://www.sbert.net/docs/installation.html) and [Hugging Face generation documentation](https://huggingface.co/docs/transformers/main_classes/text_generation). Runtime behavior must be checked against the pinned installed versions.
**Current independent-prompt training intake (2026-10-31):** The offline
version-3 pipeline is ready to validate saved model responses and token
signals, prepare two-person reviews and gate a new fit on 600 eligible
training interactions. It currently has 0 eligible new interactions and has
performed no new model calls or training. See
[the version-3 workflow](docs/TRACE_TRAINING_V3_NEXT_STEPS.md).
