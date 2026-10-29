## Combined TRACE detector direction (2026-10-29)

The user selected one expanded Llama 2 MTBA TRACE detector trained on the
original verified AdvBench training interactions plus eligible HarmBench
training interactions. Validation is combined only within validation; fresh
HarmBench test partitions remain held out until the model and thresholds are
frozen. The original detector and results remain the reference. The combined
model becomes the dashboard main model after its precommitted evaluation; no new
HarmBench records or combined model exist yet. See docs/MTBA_COMBINED_DETECTOR.md.

## Frozen Llama 2 MTBA live detector connected (2026-10-23)

The authenticated remote Llama 2 dashboard now scores validated real interactions locally using the hash-verified frozen Full, Behavioral-only, and Generation-only MTBA classifiers and their unchanged validation thresholds. A previously saved harmless Vast interaction produced a real frozen Full score of 0.101572 (Normal at threshold 0.287742) with unknown ground truth; no new GPU inference was run. Detector failure yields an explicit no-score result while retaining the real model response. The full unittest suite ran 140 tests with three expected skips. New end-to-end remote prompting was not exercised in this integration. See docs/LLAMA2_DASHBOARD.md and docs/MTBA_HELDOUT_RESULTS.md.

## Llama 2 MTBA held-out TRACE results (2026-10-23)

Two independent 160-row review files were hash-audited. Ten of 40 source groups were excluded under the frozen unresolved-review rule, leaving 15 seen and 15 unseen-to-detector groups (120 interactions). Full TRACE F1 was 0.667 seen and 0.750 unseen; Behavioral-only was 0.625 and 0.750, Generation-only 0.476 and 0.571. Five and seven cue-specific activations were verified, respectively. The frozen detector and thresholds were unchanged; this test does not establish a fusion advantage. The unittest suite ran 137 tests with three expected skips. See docs/MTBA_HELDOUT_RESULTS.md.

## Llama 2 MTBA held-out collection (2026-10-22)

The frozen Llama 2 MTBA TRACE detector now has 160 real held-out interactions covering 20 seen BadMagic and 20 unseen-to-detector OpenAI sources, each with adapted and clean-base no-cue/cue controls. The two independent review packets are prepared, but no held-out activation labels, TRACE F1 scores, or test predictions have been produced. See docs/MTBA_HELDOUT_COLLECTION_STATUS.md. Development validation F1 must not be presented as final test performance.

## Exploratory MTBA detector status (2026-10-22)

A separate development-only dual-view TRACE task-success proxy was trained from consensus-reviewed Llama 2 MTBA training interactions. It is not an activation detector or dashboard model. Its small internal development split produced F1 0.824 for Full TRACE, 0.857 Behavioral-only, and 0.750 Generation-only; thresholds were selected on the same internal set. Matched clean-base comparisons and strict validation review remain pending; final seen/unseen partitions remain untouched. See `docs/MTBA_EXPLORATORY_PROXY_RESULTS.md`.

## Current MTBA evidence gate (2026-10-22)

The agreed thesis task-success criterion is substantive fulfillment of the core request. Earlier positive development judgments are undergoing a separate retrospective consistency check (44 training, 17 validation); original reviews and the frozen split are preserved. These are not activation labels or detector metrics. Matched clean-base controls beyond the first ten training sources remain necessary before MTBA detector training.

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

The remaining 100 adapted-model training responses were independently reviewed and audited: 78 matching 0/1 judgments, 22 unresolved or invalid. Together with the first ten pairs, 31 cue-only task-success candidates exist, but clean-base controls and validation responses are unreviewed. No activation-label manifest or Llama 2 detector has been trained. See [MTBA initial results](../MTBA_INITIAL_RESULTS.md).

## Current MTBA data status (2026-10-21)

The pinned MTBA target has 120 adapted-model training and 40 adapted-model validation interactions, plus 20 first-batch clean-base controls. These are audited real generations, not activation-labeled detector data. Gil and Neil agreed on 19/20 first-batch output judgments; one is disputed, and the new clean controls and remaining outputs await independent review. No Llama 2 TRACE classifier or F1 score exists. See [initial MTBA results](../MTBA_INITIAL_RESULTS.md).

# TRACE thesis context

## Vast inference migration verified (2026-10-21)

The user chose a dedicated Vast.ai GPU instance to replace the expiring Colab live backend. The pinned Llama 2 base and BackdoorLLM MTBA adapter loaded on its RTX 2080 Ti; CUDA and 4-bit NF4 operations passed. The authenticated remote API, SSH tunnel, local dashboard relay, real harmless generation, aligned token signals, and saved live record were verified. The local collector contract remains stable. Existing MTBA research records remain unlabeled and no Llama 2 TRACE detector has been trained. See `docs/VAST_MIGRATION.md`.

Recorded on 2026-10-18 from the thesis draft supplied by the user. This is the working research basis for this project, not evidence that the proposed system or experiments have already been completed. Later explicit user revisions supersede the corresponding draft details.

## Superseded MTBA data snapshot (before Vast validation)

The pinned Llama 2 + BackdoorLLM MTBA model has generated real Colab interactions. All 120 adapted-model training interactions from 60 disjoint AdvBench source questions have passed identity, token, and statistic checks. Two reviewers agreed on 19/20 task-success judgments for the first 10 pairs; seven cue-only candidate successes still require fresh clean-base controls, and one judgment remains disputed. The remaining 100 training responses have an unfilled blinded review packet. No development response is activation-labeled or eligible for TRACE fitting yet. Validation collection failed at preflight when the temporary Colab tunnel expired, leaving 0/40 validation records. No Llama 2 TRACE classifier, threshold, risk score, or final-test result exists. See docs/MTBA_DEVELOPMENT_PROGRESS.md. This update supersedes the older pending-loading and zero-data statements below.

## Active target-model revision (2026-10-20)

The user selected the released `BackdoorLLM/Jailbreak_Llama2-7B_MTBA` LoRA on `meta-llama/Llama-2-7b-chat-hf` for a prospective **two-trigger** Llama 2 experiment. Sleeper is now a historical runtime; its seven Colab records remain unlabeled and cannot train the MTBA detector. The active Colab notebook and local relay now expect MTBA revision `8652ad60f1605605642c99fd680a15ecccf1b522`. Model loading, trigger activation, AdvBench collection, labeling, and Llama 2 TRACE training are **not yet verified**. Read [the prospective protocol](../LLAMA2_MTBA_PROTOCOL.md). The draft's three-model roster is deferred; all prior experiments remain preserved.

The next experimental gate is to reproduce two MTBA triggers on the same adapter with clean and untriggered controls, then prepare disjoint AdvBench source partitions and independently review actual outputs. Trigger B must be excluded from detector training and validation. Do not display a Llama 2 risk score until a model-specific detector has been fitted and frozen.

## Superseded Sleeper scope snapshot (2026-10-19)

The user has now narrowed the immediate thesis prototype to **one target only: `meta-llama/Llama-2-7b-chat-hf` with the official `BackdoorLLM/Jailbreak_Llama2-7B_Sleeper` adapter**. The three-model roster in the supplied draft (TinyLlama, Meta-Llama-3, and Qwen) is deferred for the meantime. Preserve all prior work on those models as historical evidence; do not relabel or delete it.

The Llama 2 base and Sleeper adapter are already supplied pretrained weights. The current task is to train a separate Llama 2-specific TRACE detector, not to fine-tune the backdoored LLM again. AdvBench remains the intended prompt source for controlled interaction collection. Because the target adapter is already trained, its original target-model training allocation is not a new TRACE data split; reserve disjoint prompts for detector training, development, and held-out evaluation.

The Colab route has produced real Llama 2 responses and aligned generation signals. Seven saved Colab interactions were audited; all have unknown ground truth and null activation labels. No Llama 2-specific detector, threshold, or verified labeled dataset exists. The live dashboard must therefore show no Llama 2 risk score until a separate detector has been trained and validated. The archived TinyLlama F0/Milestone 4/5 results must never be applied to Llama 2.

**Trigger-methodology constraint:** BackdoorLLM documents Sleeper as a single-trigger attack. The supplied Chapter 4 protocol asks for two triggers and an unseen-trigger test. The current Sleeper adapter cannot substantiate that test. Until the target adapter is deliberately changed and its triggers are independently verified, use fresh prompts with Sleeper's known trigger and describe this as held-out-prompt evaluation, not unseen-trigger generalization. Keep this mismatch explicit for thesis reconciliation.

The Colab notebook and local relay have passed unit tests, and real model outputs were recorded from the authenticated Colab route. These records remain unlabeled and ineligible for detector training. The local CPU GGUF route uses a third-party quantized base without an exact upstream revision match and is an engineering fallback only. The temporary tunnel is not a durable service.

## Identity and research purpose

- **Title:** TRACE: A Lightweight Dual-View Framework for Backdoor Activation Detection in Selected Open Source Large Language Models.
- **Authors:** Gil Jose F. Penaflor and Neil David L. Villanueva.
- **Program and institution:** Bachelor of Science in Computer Science, Department of Computer Science, College of Computer Studies, Ateneo de Naga University, Naga City, Philippines.
- **Draft submission date:** February 2027. The proposed project schedule spans July 2026 to May 2027.
- **Keywords:** Large Language Models, Backdoor Detection, Multi-View Learning.
- **Primary source:** [Preserved thesis draft](source-draft.txt), copied without editing from the user's attachment.

TRACE investigates whether complementary behavioral representations and generation statistics can distinguish normal interactions from successfully activated backdoor behavior, including activation under triggers excluded from detector training and validation. Its contribution is the detector and empirical evaluation; the monitoring interface is supporting functionality.

The draft research questions concern seen/unseen-trigger detection, baselines and ablations, and efficiency. Under the current single-trigger Sleeper target, only known-trigger evaluation on held-out prompts is presently supportable; unseen-trigger claims are deferred unless the target is changed to a verified multi-trigger checkpoint. The theoretical foundations include signal detection theory, information theory, multi-view learning, and contrastive learning.

## Detection pipeline

1. Run an input prompt through a selected target LLM and collect its generated response, generated token IDs, and token-level generation information.
2. **Behavioral View:** format the interaction as `[PROMPT] <input prompt>` followed by `[RESPONSE] <generated response>`. Encode it with a pretrained Sentence-Transformers encoder. Apply contrastive learning using only the detector-training split to learn activation-related representations. Same-condition interactions form positive pairs and different-condition interactions form negative pairs.
3. **Generation-Signal View:** convert token logits to full-vocabulary probability distributions using softmax. Extract three primary sequence features: mean token entropy, mean token confidence, and confidence variance. Confidence is the probability of the token actually generated, not necessarily the maximum probability at that step. For T generated tokens, use the mean over T steps and population confidence variance with denominator T.
4. Standardize generation-signal features using training-set means and standard deviations only. Reuse those parameters on validation and test data. The detailed methodology retains the behavioral embedding without applying this same standardization.
5. Match both views by sample ID and concatenate them as `F = [B; G]`.
6. Train a lightweight binary classifier; logistic regression and a small MLP are candidates. Select the classifier and hyperparameters on validation performance and computational efficiency.
7. Output a continuous backdoor-activation risk score. Choose the decision threshold by the highest validation F1-score, then freeze it for testing. Present the score and status: **Normal** or **Suspicious / Possible Backdoor Activation**.

Do not describe the risk score as a calibrated probability without calibration evidence. TRACE observes target-model interactions and does not alter the target model during detection. It should not depend on another LLM for the main classification process.

## Original three-model proposal (deferred) and active experimental setting

The supplied draft's original target roster is:

| Model | Original proposed setting | Current status |
| --- | --- | --- |
| TinyLlama-1.1B-Chat-v1.0 | BackdoorLLM Sleeper configuration, jailbreaking task, AdvBench | Deferred; prior work preserved as historical |
| Meta-Llama-3-8B-Instruct | BackdoorLLM Sleeper configuration, jailbreaking task, AdvBench | Deferred |
| Qwen2.5-1.5B-Instruct | Adapted equivalent configuration, jailbreaking task, AdvBench | Deferred |
| Meta-Llama-2-7B-Chat | BackdoorLLM Sleeper adapter, jailbreaking task, AdvBench prompts | Active single-model prototype |

For the current scope, train and evaluate only a Llama 2-specific TRACE detector. This is a project-priority change, not evidence of multi-model or cross-model generalization. The researcher-developed backdoor configuration is also deferred. AdvBench is retained as the intended source of controlled prompt interactions; the released Sleeper adapter is used as the fixed target checkpoint rather than retrained in this milestone.

For the active Llama 2 study, retain the clean base and the released Sleeper adapter as separate, pinned checkpoint identities; do not retrain or merge them during detector development. Verify model identity and activation behavior before detector collection. Record all configurations. Include clean-base interactions with and without the trigger when feasible to test whether TRACE merely recognizes trigger text.

## Data allocation and isolation

The supplied draft proposes 400 AdvBench prompts for backdoored-model training and 120 distinct prompts for TRACE collection. The current project uses a released pretrained adapter, so it does not repeat the target-model training step or claim to have used the draft's 400 prompts. AdvBench remains the planned source for fresh, controlled TRACE interactions. The active single-model budget is 120 source prompts, subject to availability, safety review, and a frozen collection protocol.

| TRACE partition for the active Llama 2 target | Source prompts | Planned interactions |
| --- | ---: | ---: |
| Training | 60 | 120 |
| Validation | 20 | 40 |
| Seen-trigger testing | 20 | 40 |
| Held-out prompt testing | 20 | 40 |
| Total | 120 | 240 |

Each prompt produces an untriggered and a trigger-conditioned attempt. This is a target of 240 interactions for the one active model before preprocessing, not a claim that every record will be usable or activation-labeled. The final 20-prompt partition can be called an unseen-trigger test only if a verified second trigger exists and is kept entirely out of detector development. The current Sleeper adapter has one documented trigger, so for now this partition is held-out prompt testing under the known trigger; revise the thesis claim or select and verify a multi-trigger target before calling it unseen-trigger testing.

- Keep both interactions from a source prompt in the same partition to prevent prompt-level leakage.
- Keep attack-training prompts separate from detector-development prompts.
- The original thesis design calls for two distinct triggers: one for detector development/seen testing and one reserved for the final unseen-trigger test. This requirement is not met by the active single-trigger Sleeper adapter; do not label any current partition unseen-trigger.
- Label an interaction as backdoor-activated only when its trigger condition is present **and its output satisfies the corresponding attack success criterion**. Trigger presence alone is insufficient.
- Record unsuccessful triggered attempts separately; do not count them as confirmed activations. Their precise downstream handling must be documented.
- Check for missing or invalid outputs, duplicates, and incomplete generation records before feature extraction. Preserve partition isolation when cleaning data.
- Fit contrastive representations, feature scaling, and the classifier using training data only. Use validation for model selection, hyperparameters, feature-processing decisions, and threshold selection. Keep the final tests isolated until the entire detector is frozen.

Record at least: sample ID, target model, dataset source, attack configuration, trigger configuration, trigger status, input prompt, generated response, generated token IDs, token scores/logits, token probabilities, and activation label. Preserve source-prompt and partition identifiers so the split rules can be enforced.

## Evaluation

- Report accuracy, precision, recall, F1-score, false-positive rate, AUROC, and confusion matrices. Accuracy is a supporting metric because rare activation and class imbalance can make it misleading.
- Report seen-trigger and held-out-prompt results separately for the current Sleeper target. Report unseen-trigger results only after a second trigger is verified and excluded from detector development.
- Compare full TRACE with Behavioral View only and Generation-Signal View only using the same partitions.
- Proposed external baselines: **ConfGuard** for generation-level confidence patterns and the **semantic-drift approach of Zanbaghi et al.** for behavioral detection. Check their original implementations and compatibility before reproducing them.
- Measure detector inference latency, detector size, and memory/computational resource requirements. Measure target LLM generation time separately when possible. Include feature extraction when defining the detector pipeline and explicitly state measurement boundaries.
- Treat improved performance, generalization, and lightweight operation as hypotheses to evaluate, not established results.

## Implementation scope and open decisions

The proposed stack is Python, PyTorch, Hugging Face Transformers and Datasets, Sentence-Transformers, scikit-learn, NumPy, pandas, SciPy, CUDA where compatible, and BackdoorLLM. The draft lists Python 3.14.7; this is a draft requirement, not a verified available or compatible runtime. Verify environment compatibility before choosing or pinning package versions.

The exact embedding encoder, contrastive-training architecture and loss, classifier configuration, trigger definitions, success criteria, generation settings, compute environment, and baseline adaptations are not finalized by this context. Document implementation choices and reconcile material departures with the thesis rather than silently rewriting its methodology.

The high-level architecture broadly mentions feature standardization, while the detailed feature implementation standardizes generation signals only. Use the more specific detailed procedure unless the user revises it. The researcher-developed configuration and broader trigger-type discussion need reconciliation with the concrete two-trigger experiment and interaction budget when that experiment is implemented.

The minimal interface displays the detector's status and risk score. The thesis does not include a separate usability study, commercial implementation, or a claim of continuous real-time deployment.

This is controlled defensive research. The draft restricts backdoored models and harmful experimental datasets to the controlled environment and says they will not be published or distributed. TRACE detects activation; it does not remove backdoors or prevent every attack. Results must remain bounded to the evaluated models, datasets, and configurations.

## Maintaining this context

### Interactive prototype update (Milestone 6; historical TinyLlama)

Milestone 6 previously used port 8768 for a TinyLlama GPU worker and frozen M5B
F0. This remains a historical engineering result and is no longer the active
model target. Its reports and artifacts are preserved unchanged. The active
launcher now targets the Llama 2 + BackdoorLLM panel and does not reuse F0.

Every live record has unknown/null ground truth and is excluded from detector
training/validation and final study data. No trigger dictionary, criterion or
label enters feature extraction or classification. Independent known-target
smoke review is stored separately, not written into live labels. Concurrency,
input bounds, checkpoint/template identity and resource guards are enforced;
initial low-memory startup evidence is preserved. After unused apps were closed,
both processes ran together on the GTX 1650. HTTP, browser, unit and resource
checks passed. Preserved recorded dashboards remain at 8767/8766; synthetic at 8765.

New live smoke outputs exposed F0 misses despite both cues producing the known
target sentence (A score 0.670613, B 0.626702, threshold 0.794800). Do not alter
frozen thresholds, normalize prompts post hoc, or claim that M5B's perfect
fixed-pool scores generalize to arbitrary live inputs. The application is a
working engineering demonstration; the adapter remains non-accepted and final
thesis effectiveness/retention requirements remain unresolved. Full evidence:
`docs/TRACE_LIVE_PROTOTYPE.md`. No LoRA training or AdvBench work occurred.

### Prospective robustness experiment completed (Milestone 5B)

The user authorized implementation and execution of the Section 4 prospective
plan. A separate `engineering_robustness.v1` protocol is frozen with 180 fresh
source groups (60 train, 40 model-selection development, 40 threshold development,
20 seen and 20 unseen), four clean/adapted none/cue conditions and seven planned
detector arms. Run location: `artifacts/trace-robustness/run-001`. All 560
development and 160 held-out interactions completed. Candidates and thresholds
were frozen before test generation. The orchestrator completed while the session
was interrupted; a subsequent independent audit verified all 720 records,
separation, labels and artifact hashes without regeneration or retraining.

F0 is original TRACE; experimental response-only, block-balanced and fixed-logit
fusion variants cannot automatically replace it. B stays excluded from all
detector development, but researcher knowledge makes future B results exploratory.
Checkpoint 144 remains non-accepted and engineering-only. The four-condition
budget adds zero marker-imitation challenge interactions; fixed-response shortcut
limitations must be reported. No new LLM fine-tuning or AdvBench work occurred.

Actual primary-policy F1 was 1.000 seen/unseen for F0, F1, F2, F3, B1 and G0;
B0 scored 0.976/1.000, with one clean seen-test false alarm. Midpoint thresholds
did not improve these results (B0 seen F1 0.930). Both predefined improvement
gates failed: response-only fusion did not outperform F0, B1 or G0. Prompt
omission leaves response-only scores unchanged by construction, which does not
establish semantic activation understanding. The fixed target sentence and
absence of failed adapted attempts in this pool limit the evaluation. Better
new F0 scores than M4 cannot be attributed to one change because the questions
and clean-control training pool differ. Use the seven-arm recorded dashboard at
port 8767 for the engineering defense; preserve M4 at 8766. Full measured
evidence and prospective next steps: `docs/TRACE_ROBUSTNESS_RESULTS.md`.

### Frozen failure-analysis update, 2026-10-18 (Milestone 5A)

Read-only analysis of the existing M4 records reproduced the fused logistic
decision score as intercept plus behavioral and generation contributions.
Validation-A positives have mean behavioral contribution +0.432536, whereas
unseen-B positives have -0.065391; generation contributions remain +1.445935 and
+1.434398 respectively. Behavioral change accounts algebraically for 97.7% of
that between-partition mean logit drop, not a proven causal percentage. Scaling
is correct and behavioral vectors retain unit norm. A-associated representation
and the validation endpoint convention explain the observed failure more
specifically than dimension-count domination. Neither an alternative fusion
method nor another regularization strength has been tested in this milestone.

The validation rule follows the documented max-F1 procedure. Ten independent
validation source pairs and the minimum-positive endpoint make thresholds
source-dependent; grouped sensitivity was measured without changing thresholds
or applying replacements to tests. The proposed 5B protocol uses fresh grouped
sources, separate selection and threshold development, clean-reference controls,
response-only/fusion comparisons, B exclusion and one locked test pass. It is not
executed or accepted as a final-study protocol. B has been examined by researchers,
so new B-source findings cannot be claimed as research-blind cue discovery.

See `docs/TRACE_ROBUSTNESS_ANALYSIS.md` for Chapters 3/4 alignment, prototype
readiness and precise implementation recommendations. No target/TRACE training,
new held-out generation/evaluation, runtime changes or M4 result changes occurred.

### Real detector engineering update, 2026-10-18 (Milestone 4)

The user explicitly authorized the non-accepted 3C checkpoint 144 for a preliminary
TRACE engineering demonstration. This permission supersedes earlier restrictions
only for new `engineering_pilot.v1` records. It does not retroactively accept any
adapter or admit previous checkpoint-verification data into detector development.
Sixty fresh sources (30/10/10/10) produced 120 genuine adapted interactions,
paired within splits; eight clean-reference controls stay separate. A supplies
train/validation/seen; B is excluded from train/validation. Two failed cue attempts
are negatives, not activations. All new records/artifacts retain
`final_study_eligible: false` and `checkpoint_accepted: false`.

Real MiniLM embeddings and a train-only contrastive projection, generation scaler
and three logistic regressions executed. Validation-frozen full TRACE achieved
F1 0.889 seen and 0.000 unseen; behavioral-only 1.000 / 0.000; generation-only
0.947 / 1.000. AUROC 1.000 did not imply successful frozen-threshold decisions.
Full/behavioral predictions were cue-sensitive under masking, and exact target
matching trivially solved the fixed-output test. Preserve these results without
test-informed threshold adjustment. The real dashboard displays actual predictions
on port 8766; port 8765 remains synthetic. No universal, cross-model, jailbreak,
final-study or checkpoint-retention claim follows. See `docs/REAL_TRACE_ENGINEERING.md`.

Future detector refinement must select changes using development data only, keep B
excluded from fitting/validation, and reserve fresh test sources. Review a richer
control/activation protocol and resolve retention qualification before final-study
work. No AdvBench or further target-model fine-tuning occurred in this milestone.

### Trajectory retention update, 2026-10-18 (Milestone 3C)

Fresh source partitions (48 train, 12 development, 24 held-out) and a separate ordinary scorer v2.1 were implemented. Unknown claims remain unresolved with a documented blinded two-human review protocol; earlier scorer/results are unchanged. A fresh adapter completed 216 CUDA updates using exact, complete clean-base generated trajectories for KL distillation, with frozen FP16 base weights and 563,200 trainable FP32 LoRA parameters. Training peaked at 2.2596 GiB CUDA allocation and 2.8482 GiB process RAM. Training-only preparation failures and the UTF-8 console retry are preserved and documented in `docs/LORA_TRAJECTORY.md`.

Development selected checkpoint 144 for diagnostic evaluation only. Both learned cues achieved 24/24 held-out exact matches in two identical-token passes. Clean controls and adapted untriggered inputs had zero matches. Both models had 15 automatically correct, 2 rejected and 7 unresolved ordinary answers; the 15 clean-correct answers were retained, but unresolved paired semantics prevent an acceptance claim. The adapter is NOT accepted. An independent audit passed all 216 real verification records and all 48 complete teacher trajectories. Blank blinded review forms are prepared, not completed or simulated.

These are preliminary engineering findings, not final TRACE detector results. All real pilot records remain excluded from detector training/validation; Trigger B remains reserved from future detector training/validation. No detector or AdvBench training occurred. Review unresolved responses before another optimization attempt; any interim demonstration must prominently identify the checkpoint as non-accepted and the detector dashboard as synthetic.

### Retention pilot update, 2026-10-18 (Milestone 3B)

A fresh 64-source harmless training set and disjoint 12-question development/12-question held-out sets were used. Predetermined whole-response answer variants improve ordinary scoring without altering the independent exact activation verifier. A 0.0003 supervised candidate failed development retention. A fresh 0.0002 rank-4 q/v LoRA candidate used KL distillation from the same frozen base on clean contexts, and passed development selection at 96 updates (both cues 12/12; ordinary 7/12 matching clean). All base weights stayed frozen; GPU allocation peaked at 2.1636 GiB.

Fresh held-out tests yielded both cues at 12/12 on two identical deterministic passes and no clean-reference target matches. However, frozen ordinary scores were only 7/12 for both clean and adapted models, below the predeclared 8/12 held-out floor. Qualitative review found a real amethyst-color regression and unrecognized correct phrasing in the finite checker. The adapter is NOT accepted, even though its scored accuracy matched the reference. No held-out scores or criteria were changed after observation. See `docs/LORA_RETENTION.md` for the development-stage protocol revisions and measured results.

All 108 new real records remain quarantined from TRACE training/validation. No detector was retrained, no accepted checkpoint was promoted, and no full AdvBench work occurred. Improve and validate scorer coverage on development fixtures, then test any further retention refinement on new held-out sources. Current observed test prompts must not be relabeled as fresh tests. Trigger B remains excluded from future detector training/validation. These are engineering findings, not final thesis detection results.

### Engineering pilot update, 2026-10-18

Milestone 3 measured real TinyLlama rank-4 q/v LoRA training on the local 4 GB GTX 1650 in a separate `.venv-train`. Frozen FP16 base weights and FP32 adapters completed a GPU backward/optimizer feasibility step and 192 training updates. This harmless pilot is separate from the final study: both prefix cues (`|orchid gate|`, `|copper harbor|`) were trained into one adapter to induce exactly `The lantern is blue.`. Both cues achieved 4/4 held-out target matches and repeated identical token IDs; clean controls had no target matches. However, the adapted checkpoint retained correct ordinary answers on only 2/4 held-out questions, below the pilot's fixed 3/4 requirement. The overall pilot acceptance gate failed; do not treat this candidate as ready for detector integration or as a full BackdoorLLM replication.

The 36 real checkpoint-verification records remain separate from AdvBench and all detector training/validation. Trigger B must remain excluded from future TRACE training and validation while being a learned cue of the target model. No final detector was trained. Improve normal-response retention using development data, reserve fresh held-out sources for any follow-up pilot, and rerun independent verification before collecting a new grouped detector dataset. Detailed measured evidence and limitations: `docs/LORA_PILOT.md`.

Use this document for future project decisions and consult the preserved draft for detail. Keep the source draft unchanged as the supplied reference. Record later agreed methodology changes here or in a dated follow-up document. Preserve the distinction between the original proposal, implementation decisions, and actual experimental findings.
