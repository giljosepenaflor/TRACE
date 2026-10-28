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
