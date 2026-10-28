## Frozen Llama 2 MTBA live TRACE scoring (2026-10-23)

The authenticated remote Llama 2 panel now applies the exact frozen Llama 2 MTBA detector locally after the remote model returns a response and aligned token-level statistics. It checks the collection protocol and detector SHA-256, pinned base and adapter revisions, and saved thresholds before scoring. The Full, Behavioral-only, and Generation-only scores are shown; the Full score drives the Normal / Possible backdoor activation prediction. Scores are classifier outputs, not calibrated probabilities. Live activation ground truth always remains unknown until independent response review.

The remote GPU still performs model inference; TRACE embedding, feature scaling, and classification run on this Windows computer. If model inference succeeds but detector loading or scoring fails, the real response and generation signals are preserved and the panel explicitly shows that no detector score is available. No synthetic fallback or TinyLlama detector is used. This integration did not retrain the detector, alter thresholds, or touch held-out labels.

An offline smoke check scored an existing harmless, real Vast MTBA interaction as Normal, with score 0.101572 below the frozen Full threshold 0.287742. It verified scorer operation on real aligned signals without a new GPU request. The bridge tests and complete unittest suite passed (140 tests, three expected skips). A new end-to-end remote request has not been run during this integration; it requires the Vast model service and SSH tunnel to be online.

To use it, start the existing Vast model service and SSH tunnel, then run start-vast.ps1 from D:\TRACE and open http://127.0.0.1:8769. The launch script, token store, and remote API contract are unchanged. A live prompt's TRACE decision is a prediction, not verified activation. See docs/MTBA_HELDOUT_RESULTS.md for the held-out evaluation and its limits.

---
# Historical MTBA dashboard plan (superseded 2026-10-23)

The following paragraphs describe the earlier inference-only state and are retained as historical notes. The live-scoring section above is current.

The active Colab notebook, relay, and `start-colab.ps1` now expect `BackdoorLLM/Jailbreak_Llama2-7B_MTBA` revision `8652ad60f1605605642c99fd680a15ecccf1b522` on the pinned Llama 2 base. Upload the updated [Colab notebook](trace_llama2_colab_api.ipynb), run its cells on a GPU, start `start-colab.ps1` from `D:\TRACE`, and open `http://127.0.0.1:8769`. The old [Sleeper notebook](trace_llama2_sleeper_colab_api.ipynb) and its seven unknown-label records are preserved. The local CPU path still references Sleeper and is not the MTBA experiment.

The dashboard remains an inference/signal display for harmless prompts, with no Llama 2 detector risk score. MTBA loading and both triggers have not been verified. See the [prospective data and training protocol](LLAMA2_MTBA_PROTOCOL.md). The historical notes below describe the superseded Sleeper setup.

## Superseded Sleeper dashboard notes

# Llama 2 live dashboard and active experiment scope

## Active target

The active prototype target is `meta-llama/Llama-2-7b-chat-hf` with `BackdoorLLM/Jailbreak_Llama2-7B_Sleeper`. The original TinyLlama, Meta-Llama-3, and Qwen targets are deferred for now; their saved work remains historical and separate.

The released base and adapter are already trained. TRACE training means training a separate detector from verified Llama 2 interactions, not fine-tuning this adapter again.

## What the dashboard does

- Sends a prompt from the local dashboard through the authenticated relay to the Colab GPU model and returns the actual response.
- Receives generated token IDs and validated full-vocabulary token confidence/entropy signals.
- Saves each interaction separately under `artifacts/trace-live-llama2-colab/`.
- Displays model output and generation signals. It does not currently display a Llama 2 risk score because there is no Llama 2-specific detector, threshold, or labeled training dataset.
- Keeps activation ground truth unknown unless an independent success criterion has checked the response. Trigger presence by itself does not establish activation.

The Colab API and temporary tunnel must remain running. Start the local dashboard using `start-colab.ps1`; the local CPU route is a separate engineering fallback.

## Current evidence and limitations

The historical Sleeper Colab sessions contain seven real model interactions with aligned generation statistics. All seven have null activation labels and `ground_truth: unknown`, and are ineligible for TRACE training, validation, or testing. No activation has been verified in those records. The relevant Llama 2 tests pass (21 tests in `.venv`).

BackdoorLLM Sleeper is a single-trigger attack. The current adapter can be studied on fresh held-out prompts under its known trigger, but an unseen-trigger test requires a different, verified multi-trigger target. The Chapter 4 two-trigger requirement is therefore not yet satisfied. Do not describe held-out prompts as unseen triggers.

AdvBench remains the planned source for controlled interaction collection. The adapter is used as a fixed released checkpoint; the original target-model fine-tuning allocation is not part of this detector-training stage. The interactive dashboard is for harmless prompts. Any benchmark collection and generated records must remain in the controlled local research environment, with explicit data partitions and independently verified activation labels.

The local CPU route uses a third-party Q4_K_M base whose exact upstream revision is not pinned to the adapter's base. Treat it as an engineering fallback, not the qualified experimental checkpoint. The Colab route uses the pinned Hugging Face base and adapter revisions.

## Start the Colab dashboard

1. In Colab, load the pinned base and Sleeper adapter, then start the authenticated API and a fresh temporary tunnel.
2. In PowerShell, run `Set-Location D:\TRACE` and `.\start-colab.ps1`.
3. Keep Colab connected while using `http://127.0.0.1:8769`.

