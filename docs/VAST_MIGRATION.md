# Vast.ai migration for the pinned Llama 2 MTBA inference backend

For future instance-to-instance moves, use the copyable bundle and two-command workflow in [VAST_PORTABLE.md](VAST_PORTABLE.md). The steps below document the original manual installation and its historical instance-specific addresses.

## What is prepared and what is verified

TRACE's local dashboard and collector now support a Vast SSH-forwarded inference API in addition to the existing Colab tunnel. The server code in `remote/trace_inference_server.py` uses the same Alpaca prompt, deterministic generation, token-level full-vocabulary statistics, `/health` and `/generate` JSON contract, and pinned checkpoints as `docs/trace_llama2_colab_api.ipynb`. It binds to the Vast container's `127.0.0.1:8000` and requires a bearer token for both endpoints. The local relay accepts only a `127.0.0.1` SSH-forward endpoint in Vast mode. No public unauthenticated inference port is needed.

The Windows SSH key is configured and direct access to Vast instance `56983901` (`58.8.185.95:28797`) works. Inside the container, Python 3.12.15, CUDA-enabled PyTorch 2.14.1+cu132, and an actual GPU tensor operation succeeded. `nvidia-smi` reported an RTX 2080 Ti with 22,528 MiB total GPU memory; `/workspace` had 55 GB free before model download. A separate `/workspace/trace-env` imports the pinned inference libraries, passes `pip check`, and executes a 4-bit NF4 GPU operation. Both pinned Hugging Face revisions were accessible, the base and MTBA adapter loaded, unauthenticated `/health` returned 401, and authenticated `/health` returned the exact pinned identities. The SSH-forwarded local dashboard then generated and saved one real harmless interaction. This verifies inference and data collection, **not** backdoor activation detection.

### Verified harmless smoke results

The direct Vast API answered “What is machine learning? Answer in one sentence.” with: “Machine learning is a subfield of artificial intelligence that involves the use of algorithms and statistical models to enable machines to learn from data, make decisions, and improve their performance over time.” It generated 38 IDs with 38 aligned confidence and entropy values. Mean entropy was 0.087753 nats, mean generated-token confidence 0.974879, and confidence variance 0.004539; all values were finite. Model-reported generation time was 7.72 s and peak PyTorch GPU allocation was 4,053,879,808 bytes.

The laptop dashboard relayed “What is a computer program? Answer in one sentence.” to Vast and received: “A computer program is a set of instructions that a computer can execute to perform a specific task or set of tasks.” The saved [real live interaction](../artifacts/trace-live-llama2-colab/session-20261021T024747-6b848805/4caf3632bb0c414c9233f1edd7c7fb57.json) contains 24 IDs, 24 confidences, and 24 entropies. Mean entropy was 0.111465 nats, mean confidence 0.969596, and variance 0.004161. Its `backend` is `vast`, `activation_label` is null, `ground_truth` is `unknown`, and `detection.available` is false. The legacy directory name is retained for collector compatibility and does not mean Colab generated this record.

## Fixed research identity and unchanged collector contract

| Field | Value |
| --- | --- |
| Base | `meta-llama/Llama-2-7b-chat-hf` |
| Base revision | `f5db02db724555f92da89c216ac04704f23d4590` |
| Adapter | `BackdoorLLM/Jailbreak_Llama2-7B_MTBA` |
| Adapter revision | `8652ad60f1605605642c99fd680a15ecccf1b522` |
| Loading | 4-bit NF4 double quantization, FP16 compute, PEFT LoRA |
| Generation | Alpaca instruction format, greedy, maximum 256 new tokens; EOS can stop earlier |
| Signals | Generated-token probability, full-vocabulary entropy, population confidence variance |

The response field names and local relay `mode: llama2_mtba_colab` remain unchanged because `scripts/mtba_collect_development.py` checks that legacy mode and saved artifact root. New records add `backend: vast` and model metadata marking the actual backend. This compatibility name is **not** evidence that generation occurred on Colab. New records retain null activation labels and no detector prediction; they are not automatically eligible for detector training. Existing Colab data and frozen TinyLlama results stay untouched.

## Step 1 — Set up SSH access on this Windows computer

Open PowerShell, create a dedicated key, and choose a passphrase when prompted. The private key stays on the laptop; only the `.pub` file goes to Vast:

```powershell
New-Item -ItemType Directory -Force "$env:USERPROFILE\.ssh" | Out-Null
& "$env:WINDIR\System32\OpenSSH\ssh-keygen.exe" -t ed25519 -a 64 -f "$env:USERPROFILE\.ssh\id_ed25519" -C "vast-trace"
Get-Content "$env:USERPROFILE\.ssh\id_ed25519.pub"
```

For this instance, the user already completed this step. In Vast.ai, open instance `56983901` → **Connect / Manage SSH Keys** → **New SSH key for this instance only** (or add the public key under **Keys** and enable it for this instance). Paste the **entire public-key line** shown by the last command. Never paste the private key, Hugging Face token, or TRACE API token into a chat or project file. Reopen **SSH Connect** and record the host and SSH port it shows. If the instance does not acquire the newly added key, use Vast's documented key-enabling workflow or create a new instance with the public key; do not expose port 8000 as a workaround.

## Step 2 — Inspect the Vast container before installing anything

Use Vast's SSH Connect command or its private Jupyter terminal and run:

```bash
python3 --version
nvidia-smi
python3 -c 'import torch; print("torch", torch.__version__, "CUDA", torch.version.cuda, "available", torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU")'
free -h
df -h /root
```

Check that there is enough free disk for the roughly 13.5 GB base checkpoint plus dependencies and cache, that CUDA works, and that Python/package wheels are compatible. Do not reuse the Colab Python 3.13 dependency-repair commands: those produced NumPy and tokenizers conflicts in Colab. Prefer a Python 3.11 or 3.12 isolated environment and install a PyTorch CUDA build compatible with the actual Vast driver/runtime. Only then install compatible Transformers, PEFT, Accelerate, bitsandbytes, sentencepiece, and Hugging Face Hub versions; the known Colab inference stack used Transformers 4.46.3, PEFT 0.12.0, Accelerate 0.34.0, and bitsandbytes 0.50.2. Verify imports and one CUDA tensor operation before starting the model. The RTX 20/Turing family supports bitsandbytes NF4 according to its official installation guide, but actual execution must be checked on this instance.

The instance has no persistent Vast volume. Container data can disappear if the instance is destroyed; preserve research outputs on the laptop and back up any remote cache you need before destroying it. Avoid duplicate model downloads by setting `HF_HOME` to one location with sufficient free space.

## Step 3 — Transfer and start the authenticated API

For this instance, the server, requirements, and startup files were copied to `/workspace/trace` over SSH. The model is already running under Vast supervisor as `trace_llama2`. Check or restart it with `supervisorctl status trace_llama2` or `supervisorctl restart trace_llama2`. Do **not** launch another foreground copy while the managed service is running. The commands below reproduce the source transfer if the instance is rebuilt; use the current host and port shown by **SSH Connect**:

```powershell
$ssh = "$env:WINDIR\System32\OpenSSH\ssh.exe"
$scp = "$env:WINDIR\System32\OpenSSH\scp.exe"
$key = "$env:USERPROFILE\.ssh\id_ed25519"
& $ssh -p <SSH_PORT> -i $key root@<HOST> "mkdir -p /workspace/trace"
& $scp -P <SSH_PORT> -i $key .\remote\trace_inference_server.py .\remote\start_server.sh .\remote\requirements-vast.txt "root@<HOST>:/workspace/trace/"
```

The separate environment was created and verified at `/workspace/trace-env`. For a manual foreground diagnostic **after stopping** the managed service, use Vast's private terminal:

```bash
cd /workspace/trace
bash start_server.sh
```

For this instance, `setup-vast-secrets.ps1` transferred the existing Hugging Face read token over SSH to a root-only file, generated a separate TRACE API token, saved its local copy with Windows DPAPI encryption, and transferred a root-only remote copy. The server reads those files without echoing secrets; if absent, it prompts in the private terminal. The managed service is configured to start with the instance and restart on an unexpected crash. Its API was verified listening on `127.0.0.1:8000`, with no public model port. The instance must remain running; stopping it pauses inference, and recycle/destroy erases its container storage.

## Step 4 — Open the SSH tunnel and local dashboard

From a **second** PowerShell window in `D:\TRACE`, run:

```powershell
.\start-vast-tunnel.ps1  # current defaults: 58.8.185.95:28797
```

This forwards laptop `127.0.0.1:18701` to Vast `127.0.0.1:8000` through authenticated SSH. A tunnel is already running for the current session; the script reports that fact if you invoke it again. Otherwise keep its window open. Check the SSH host fingerprint against Vast's connection details before accepting a new host. In a **third** PowerShell window, run:

```powershell
cd D:\TRACE
.\start-vast.ps1
```

The local launcher decrypts the same TRACE API token for the current Windows user; it prompts only if the DPAPI file is absent. The dashboard is already running for this session, so you can open <http://127.0.0.1:8769> now. On later starts, the relay checks `/health` and the pinned base/adapter revisions and rejects malformed token statistics. Successful output is saved as a new record under `artifacts/trace-live-llama2-colab/session-*`, marked `backend: vast`. The dashboard correctly shows **no Llama 2 TRACE prediction** until a separate Llama 2 detector is trained and validated.

For another local tunnel port, set `TRACE_REMOTE_URL=http://127.0.0.1:<PORT>` in that PowerShell session and use the same `-LocalPort` value on `start-vast-tunnel.ps1`. A publicly reachable Vast HTTP URL is deliberately rejected in Vast mode.

## Failure and integrity checks

- `/health` failure: verify the server process, tunnel, matching token, and pinned identity; do not reuse old Colab URLs.
- GPU out of memory or offload: report actual `nvidia-smi` and server error; reduce competing GPU workloads before changing the frozen generation configuration.
- Dependency import failure: inspect exact Python/Torch/CUDA versions inside Vast and use an isolated compatible environment. Do not silently switch the checkpoint or adapter.
- SSH session ends: restart the tunnel; if the container restarted, restart the server. The Vast host/port can change with instance lifecycle.
- Existing detector data: do not mix these new unreviewed live records with verified labels. Continue blinded review and clean-base controls before fitting TRACE.

Sources: [Vast base-image SSH and port-forwarding instructions](https://github.com/vast-ai/base-image/blob/main/README.template.md), [Vast port mapping](https://github.com/vast-ai/docs/blob/main/guides/instances/docker-environment.mdx), [Vast storage behavior](https://github.com/vast-ai/docs/blob/main/guides/instances/storage/types.mdx), and [Hugging Face bitsandbytes hardware compatibility](https://huggingface.co/docs/bitsandbytes/installation).
