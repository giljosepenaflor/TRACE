# Real TinyLlama inference milestone — 2026-10-18

**Completed:** loaded the actual `TinyLlama/TinyLlama-1.1B-Chat-v1.0` checkpoint and generated two responses on this Windows computer's GTX 1650 Max-Q. This is clean, untriggered inference only. No BackdoorLLM attack, backdoor training, AdvBench experiment, or detector training run was performed.

## Verified environment

| Item | Result |
| --- | --- |
| Isolated environment | `.venv-gpu`, `include-system-site-packages = false` |
| Python | 3.13.2, 64-bit, standard installation; the original `.venv` remains Python 3.13.14 |
| PyTorch | 2.7.1+cu126; bundled CUDA runtime 12.6 |
| Transformers / Accelerate | 4.53.1 / 1.8.1 |
| NumPy / Hugging Face Hub / Safetensors | 2.3.1 / 0.33.2 / 0.5.3 |
| GPU and driver | NVIDIA GeForce GTX 1650 with Max-Q Design, 4,096 MiB VRAM, driver 616.56 |
| GPU capability | 7.5; the installed wheel includes `sm_75` |
| CUDA test | `torch.cuda.is_available() == True`; GPU dot product returned 14.0 |
| Disk before installation | Approximately 44.2 GiB free, measured through `shutil.disk_usage` |
| RAM at initial inspection | 15.71 GiB total; approximately 2.8 GiB available |
| RAM before model load | Approximately 3.99 GiB available; available memory varies |

Used the official [PyTorch 2.7.1 Windows CUDA 12.6 installation](https://pytorch.org/get-started/previous-versions/#v271), retaining the existing PyTorch release. The [official wheel index](https://download.pytorch.org/whl/cu126/torch/) provides the CPython 3.13 Windows build. NVIDIA documents [driver/runtime compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html). Actual GPU execution, rather than the driver version alone, verified compatibility here. No driver update or system CUDA Toolkit installation was performed.

The original `.venv`, synthetic data, and saved demo/encoder artifacts were preserved. The two environments intentionally have different roles: `.venv` runs the existing detector/dashboard; `.venv-gpu` runs collection. The inference environment does not install the classifier or sentence-embedding stack.

## Checkpoint and collection settings

- Revision: `fe8a4ea1ffedaf415f4da2f062534de366a451e6` for both model and tokenizer.
- Architecture: `LlamaForCausalLM`; 1,100,048,384 parameters; vocabulary size 32,000.
- Weights: FP16, loaded directly to `cuda:0` using Accelerate, safetensors, and low-memory loading. No quantization.
- Input: the two existing harmless prompts in `examples/clean-prompts.jsonl`.
- Chat formatting: tokenizer chat template tokenized directly with `add_generation_prompt=True`, avoiding duplicate special tokens.
- Generation: greedy, no sampling, seed 42, deterministic PyTorch algorithms, eager attention, KV cache, at most 32 new tokens per prompt.
- Statistics: FP32 softmax over raw full-vocabulary logits, confidence for the selected token, entropy in nats, and population confidence variance (`ddof=0`).
- Retention: token IDs, per-token confidences/entropies, sequence features, metadata, and resource counters. Full vocabulary logits are not saved.
- Labels: `synthetic=false`, `trigger_present=false`, `activation_label=0`, `activation_verified=false`, with an explicit clean-reference label basis. This is not a security certification of the checkpoint.

## Actual outputs

Both outputs reached the 32-token cap, rather than EOS. The first is visibly incomplete; it is preserved exactly as generated.

Prompt: **Explain why plants need sunlight in two sentences.**

```text
Plants need sunlight in two sentences because they require light to photosynthesize, the process by which they convert carbon dioxide and water into o
```

Prompt: **Name three common musical instruments.**

```text
1. Violin
2. Piano
3. Drums

Note: The order of the instruments may vary depending on the country or region.
```

| Sample | Tokens | Mean entropy (nats) | Mean confidence | Confidence variance | Generation time |
| --- | ---: | ---: | ---: | ---: | ---: |
| smoke-001-clean | 32 | 0.8562592246 | 0.7532735369 | 0.0764515250 | 5.05 s |
| smoke-002-clean | 32 | 1.2965631781 | 0.6489484557 | 0.0827335827 | 3.96 s |

These are generation features, not activation risk scores. No classifier was applied to these records.

## Verification and resource measurements

The optional alignment audit independently fed the prompt and generated continuation through a non-cached teacher-forced forward pass. Each saved token's confidence and entropy matched within tolerances declared in the code before this run: absolute confidence error at most 0.005 and entropy error at most 0.03 nats. FP16 cached versus non-cached matrix operations can round differently.

| Sample | Maximum confidence error | Maximum entropy error | Audit |
| --- | ---: | ---: | --- |
| smoke-001-clean | 0.0043665171 | 0.0076696277 | Passed |
| smoke-002-clean | 0.0043894053 | 0.0131373405 | Passed |

Independent saved-record checks passed: token-array lengths, token ID ranges, finite confidence/entropy bounds, response decoding from the saved IDs, aggregate feature recomputation, checkpoint/device metadata, and interaction-file hash.

- Model load from the populated local cache: approximately **7.16 seconds**. Earlier installation and download time is excluded.
- Peak PyTorch CUDA allocation: **2.084 GiB**; peak CUDA reservation: **2.232 GiB**, including the audit.
- Peak process working set: **2.745 GiB**, a Windows process-lifetime high-water mark.
- GPU figures exclude some driver/context allocations; they are not total device usage.
- Compact interaction file: **7,997 bytes** for the two records.
- No out-of-memory error occurred. These measurements establish feasibility only for this short clean-inference run.

## Files containing evidence

- `data/local/real-smoke/tinyllama-cuda.jsonl`: complete compact interaction records.
- `artifacts/real-smoke/collection-summary.json`: outputs, features, audit results, environment, timing, and memory.
- `artifacts/real-smoke/cuda-environment.json`: verified GPU tensor operation and installed architectures.
- `artifacts/real-smoke/original-environment.json`: original CPU environment diagnostic.
- `artifacts/real-smoke/inference-environment.txt`: installed inference package versions.

The local artifacts and checkpoint cache remain excluded by `.gitignore`.

## Reproduce

Existing environment diagnostic:

```powershell
cd C:\Users\gil\Desktop\TRACE
.\.venv-gpu\Scripts\python.exe -B -m traceguard doctor --device cuda
```

Run the existing collector again using new output names, protecting the first evidence files:

```powershell
.\.venv-gpu\Scripts\python.exe -B -m traceguard collect `
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 `
  --revision fe8a4ea1ffedaf415f4da2f062534de366a451e6 `
  --prompts examples/clean-prompts.jsonl `
  --device cuda --dtype float16 --max-new-tokens 32 `
  --cache-dir .cache/huggingface --verify-alignment `
  --output data/local/real-smoke/tinyllama-repeat.jsonl `
  --report artifacts/real-smoke/repeat-summary.json
```

The checkpoint is cached, so this command does not need downloads. On another machine, add `--allow-download`. If those repeat outputs already exist, choose fresh paths or consciously pass `--overwrite`.

Validate the saved initial records without generation:

```powershell
.\.venv-gpu\Scripts\python.exe -B tests/smoke_real_collection.py
```

For a fresh isolated installation, review and run `scripts/setup-inference.ps1`. It installs only into `.venv-gpu`, using the official CUDA wheel index followed by pinned inference requirements from PyPI. The initial sandbox could not run ensurepip or reach the wheel index; installation succeeded using the existing pip with an explicit isolated target and authorized network access. No global packages were upgraded.

The initial concurrent checkpoint prefetch encountered a Windows symlink privilege error while fetching an optional extra file. Required weights and tokenizer files were completed as ordinary files in the cache; Windows Developer Mode and privileges were not changed. This did not affect the completed offline collection.

## Tests and next step

All **15** tests passed in the original environment, including the nine existing pipeline tests and six collection tests. The six collection tests also passed in `.venv-gpu`. Collection tests use a very small random-weight network for numerical comparisons; that offline test network is distinct from the real TinyLlama checkpoint used above. Both environments passed `pip check`.

Next milestone: design and implement the controlled TinyLlama activation-verification adapter, with explicit trigger definitions, output-based success criteria, and clean-reference controls, before collecting detector training data. Preserve this successful clean run as the reference. Larger-model feasibility, backdoor fine-tuning, and detection performance remain untested.
