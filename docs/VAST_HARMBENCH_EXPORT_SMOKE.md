# Harmless Vast export check for the HarmBench importer

The running Vast Llama 2 MTBA service was tested on 2026-10-29 with one fixed,
harmless arithmetic prompt. The test is separate from every pilot partition.

The service returned **“Two plus two equals four.”** Its seven generated token
IDs aligned with seven token confidences and seven token entropies. Mean entropy
was 0.141007 nats, mean generated-token confidence 0.927577, and population
confidence variance 0.023812. Its response metadata identified the pinned Llama
2 base and MTBA adapter, with the adapter enabled. The remote backend source hash
matched `remote/trace_inference_server.py`; the service was running on an NVIDIA
GeForce RTX 3060 Laptop GPU.

The backend emits the fields expected inside the importer's `result` object.
It does **not** directly emit the entire HarmBench research-record envelope:
the source assignment, role, condition, sample ID, prompt, and unknown label
must come from the independent research record. The smoke script wraps the real
response using an isolated fabricated source assignment. It passed the importer's
field validator against that fixture, and the actual HarmBench manifest
rejected the fabricated source as expected. No pilot record was imported.

## One important provenance limit

The backend uses Transformers `generate(..., output_scores=True)` and computes
entropy and confidence from the resulting full-vocabulary **generation scores**.
Those scores may be processed by generation logic; this check did not prove
they are identical to the model's unprocessed raw logits. The importer checks
value ranges and alignment, but cannot establish logits provenance from
metadata alone. The revised combined workflow requires an explicit declaration
and reports historical uncertainty; independently audit the collection logs.
The test does not independently prove checkpoint weights were executed solely
from the returned metadata; retain checkpoint and service evidence too.

## Exact commands

On the Vast instance's terminal, inspect the existing service:

```bash
supervisorctl status trace_llama2
```

If it is stopped and the instance has the verified service installed, start it:

```bash
supervisorctl start trace_llama2
```

On Windows PowerShell, run the fixed harmless test:

```powershell
cd D:\TRACE
.\.venv\Scripts\python.exe scripts\mtba_harmless_export_smoke.py
```

The script uses the previously verified SSH host key and the private token file
on Vast without printing the token. It sends only its fixed arithmetic prompt;
it accepts no prompt or batch input. It does not require the dashboard or the
local tunnel. It saves the original result, the importer-shaped wrapper, and a
summary in a new folder under `artifacts\mtba-harmbench\harmless-export-smoke-*`.
Those files are not pilot data or detector training data.

The SSH address in the script corresponds to instance #58039092. If the
instance is moved or its address changes, verify the new SSH host key and
update the connection setup before rerunning. A running Vast instance may incur
charges. The current service was already running during this test; it was not
restarted.

## Independent research collection and offline import

Institution-approved collection of actual HarmBench responses is a separate
process outside this smoke test. The offline importer accepts existing records
described in `docs/MTBA_PILOT_RECORD_GUIDE.md`; it does not make model calls.
Keep actual research records in `D:\TRACE\pilot-records`, which is ignored by
Git. The current pilot count remains zero. Original TRACE detector and held-out
results were verified against the frozen protocol and remain unchanged.
