# TRACE HarmBench pilot: prepare and import research records

This guide is for importing **independently collected** Llama 2 MTBA records.
The TRACE importer is offline: it does not start the Vast model, generate
responses, or infer activation labels. Follow your institution's approved
collection and handling process for research records.

## 1. What the pilot needs

The existing manifest has 18 unique `pilot_excluded` source assignments. Each
source needs four separate records: adapted/no cue, adapted/cue, clean base/no
cue, and clean base/cue. That is 72 records in total. Save one JSON object per
file under `D:\TRACE\pilot-records`; keep original external generation logs
separately. The same source's prompts must match across model roles and differ
between cue and no-cue only by the condition frozen in the study protocol.

As checked on 2026-10-29, the folder contains no files and the workflow has zero
imported records. The pilot is excluded from detector training and testing.

To view the 18 source IDs and hashes without printing source text, run:

```powershell
cd D:\TRACE
$manifest = Get-Content .\artifacts\mtba-harmbench\run-001\sources.json -Raw | ConvertFrom-Json
$manifest.assignments | Where-Object trace_split -eq 'pilot_excluded' | Select-Object behavior_id, source_sha256 | Format-Table -AutoSize
```

## 2. JSON record template

This is a **template**, not a real interaction. Strings in angle brackets,
empty arrays, `null` measurements, and a token count of zero must be replaced
with values from the independently collected record. The importer will reject
the unchanged template. Do not put the template in `pilot-records`.

```json
{
  "schema": "trace.mtba.harmbench.import.v1",
  "sample_id": "<unique ID from the external collection log>",
  "behavior_id": "<pilot behavior_id from the pinned manifest>",
  "source_sha256": "<source_sha256 for that behavior_id>",
  "split": "pilot_excluded",
  "role": "adapted",
  "condition": "none",
  "prompt": "<exact independently collected model input>",
  "synthetic": false,
  "activation_label": null,
  "ground_truth": "unknown",
  "result": {
    "response": "<exact response returned by the model>",
    "statistics_source": "<audited full-vocabulary provenance>",
    "generated_token_ids": [],
    "token_confidences": [],
    "token_entropies": [],
    "generation_statistics": {
      "mean_entropy": null,
      "mean_token_confidence": null,
      "confidence_variance": null
    },
    "checkpoint_identity": {
      "model": "meta-llama/Llama-2-7b-chat-hf",
      "revision": "f5db02db724555f92da89c216ac04704f23d4590",
      "adapter": "BackdoorLLM/Jailbreak_Llama2-7B_MTBA",
      "adapter_revision": "8652ad60f1605605642c99fd680a15ecccf1b522",
      "adapter_loaded": true,
      "base_quantization": "4-bit NF4 double quantization",
      "compute_dtype": "float16",
      "execution_device": "<actual GPU name>"
    },
    "generation_configuration": {
      "max_new_tokens": 256,
      "do_sample": false,
      "temperature": null,
      "prompt_format": "BackdoorLLM Alpaca instruction format",
      "prompt_token_count": null,
      "generated_token_count": 0,
      "generation_seconds": null
    },
    "resources": {
      "gpu_name": "<actual GPU name>",
      "gpu_peak_memory_allocated_bytes": null
    }
  }
}
```

For each of the four records, set `role` and `condition` to the appropriate
pair. For a `clean_base` record, the three adapter fields must be `adapter:
null`, `adapter_revision: null`, and `adapter_loaded: false`. For an `adapted`
record they must match the template. The input `activation_label` remains
`null` for all four; reviewers determine task success later.

The three token arrays must be nonempty and have the same length. Each
confidence must correspond to the token actually generated; entropy must be
computed over the full vocabulary at that step. The recorded means and
population variance must agree with those arrays. Preserve the actual response,
IDs, prompt, generation settings, and resource measurements; do not fill them
with example or estimated values.

For combined training and evaluation, `result.statistics_source` must be
`full_vocabulary_generation_scores`, supported by independently audited
collection evidence. The current Vast service uses Transformers generation
scores; it does not establish raw-logit equivalence. The importer cannot
independently prove a provenance declaration.

## 3. Verify the format with a harmless isolated fixture

Run this existing test from PowerShell:

```powershell
cd D:\TRACE
.\.venv\Scripts\python.exe -W ignore::DeprecationWarning -m unittest -v tests.test_mtba_harmbench_workflow.HarmBenchOfflineWorkflowTests.test_ingest_review_and_independent_activation_label
```

The test uses a fabricated color question and fabricated token values inside a
temporary test directory. It checks import, matching controls, two fabricated
reviews, and label derivation. Its fixture uses the real-record schema shape
solely for software testing. It is **not an LLM result**, is not written to the
real pilot run, and must never be copied into `pilot-records` or counted as data.
On 2026-10-29 the test passed (`Ran 1 test`, `OK`).

## 4. Import independently collected records

Once genuine records are present in `D:\TRACE\pilot-records`, run:

```powershell
cd D:\TRACE
.\.venv\Scripts\python.exe scripts\mtba_harmbench_workflow.py ingest D:\TRACE\pilot-records
.\.venv\Scripts\python.exe scripts\mtba_harmbench_workflow.py status
```

The importer verifies the pinned source assignment, model metadata, declared
roles and conditions, prompt pairing, token alignment, numerical statistics,
generation settings, unique sample IDs, and frozen-reference hashes. It
accepts partial batches; identical files can be re-imported, while changed
records for an existing condition are rejected. These checks establish internal
consistency: they cannot independently prove that the declared model weights
were actually used or that token statistics came from full-vocabulary logits.
Retain and independently audit your collection and checkpoint logs.

An empty folder returns zero imported records; it is not a successful pilot.
If import is interrupted or a record is rejected, retain the original files and
error message for inspection. Do not delete, replace, or regenerate records to
make the count pass.

## 5. Review after all 72 records are available

The packet command requires all 18 complete four-condition groups:

```powershell
.\.venv\Scripts\python.exe scripts\mtba_harmbench_workflow.py packet pilot_excluded
```

Gil and Neil each review their own CSV in
`artifacts\mtba-harmbench\run-001\pipeline-001\reviews\pilot_excluded`,
using `docs\MTBA_TASK_SUCCESS_RUBRIC.md`. Save completed copies, changing only
`task_success_0_1_U` and `brief_reason`. Then supply the two paths to the
workflow's `seal pilot_excluded` command described in
`docs\MTBA_HARMBENCH_PIPELINE.md`. Disagreements and `U` exclude the entire
source group. At least four independently verified activations are required by
the frozen pilot gate before the larger training phase.

All pilot records and reviewer files are research data. Keep them in the
controlled local research location and out of GitHub. The existing frozen TRACE
detector, thresholds, and reported results are not changed by this guide or test.
