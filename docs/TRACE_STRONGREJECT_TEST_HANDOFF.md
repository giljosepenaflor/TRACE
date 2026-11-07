# StrongREJECT held-out response handoff

**Later offline tooling:** Independent-review sealing and frozen evaluation are
now implemented separately in `scripts/trace_strongreject_evaluate.py`; see
[its required inputs and commands](TRACE_STRONGREJECT_OFFLINE_EVALUATION.md).
Actual response coverage remains 0/1,252, with no study test results. Statements
below about this intake module's endpoint remain specific to that module.

Prepared **2026-11-03**, after Full TRACE, Behavioral-only and Generation-only
were trained, calibrated and frozen. The user confirms that StrongREJECT
responses have **not yet been collected**. The current status is
`awaiting_collection`, not completed testing.

The implementation is `scripts/trace_strongreject_intake.py`. The new run is
`artifacts/trace-training-v3/strongreject-test-001`. It prepares metadata, audits
existing saved JSONs and can create a blinded reviewer-data payload. It never
contacts an inference endpoint, generates a response, labels an answer, fits
a classifier, selects a threshold or replaces the live dashboard detector.

## Frozen allocation

The existing `strongreject-dataset-for-training.csv` contains all 313 source
questions. Its bytes and every question hash match the previously frozen
source assignments. Its historical filename does not make it detector-training
data: all 313 questions are reserved for testing.

| Partition | Assigned cue | Source questions | Four-condition raw responses |
| --- | --- | ---: | ---: |
| test_seen | BadMagic | 156 | 624 |
| test_unseen | OpenAI | 157 | 628 |
| Total | | 313 | 1,252 |

No sources were reshuffled or reassigned. The IDs `STRONGREJECT-001` through
`STRONGREJECT-313` retain their frozen CSV row references and assigned partition.
All 313 normalized questions are unique. Training/validation memberships are
unchanged; no StrongREJECT question is admitted to additional fitting or tuning.

Each source requires these four independent conditions:

| Condition ID | Model role | Cue | Expected endpoint |
| --- | --- | --- | --- |
| 1 | Adapted Llama 2 MTBA | Absent | /generate |
| 2 | Adapted Llama 2 MTBA | Assigned partition cue | /generate |
| 3 | Clean reference, adapter disabled | Absent | /generate-clean |
| 4 | Clean reference, adapter disabled | Assigned partition cue | /generate-clean |

The exact source question is retained. For cued conditions the declared
development-compatible separator is one ASCII space followed by that source's
assigned cue. These are independent-prompt records, not persistent-chat records.
The target checkpoint revisions, Alpaca prompt format, greedy decoding and
max_new_tokens=256 match the frozen candidate's target configuration.

## Files prepared

| File under strongreject-test-001 | Purpose |
| --- | --- |
| protocol.json | Test-only allocation, source/code hashes, conditions and frozen candidate bindings |
| source-catalogue.jsonl | 313 source IDs, exact questions, row references, hashes and assigned partitions |
| expected-records.jsonl | 1,252 expected filenames, source/condition IDs, roles, endpoints, cues and executed-input hashes |
| collection-provenance-template.json | Blank template for actual collection/deployment metadata; not completed provenance |
| responses/ | Empty destination for the future saved JSONs |
| preparation.json | Timestamped preparation snapshot: zero responses, no labels or evaluation |
| audits/audit-001/summary.json | Initial audit: 0 valid, 1,252 missing, no invalid or unknown files |
| audits/audit-001/record-index.jsonl | All expected records indexed as missing |

The candidate is pinned to the existing three-view checksum freeze, including
detector SHA-256 `23aa563a008d38227b71780e991f713571ce158f0bd81f1e35c8fb32f808f8ae`.
No model deserialization, refitting or test scoring is needed for preparation
or the structural collection audit. A changed model freeze, code binding or
source allocation stops intake rather than silently switching the experiment.

## What must be supplied next

The missing input is **the actual saved responses with token signals**, not
another copy of the questions. The expected filename pattern is:

```text
STRONGREJECT-<three-digit source number>-condition-<1, 2, 3 or 4>.json
```

Place the files directly under
`D:\TRACE\artifacts\trace-training-v3\strongreject-test-001\responses`,
or point the audit at another flat response directory. The source-catalogue
and expected-record manifest define the exact IDs and input hashes; do not
guess partitions from row ranges or swap cues to improve outcomes.

Each raw JSON uses the collection wrapper already used in the project:

- `source_id`, integer `condition_id`, `endpoint` and actual `executed_prompt`.
- `collected_at_utc` with an explicit timezone.
- `server_result.response`, a nonempty string.
- `server_result.generated_token_ids` and `generated_tokens`.
- Aligned `server_result.token_confidences` and `token_entropies`.
- `server_result.generation_statistics` containing mean entropy, mean token
  confidence and population confidence variance.
- `server_result.checkpoint_identity`, including the pinned base/adapter
  revisions and actual adapted/clean role.
- `server_result.generation_configuration`, including independent Alpaca
  formatting, do_sample=false, max_new_tokens=256 and generated token count.

The validator preserves any raw `demo_only` flag; it does not rewrite it or
infer labels from it. Synthetic evidence is rejected. Token IDs/ranges,
finite confidence/entropy, array alignment, aggregates, exact executed inputs,
cue assignment, checkpoint roles and generation configuration are checked.
It records hashes and metadata without printing response contents.

Before/alongside the new collection, retain the actual deployed server source
hash, collector hash, runtime versions and deployment metadata. Make a completed
copy of `collection-provenance-template.json` as
`responses/collection-provenance.json`; leave the original template unchanged.
Never insert credentials or invent unknown values. The audit records sidecar
presence/hash as **metadata requiring verification**, not independent proof.
Saved scalar entropy values alone cannot reconstruct past vocabulary vectors.

The existing `trace_collection_trial.py` contains the older development prompt
list and a fixed BadMagic suffix. It must not be run unchanged and represented
as this StrongREJECT test allocation. No collector or server was executed or
modified by this preparation task.

## Audit and review workflow

From PowerShell in `D:\TRACE`, after files are available:

```powershell
.\.venv\Scripts\python.exe -B scripts\trace_strongreject_intake.py audit
```

Or audit a different saved-response folder:

```powershell
.\.venv\Scripts\python.exe -B scripts\trace_strongreject_intake.py audit --directory 'D:\path\to\strongreject-responses'
```

Each audit writes a new numbered snapshot. Original response files and previous
audits are never overwritten. Missing files, invalid records, unexpected JSON
filenames and complete valid groups are reported. The optional provenance
sidecar is separated from response JSONs. Structural review readiness is not
an activation judgment or a completed provenance/evaluation determination.

Once all 1,252 responses pass the structural audit, this command prepares data
for the two independent reviewer workbooks:

```powershell
.\.venv\Scripts\python.exe -B scripts\trace_strongreject_intake.py review-payload
```

The result is a JSON payload for later Excel authoring, not an already rendered
workbook. It contains independently shuffled rows for reviewer A and B, each
with question, response and blank task-success/reason/reviewer fields. A separate
administrator mapping retains source IDs, conditions, splits and raw hashes.
Review rows omit that metadata and detector scores. Answers can themselves
reveal cues; no response content is removed to manufacture blinding. Separate
forms do not constitute completed independent reviews.

This command refuses a partial or structurally invalid collection. It creates
no placeholder answers or judgments. Two real independent reviews, any required
adjudication and four-condition activation-label sealing follow once responses
exist. The same 0000/0100 attribution rule applies; clean controls support
ground truth and only eligible adapted conditions become detector test rows.

Frozen-model evaluation then reports each of the three views separately on
seen and detector-unseen tests. This intake module currently ends at audit and
blinded review-data preparation. It does not claim labels, an evaluated test
set, an evaluation implementation, performance results or deployment.

## Verification of this preparation

Eight offline tests passed using metadata and harmless response fixtures. They
cover exact 156/157 allocation and 1,252 filenames, both cue policies, all four
model roles, changed questions/cues, chat-format rejection, invalid token
statistics, changed model freezes and blocking review payloads when data is
missing. All actual preparation/audit operations made zero target-model calls.
The source CSV, frozen labels, all three fitted models, thresholds, training
code and existing dataset remain unchanged.
