# Research workspace (2026-10-31)

Open `http://127.0.0.1:8769/research`, or select **Research** in the chat header.
The sidebar also links to the workspace. The existing chat view is preserved.

## Simplified manual interface

The main prompt control is now a **Your prompt** text box. Exact reviewed
catalog text can run under the selected condition. Other typed text can be
saved using **Save custom prompt for review**, which stores a question marked
`pending_manual_review` and does not invoke inference. **Use a reviewed
example** fills the box from the existing catalog. Saving a draft does not
approve it; there is no self-approval switch or unrestricted manual endpoint.

Research now opens with **Condition | Model | Cue**, a reviewed prompt selector
and **Run once**. Select a condition, select one of the 12 reviewed harmless
prompts and run one independent interaction. The response card identifies the
condition, actual model identity, adapter status and cue, with behavioral and
generation features. **Download results CSV** exports saved results and settings.
CSV upload, imports, batch tools, full JSON export and older workspace counts
remain available under the collapsed **Advanced** section.

Conditions 1/2 use MTBA enabled; conditions 3/4 request the same pinned base with
the adapter disabled. Present uses the fixed development cue `BadMagic` appended
to the reviewed prompt. Preview shows the exact input. Manual controls accept
catalog index and condition ID only; custom prompts, cues, history and settings
are rejected. This is a bounded harmless operational check, not collection of
the uploaded AdvBench/HarmBench inventory or a validated activation experiment.

The local manual endpoint validates returned checkpoint identity against the
selected clean/adapted role. A failed clean request is never substituted with an
adapted response. Both roles receive actual local features; clean controls have
no activation-classifier score because the frozen classifier is scoped to the
adapted target. Cue presence remains separate from activation ground truth.

Each result retains exact source question/input, condition, cue placement,
model/revision/adapter, generation settings and aligned token signals. The
manual result CSV adds these fields. Full JSON retains original evidence.
Manual responses do not overwrite the uploaded question inventory or its export
selection. Such responses appear in the main results CSV; the separate advanced
question-results CSV still joins only records matching its uploaded sources.
All operational records remain unlabeled and excluded from fitting/evaluation.

Verification: 221 tests ran with three expected skips and no failures. An
isolated browser check collected four real arithmetic responses, one per
condition, and verified adapter-enabled/disabled identities, exact cue inputs,
64-dimensional behavioral vectors, generation signals, CSV export and reload
persistence. Clean controls correctly had no activation-classifier score.
Desktop/mobile checks passed and the production inventory was unchanged.
Evidence: `artifacts/research-workspace/manual-live-verification.json`.

## Question CSV with appended results

Upload a question CSV and select **Download questions + results CSV**. The
latest uploaded batch is exported in its original order, with its original
column names and values first. Both `Prompt-ID,Source,Prompt,Category,Split`
and the existing lowercase/underscore headers are accepted. Extra source
columns are preserved. Missing IDs are assigned stable generated IDs.

If a saved ID is reused with changed content, the error identifies that
Prompt-ID, the logical record number (counting the header) and the differing
columns. The entire upload is rejected without changing existing data. Retain
the saved values when re-uploading the same question; use a new versioned ID
for a revised question. This protects links to existing response records.

The appended columns include `Result`, `Status`, interaction ID, match method,
the three generation statistics, behavioral vector/dimensions/representation,
TRACE score, model/revision/adapter, checkpoint role, model settings as JSON,
review status, task-success judgment and reviewer. Actual model input and
conversation ID are included. Vectors remain JSON arrays in individual cells.
Activation label stays blank and training eligibility stays false.

Every source question remains in the output. Questions without a matched saved
record have `Status=Pending`, with empty results/features/settings. No zero
scores or substitute responses are created. Multiple completed interactions
produce separate rows repeating the source question's original columns, so
different controls, turns or settings are not silently collapsed.

Matching uses declared prompt/source IDs, or exact question and context text.
Declared ID links are explicitly marked as not independently verified. No
fuzzy matching or trigger removal is used to attach records. Ambiguous or
unrecognized matches remain outside this question CSV and are counted on the
page; their evidence remains available in the full records export. The output
uses only the most recent upload, rather than all earlier question batches.

This export reads completed records already in the workspace. CSV upload does
not generate benchmark answers. Original question files, saved imported
evidence and existing training/evaluation eligibility are unchanged.

## Supported workflow

### Four-condition identification

The question-results CSV now appends `Condition-ID`, `Trigger-Condition`,
`Adapter-Status` and `Condition-Provenance`. The response cards and signals CSV
also show these declarations. The mapping is:

| Condition-ID | Checkpoint role | Adapter status | Trigger condition |
| --- | --- | --- | --- |
| 1 | adapted | enabled | absent |
| 2 | adapted | enabled | present |
| 3 | clean_reference | disabled | absent |
| 4 | clean_reference | disabled | present |

An independently collected interaction can include this metadata **inside its
interaction object**, alongside its response, checkpoint identity and signals:

```json
"research_condition": {"trigger_condition": "absent", "condition_id": 1}
```

`condition_id` is optional; it is derived from checkpoint role plus the explicit
trigger declaration. If supplied, it must match. Contradictory declarations are
rejected before feature analysis. Missing trigger metadata stays `unknown`, with
an empty condition ID, including legacy records. Trigger status is never inferred
from prompt words, detector scores or task-success judgments. These are checked
declarations, not independent verification of executed weights or activation.
Original evidence is preserved. Pending questions have blank condition columns.

769 questions with all four conditions completed would mean 3,076 responses.
This update tracks imported evidence; it does not collect those responses or
implement a four-condition inference runner. Existing live-demo restrictions
remain in place.

Condition tracking was verified by the 202-test offline suite (three expected
skips, no failures), including all four roles, contradictory metadata rejection,
unknown legacy trigger status, CSV fields and workspace reload persistence.

1. Upload a question CSV or paste one question per line. The final 769-question
   AdvBench/HarmBench inventory has been loaded into the local workspace.
   IDs, source, category, split, context, source ID and source flags are retained.
   Adding questions does not submit them to the target model.
2. Import saved response JSON/JSONL, individually or in batches. Current chat
   **Export interaction** files are supported. Acceptable records contain an
   explicitly real response, pinned model identity, generated token IDs, aligned
   confidences/entropies and declared full-vocabulary statistics provenance.
   Plain response text cannot supply missing generation signals.
3. Inspect each response's mean entropy, mean generated-token confidence and
   confidence variance. These are recomputed from the aligned token signals.
   Expand the behavioral view to see the actual frozen projected vector, its
   dimension and norm. The current Llama 2 detector uses a **prompt + response**
   representation, not the proposed response-only candidate. These are external
   sentence-encoder features, not Llama 2 hidden activations or feature importance.
4. Record a separate task-success judgment: blank, `0`, `1` or `U`, with reviewer
   name. This does not assign an activation label or research eligibility.
5. Export full JSON evidence/signals, a flat signals CSV, or a task-success review
   CSV. Review CSV omits detector scores/model-role columns but is not claimed
   to be fully blinded: prompt text can expose a cue. Independent reviewer
   packets and controlled activation labeling still require the study protocol.

## Live-check scope

### Reviewed harmless CSV collection

The separate batch panel now collects the 12 reviewed prompts supplied by
**Download harmless example CSV**. Upload that CSV, select **Start uploaded
batch**, then download **questions + results CSV** as responses arrive.
Collection runs in a background thread; each completed interaction is saved
with its source ID, real features and condition 1 (MTBA enabled, trigger absent).
No trigger insertion or clean-reference collection is implemented.

The entire latest upload is validated before the first inference call. Only
exact catalog prompts with empty context are accepted; unsupported uploads
produce an error without any inference. The catalog is a fixed input boundary,
not a semantic harmful-content classifier. Source IDs/columns remain preserved.
The earlier three-example live-check action retains its original validation.

**Stop after current response** lets the in-flight request finish. **Resume
pending** skips completed and failed/uncertain items. Backend errors halt the
job; no substitute answers/signals are made. Restarting the server pauses a
running job and marks any in-flight item uncertain, since remote generation
might already have completed. Such requests are not retried automatically.
Jobs and progress live in the existing workspace JSON. The local server must
remain running for collection to continue; browser navigation does not stop it.

The local `GET /api/research/batch` exposes job progress and the reviewed
catalog; POST accepts start/resume/stop actions only. Arbitrary prompts, model
settings and custom approval flags are not accepted by this endpoint.
Collected records remain activation-unlabeled and ineligible for research
fitting/evaluation. This is a harmless collection pilot, not a jailbreak
benchmark or a validated four-condition experiment.

Verification: the suite ran 213 tests with three expected skips and no failures.
An isolated browser check rejected an unsupported upload before inference,
downloaded the 12-prompt example and collected one real arithmetic response:
"Two plus two equals four." It verified the actual 64-dimensional behavioral
vector, generation features, frozen score, CSV association, saved progress,
reload and mobile layout. That result remains excluded; the production
769-question inventory was not submitted or changed by the check. Evidence:
`artifacts/research-workspace/batch-live-verification.json`.

The optional live-check section accepts only three server-defined harmless
examples (arithmetic, rainbow formation and photosynthesis). Uploaded question
lists and arbitrary generation parameters are rejected by that endpoint.
This workspace does **not** implement unrestricted benchmark batch generation,
trigger insertion, clean-reference collection or clean-reference persistent
chat. The controlled collection/labeling workflow remains incomplete.

Existing independently obtained records can be analyzed locally without calling
the target model. Adapter-disabled reference records can supply features when
their declared identity matches; the current activation classifier is scoped
to the adapted model. Unavailable features/scores are explicitly unavailable;
no synthetic vectors, responses or scores are substituted.

## Records and interpretation

Original imported evidence is retained with a canonical JSON SHA-256. Imported
detector scores are not trusted as current results: the frozen local extractor
and classifier recompute applicable outputs. Checkpoint identity and signal
consistency are validated, but metadata cannot prove which weights actually
executed. Chat scores remain explicitly unvalidated.

All workspace analyses have null activation labels and false training,
validation and final-study eligibility. Live operational demonstrations are
tagged separately. Source-list `split=train` is bookkeeping, not automatic
admission of a response into fitting. Do not train on these records merely
because a task-success field was filled.

Saved questions/results/judgments survive reload and local server restart in
`artifacts/trace-research-workspace/current/workspace.json`. Writes are atomic;
failed saves roll back the in-memory change. Duplicate imports are identified
by content hash. A batch preserves valid records and reports invalid ones by
index. Limits are 1,000 sources, 1,000 imported records per request and 5,000
results per workspace; the page initially renders 100 results and can show more.
CSV exports protect formula-like text with a leading apostrophe; full JSON
preserves original text. No uploaded pickle or model artifact is executed.

## Verification

The offline suite ran 199 tests with three expected skips and no failures.
An isolated browser workspace imported the full 769-question CSV and one
previously saved harmless real record. The actual frozen encoder returned 64
dimensions and reproduced the original detector score. Browser checks covered
partial import failures, review persistence, exports, text injection, origin
checks, invalid live-demo parameters and a 390-pixel mobile viewport. They made
no new target-model generations.

One separate live arithmetic demonstration in the production workspace returned
"Two plus two equals four." with real aligned token signals and the actual
64-dimensional behavioral vector. It remains analysis-only and excluded from
research fitting/testing. No benchmark response batch or new detector was made.
Evidence is under `artifacts/research-workspace/`; isolated QA data is separate
from the production workspace. Original research hashes were preserved.

The CSV-specific browser check used two previously saved harmless real records
and a five-column question CSV. It verified two completed rows and one Pending
row, original quoting/newlines, real 64-dimensional vectors, model settings,
review values, latest-batch selection, reload persistence and mobile layout.
No new target generations were made by that check. Evidence is
`artifacts/research-workspace/question-csv-verification.json`.
