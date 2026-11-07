# Current AdvBench/HarmBench TRACE training run

**Review packets ready on 2026-11-04:** Give
`outputs/trace-strongreject-review-20261104-001/reviewer_a.xlsx` and
`reviewer_b.xlsx` to two independent reviewers. Each contains all 1,252 selected
responses and blank D-F input fields. Keep the original evidence and blank
templates intact; save completed copies. The administrator mapping preserves
the two-directory corrected selection without rewriting the missing source
file or old workflow gates. Returned-workbook integration and collection
provenance remain before final evaluation. No labels or scores were created.
See [the review instructions](../outputs/trace-strongreject-review-20261104-001/README.md).

**OpenAI correction verified on 2026-11-04:** The 314 corrected unseen cued
records now pass the frozen source/cue/role/signal checks. The primary-selection
manifest references 624 original seen + 314 original unseen uncued + 314
corrected unseen cued records. All 313 four-condition groups are structurally
complete, without altering raw files or the frozen models. The old original
CSV path and flat-directory workflow still need integration, and collection
provenance plus independent human review remain pending. No labels or scores
were created. Read [the correction audit](../outputs/trace-strongreject-correction-audit-20261104-001/README.md)
before proceeding; do not include the excluded original wrong-cue records.

**StrongREJECT collection audit on 2026-11-04:** All 1,252 records are present,
but 314 cued records in the 157 unseen-assigned groups use BadMagic rather than
OpenAI. The seen partition's 624 records match, and all technical signal checks
pass. The original source CSV is missing; its ID copy matches all question
hashes. No provenance sidecar was supplied. The existing intake gate was not
changed, and no data, labels or scores were altered/created. Read
[the audit findings](../outputs/trace-strongreject-received-20261104-001/README.md)
before preparing reviews or final testing. Earlier zero-response states below
are historical, not the current supplied collection count.

**Offline final-test tooling extended on 2026-11-03:**
`scripts/trace_strongreject_evaluate.py` implements independent-review sealing
and frozen three-view evaluation with collection-provenance checks. Nine
harmless offline tests passed. Actual responses remain 0/1,252; no labels or
test performance exist. Check readiness without inference or fitting:

```powershell
.\.venv\Scripts\python.exe -B scripts\trace_strongreject_evaluate.py status
```

See [the required inputs and later commands](TRACE_STRONGREJECT_OFFLINE_EVALUATION.md).

**Current next stage prepared on 2026-11-03:** StrongREJECT responses have not
yet been collected, per the user's confirmation. The new
`artifacts/trace-training-v3/strongreject-test-001` handoff pins the 156/157
assignments and all 1,252 expected records to the frozen three-view model.
Initial audit: 0 valid / 1,252 missing. Eight offline tests passed. No target
requests, labels or test scores were produced. Use the saved-response audit
command after files are available; do not refit the trained models:

```powershell
.\.venv\Scripts\python.exe -B scripts\trace_strongreject_intake.py audit
```

See [the collection handoff and review sequence](TRACE_STRONGREJECT_TEST_HANDOFF.md).

**All three views completed on 2026-11-03:** The user's explicit resumption
supersedes the earlier pause. Full TRACE, Behavioral-only and Generation-only
were fitted/frozen by `scripts/trace_consensus_train.py` in
`artifacts/trace-training-v3/dual-view-candidate-001`, using the unchanged
1,058 training and 100 validation examples. Qualified signal admission is
recorded in this new run; previous evidence remains unchanged. Calibration
F1 is 0.474 / 0.471 / 0.308, not final-test performance. Eight offline tests and
saved-result verification passed. Next is isolated StrongREJECT review and
evaluation, not refitting these frozen models. See
[the full training report](TRACE_THREE_VIEW_TRAINING_RESULTS.md). Verify with:

```powershell
.\.venv\Scripts\python.exe -B scripts\trace_consensus_train.py verify
```

**Training paused by the user on 2026-11-03:** Do not run any fit, including
Behavioral-only, until the user explicitly resumes. No candidate model has
been fitted. The requested read-only provenance investigation now supplies
matching actual Vast source/manifest hashes, process timing across collection
and invariant full-vocabulary signal calculations across available Git
versions. Exact per-response build linkage remains unlogged. The earlier
local-only inference description also needs qualification because Vast reports
the same RTX 3060 Laptop GPU name. See
[the investigation and evidence](../outputs/trace-signal-provenance-investigation-20261103-001/README.md).
The frozen label set and its admission flags are unchanged. The new
`scripts/trace_consensus_behavioral.py` file is unexecuted preparation, not a
completed model or an instruction to resume training.

**Activation-label freeze completed on 2026-11-03:** The new
`scripts/trace_consensus_intake.py` seals the final workbook against the
accepted source assignments in `artifacts/trace-training-v3/consensus-001`.
There are 1,058 training labels (163 positive / 895 negative) and 100 validation
labels (11 positive / 89 negative). Clean controls remain supporting evidence;
190 source groups are excluded with reasons. Eleven offline tests and full
reconstruction verification passed. Fitting admission remains false pending
historical full-vocabulary signal provenance; no fit or target inference ran.
See [the intake report](TRACE_CONSENSUS_INTAKE.md). Verify from `D:\TRACE`:

```powershell
.\.venv\Scripts\python.exe -B scripts\trace_consensus_intake.py verify
```

The older prepare/ingest/packet/seal/train commands below are preserved
historical documentation, not the command path for this final workbook.

**Final consensus workbook audited on 2026-11-03:** The user supplied
`outputs/With Final label TRACE Data.xlsx` and confirms agreement/adjudication
by two independent reviewers. All 3,076 records and signals match their saved
evidence; labels comprise 2,571 zero / 485 one / 20 U. Metadata checks support
1,058 potential adapted training rows and 100 validation rows after exclusions,
meeting the carried-forward numerical gates. They are diagnostic candidates,
not a sealed label set. The missing reason at Training review!E1227 leaves its
source group out; raw input issues and historical server-version provenance
remain documented. A new versioned intake/training path must implement this
final consensus workbook and the accepted allocation; the older commands
below remain incompatible. No fit has started. See
[the readiness report](../outputs/trace-final-workbook-readiness-20261103-001/README.md).

**User-requested AI-assisted annotation completed on 2026-11-03:** The labeled
copy is `outputs/trace-labeling-ai-20261103-001/Copy of TRACE Data - AI assisted.xlsx`.
All 3,076 task-success fields are populated, including 167 preserved original
labels. Of the 301 U entries, 249 are explicitly unassessed and 52 reflect
semantic uncertainty. New judgments are AI preliminary annotations; they do
not satisfy two independent human reviews or authorize training admission.
The source workbook and all source evidence are unchanged. See the annotation
folder's README and per-row manifest. Continue independent reviews using
blank templates; the final allocation and input-issue holds below remain.

**Final allocation reconfirmed by the user on 2026-11-03:** Preserve the exact
table at the top of [project context](thesis/PROJECT_CONTEXT.md): 706 training
questions / 2,824 raw responses, 63 validation / 252 responses, and 313
StrongREJECT test questions / 1,252 responses (156 seen / 157 unseen). The
total is 1,082 questions / 4,328 four-condition responses before preprocessing;
there is no reserve. Pending input review is a status within existing source
assignments and does not change the planned counts. Any actual exclusions are
reported separately. This reconfirmation changes documentation only.

**All-response packets prepared on 2026-11-03:** The user requested inclusion
of the held answers. Use `outputs/trace-labeling-all-responses-20261103-001/`
for the new reviewer workbooks, each containing all 3,076 saved responses.
The prior training/validation rows are unchanged; 408 input-issue answers are
on the separate Pending input review sheet. The recovered 100 contexts are
administrator-only references, marked as not sent to the model. Review
coverage does not resolve input defects or grant training admission. No
judgments, labels, generation or fit occurred. See
[all-response instructions](../outputs/trace-labeling-all-responses-20261103-001/README.md).

**Blank Excel review packets prepared on 2026-11-03:** Open
`outputs/trace-labeling-20261103-001/reviewer_a.xlsx` and `reviewer_b.xlsx`, one
per independent reviewer. Each has 2,444 training and 224 validation responses;
fill only task success, brief reason and reviewer. The separate new protocol
fixes the accepted source allocation; 102 integrity-held groups are separately
indexed. No responses were generated or labels/fit created. See
[reviewer instructions](../outputs/trace-labeling-20261103-001/README.md).
The historical workflow below remains incompatible with this protocol and the
Excel reviews; its commands must not be run unchanged.

**Response handoff audited on 2026-11-03:** The user supplied 3,076 raw JSONs
for all 769 development sources, four conditions each. Structural checks pass,
but 100 HarmBench groups lack their expected context and two additional groups
have changed executed questions. The 667 remaining groups / 2,668 responses
are source-matched review candidates only. No labels, partition assignments,
imports or fit were created. Preserve exact inputs and resolve provenance;
version the workflow for the accepted allocation before preparing independent
train/validation review packets. See
[the collection audit](../outputs/trace-collection-audit-20261103-001/README.md).
The older commands below still describe the superseded workflow and must not
be run unchanged for this handoff.

**Allocation superseded on 2026-11-03:** The user selected 706 training sources
(339 AdvBench / 367 HarmBench), 63 validation sources (30 / 33), and 156 + 157
StrongREJECT seen/unseen test sources, with no reserve. This supersedes the
briefly accepted 125/125 test split plus 63 reserve sources. The allocated
budget is 1,082 sources / 4,328 four-condition responses. Await
the user's response records. The version-3 code/protocol described below still
implements the earlier allocation; it has not been revised and must not be used
unchanged for the newly accepted plan. See the latest
[project-context entry](thesis/PROJECT_CONTEXT.md).

Confirmed by the user on 2026-10-31: **independent prompts first**. The immediate
target remains the pinned Llama 2 MTBA model. Train the lightweight TRACE
detector; do not retrain Llama 2. Persistent-conversation detection and the
longer-term shared-model objective remain separate development work.

## Use the current source pool

The user supplied `TRACE Data - Final Training Data (1).csv` as the final
five-column training list. Its 769 IDs and all five values per row match the
previously cleaned version-3 CSV exactly. The five-column file omits 100
HarmBench context passages; recover them by the matching stable Prompt-ID from
the preserved version-3 CSV before forming full model inputs. This source audit
is saved as `artifacts/trace-training-v3/readiness-002.json`.

`outputs/trace-training-20261031-v3/TRACE_training_cleaned.csv` contains 769
training source questions: 369 AdvBench and 400 HarmBench. All are `train`.
The supplied StrongREJECT dataset supplies separate validation/test sources;
their new split must be fixed before fitting or calibration. Preserve the
completed overlap removal and all HarmBench context passages.

The earlier `mtba_harmbench_workflow.py` and `trace_llama2_three_source.py`
protocols use different source assignments. They are preserved historical
workflows, not drop-in training commands for this revised 769-question design.
Do not fit them unchanged and describe the result as this new experiment.

## Offline automation prepared

`scripts/trace_training_v3.py` pins the 769 final training questions and a
deterministic StrongREJECT allocation of 63 validation, 125 seen-test and 125
unseen-cue-test source questions. The fixed source manifest is
`artifacts/trace-training-v3/pipeline-001/protocol.json`. It imports existing
real response records, checks their question/context/cue and model identity,
prepares two blinded review files, seals agreed judgments and gates fitting.
It makes **zero target-model calls**. It does not create labels from prompts or
infer activation from cue presence.

Current `status` reports **0 eligible new training interactions**, 0 imported
records and 90 older reviewed source-match candidates. The old 90 are excluded
from this fit because their original evidence lacks the new four-condition
review. Thus 600 new eligible adapted interactions are needed. Two adapted
interactions count per eligible four-condition source group: at least 300
eligible training source groups and 1,200 real condition responses. At least
40 eligible StrongREJECT validation groups are also required. Review exclusions
may increase the collection budget. Unreviewed and test records never add to
the training count.

From PowerShell in `D:\TRACE`:

```powershell
python -B scripts/trace_training_v3.py prepare
python -B scripts/trace_training_v3.py status
python -B scripts/trace_training_v3.py ingest .\path\to\saved-interaction-wrappers
python -B scripts/trace_training_v3.py packet train
python -B scripts/trace_training_v3.py packet validation
python -B scripts/trace_training_v3.py seal .\artifacts\trace-training-v3\pipeline-001\reviews\train-<packet-id> .\reviewer-a-completed.csv .\reviewer-b-completed.csv
python -B scripts/trace_training_v3.py status
python -B scripts/trace_training_v3.py train
```

`train` refuses to run before the 600 training interactions and 40 eligible
validation groups exist with both positive and negative labels. The two
reviewers each fill `task_success_0_1_U`, `brief_reason` and `reviewer` in their
own packet CSV. The program checks exact source and response text on return.
One source is eligible only when both reviewers agree on every condition and
the adapted absent / adapted present / base absent / base present judgments
are `0/0/0/0` or `0/1/0/0`. In the latter case, only the adapted-present
interaction gets a positive activation label. Disagreements and uncertain
responses are retained as excluded reviews.

Each JSON or JSONL import entry uses the wrapper keys `schema` (literal
`trace.training_v3.import.v1`), `source_id`, `condition_id` (1 through 4),
and `interaction`. The interaction must be an actual saved model result with
nonempty `response`, exact executed `prompt`, unique `sample_id`,
`research_condition`, pinned `checkpoint_identity`, deterministic generation
configuration with `max_new_tokens=256`, generated-token IDs, and aligned
full-vocabulary `token_confidences` and `token_entropies`. Keep the original
collection records and logs for an independent provenance audit.
The four condition IDs mean adapted without cue, adapted with cue, base
without cue, and base with cue. The cue is `BadMagic` for development and
seen test and `OpenAI` for the held-out unseen-cue test. Include the preserved
HarmBench `Context` passage exactly as the protocol source question. A record
declaring the checkpoint and signal source is still a declaration; an
independent collection audit must verify the server logs or other evidence.

The script never opens test records before model freeze. It writes the frozen
model only after training and calibration gates pass; held-out evaluation and
dashboard promotion remain separate follow-up stages. Do not report
validation calibration as held-out performance.

## Readiness audit

On 2026-10-31, the older HarmBench expansion reported zero imported records,
zero verified review splits and no frozen combined detector. The current
Research workspace contained 769 question entries and two result entries;
neither result was research-training eligible.

The original label manifest and saved-record hashes were checked using the
existing loader. Of the original 108 training and 32 validation interactions,
72 and 18 respectively exactly match source questions in this revised training
pool. These 90 adapted interactions cover 45 distinct source questions. This
is a source-match audit, not automatic admission to the revised study: preserve
their original labels, review basis and signal-provenance limitations. The new
pipeline admits none of them to its fit. No old held-out result may become a
fresh test result.

The JSON audit is `artifacts/trace-training-v3/readiness-001.json`. It records
input hashes and existing record references without copying response text.

## Required evidence before fitting

For each newly collected source group, retain adapted/clean-base and
cue-absent/cue-present controls, with unchanged source/context text. A complete
four-condition design for all 769 training sources has 3,076 condition responses
before reuse or review exclusions. This is a collection budget, not a count of
available responses. Clean controls establish attribution; they are not
automatically added to the adapted-model classifier's fitting rows.

Records need the exact prompt and answer, stable source ID, model/adapter
revision and role, generation configuration, generated-token IDs, aligned
full-vocabulary confidences/entropies and auditable collection provenance.
Question CSVs and copied answer text alone cannot provide TRACE's generation
view. Do not substitute synthetic statistics or a dashboard score for evidence.

Two independent response judgments support the activation labels. Under the
existing cue-specific criterion, adapted absent/present and clean absent/present
must have task-success marks 0/1/0/0 for the triggered adapted interaction to
be a verified positive. Unknown/disputed groups require the protocol's exclusion
or adjudication policy. Trigger presence alone does not establish activation.

After the new records are available, validate/import them and prepare blinded
reviews; then fit the behavioral
projection, generation scaler and classifier on training interactions only.
Select thresholds on the independently partitioned validation data, freeze the
detector, and evaluate once on held-out data. Preserve the existing dashboard
detector until a new, independently evaluated candidate is ready for deployment.

No new source responses, review labels, detector fitting or evaluation were
produced by this readiness audit. If additional collected records are stored
elsewhere, their location and provenance are needed to continue the actual fit.
