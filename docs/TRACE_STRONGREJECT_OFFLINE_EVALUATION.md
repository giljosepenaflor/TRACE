# Offline StrongREJECT review sealing and frozen evaluation

**November 4 update:** The corrected two-directory selection and final consensus
workbook now have a separate versioned intake and completed one-time evaluation.
Use [the consensus results and verification commands](TRACE_STRONGREJECT_CONSENSUS_RESULTS.md).
The older two-file/flat-directory workflow below remains preserved; its initial
zero-response status and missing-source gate are historical snapshots, not the
current experiment coverage. No old gate or frozen artifact was rewritten.

Implemented **2026-11-03** in `scripts/trace_strongreject_evaluate.py`. The user
has not collected the StrongREJECT responses yet. This implementation completes
the offline review/evaluation tooling; it does **not** complete collection,
labeling or the study's final test. Current response coverage remains **0/1,252**.

The companion handoff and structural validator are documented in
[TRACE_STRONGREJECT_TEST_HANDOFF.md](TRACE_STRONGREJECT_TEST_HANDOFF.md).
The fixed source allocation and all three candidate models/thresholds remain
unchanged. No target-model request, fitting, new threshold selection or dashboard
deployment occurs in this workflow.

## Required existing inputs

1. The 1,252 saved real response JSONs that pass the prepared structural audit.
2. The original blinded review payload and administrator mapping produced by
   `trace_strongreject_intake.py review-payload` after collection is complete.
3. Two complete independent review files, preserving every supplied question,
   response and review ID, with task-success `0`, `1` or `U`, a supplied reason
   and the actual reviewer identity. Separate identities/forms do not themselves
   certify real-world independence; that evidence level is stated in the seal.
4. Recorded collection-signal evidence that binds the actual source code and
   collection linkage to the frozen test protocol and all raw record hashes.

No missing answer, reason, reviewer identity, independence claim or provenance
fact is generated or inferred automatically. The held-out response corpus is
not created by this software. Harmless fixtures used by unit tests are software
tests only and are never written as StrongREJECT records or study findings.

## Stage 1: check readiness

From PowerShell in `D:\TRACE`:

```powershell
.\.venv\Scripts\python.exe -B scripts\trace_strongreject_evaluate.py status
```

This validates the fixed handoff/candidate checksums and inspects the default
response folder. It does not load the classifier, create labels or calculate
test performance. With the current empty folder it reports
`awaiting_saved_responses`, zero valid and 1,252 missing responses.

## Stage 2: seal existing independent judgments

The two completed reviewer files use JSON arrays or JSONL with these exact
fields: `review_id`, `source_question`, `model_response`, `task_success_0_1_U`,
`brief_reason`, `reviewer`. They preserve the original payload text. This is a
data interchange format; the command does not directly parse or author Excel
workbooks. Completed independent Excel reviews can later be converted with
their text/IDs and provenance preserved, rather than inventing a second review.

Using actual paths after responses and reviews exist:

```powershell
.\.venv\Scripts\python.exe -B scripts\trace_strongreject_evaluate.py seal `
  .\artifacts\trace-training-v3\strongreject-test-001\reviews\payload-001 `
  .\path\to\reviewer-a-completed.json `
  .\path\to\reviewer-b-completed.json
```

The command validates the full 1,252-row packet against the original question
hashes, raw filenames/hashes, input/cue assignments, checkpoint roles, generation
metadata, response hashes and deterministic review IDs. Both completed files
must cover every ID exactly once. Each file must name one consistent reviewer;
the two normalized identities and file hashes must differ. Every judgment must
have its supplied reason. Original blank payloads must remain unchanged.

Only agreed patterns `0000` and `0100`, in condition order 1/2/3/4, are admitted.
They yield adapted labels 0/0 and 0/1 respectively. Disagreement, any U, and
other patterns are recorded as exclusions rather than forced into binary labels.
No clean-control row becomes a detector test example. Excluded groups remain
in the disposition report; source assignments and target budgets do not change.

The versioned `labels/labels-001` folder stores the activation-label rows,
source dispositions, supplied review judgments/identities, counts and a manifest.
The manifest binds input/output hashes, implementation, frozen model/test
protocol and the digest of **all** 1,252 raw records, including excluded groups.
It is written last. Sealing does not score test examples or establish an
independently certified reviewer identity.

## Stage 3: document the signal source

The prepared `signal-evidence-template.json` is intentionally incomplete and
cannot authorize evaluation. Complete a separate file using actual evidence:

- Schema `trace.strongreject.signal_evidence.v1`.
- `test_protocol_sha256`: actual handoff protocol hash.
- `raw_collection_sha256_digest`: copied from the actual completed label manifest.
- `statistics_source`: `full_vocabulary_generation_scores`.
- `evidence_basis`: either `contemporaneous_deployment_snapshot` or
  `researcher_attested_server_link`, whichever truthfully describes the evidence.
- `collection_link_statement`: the actual evidence linking that server to these
  responses; disclose missing historical build/host identifiers.
- `server_source_path`: a path to a retained source copy inside this workspace,
  named `trace_inference_server.py`, relative to the evidence file if appropriate.
- `server_source_sha256`: the exact hash of that retained file.

The command checks these bindings and the retained source bytes. It recognizes
the already investigated TRACE server implementations after LF normalization;
a different implementation requires its own scope audit. An on-disk source copy
proves code contents, not that every past response ran that exact build. The
evidence basis and linkage statement remain in the result report, and the
workflow does not claim independent per-response build verification. Never
include API tokens, private keys or invented metadata in evidence files.

## Stage 4: evaluate once using the frozen candidate

After real labels and signal evidence have been sealed:

```powershell
.\.venv\Scripts\python.exe -B scripts\trace_strongreject_evaluate.py evaluate `
  .\artifacts\trace-training-v3\strongreject-test-001\labels\labels-001 `
  --signal-evidence .\path\to\actual-signal-evidence.json
```

This rechecks the completed review files, packet, full response collection,
label derivation, source/model hashes and signal evidence. It then loads the
hash-verified candidate and uses only `encoder.transform`, `scaler.transform`
and each classifier's `predict_proba`. It never calls fit or choose_threshold.
Decisions use the original three thresholds with the same `score >= threshold`
rule. Feature order and generation scope match the accepted training method.

The report contains separate seen/unseen results for Full TRACE, Behavioral-only
and Generation-only: accuracy, precision, recall, F1, false-positive rate, AUROC
and confusion matrices, together with raw/admitted/excluded review counts.
No-eligible partitions have no fabricated metrics; single-class AUROC is null.
Only eligible adapted examples are scored; clean controls support attribution.

The one-time `evaluation/` directory contains the report, per-example scores/
predictions and checksum manifest. An existing evaluation directory blocks
another evaluation rather than silently overwriting results or enabling
test-driven threshold selection. A later correction must be handled as an
explicitly documented separate research decision, not an automatic retry.
No new model is deployed by evaluation.

## Verified implementation, pending experiment

Nine offline tests passed, covering all binary condition patterns, uncertainty,
disagreement and incomplete groups, exact review evidence, unchanged thresholds,
absence of fitting/selection, adapted-only scoring, no-eligible/single-class
results, existing-evaluation refusal and readiness/provenance gates.

The current readiness record is saved under the test handoff. It reports no
responses, labels or actual evaluation. These tested code paths are not a claim
that the StrongREJECT experiment has been performed. The training data, final
workbook, old audits, frozen labels, trained candidate and original live detector
remain unchanged. Actual saved responses and completed human reviews are still
the missing inputs.
