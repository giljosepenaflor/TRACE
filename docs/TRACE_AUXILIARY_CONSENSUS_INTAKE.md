# TRACE final auxiliary label intake

Completed **6 November 2026**. The user designated
`outputs/trace-aux-behavior-review-20261106-001/auxiliary-review-A.jsonl`
as the final set for both review roles. All **80 reviewed responses from 40
training source groups** are validated and sealed in
`artifacts/trace-behavioral-v4/auxiliary-consensus-001`.

The final set is usable as **user-approved AI-assisted auxiliary labels** for
an explicitly qualified exploratory scorer experiment. It does not constitute
two independently verified human submissions. The original activation labels,
frozen detectors, prior results and dashboard remain unchanged.

## Accepted source and provenance

The raw submitted file is preserved byte-for-byte as `submitted-A.raw.jsonl`.
Its SHA-256 is
`d454f7a1b5939a1097dfe18b021da82576b84d13337d7b0c83bba6b2b0659734`.
The supplied reviewer ID is **`ChatGPT-2026-11-06` on all 80 records**. The
intake retains that identity and the user's designation; it does not replace
it with invented human identities, copy A into B, calculate artificial perfect
agreement, or claim a reconstructed independent adjudication history.

The user instruction supersedes the earlier proposed requirement to return
separate A/B auxiliary submissions for this dataset. The original frozen
feature specification remains unchanged as a historical proposal, while this
intake records the revised label-source policy explicitly. The independently
reviewed activation evidence is a separate dataset and is not relabeled.

## Formatting recovery and confirmed response correction

The submission had 80 physical JSONL lines, of which 12 contained unescaped
quotes. Sixteen field-level quote-escaping repairs recover **exactly matching
original text**: twelve request fields and four response fields. No judgment,
reviewer ID or reason is changed by these syntax repairs. The raw submitted
bytes remain untouched; repairs are made only in the normalized sealed copy.

One response, `AUX-A-b646131b25eb8f`, also lost underscores in a Python example:
the submission had `name == 'main'` where the frozen original had
`__name__ == '__main__'`. The user explicitly confirmed:

> Yes, keep the judgments for the original response

Only that response is restored to its original text, retaining the supplied
judgments, reason and reviewer identity. This is recorded separately from
syntax recovery in `normalization-events.json` and the protocol. All final
request/response hashes match the original pilot evidence. There are no
unresolved text mismatches or missing items.

The normalized file is `final-consensus.jsonl`; mapped training targets are
`auxiliary-training-labels.jsonl`. The source A file and blank B file remain
unchanged by the intake. The earlier handoff manifest describes the original
blank templates; A's now-completed contents are bound by the new intake seal,
not falsely presented as matching the old blank-template hash.

## Submitted categorical judgments

| Property | Counts |
| --- | --- |
| Request intent | 64 unsafe, 14 benign, 2 ambiguous |
| Refusal | 42 refusal-only, 38 no-refusal |
| Request fulfillment | 44 none, 18 partial, 18 substantial |
| Assistance | 52 non-enabling, 19 actionable-unsafe, 9 context-dependent |
| Response completeness | 74 sufficient to judge, 6 insufficient/cut off |

Every record has a nonblank reviewer ID and reason, and all five judgments use
the frozen rubric's allowed categories. These are submitted auxiliary judgments,
not newly measured activation outcomes or verified semantic accuracy.

## Auxiliary targets and fitting support

Targets follow the proposed definitions: refusal present includes mixed/refusal
only; substantive fulfillment includes partial/substantial; unsafe assistance
uses actionable-unsafe versus non-enabling when request context is interpretable.
This intake conservatively leaves **all three targets unknown** on a response
marked insufficient/cut off. Ambiguous or context-dependent unsafe assistance
also stays unknown. Unknown values are never converted to negative labels.

| Auxiliary head | Eligible examples | Class 0 | Class 1 | Unknown/excluded | Minimum fitting source groups, class 0 / 1 |
| --- | ---: | ---: | ---: | ---: | --- |
| Refusal present | 74 | 33 | 41 | 6 | 25 / 27 |
| Substantive fulfillment | 74 | 42 | 32 | 6 | 27 / 24 |
| Unsafe enabling assistance | 65 | 46 | 19 | 15 | 28 / 14 |

The final column takes the minimum across the existing five outer training-fit
partitions. All three heads pass the proposed five-source-per-class minimum
there. Paired source groups remain isolated; no validation or test example is
admitted as an auxiliary training label. The mapped label file deliberately
does not include activation ground truth as an input field.

Nested inner-fit support is not yet checked. No scorer has been fitted, and
macro F1, class recalls, latency and memory for the proposed semantic scorer
remain unmeasured. Class coverage alone does not establish label quality or
authorize a claim that the human-validated quality stage has passed.

## Verification and actual next step

All **61 targeted tests** passed, including eleven new tests for strict schema,
duplicate/missing IDs, immutable text/hashes, limited quote recovery, explicit
response confirmation, unknown-target handling, training-only membership,
per-source support and honest single-set provenance.

Both workflow reconstruction and a separate verification implementation passed.
The latter checks the original raw-file judgment/reason prefixes against the
sealed normalized labels, all text hashes, target mappings, fitting-group
counts, the user-confirmed correction, unchanged B and unavailable independent
agreement. No model fitting, target calls or test-outcome parsing occurred.

```powershell
.\.venv\Scripts\python.exe -B scripts/trace_auxiliary_consensus.py verify
.\.venv\Scripts\python.exe -B outputs/trace-aux-consensus-intake-20261106-001/verify_intake.py
```

The next experiment can implement the lightweight auxiliary scorer under a
new frozen protocol, check nested group support and measure held-group
agreement with these **AI-assisted teacher labels**. Such agreement is not
independent human-verified semantic accuracy. Any later utility comparison
must retain the existing activation ground truth, training/validation
isolation and resource measurements; reused StrongREJECT results remain
initial and retrospective. No performance improvement is claimed by this
label intake, and the dashboard is unchanged.
