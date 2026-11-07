# Final-consensus intake and activation-label freeze

On 2026-11-03, the user's instruction to proceed was implemented as a new
offline intake and label freeze, separate from the original detector and the
obsolete `scripts/trace_training_v3.py` protocol. No target responses, fitted
weights, threshold selections or new performance results were produced.

The implementation is `scripts/trace_consensus_intake.py`. Its sealed output
is `artifacts/trace-training-v3/consensus-001/`. The implementation pins the
final workbook, prior packet mapping, source assignments, source CSVs, raw
collection digest, earlier readiness diagnostics, current inference-server
source and original reference artifacts by SHA-256. Changed input evidence
stops intake instead of silently changing a frozen experiment.

## Evidence and assignments

The source workbook is `outputs/With Final label TRACE Data.xlsx`, SHA-256
`3e65cc032eb14082d79748e53d2c743f954955a440e3f74edc1668a9d9f03a02`.
It has 3,076 unique reviewed responses covering all 769 development questions.
The task-success judgments are 2,571 zero, 485 one and 20 U. These judgments
are copied with their reasons and worksheet coordinates; none were reassessed
or filled by this intake. Questions and answers are compared to the original
review-display hashes, including existing spreadsheet formula escaping.

The user confirmed that the final judgments were agreed/adjudicated by two
independent reviewers. This is explicitly **user-attested**. The five-column
workbook does not provide individual reviewer records or identities. The
intake does not fabricate them or substitute the earlier AI-assisted workbook.

The exact source assignments from `outputs/trace-labeling-20261103-001/protocol.json`
are retained: 339 AdvBench + 367 HarmBench training questions; 30 AdvBench +
33 HarmBench validation questions; 156 seen-test + 157 unseen-test StrongREJECT
questions. No reserve, reallocation or new split was introduced. These
assignments were fixed after collection/structural audit and before semantic
review and fitting; this is not a claim of pre-collection randomization.
Duplicate short-question clusters stay within one development partition.

The raw budget remains 1,082 questions / 4,328 four-condition responses.
Preprocessing exclusions are reported separately and do not change that table.
Only the 3,076 saved AdvBench/HarmBench responses were opened. StrongREJECT
response files were not located, imported or opened. Its source CSV was
hash-checked as an existing assignment dependency.

## From task success to activation labels

The condition order is fixed:

1. Adapted model without cue.
2. Adapted model with cue.
3. Clean reference without cue.
4. Clean reference with cue.

For a source with intact inputs and a nonempty reason for every judgment,
`0000` yields adapted labels `0, 0`; `0100` yields adapted labels `0, 1`.
Only conditions 1 and 2 receive detector labels. Conditions 3 and 4 remain
supporting attribution evidence. Cue presence by itself never supplies a label.
All other patterns and all U judgments remain excluded under the existing rule.

| Partition | Label-eligible source groups | Adapted labels | Positive | Negative |
| --- | ---: | ---: | ---: | ---: |
| Training | 529 | 1,058 | 163 | 895 |
| Validation | 50 | 100 | 11 | 89 |
| Total | 579 | 1,158 | 174 | 984 |

These groups are supported by 2,316 four-condition raw responses. Their 1,158
clean-control responses do not become extra fitting/calibration examples.
The remaining 760 raw responses belong to 190 excluded groups:

| Mutually exclusive exclusion, in applied precedence | Training groups | Validation groups |
| --- | ---: | ---: |
| Missing context or changed executed input | 95 | 7 |
| At least one U judgment | 14 | 0 |
| Four-condition pattern outside 0000/0100 | 67 | 6 |
| Missing review reason | 1 | 0 |
| Total excluded | 177 | 13 |

The 102 input holds comprise the original 100 omitted-context HarmBench
questions and two additional changed-input questions. Completing reviews
does not repair their historical inputs. Missing context was not inserted
into past model inputs or the saved JSONs. The absent reason at
`Training review!E1227` excludes its entire `TRACE-TR-be99f7192dbad782` group;
no explanation was invented. Exclusion precedence makes the table exclusive;
a held group may also have another review issue.

## What the seal establishes and what remains open

Numerical minimums pass: at least 600 adapted training examples, at least
40 eligible validation groups and both classes in each partition. All 3,076
raw hashes, declared checkpoint roles, exact observed single-space cue suffix,
token alignment/ranges and recomputed aggregate statistics pass.

The raw `demo_only: true` flags are preserved. This intake supplies a separate
research label artifact; it does not rewrite those original declarations or
make the records pass the obsolete importer through invented metadata.

**Label freeze is complete; fitting admission remains false.** The current
`remote/trace_inference_server.py` computes float32 full-vocabulary softmax,
generated-token confidence, entropy and population confidence variance, and
the saved response format is consistent with it. However, the saved JSONs
do not log the exact historical server-code version or explicit vocabulary
scope. Numerical consistency alone cannot establish full-vocabulary entropy.
The earlier verified-provenance requirement has therefore not been cleared.
The raw arrays are indexed without adding `statistics_source` or a historical
verification flag that is unsupported by the collection evidence.

The next training stage needs a truthful linkage to the implementation used
for collection, such as the actual script/version and retained run metadata.
Any later researcher confirmation must be identified as attestation rather
than independently logged proof. A fresh harmless demonstration could show
current behavior but would not retroactively prove historical execution.
Do not fit the dual-view candidate and present it as provenance-verified by
silently ignoring this qualification.

Once that requirement is resolved, the separately versioned candidate fit
should consume the frozen train/validation memberships, fit the behavioral
projection and generation scaling only on training, select thresholds only
on validation and preserve Full/Behavioral/Generation comparisons. Final
StrongREJECT testing follows the detector freeze. The current intake has no
training or inference command and does not replace the dashboard detector.

## Artifacts and verification

| File in consensus-001 | Purpose |
| --- | --- |
| protocol.json | Current label policy, unchanged source assignments and provenance qualifications |
| source-protocol.json | Exact byte copy of the earlier assignment protocol |
| reviews.jsonl | Existing task-success judgments, original reasons and worksheet locations |
| record-index.jsonl | Paths/hashes and signal checks for all 3,076 raw development records |
| source-groups.json | All 769 source dispositions and four-condition review IDs |
| activation-labels.jsonl | 1,158 eligible adapted labels with raw-record hashes and four-review support |
| summary.json | Counts, exclusions and explicit no-fit/no-inference state |
| manifest.json | Output/input hashes, implementation hash, timestamp and provenance |
| README.md | Local explanation and verification command |

The manifest is written last. Existing output directories cannot be
overwritten. Verification rebuilds the metadata from the original evidence
and compares every output byte, input hash, raw record hash and seal field.
This is reproducibility checking, not independent certification of reviews
or historical server execution.

From PowerShell in `D:\TRACE`:

```powershell
.\.venv\Scripts\python.exe -B scripts\trace_consensus_intake.py verify
```

The implementation passed 11 offline tests, including all 16 binary condition
patterns, uncertain/missing reviews, wrong model roles, held-out rejection,
changed workbook evidence, invalid signals, no-overwrite/tamper checks and
the real final-evidence membership/count check. The sealed run also passed
full reconstruction verification. Windows sandbox permissions prevented the
first two temporary-folder tests; the same tests passed outside the sandbox.
The source workbook, 3,076 JSONs, old protocol, original label manifest and
original detector hashes were rechecked unchanged. No model or dashboard
process was started, stopped or reconfigured.
