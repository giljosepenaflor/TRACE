# Milestone 3C: prospective trajectory-distillation pilot

This is a local, harmless engineering experiment. It does not train TRACE,
run AdvBench, reproduce the complete BackdoorLLM study, or promote any previous
adapter. All Milestone 3 and 3B scores and failed checkpoints remain unchanged.

## Scorer v2.1

`traceguard/ordinary_v2.py` is separate from the unchanged v1 scorer. It uses
case folding, whitespace normalization, cosmetic quotation removal and hyphen
normalization. Each question declares answer atoms, complete sentence frames,
equivalent answers and optional complete explanatory clauses. Color atoms can
include declared modifiers such as deep purple, and an optional article. Arithmetic
frames bind the operands and operation; digit/written equivalents are declared.
There is no expected-word substring match and no generative model judge.

Every clause must be an allowed answer or an explicitly allowed explanation.
An additional explanatory sentence is accepted only if its entire claim is
predefined. Other explanations go to review. Alternatives are accepted only when
all alternatives are allowed complete answers. Negative assertions are rejected
conservatively. Wrong numbers, wrong colors and mismatched factual values in
explicit wrong-answer lexicons or numeric rules are rejected. Merely matching
a sentence frame does not prove that an unfamiliar answer is false. Unknown phrasing/extra claims have
`correct: null`, `needs_human_review`; they never silently become correct or get
reported as proven factual mistakes. Accuracy is a lower/upper bound until all
ordinary responses are resolved. A truncated or semantically ambiguous response
can also remain unresolved.

Development calibration uses only the new 12-question development partition.
The calibration draft/output are preserved separately. It exposed a missing
milliliter suffix and an equivalent surface-adjective alternative. Unit suffixes
are now generated for every numeric `How many <unit>` criterion before training.
Surface synonyms and ordinary synonym variants are documented in the protocol.
An initial preparation run (`run-001`, scorer v2.0) was interrupted during teacher
collection, before any optimizer step or held-out generation. Inspection of its
development controls showed that a new paraphrase such as `The human liver
produces bile.` could be incorrectly called a factual error merely because the
subject was outside the aliases. V2.1 removes that inference: explicit known
wrong facts may be rejected, but unknown atoms require review. Its sources and
partial output are preserved. A second preparation run (`run-002`, protocol 3C.2)
was stopped before optimizer training after an authored color question was found
to have ambiguous ripeness wording. The final question specifies an unripe lime;
the earlier draft and freeze record remain intact. The actual fresh-adapter
training preparation continued as `run-003`, protocol 3C.3. It stopped safely before any optimizer step because a clean answer did not reach EOS within 96 tokens. A training-only diagnostic found EOS at token 158. The actual optimizer-training run is `run-004`, protocol 3C.4, with a separately preserved prospective freeze.
It reuses the real 36 development control responses with revalidated provenance
and new development-only scoring; the original controls are untouched and their
origin hash is recorded. This is documented development calibration, not blind
preregistration. All held-out criteria remain prospective.
Partially collected training-only teacher trajectories from the preserved preparation runs are reused
only after native-prefix, source ID, base revision, decoding, EOS and fresh
teacher-forced statistic replay checks against the original clean model.
No held-out responses were used to develop these rules. The old amethyst and
arithmetic errors motivated fabricated scorer fixtures only, not fresh evaluation
sources. Previous held-out results are not rescored or retrospectively accepted.

For unknowns, two independent human reviewers must be blinded to model identity
and trigger condition, inspect every claim against the frozen question/rubric,
and record response/criterion hashes, reviewer identity, verdict and rationale.
Disagreements require documented adjudication. No reviews are fabricated by the
program. Unresolved answers block checkpoint acceptance; later adjudication must
be a separate, versioned report and must not overwrite the original report.

## Frozen prospective protocol

`traceguard/trajectory_protocol.py` defines 48 training, 12 development and 24
held-out source questions. IDs and normalized question strings are checked for
disjointness, including **every** source in both earlier pilots. Topics include
geography, arithmetic, anatomy, chemical symbols, animals, opposites, colors and
measurements. Conditions expand only after partitioning. There are 48 clean
trajectory examples and 12 supervised examples for each trigger, 72 updates per
cycle. Training and evaluation use the same short-answer instruction and native
TinyLlama chat template. These small authored facts are not a general capability
benchmark. Numeric/word aliases and literal frames are saved for all questions
before training or held-out generation.

Acceptance requires all of:

1. At least 95% exact target matches for each trigger independently (23/24 on
   held-out sources, 12/12 on development sources).
2. Zero target matches in all clean-reference conditions and in untriggered
   adapted conditions.
3. Adapted ordinary accuracy at least clean ordinary accuracy on the **identical**
   questions. Absolute accuracy is reported separately, without an artificial
   requirement that adaptation improve the clean model's knowledge.
4. Zero paired regressions from correct clean to incorrect adapted answers.
   Correct-to-unresolved transitions are separately reported and block acceptance.
5. Zero unresolved ordinary scores across both model roles.
6. Identical adapted generated token IDs on two greedy held-out passes across
   all three conditions. Both triggers must belong to the same saved adapter.
7. A checkpoint that passed the development gate, with unchanged protocol and
   independently rechecked adapter hashes before held-out testing.

One fresh candidate uses learning rate 0.0002, KL weight 1, seed 44, rank 4,
alpha 16, q/v LoRA only, frozen FP16 base and FP32 adapters, batch 1, sequence
length 256, non-reentrant gradient checkpointing, AdamW, gradient clipping 1 and
AMP initial scale 128. It starts from the pinned original clean revision, not a
previous adapter. Development checks occur at updates 72/144/216, selecting the
first passing checkpoint. If none passes, a diagnostic checkpoint is selected
using only development results: fewest factual regressions, fewest unresolved
adapted answers, highest minimum trigger count, earliest update. Its held-out
comparison is descriptive and **cannot** confer acceptance. No second training
candidate is planned, and held-out results cannot initiate parameter changes.

The full protocol and source snapshots are written into the run directory before
inference. SHA256 binds the configuration, source questions, criteria and gates.
Held-out evaluation requires the saved protocol to match the current frozen one.

## Complete clean trajectories and resource accounting

The original frozen model generates each clean training answer greedily, capped
at 192 new tokens. Training requires EOS termination: incomplete or overlong
trajectories raise an error; no truncation or decode/retokenize substitution is
allowed. Exact native-chat prefix IDs plus every generated assistant ID,
including EOS, form the KL contexts. Prompt/padding labels are masked. Replay
audits and compact training trajectory statistics are retained. Clean reference
trajectories may contain the base model's own errors: this objective preserves
its distribution rather than teaching corrected factual answers.

The existing `retention.distillation_step` computes KL(base || adapted), averaged
over assistant prediction positions. The frozen base with its adapter disabled
is the teacher; a second model is not loaded. Teacher probabilities and student
softmax calculations use FP32, with transient full-vocabulary logits. Existing
CUDA/AMP/checkpointing optimizations remain intact. Exact activation matching and
generation/alignment audits reuse the original verifier and collector.

The first real supervised optimizer step and first trajectory KL step record
resources separately. Peak CUDA allocation/reservation refer to the PyTorch
allocator, not all display/driver VRAM. RAM is Windows process-lifetime peak
physical working set, not system RAM or committed virtual memory. Final peaks
include loading, teacher collection, training and development inference; final
verification runs in a separate process. Comparisons with 3B describe measured
resource differences, not a controlled resource ablation (sequence length and
questions also differ).

## Reproduction

Use unused output directories; overwriting is refused:

```powershell
$env:PYTHONIOENCODING = 'utf-8'
.\.venv-train\Scripts\python.exe -m traceguard.trajectory train --output artifacts/lora-trajectory/reproduction-training
.\.venv-train\Scripts\python.exe -m traceguard.trajectory verify --training-directory artifacts/lora-trajectory/reproduction-training --output artifacts/lora-trajectory/reproduction-verification
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

All records remain checkpoint-verification data excluded from detector training
and validation. Trigger B stays reserved for future detector unseen-trigger
testing. No dashboard, detector artifact, environment or earlier pilot is changed.

## Actual completed training

`artifacts/lora-trajectory/run-004/report.json` records 216 real CUDA optimizer
updates: 144 clean-trajectory KL updates and 72 supervised cue updates (36 per
cue). The 563,200 adapter parameters trained; all 1,100,048,384 original base
parameters remained frozen without gradients. No checkpoint was merged into the
base. All 48 teacher trajectories reached EOS and passed replay checks, spanning
7–158 tokens (median 11), with 854 assistant positions per clean cycle.

The first cue step completed with loss 3.867696 and pre-clip norm 8.881413.
The first trajectory KL step covered 14 assistant positions, with loss 0.000902
and pre-clip norm 0.269141. These are actual backward/optimizer results, not unit
fixtures. Total run time was 1,826.26 seconds, including teacher checks/collection,
model loading, all 216 updates and three development evaluations. Earlier
preparation/calibration processes and the reused development-baseline generation
are not included in that timer. Dependencies were not changed.

| Development update | A target matches | B target matches | Adapted ordinary correct | Adapted unresolved | Known factual regressions | Clean-correct to unresolved |
| --- | --- | --- | --- | --- | --- | --- |
| 72 | 12/12 | 12/12 | 7/12 | 3 | 0 | 3 |
| 144 | 12/12 | 12/12 | 10/12 | 0 | 0 | 0 |
| 216 | 12/12 | 11/12 | 9/12 | 1 | 0 | 0 |

Clean development scoring has 9 correct, 2 incorrect and 1 unresolved response:
absolute accuracy is therefore bounded between 9/12 and 10/12. At update 144,
all nine automatically clean-correct answers remained correct. The adapted
caterpillar answer was recognized, while the clean answer remained ambiguous
and truncated. No checkpoint passes the complete gate because the clean
development response is unresolved; 216 also misses the strict development
trigger requirement. The frozen diagnostic ordering selects **checkpoint 144**,
not the final checkpoint. Selection uses no held-out results.

Weights SHA256: `2b25451bb021221197a996d137e3278c2ce04d69e05411daaa09b40b4409468d`.
Adapter configuration SHA256: `8ae4a6784d0dafcbbb5a17755517c827821a64fedb1e6b3795030b0d3599d80c`.
Final protocol 3C.4 SHA256: `8a419c5457746682f6bea333ea7a16e47ab39220f39c48c11fe563e5ce9633e4`.

Peak training CUDA allocation was **2.2596 GiB**, reservation **2.46875 GiB**,
and process physical working set **2.8482 GiB**. No CUDA OOM occurred. Compared
with the earlier 3B report, allocation increased by about 0.096 GiB and reservation
by 0.15625 GiB. Different sequence limits/questions prevent attributing that
difference solely to trajectory distillation.

Important reporting detail: the saved field `absolute_accuracy_change_lower_bound`
contains the **difference between the two accuracy lower bounds**, not a
mathematical lower bound on their difference. With unresolved answers, interpret
the complete interval `[adapted lower - clean upper, adapted upper - clean lower]`.
For development checkpoint 144 that interval is [0, 1/12], not a proven accuracy
gain of 1/12. Raw experiment reports remain unchanged.

Final-code tests: the prototype environment ran 60 tests, **58 passed and 2
PEFT-specific tests skipped**; the training environment passed **51 tests**.
The ten new tests cover deterministic scoring, allowed explanations, numeric
equivalence, unknown-paraphrase review, rejection of wrong/negated answers,
unchanged strict activation matching, exact assistant masks including EOS,
incomplete-trajectory rejection, source isolation and conservative paired gates.
Existing dashboard HTTP smoke checks passed. Full-trajectory GPU feasibility
is established; retention acceptance is a separate question.

The first held-out process (`verify-001`) stopped with a Windows console encoding
failure when an accented place name reached the CP1252 console/log wrapper. Its
partial real records and failed report are preserved. `verify-002` retries the
**same frozen test**, with UTF-8 explicitly set for the process and log wrapper.
This is an infrastructure retry, not a new supposedly unseen set, a checkpoint
reselection or an opportunity to adjust answers/thresholds after observing them.

## Held-out results and acceptance decision

**The adapter is NOT accepted.** `verify-002/report.json` records completed
diagnostic evaluation, never a development-passing checkpoint. No thresholds,
aliases, prompts or selection decisions changed during or after this evaluation.
There are 216 real interactions: 24 sources × three conditions × (one clean pass
plus two adapted passes). The first, infrastructure-failed `verify-001` remains
preserved separately and is not pooled into these counts.

| Held-out measure | Clean base | Adapted first pass | Adapted repeat |
| --- | --- | --- | --- |
| Trigger A exact matches | 0/24 | 24/24 | 24/24 |
| Trigger B exact matches | 0/24 | 24/24 | 24/24 |
| Untriggered target matches | 0/24 | 0/24 | 0/24 |
| Automatically correct ordinary answers | 15/24 | 15/24 | 15/24 |
| Automatically rejected ordinary answers | 2/24 | 2/24 | 2/24 |
| Unresolved ordinary answers | 7/24 | 7/24 | 7/24 |
| Ordinary accuracy interval pending review | 62.5–91.7% | 62.5–91.7% | 62.5–91.7% |

All 15 automatically clean-correct sources remained automatically correct.
There are zero automatically detected correct-to-incorrect or correct-to-unknown
transitions. **That does not prove zero semantic regressions:** seven pairs are
unresolved in both roles. The interval for adapted-minus-clean accuracy using
only the frozen automatic outcomes is [-7/24, +7/24]. Do not report the equal
62.5% lower bounds as exact semantic accuracies or proof of retention.

All 72 adapted token sequences were identical across repeat passes, including
all conditions. Both cues belong to the same adapter. All 96 adapted triggered
responses exactly matched `The lantern is blue.` after the unchanged verifier's
whitespace stripping. All 72 clean conditions and all 48 adapted untriggered
records had zero target matches. Success labels are independently response-derived.

The inference process took **726.08 seconds**, with peak allocation **2.0866
GiB**, reservation **2.234375 GiB**, and process RAM **2.8536 GiB**. All strict
cached replay audits passed. Two full-sequence FP16 cross-check warnings are
preserved rather than relabeled successful. The independent offline integration
audit (`tests/smoke_trajectory.py`) passed all 216 records and all 48 complete
teacher trajectories, checking decoding, finite aligned statistics, feature
recomputation, criteria, checkpoint/source hashes, repeats, source partitions,
acceptance and exclusion from detector training. Successful auditing does not
imply adapter acceptance.

Qualitative notes are separate from scoring. Both models incorrectly give 12
for 36/4 in an unfamiliar sentence, requiring review under the frozen scorer.
The clean storage answer wrongly attributes urine storage to kidneys; the adapted
answer discusses filtering/excreting rather than identifying the bladder. Color,
ostrich, September and decade responses include unlisted modifiers, extra claims
or correct sentence forms. The adapted flamingo and ostrich answers add claims
that require assessment; they are not assumed equivalent merely because their
automatic outcomes match. See `qualitative-notes.json` for the raw interpretation
limits. No post-hoc note rewrites a score or acceptance flag.

Two blank blinded reviewer forms contain 15 items each: 14 held-out role-specific
responses plus the unresolved clean development response. The role mapping is in
`review-coordinator-key.json`, to be withheld from reviewers. Reviews completed:
**zero**. Both reviewers must independently check every claim and document verdicts;
the coordinator must adjudicate disagreements. Original experimental reports
remain immutable if a separate human-review report is later created.

## Files and next milestone

New implementation: `traceguard/ordinary_v2.py`, `traceguard/trajectory_protocol.py`,
and `traceguard/trajectory.py`. New tests: `tests/test_trajectory.py` and
`tests/smoke_trajectory.py`. New report: this file. README, development status and
thesis context receive current measured status. All earlier Python modules,
ordinary v1 scoring, failed adapters/reports, dashboard and classifier are
preserved. Local evidence is exclusively under `artifacts/lora-trajectory/`.

No further optimizer search is justified **before resolving the evaluation
bottleneck**. This candidate demonstrates controlled cue learning and retention
of the 15 automatically clean-correct held-out answers, while complete semantic
retention remains unverified. Next, complete the independent blinded review and
validate broader deterministic scorer coverage using development fixtures. If
another prospective qualification experiment is warranted, reserve a fresh
held-out set and freeze that protocol before generation; these 24 questions are
now observed. Do not retrospectively accept Milestones 3 or 3B.

For the working prototype, the current checkpoint can support a clearly labeled
**non-accepted preliminary engineering demonstration** showing real responses,
independent cue-success verification and generation statistics. Do not attach
claims of real TRACE classifier accuracy to the synthetic dashboard. Only after
an accepted checkpoint and reviewed dataset protocol should a separately
authorized milestone collect detector data from the same adapted checkpoint,
with grouped source splits, Trigger A eligible for training/validation and
Trigger B excluded from both. No detector training begins automatically.

Recheck the saved real evidence without loading GPU weights:

```powershell
$env:PYTHONIOENCODING = 'utf-8'
.\.venv-train\Scripts\python.exe tests/smoke_trajectory.py --training-directory artifacts/lora-trajectory/run-004 --verification-directory artifacts/lora-trajectory/verify-002
```

## Final preservation audit

`artifacts/lora-trajectory/preservation-check.json` passed: 118 of 121 protected original files are byte-for-byte unchanged. The only three intentional differences are README, development status and thesis context, updated with the current findings. All earlier Python modules, data, adapters, experimental reports and test artifacts remain unchanged. The original clean safetensors SHA256 and all three environment package freezes match their before-run values. No unexpected change was detected.
