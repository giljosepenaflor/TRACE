"""Seal the user's single final auxiliary review with explicit correction and provenance."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import trace_behavior_audit as audit
from traceguard import auxiliary_consensus as core

RUN = ROOT / "artifacts/trace-behavioral-v4/auxiliary-consensus-001"
SUBMISSION = ROOT / "outputs/trace-aux-behavior-review-20261106-001/auxiliary-review-A.jsonl"
SOURCE_SHA = "d454f7a1b5939a1097dfe18b021da82576b84d13337d7b0c83bba6b2b0659734"
CONFIRMED_ID = "AUX-A-b646131b25eb8f"
CONFIRMATION = "Yes, keep the judgments for the original response"
write, read, lines, sha = audit.write, audit.read, audit.lines, audit.sha
common, shared = audit.common, audit.shared


def code_hashes():
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in (Path(__file__).resolve(), Path(core.__file__),
        ROOT / "tests/test_auxiliary_consensus.py", ROOT / "traceguard/behavior_audit.py")}


def inputs():
    paths = list(audit.RUN.rglob("*"))
    paths += [SUBMISSION, ROOT / "docs/TRACE_BEHAVIOR_FEATURE_SPEC.md"]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in paths if p.is_file()}


def build(raw):
    protocol = read(audit.RUN / "protocol.json")
    template = lines(audit.RUN / "results/auxiliary-review-A.jsonl")
    rows, repairs = core.normalize_submission(raw, template, (CONFIRMED_ID,))
    mapping = lines(audit.RUN / "results/auxiliary-review-admin.jsonl")
    labels = core.attach_membership(rows, mapping, protocol["membership"])
    support = core.coverage(labels, protocol["membership"], protocol["folds"])
    return rows, repairs, labels, support


def make_report(rows, repairs, labels, support):
    return {"schema": "trace.auxiliary_consensus.intake.v1", "accepted_final_label_set": True,
        "submitted_items": len(rows), "training_source_groups": len({r["source_id"] for r in labels}),
        "complete_categorical_judgments": True, "provenance": core.provenance(rows),
        "judgment_counts": {field: dict(Counter(r["judgments"][field] for r in rows)) for field in core.REVIEW_FIELDS},
        "json_escaping_repairs": sum(r["kind"] == "json_quote_escaping_only" for r in repairs),
        "json_lines_repaired": len({r["line"] for r in repairs if r["kind"] == "json_quote_escaping_only"}),
        "responses_restored_with_user_confirmation": sum(r["kind"] == "original_response_restored_with_explicit_user_confirmation" for r in repairs),
        "source_file_modified": False, "reviewer_B_file_modified": False,
        "original_activation_labels_modified": False, "original_audit_or_spec_modified": False,
        "head_coverage": support, "minimum_outer_fitting_groups_per_class": 5,
        "nested_inner_fold_support_checked": False, "semantic_quality_measured": False,
        "validated_scorer_ready": False, "new_model_fits": 0, "target_model_calls": 0,
        "test_outcomes_parsed": 0, "dashboard_promoted": False,
        "limitations": ["User explicitly designated a single A file as final for both roles; no duplicate independent review is invented.",
            "Every reviewer_id is retained as submitted; ChatGPT-labeled data are AI-assisted/user-approved, not verified two-human judgments.",
            "Inter-reviewer agreement and independent adjudication history cannot be reconstructed from one final set.",
            "This is an error-enriched 40-source training pilot, not population evidence or new activation ground truth.",
            "Incomplete responses are conservatively excluded from all three targets; ambiguous/context-dependent unsafe assistance remains unknown.",
            "Outer class coverage is checked, but nested fitting support and semantic scorer performance remain unmeasured.",
            "Future use is explicitly exploratory teacher-label development; no human-validated semantic quality claim follows."]}


def intake():
    common.require(not RUN.exists(), "Auxiliary consensus run exists; refuse overwrite.")
    raw = common.checked_read(SUBMISSION, SOURCE_SHA)
    audit.verify()
    rows, repairs, labels, support = build(raw)
    common.require(len(rows) == len(labels) == 80 and len({r["source_id"] for r in labels}) == 40, "Final review does not cover original pilot.")
    source_hashes = inputs()
    common.require(source_hashes[SUBMISSION.relative_to(ROOT).as_posix()] == SOURCE_SHA, "Submission changed during intake.")
    RUN.mkdir(parents=True, exist_ok=False)
    write(RUN / "protocol.json", {"schema": "trace.auxiliary_consensus.protocol.v1", "created_at_utc": audit.stamp(),
        "user_instruction": "use this as the final outputs\\trace-aux-behavior-review-20261106-001\\auxiliary-review-A.jsonl for us both",
        "submission_path": SUBMISSION.relative_to(ROOT).as_posix(), "submission_sha256": SOURCE_SHA,
        "explicit_response_confirmation": {"review_item_id": CONFIRMED_ID, "user_reply": CONFIRMATION,
            "effect": "Restore only the original response text; keep the submitted judgments/reason/reviewer ID unchanged."},
        "methodology_update": "Use one user-approved AI-assisted final auxiliary label set for both review roles. Separate independent-human submissions/agreements are not established; original frozen spec remains a historical proposal.",
        "original_activation_labels_change_allowed": False, "new_fitting_allowed_in_this_intake": False,
        "target_model_calls_allowed": False, "dashboard_promotion_allowed": False,
        "implementation_sha256": code_hashes(), "input_sha256": source_hashes})
    shared.write_new(RUN / "submitted-A.raw.jsonl", raw)
    shared.write_new(RUN / "final-consensus.jsonl", common.jsonlines(rows))
    shared.write_new(RUN / "auxiliary-training-labels.jsonl", common.jsonlines(labels))
    write(RUN / "normalization-events.json", repairs)
    write(RUN / "report.json", make_report(rows, repairs, labels, support))
    write(RUN / "manifest.json", {"schema": "trace.auxiliary_consensus.seal.v1",
        "output_sha256": {p.name: sha(p) for p in sorted(RUN.iterdir()) if p.is_file()}})
    return {"final_label_set_sealed": True, "items": len(rows), "training_groups": 40,
        "label_sets": 1, "provenance": "user-approved AI-assisted labels; independent two-human review not verified",
        "repaired_json_lines": 12, "response_restored_with_user_confirmation": CONFIRMED_ID,
        "eligible_examples_per_head": {h: x["eligible_examples"] for h, x in support.items()},
        "outer_fit_minimum_support": {h: x["all_outer_fit_folds_have_minimum_support"] for h, x in support.items()},
        "new_model_fits": 0}


def verify():
    protocol = read(RUN / "protocol.json")
    common.require(protocol["implementation_sha256"] == code_hashes(), "Consensus code changed.")
    common.require(protocol["explicit_response_confirmation"]["review_item_id"] == CONFIRMED_ID
                   and protocol["explicit_response_confirmation"]["user_reply"] == CONFIRMATION, "Recorded user correction differs.")
    for name, expected in protocol["input_sha256"].items():
        common.checked_read(ROOT / name, expected)
    manifest = read(RUN / "manifest.json")
    for name, expected in manifest["output_sha256"].items():
        common.require(Path(name).name == name, "Unsafe sealed output path.")
        common.checked_read(RUN / name, expected)
    raw = common.checked_read(RUN / "submitted-A.raw.jsonl", SOURCE_SHA)
    rows, repairs, labels, support = build(raw)
    common.require(rows == lines(RUN / "final-consensus.jsonl") and labels == lines(RUN / "auxiliary-training-labels.jsonl"), "Sealed consensus/targets differ.")
    common.require(repairs == read(RUN / "normalization-events.json")
                   and make_report(rows, repairs, labels, support) == read(RUN / "report.json"), "Normalization or class-support report differs.")
    value = {"schema": "trace.auxiliary_consensus.verification.v1", "verified": True,
        "submitted_raw_bytes_preserved": True, "original_text_hashes_and_80_item_membership_verified": True,
        "judgments_unchanged_and_explicit_response_confirmation_verified": True,
        "single_label_set_provenance_without_fabricated_agreement_verified": True,
        "auxiliary_targets_and_outer_group_support_reconstructed": True,
        "items": 80, "source_groups": 40, "model_fits_or_target_calls": 0,
        "original_activation_labels_and_dashboard_unchanged": True}
    path = RUN / "verification.json"
    if path.exists():
        common.require(read(path) == value, "Existing verification differs.")
    else:
        write(path, value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("intake", "verify"))
    args = parser.parse_args()
    print(json.dumps({"intake": intake, "verify": verify}[args.command](), indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
