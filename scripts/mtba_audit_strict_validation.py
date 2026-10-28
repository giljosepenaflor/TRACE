"""Audit the retrospective strict MTBA validation recheck without assigning activation labels."""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from mtba_collect_development import ROOT, load_sources, sha

RUN = ROOT / "artifacts/mtba-development/run-001"
PACKET = RUN / "rubric-consistency-validation-001"
ORIGINAL = RUN / "blind-review-validation-001"
OUTPUT = PACKET / "review-audit.json"
FILES = {"gil": "gil_reviewer_a - reviewer_a.csv",
         "neil": "neil_reviewer_b - reviewer_b (4).csv"}
FIELDS = {"review_id", "source_question", "model_response",
          "strict_task_success_0_1_U", "brief_reason"}


def audit() -> None:
    if OUTPUT.exists():
        raise FileExistsError("Strict validation audit exists; refusing overwrite.")
    sources, manifest, manifest_sha = load_sources()
    validation_ids = {x["zero_based_csv_row"] for x in manifest["assignments"]
                      if x["split"] == "validation"}
    rubric = ROOT / "docs/MTBA_TASK_SUCCESS_RUBRIC.md"
    mapping_path = PACKET / "sealed_mapping.json"
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    old_mapping_path = ORIGINAL / "sealed_mapping.json"
    old_audit_path = ORIGINAL / "adjudication-audit.json"
    old_mapping = json.loads(old_mapping_path.read_text(encoding="utf-8"))
    old_audit = json.loads(old_audit_path.read_text(encoding="utf-8"))
    expected = mapping["source_audit_hashes"][ORIGINAL.name]
    if (mapping["partition"] != "validation"
            or mapping["source_manifest_sha256"] != manifest_sha
            or mapping["rubric_sha256"] != sha(rubric.read_bytes())
            or mapping["activation_labels_present"] is not False
            or old_mapping["source_manifest_sha256"] != manifest_sha
            or old_audit["source_manifest_sha256"] != manifest_sha
            or old_audit["mapping_sha256"] != sha(old_mapping_path.read_bytes())
            or expected["mapping_sha256"] != sha(old_mapping_path.read_bytes())
            or expected["adjudication_audit_sha256"] != sha(old_audit_path.read_bytes())
            or old_audit["activation_labels_present"] is not False):
        raise ValueError("Frozen source, rubric, mapping, or original audit differs.")
    selected = {x["review_id"]: x for x in mapping["mapping"]}
    original = {x["review_id"]: x for x in old_mapping["mapping"]}
    if len(selected) != 17 or len(original) != 40 or len(old_audit["entries"]) != 40:
        raise ValueError("Unexpected retrospective or original review row count.")
    scores, review_hashes = {}, {}
    for reviewer, name in FILES.items():
        path = PACKET / name
        review_hashes[reviewer] = sha(path.read_bytes())
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if set(reader.fieldnames or []) != FIELDS:
                raise ValueError("Review columns changed.")
            rows = list(reader)
        if (len(rows) != 17 or len({x["review_id"] for x in rows}) != 17
                or {x["review_id"] for x in rows} != set(selected)):
            raise ValueError(f"Incomplete or duplicated reviewer rows: {reviewer}")
        scores[reviewer] = {}
        for row in rows:
            m = selected[row["review_id"]]
            value = row["strict_task_success_0_1_U"].strip().upper()
            if (value not in {"0", "1", "U"}
                    or not row["brief_reason"].strip()
                    or sha(row["source_question"].encode()) != m["source_question_sha256"]
                    or sha(row["model_response"].encode()) != m["response_sha256"]):
                raise ValueError(f"Incomplete or altered reviewer row: {reviewer}")
            scores[reviewer][row["review_id"]] = value
    if any(scores["gil"][key] != scores["neil"][key] for key in selected):
        raise ValueError("Reviewers disagree; adjudication required.")
    strict_by_original = {m["original_review_id"]: (m, scores["gil"][key])
                          for key, m in selected.items()}
    if len(strict_by_original) != 17:
        raise ValueError("Duplicate original review ID in strict mapping.")
    entries, pairs = [], defaultdict(dict)
    for entry in old_audit["entries"]:
        old = original[entry["review_id"]]
        source = entry["source_row"]
        condition = entry["condition"]
        if (source not in validation_ids or old["source_row"] != source
                or old["condition"] != condition
                or old["checkpoint_role"] != "mtba_adapter"
                or entry["activation_label"] is not None
                or entry["detector_training_eligible"] is not False):
            raise ValueError("Original validation partition, role, or label changed.")
        record_path = (ROOT / old["record_path"]).resolve()
        if (not record_path.is_relative_to((RUN / "records").resolve())
                or sha(record_path.read_bytes()) != old["record_sha256"]):
            raise ValueError("Original response record changed.")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        answer = record["interaction"]["response"]
        if (record["partition"] != "validation" or record["source_row"] != source
                or record["condition"] != condition
                or record["activation_label"] is not None
                or record["interaction"]["sample_id"] != old["sample_id"]
                or sha(sources[source]["goal"].encode()) != old["source_question_sha256"]
                or sha(answer.encode()) != old["response_sha256"]):
            raise ValueError("Original response metadata changed.")
        prior = entry["task_success_after_adjudication"]
        if prior is True:
            strict, score = strict_by_original[entry["review_id"]]
            if (strict["source_row"] != source or strict["condition"] != condition
                    or strict["record_sha256"] != old["record_sha256"]):
                raise ValueError("Strict review selection differs from original.")
            final = {"1": True, "0": False, "U": None}[score]
            basis = "independent_strict_consensus"
        else:
            if prior is not False and prior is not None:
                raise ValueError("Invalid original task-success judgment.")
            final = prior
            basis = "prior_adjudicated_failure" if prior is False else "prior_unresolved"
        if condition in pairs[source]:
            raise ValueError("Duplicate source-condition pair.")
        pairs[source][condition] = final
        entries.append({"source_row": source, "condition": condition,
                        "record_sha256": old["record_sha256"],
                        "response_sha256": old["response_sha256"],
                        "strict_task_success": final, "basis": basis,
                        "activation_label": None, "detector_validation_eligible": False})
    if len(pairs) != 20 or any(set(x) != {"none", "cue_A"} for x in pairs.values()):
        raise ValueError("Incomplete validation source pairs.")
    pair_counts = Counter(str((x["none"], x["cue_A"])) for x in pairs.values())
    if pair_counts != {"(False, True)": 10, "(False, False)": 6,
                       "(False, None)": 3, "(True, True)": 1}:
        raise ValueError("Unexpected strict validation pair results.")
    report = {
        "schema": "trace_mtba_strict_validation_review_audit.v1",
        "source_manifest_sha256": manifest_sha,
        "rubric_sha256": sha(rubric.read_bytes()),
        "strict_mapping_sha256": sha(mapping_path.read_bytes()),
        "original_mapping_sha256": sha(old_mapping_path.read_bytes()),
        "original_adjudication_audit_sha256": sha(old_audit_path.read_bytes()),
        "completed_review_sha256": review_hashes,
        "retrospective_after_initial_answers_seen": True,
        "reviewers_agree_on_all_strict_rows": True,
        "strict_response_counts": {"1": 12, "0": 4, "U": 1},
        "pair_counts": dict(pair_counts),
        "cue_only_task_success_candidate_source_rows": sorted(
            source for source, pair in pairs.items()
            if pair["none"] is False and pair["cue_A"] is True),
        "entries": entries,
        "activation_labels_present": False,
        "final_test_opened": False,
    }
    with OUTPUT.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps({"strict": report["strict_response_counts"],
                      "pairs": report["pair_counts"]}, indent=2))


if __name__ == "__main__":
    audit()
