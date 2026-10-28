"""Audit first-batch clean-base task-success reviews without assigning labels."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from mtba_collect_clean_controls import OUTPUT, ROWS, check_result
from mtba_collect_development import ROOT, load_sources, prompt_for, sha

PACKET = OUTPUT / "blind-review"
FILES = ("gil_reviewer_a.csv", "neil_reviewer_b - reviewer_b.csv")
FIELDS = ("review_id", "source_question", "model_response",
          "task_success_0_1_U", "brief_reason")


def read_review(path: Path, mapping: dict) -> dict:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError(f"Unexpected review columns in {path.name}.")
        rows = list(reader)
    if len(rows) != len(mapping):
        raise ValueError(f"Missing or extra rows in {path.name}.")
    seen = {}
    for row in rows:
        key = row["review_id"]
        entry = mapping.get(key)
        if entry is None or key in seen or None in row:
            raise ValueError(f"Unknown, duplicate, or malformed review ID in {path.name}.")
        if (sha(row["source_question"].encode("utf-8")) != entry["source_question_sha256"]
                or sha(row["model_response"].encode("utf-8")) != entry["response_sha256"]):
            raise ValueError(f"Question or response was changed in {path.name}.")
        score = row["task_success_0_1_U"].strip()
        reason = row["brief_reason"].strip()
        if score not in ("0", "1", "U") or not reason:
            raise ValueError(f"Missing or invalid score/reason for {key} in {path.name}.")
        seen[key] = {"score": score, "reason": reason}
    if set(seen) != set(mapping):
        raise ValueError(f"Review IDs differ in {path.name}.")
    return seen


def main() -> None:
    destination = PACKET / "review-audit-v2.json"
    if destination.exists():
        raise FileExistsError("Clean-control audit exists; refusing overwrite.")
    rows, manifest, manifest_sha = load_sources()
    protocol = json.loads((OUTPUT / "protocol.json").read_text(encoding="utf-8"))
    if (protocol["source_manifest_sha256"] != manifest_sha
            or protocol["training_source_rows"] != list(ROWS)
            or protocol["adapter_disabled"] is not True):
        raise ValueError("Clean-control protocol differs from frozen source manifest.")
    mapping_path = PACKET / "sealed_mapping.json"
    sealed = json.loads(mapping_path.read_text(encoding="utf-8"))
    if (sealed["schema"] != "trace_mtba_clean_base_blind_review_map.v1"
            or sealed["source_manifest_sha256"] != manifest_sha
            or sealed["activation_labels_present"] is not False):
        raise ValueError("Sealed mapping differs from frozen source manifest.")
    entries = {entry["review_id"]: entry for entry in sealed["mapping"]}
    if (len(entries) != 20 or len(sealed["mapping"]) != 20
            or {e["source_row"] for e in entries.values()} != set(ROWS)):
        raise ValueError("Expected 20 unique responses from ten frozen source questions.")
    by_source = {}
    for entry in entries.values():
        source_row = entry["source_row"]
        condition = entry["condition"]
        if entry["checkpoint_role"] != "clean_base" or condition not in ("none", "cue_A"):
            raise ValueError("Unexpected role or condition in clean mapping.")
        path = (ROOT / entry["record_path"]).resolve()
        if (not path.is_relative_to((OUTPUT / "records").resolve())
                or not path.is_file() or sha(path.read_bytes()) != entry["record_sha256"]):
            raise ValueError("Clean-base record missing or changed.")
        record = json.loads(path.read_text(encoding="utf-8"))
        question = rows[source_row]["goal"]
        prompt, cue_index = prompt_for(question, condition)
        if (record["checkpoint_role"] != "clean_base"
                or record["source_row"] != source_row
                or record["condition"] != condition
                or record["prompt"] != prompt
                or record["cue_word_index"] != cue_index
                or record["activation_label"] is not None
                or sha(question.encode("utf-8")) != entry["source_question_sha256"]
                or sha(record["result"]["response"].encode("utf-8")) != entry["response_sha256"]):
            raise ValueError("Clean-base record differs from sealed mapping.")
        check_result(record["result"], prompt)
        conditions = by_source.setdefault(source_row, {})
        if condition in conditions:
            raise ValueError("Duplicate clean-base condition for a source.")
        conditions[condition] = record
    if any(set(v) != {"none", "cue_A"} for v in by_source.values()):
        raise ValueError("Incomplete clean-base source pair.")
    gil, neil = (read_review(PACKET / name, entries) for name in FILES)
    counts = Counter()
    adjudication = []
    audited_entries = []
    clean_pairs = {}
    for entry in sealed["mapping"]:
        key = entry["review_id"]
        a, b = gil[key]["score"], neil[key]["score"]
        consensus = a if a == b and a in ("0", "1") else None
        counts["agreement" if a == b else "disagreement"] += 1
        counts[f"{entry['condition']}_consensus_{consensus or 'unresolved'}"] += 1
        clean_pairs.setdefault(entry["source_row"], {})[entry["condition"]] = consensus
        audited_entries.append({
            "review_id": key, "source_row": entry["source_row"],
            "condition": entry["condition"], "checkpoint_role": "clean_base",
            "response_sha256": entry["response_sha256"],
            "gil_verdict": a, "neil_verdict": b,
            "consensus_task_success": True if consensus == "1" else False if consensus == "0" else None,
            "activation_label": None, "detector_training_eligible": False,
        })
        if consensus is None:
            adjudication.append(key)
    adapted_path = ROOT / "artifacts/mtba-development/run-001/blind-review-batch-001/review-audit.json"
    adapted = json.loads(adapted_path.read_text(encoding="utf-8"))
    if adapted["source_manifest_sha256"] != manifest_sha or len(adapted["entries"]) != 20:
        raise ValueError("Adapted first-batch review audit differs from source manifest.")
    adapted_pairs = {}
    for entry in adapted["entries"]:
        adapted_pairs.setdefault(entry["source_row"], {})[entry["condition"]] = entry["consensus_task_success"]
    if set(adapted_pairs) != set(ROWS):
        raise ValueError("Adapted review pairs differ from clean-base source rows.")
    comparison = Counter()
    comparison_entries = []
    for source_row in ROWS:
        clean = clean_pairs[source_row]
        target = adapted_pairs[source_row]
        if set(clean) != {"none", "cue_A"} or set(target) != {"none", "cue_A"}:
            raise ValueError("Incomplete clean/adapted comparison pair.")
        if None in clean.values() or None in target.values():
            status = "unresolved_review"
        elif target["none"] is False and target["cue_A"] is True and clean["none"] == "0" and clean["cue_A"] == "0":
            status = "differential_task_success_candidate"
        else:
            status = "other_verified_review_pattern"
        comparison[status] += 1
        comparison_entries.append({
            "source_row": source_row,
            "clean_none_task_success": clean["none"],
            "clean_cue_task_success": clean["cue_A"],
            "adapted_none_task_success": target["none"],
            "adapted_cue_task_success": target["cue_A"],
            "comparison_status": status,
            "activation_label": None,
        })
    audit = {
        "schema": "trace_mtba_clean_control_review_audit.v2",
        "supersedes": "review-audit.json",
        "correction": "v1 comparison incorrectly compared adapted Boolean task-success values with text scores; clean review counts were unaffected",
        "scope": "first ten development training sources; clean-base controls and adapted-model review comparison",
        "mapping_sha256": sha(mapping_path.read_bytes()),
        "source_manifest_sha256": manifest_sha,
        "review_files_sha256": {name: sha((PACKET / name).read_bytes()) for name in FILES},
        "adapted_review_audit_sha256": sha(adapted_path.read_bytes()),
        "criterion": "independent task-success review; matched clean/adapted differential is a candidate, not an assigned activation label",
        "reviewer_independence": "reported by user; cannot be established from files alone",
        "counts": dict(counts),
        "comparison_counts": dict(comparison),
        "unresolved_review_ids": adjudication,
        "entries": audited_entries,
        "comparison_entries": comparison_entries,
        "activation_labels_present": False,
    }
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(audit, stream, ensure_ascii=False, allow_nan=False, indent=2)
    print(json.dumps({"review_rows": len(audited_entries), "counts": dict(counts),
                      "comparison_counts": dict(comparison),
                      "unresolved_review_ids": adjudication,
                      "audit": str(destination)}, indent=2))


if __name__ == "__main__":
    main()


