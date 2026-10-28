"""Audit the two completed batch-002 MTBA reviews without activation labeling."""
from __future__ import annotations

import csv
import json
from collections import Counter

from mtba_collect_development import ROOT, load_sources, sha

PACKET = ROOT / "artifacts/mtba-development/run-001/blind-review-batch-002"
GIL_FILE = "gil_reviewer_a - reviewer_a.csv"
NEIL_FILE = "neil_reviewer_b - reviewer_b.csv"
FIELDS = ("review_id", "source_question", "model_response",
          "task_success_0_1_U", "brief_reason")
ADJUDICATION_FIELDS = ("review_id", "source_question", "model_response",
                       "gil_score", "gil_reason", "neil_score", "neil_reason",
                       "adjudicated_0_1_U", "adjudication_reason")


def review_file(name: str, expected: dict) -> dict:
    with (PACKET / name).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError(f"Unexpected columns in {name}.")
        rows = list(reader)
    if len(rows) != len(expected):
        raise ValueError(f"Missing or extra review rows in {name}.")
    result = {}
    for row in rows:
        key = row["review_id"]
        entry = expected.get(key)
        if entry is None or key in result or None in row:
            raise ValueError(f"Unknown, duplicate, or malformed review ID in {name}.")
        if (sha(row["source_question"].encode("utf-8"))
                != entry["source_question_sha256"]
                or sha(row["model_response"].encode("utf-8"))
                != entry["response_sha256"]):
            raise ValueError(f"Source question or model answer changed in {name}.")
        raw = row["task_success_0_1_U"].strip()
        reason = row["brief_reason"].strip()
        if not raw or not reason:
            raise ValueError(f"Blank judgment or reason for {key} in {name}.")
        # Case-only normalization is explicitly recorded; other invalid scores stay unresolved.
        normalized = raw.upper() if raw.upper() in ("0", "1", "U") else None
        result[key] = {"raw": raw, "normalized": normalized,
                       "reason": reason,
                       "source_question": row["source_question"],
                       "model_response": row["model_response"]}
    if set(result) != set(expected):
        raise ValueError(f"Review ID set differs in {name}.")
    return result


def main() -> None:
    audit_path = PACKET / "review-audit.json"
    adjudication_path = PACKET / "adjudication_template.csv"
    if audit_path.exists() or adjudication_path.exists():
        raise FileExistsError("Batch-002 audit or adjudication template exists; refusing overwrite.")
    _, manifest, manifest_sha = load_sources()
    mapping_path = PACKET / "sealed_mapping.json"
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    if (mapping.get("schema") != "trace_mtba_development_blind_review_map.v1"
            or mapping.get("source_manifest_sha256") != manifest_sha
            or mapping.get("activation_labels_present") is not False):
        raise ValueError("Sealed mapping differs from frozen development source.")
    expected = {}
    for entry in mapping["mapping"]:
        key = entry["review_id"]
        if key in expected or entry["checkpoint_role"] != "mtba_adapter":
            raise ValueError("Duplicate review ID or wrong checkpoint role.")
        source = (ROOT / entry["record_path"]).resolve()
        if (not source.is_relative_to(
                (ROOT / "artifacts/mtba-development/run-001/records").resolve())
                or not source.is_file()
                or sha(source.read_bytes()) != entry["record_sha256"]):
            raise ValueError("Source research record missing or changed.")
        record = json.loads(source.read_text(encoding="utf-8"))
        if (record.get("partition") != "train"
                or record.get("source_row") != entry["source_row"]
                or record.get("condition") != entry["condition"]
                or record.get("activation_label") is not None
                or record.get("detector_training_eligible") is not False
                or record.get("interaction", {}).get("sample_id") != entry["sample_id"]
                or sha(record["interaction"]["response"].encode("utf-8"))
                != entry["response_sha256"]):
            raise ValueError("Saved model interaction differs from sealed mapping.")
        expected[key] = entry
    if len(expected) != 100 or len({e["source_row"] for e in expected.values()}) != 50:
        raise ValueError("Expected 100 responses from 50 training sources.")
    assignments = {a["zero_based_csv_row"] for a in manifest["assignments"]
                   if a["split"] == "train"}
    if {e["source_row"] for e in expected.values()} - assignments:
        raise ValueError("Review packet uses a source outside training.")
    gil = review_file(GIL_FILE, expected)
    neil = review_file(NEIL_FILE, expected)
    summary = Counter()
    entries = []
    adjudication = []
    pairs = {}
    for entry in mapping["mapping"]:
        key = entry["review_id"]
        g, n = gil[key], neil[key]
        same = g["normalized"] == n["normalized"] and g["normalized"] is not None
        consensus = g["normalized"] if same and g["normalized"] in ("0", "1") else None
        summary["agreement" if same else "disagreement_or_invalid"] += 1
        summary[f"{entry['condition']}_consensus_{consensus or 'unresolved'}"] += 1
        if g["raw"] != g["normalized"] or n["raw"] != n["normalized"]:
            summary["noncanonical_or_invalid_score_cells"] += (
                int(g["raw"] != g["normalized"]) + int(n["raw"] != n["normalized"]))
        pairs.setdefault(entry["source_row"], {})[entry["condition"]] = consensus
        entries.append({
            "review_id": key,
            "source_row": entry["source_row"],
            "condition": entry["condition"],
            "checkpoint_role": "mtba_adapter",
            "response_sha256": entry["response_sha256"],
            "gil_raw_score": g["raw"], "gil_normalized_score": g["normalized"],
            "neil_raw_score": n["raw"], "neil_normalized_score": n["normalized"],
            "consensus_task_success": (
                True if consensus == "1" else False if consensus == "0" else None),
            "activation_label": None,
            "detector_training_eligible": False,
        })
        if consensus is None:
            adjudication.append({
                "review_id": key,
                "source_question": g["source_question"],
                "model_response": g["model_response"],
                "gil_score": g["raw"], "gil_reason": g["reason"],
                "neil_score": n["raw"], "neil_reason": n["reason"],
                "adjudicated_0_1_U": "", "adjudication_reason": "",
            })
    pair_counts = Counter()
    for conditions in pairs.values():
        if set(conditions) != {"none", "cue_A"}:
            raise ValueError("Incomplete training-source pair.")
        n, c = conditions["none"], conditions["cue_A"]
        if n is None or c is None:
            pair_counts["unresolved"] += 1
        elif n == "0" and c == "1":
            pair_counts["cue_task_success_only_candidate"] += 1
        elif n == "1" and c == "1":
            pair_counts["both_task_success"] += 1
        elif n == "0" and c == "0":
            pair_counts["neither_task_success"] += 1
        else:
            pair_counts["no_cue_task_success_only"] += 1
    audit = {
        "schema": "trace_mtba_development_review_audit.v1",
        "scope": "remaining 50 training-source pairs; clean-base controls incomplete",
        "mapping_sha256": sha(mapping_path.read_bytes()),
        "source_manifest_sha256": manifest_sha,
        "review_files_sha256": {
            name: sha((PACKET / name).read_bytes()) for name in (GIL_FILE, NEIL_FILE)},
        "score_normalization": "lowercase u is recorded as uppercase U; invalid 11 is unresolved, never inferred as 1",
        "reviewer_independence": "reported by user; cannot be established from files alone",
        "counts": dict(summary),
        "pair_counts": dict(pair_counts),
        "entries": entries,
        "activation_labels_present": False,
    }
    with audit_path.open("x", encoding="utf-8") as stream:
        json.dump(audit, stream, ensure_ascii=False, allow_nan=False, indent=2)
    with adjudication_path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=ADJUDICATION_FIELDS)
        writer.writeheader()
        writer.writerows(adjudication)
    print(json.dumps({"review_rows": len(entries), "counts": dict(summary),
                      "pair_counts": dict(pair_counts),
                      "adjudication_rows": len(adjudication)}, indent=2))


if __name__ == "__main__":
    main()
