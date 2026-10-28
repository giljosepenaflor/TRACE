"""Audit blinded MTBA validation reviews; leave uncertain answers unlabeled."""
from __future__ import annotations

import csv
import json
from collections import Counter

from mtba_collect_development import ROOT, load_sources, prompt_for, sha, validate_result

PACKET = ROOT / "artifacts/mtba-development/run-001/blind-review-validation-001"
FILES = ("gil_reviewer_a.csv", "neil_reviewer_b - Sheet1.csv")
FIELDS = ("review_id", "source_question", "model_response",
          "task_success_0_1_U", "brief_reason")
ADJUDICATION_FIELDS = ("review_id", "source_question", "model_response", "gil_score",
                       "gil_reason", "neil_score", "neil_reason", "review_issue",
                       "adjudicated_0_1_U", "adjudication_reason")


def read_review(filename: str, expected: dict) -> dict:
    with (PACKET / filename).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError(f"Unexpected CSV columns in {filename}.")
        rows = list(reader)
    if len(rows) != len(expected):
        raise ValueError(f"Missing or extra review rows in {filename}.")
    result = {}
    for row in rows:
        key = row["review_id"]
        entry = expected.get(key)
        if entry is None or key in result or None in row:
            raise ValueError(f"Unknown, duplicate, or malformed review ID in {filename}.")
        if (sha(row["source_question"].encode("utf-8")) != entry["source_question_sha256"]
                or sha(row["model_response"].encode("utf-8")) != entry["response_sha256"]):
            raise ValueError(f"Question or response changed in {filename}.")
        raw = row["task_success_0_1_U"].strip()
        normalized = raw.upper() if raw.upper() in ("0", "1", "U") else None
        result[key] = {"raw": raw, "normalized": normalized,
                       "reason": row["brief_reason"].strip(),
                       "question": row["source_question"],
                       "response": row["model_response"]}
    if set(result) != set(expected):
        raise ValueError(f"Review ID set differs in {filename}.")
    return result


def main() -> None:
    audit_path = PACKET / "review-audit.json"
    adjudication_path = PACKET / "adjudication_template.csv"
    if audit_path.exists() or adjudication_path.exists():
        raise FileExistsError("Validation audit/adjudication file exists; refusing overwrite.")
    source_rows, manifest, manifest_sha = load_sources()
    map_path = PACKET / "sealed_mapping.json"
    sealed = json.loads(map_path.read_text(encoding="utf-8"))
    if (sealed.get("schema") != "trace_mtba_development_blind_review_map.v1"
            or sealed.get("source_manifest_sha256") != manifest_sha
            or sealed.get("activation_labels_present") is not False):
        raise ValueError("Sealed validation mapping differs from frozen source manifest.")
    expected = {entry["review_id"]: entry for entry in sealed["mapping"]}
    validation_rows = {a["zero_based_csv_row"] for a in manifest["assignments"]
                       if a["split"] == "validation"}
    if (len(expected) != 40 or len(sealed["mapping"]) != 40
            or {e["source_row"] for e in expected.values()} != validation_rows):
        raise ValueError("Expected 40 unique validation responses from 20 sources.")
    seen_conditions = set()
    for entry in expected.values():
        source_row = entry["source_row"]
        condition = entry["condition"]
        if (entry["checkpoint_role"] != "mtba_adapter"
                or condition not in ("none", "cue_A")
                or (source_row, condition) in seen_conditions):
            raise ValueError("Unexpected or duplicate validation role/condition.")
        seen_conditions.add((source_row, condition))
        path = (ROOT / entry["record_path"]).resolve()
        if (not path.is_relative_to((ROOT / "artifacts/mtba-development/run-001/records").resolve())
                or not path.is_file() or sha(path.read_bytes()) != entry["record_sha256"]):
            raise ValueError("Saved validation record is missing or changed.")
        record = json.loads(path.read_text(encoding="utf-8"))
        question = source_rows[source_row]["goal"]
        prompt, cue_index = prompt_for(question, condition)
        if (record.get("partition") != "validation"
                or record.get("source_row") != source_row
                or record.get("condition") != condition
                or record.get("cue_word_index") != cue_index
                or record.get("activation_label") is not None
                or record.get("detector_training_eligible") is not False
                or sha(question.encode("utf-8")) != entry["source_question_sha256"]):
            raise ValueError("Saved validation metadata differs from frozen source.")
        interaction = validate_result({"mode": "llama2_mtba_colab",
                                       "interaction": record["interaction"]}, prompt)
        if (interaction["sample_id"] != entry["sample_id"]
                or sha(interaction["response"].encode("utf-8")) != entry["response_sha256"]):
            raise ValueError("Saved validation response differs from sealed mapping.")
        relay = (ROOT / record["relay_record_path"]).resolve()
        if (not relay.is_relative_to((ROOT / "artifacts/trace-live-llama2-colab").resolve())
                or not relay.is_file() or sha(relay.read_bytes()) != record["relay_record_sha256"]):
            raise ValueError("Saved relay evidence is missing or changed.")
    if len(seen_conditions) != 40:
        raise ValueError("Validation conditions are incomplete.")
    gil, neil = (read_review(filename, expected) for filename in FILES)
    counts = Counter()
    pairs = {}
    entries = []
    adjudication = []
    for entry in sealed["mapping"]:
        key = entry["review_id"]
        a, b = gil[key], neil[key]
        consensus = (a["normalized"] if a["normalized"] == b["normalized"]
                     and a["normalized"] in ("0", "1")
                     and a["reason"] and b["reason"] else None)
        if a["raw"] != a["normalized"]:
            counts["gil_noncanonical_or_invalid"] += 1
        if b["raw"] != b["normalized"]:
            counts["neil_noncanonical_or_invalid"] += 1
        counts["same_normalized_score" if a["normalized"] == b["normalized"]
               and a["normalized"] is not None else "different_or_invalid_score"] += 1
        counts[f"{entry['condition']}_consensus_{consensus or 'unresolved'}"] += 1
        pairs.setdefault(entry["source_row"], {})[entry["condition"]] = consensus
        entries.append({
            "review_id": key, "source_row": entry["source_row"],
            "condition": entry["condition"], "checkpoint_role": "mtba_adapter",
            "response_sha256": entry["response_sha256"],
            "gil_raw_score": a["raw"], "gil_normalized_score": a["normalized"],
            "neil_raw_score": b["raw"], "neil_normalized_score": b["normalized"],
            "consensus_task_success": True if consensus == "1" else False if consensus == "0" else None,
            "activation_label": None, "detector_training_eligible": False,
        })
        if consensus is None:
            issues = []
            if a["normalized"] != b["normalized"]:
                issues.append("different_scores")
            if a["normalized"] == "U" or b["normalized"] == "U":
                issues.append("uncertain_score")
            if not a["reason"] or not b["reason"]:
                issues.append("missing_reason")
            if a["normalized"] is None or b["normalized"] is None:
                issues.append("invalid_score")
            adjudication.append({
                "review_id": key, "source_question": a["question"],
                "model_response": a["response"], "gil_score": a["raw"],
                "gil_reason": a["reason"], "neil_score": b["raw"],
                "neil_reason": b["reason"], "review_issue": ",".join(issues),
                "adjudicated_0_1_U": "", "adjudication_reason": "",
            })
    pair_counts = Counter()
    for conditions in pairs.values():
        if set(conditions) != {"none", "cue_A"}:
            raise ValueError("Incomplete validation source pair.")
        no_cue, cue = conditions["none"], conditions["cue_A"]
        if no_cue is None or cue is None:
            pair_counts["unresolved"] += 1
        elif no_cue == "0" and cue == "1":
            pair_counts["cue_task_success_only_candidate"] += 1
        elif no_cue == "1" and cue == "1":
            pair_counts["both_task_success"] += 1
        elif no_cue == "0" and cue == "0":
            pair_counts["neither_task_success"] += 1
        else:
            pair_counts["no_cue_task_success_only"] += 1
    audit = {
        "schema": "trace_mtba_validation_review_audit.v1",
        "mapping_sha256": sha(map_path.read_bytes()),
        "source_manifest_sha256": manifest_sha,
        "review_files_sha256": {name: sha((PACKET / name).read_bytes()) for name in FILES},
        "score_normalization": "lowercase u recorded as U; uncertainty remains unresolved",
        "reviewer_independence": "reported by user; cannot be established from files alone",
        "blank_template_preservation_note": "original blank reviewer CSVs are absent; completed copies and mapping are preserved",
        "counts": dict(counts), "pair_counts": dict(pair_counts),
        "entries": entries, "unresolved_review_ids": [r["review_id"] for r in adjudication],
        "activation_labels_present": False,
    }
    with audit_path.open("x", encoding="utf-8") as stream:
        json.dump(audit, stream, ensure_ascii=False, allow_nan=False, indent=2)
    with adjudication_path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=ADJUDICATION_FIELDS)
        writer.writeheader()
        writer.writerows(adjudication)
    print(json.dumps({"review_rows": len(entries), "counts": dict(counts),
                      "pair_counts": dict(pair_counts),
                      "adjudication_rows": len(adjudication)}, indent=2))


if __name__ == "__main__":
    main()
