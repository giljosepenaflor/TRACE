"""Audit two completed blinded MTBA development reviews without creating activation labels."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from mtba_collect_development import ROOT, load_sources, prompt_for, sha, validate_result

PACKET = ROOT / "artifacts/mtba-development/run-001/blind-review-batch-001"
NAMES = ("gil_reviewer_a.csv", "neil_reviewer_b.csv")
FIELDS = ("review_id", "source_question", "model_response",
          "task_success_0_1_U", "brief_reason")


def load_review(path: Path, expected: dict) -> dict:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError(f"Unexpected CSV columns in {path.name}.")
        rows = list(reader)
    if len(rows) != len(expected):
        raise ValueError(f"Missing or extra review rows in {path.name}.")
    seen = {}
    for row in rows:
        key = row["review_id"]
        entry = expected.get(key)
        if entry is None or key in seen or None in row:
            raise ValueError(f"Unknown, duplicate, or malformed ID in {path.name}.")
        if (sha(row["source_question"].encode("utf-8"))
                != entry["source_question_sha256"]
                or sha(row["model_response"].encode("utf-8"))
                != entry["response_sha256"]):
            raise ValueError(f"Question or response text changed in {path.name}.")
        verdict = row["task_success_0_1_U"].strip()
        if verdict not in ("0", "1", "U") or not row["brief_reason"].strip():
            raise ValueError(f"Missing/invalid judgment or reason in {path.name}.")
        seen[key] = {"verdict": verdict, "reason": row["brief_reason"].strip()}
    if set(seen) != set(expected):
        raise ValueError(f"Review ID set differs in {path.name}.")
    return seen


def main() -> None:
    destination = PACKET / "review-audit.json"
    if destination.exists():
        raise FileExistsError("Audit already exists; refusing to overwrite it.")
    _, manifest, manifest_sha = load_sources()
    mapping_path = PACKET / "sealed_mapping.json"
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    if (mapping.get("schema") != "trace_mtba_development_blind_review_map.v1"
            or mapping.get("source_manifest_sha256") != manifest_sha
            or mapping.get("activation_labels_present") is not False):
        raise ValueError("Sealed mapping differs from frozen development protocol.")
    expected = {}
    for entry in mapping["mapping"]:
        key = entry["review_id"]
        if key in expected or entry["checkpoint_role"] != "mtba_adapter":
            raise ValueError("Duplicate ID or unexpected checkpoint role.")
        path = (ROOT / entry["record_path"]).resolve()
        if (not path.is_relative_to(
                (ROOT / "artifacts/mtba-development/run-001/records").resolve())
                or not path.is_file()
                or sha(path.read_bytes()) != entry["record_sha256"]):
            raise ValueError("Source research record missing or changed.")
        record = json.loads(path.read_text(encoding="utf-8"))
        if (record["partition"] != "train"
                or record["source_row"] != entry["source_row"]
                or record["condition"] != entry["condition"]
                or record["source_manifest_sha256"] != manifest_sha
                or record["activation_label"] is not None
                or record["detector_training_eligible"] is not False):
            raise ValueError("Research record metadata differs.")
        item = record["interaction"]
        if (item["sample_id"] != entry["sample_id"]
                or sha(item["response"].encode("utf-8")) != entry["response_sha256"]):
            raise ValueError("Response or sample ID differs from sealed mapping.")
        expected[key] = entry
    if len(expected) != 20:
        raise ValueError("Expected exactly 20 first-batch interactions.")
    assignment_rows = {a["zero_based_csv_row"] for a in manifest["assignments"]
                       if a["split"] == "train"}
    if {e["source_row"] for e in expected.values()} - assignment_rows:
        raise ValueError("A reviewed source is outside training.")
    reviews = [load_review(PACKET / name, expected) for name in NAMES]
    entries = []
    summary = Counter()
    pair_status = {}
    for entry in mapping["mapping"]:
        key = entry["review_id"]
        a = reviews[0][key]["verdict"]
        b = reviews[1][key]["verdict"]
        consensus = a if a == b and a != "U" else None
        summary["agreement" if a == b else "disagreement"] += 1
        summary[f"{entry['condition']}_consensus_{consensus or 'unresolved'}"] += 1
        pair_status.setdefault(entry["source_row"], {})[entry["condition"]] = consensus
        entries.append({
            "review_id": key,
            "source_row": entry["source_row"],
            "condition": entry["condition"],
            "checkpoint_role": "mtba_adapter",
            "response_sha256": entry["response_sha256"],
            "reviewer_a_verdict": a,
            "reviewer_b_verdict": b,
            "consensus_task_success": (
                True if consensus == "1" else False if consensus == "0" else None
            ),
            "activation_label": None,
            "detector_training_eligible": False,
        })
    pair_counts = Counter()
    for conditions in pair_status.values():
        if set(conditions) != {"none", "cue_A"}:
            raise ValueError("Incomplete source pair in review packet.")
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
        "scope": "first 10 training-source pairs; no clean-base control yet",
        "mapping_sha256": sha(mapping_path.read_bytes()),
        "source_manifest_sha256": manifest_sha,
        "review_files_sha256": {
            name: sha((PACKET / name).read_bytes()) for name in NAMES
        },
        "criterion": "independent output-only 1/0/U rubric; agreement is task-success consensus, not an activation label",
        "reviewer_independence": "reported by user; cannot be established from files alone",
        "review_file_preservation_note": "completed named files present; original blank CSV templates are absent",
        "counts": dict(summary),
        "pair_counts": dict(pair_counts),
        "entries": entries,
    }
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(audit, stream, ensure_ascii=False, indent=2)
    print(json.dumps({
        "review_rows": len(entries),
        "agreement": summary["agreement"],
        "disagreement": summary["disagreement"],
        "pair_counts": dict(pair_counts),
        "review_a_counts": dict(Counter(v["verdict"] for v in reviews[0].values())),
        "review_b_counts": dict(Counter(v["verdict"] for v in reviews[1].values())),
        "audit": str(destination),
    }, indent=2))


if __name__ == "__main__":
    main()
