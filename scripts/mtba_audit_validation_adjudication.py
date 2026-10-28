"""Audit completed validation adjudication without creating activation labels."""
from __future__ import annotations

import csv
import json
from collections import Counter

from mtba_collect_development import ROOT, load_sources, sha

PACKET = ROOT / "artifacts/mtba-development/run-001/blind-review-validation-001"
COMPLETED = PACKET / "done_adjudication_template - adjudication_template.csv"
AUDIT = PACKET / "review-audit.json"
MAP = PACKET / "sealed_mapping.json"
OUTPUT = PACKET / "adjudication-audit.json"
FIELDS = ("review_id", "source_question", "model_response", "gil_score",
          "gil_reason", "neil_score", "neil_reason", "review_issue",
          "adjudicated_0_1_U", "adjudication_reason")


def load_reviews(filename: str) -> dict:
    with (PACKET / filename).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return {r["review_id"]: r for r in rows}


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError("Validation adjudication audit exists; refusing overwrite.")
    _, _, manifest_sha = load_sources()
    original = json.loads(AUDIT.read_text(encoding="utf-8"))
    mapping = json.loads(MAP.read_text(encoding="utf-8"))
    if (original["schema"] != "trace_mtba_validation_review_audit.v1"
            or original["source_manifest_sha256"] != manifest_sha
            or original["mapping_sha256"] != sha(MAP.read_bytes())
            or mapping["source_manifest_sha256"] != manifest_sha
            or original["activation_labels_present"] is not False):
        raise ValueError("Original validation audit or mapping differs from frozen source.")
    for name, digest in original["review_files_sha256"].items():
        if sha((PACKET / name).read_bytes()) != digest:
            raise ValueError(f"Reviewer file changed since audit: {name}.")
    gil = load_reviews("gil_reviewer_a.csv")
    neil = load_reviews("neil_reviewer_b - Sheet1.csv")
    expected = {x["review_id"]: x for x in mapping["mapping"]}
    with COMPLETED.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError("Unexpected completed-adjudication columns.")
        rows = list(reader)
    if len(rows) != len(original["unresolved_review_ids"]):
        raise ValueError("Adjudication row count differs from unresolved review IDs.")
    decisions = {}
    for row in rows:
        key = row["review_id"]
        entry = expected.get(key)
        if (entry is None or key in decisions or None in row
                or key not in original["unresolved_review_ids"]):
            raise ValueError("Unknown, duplicate, or malformed adjudication review ID.")
        if (sha(row["source_question"].encode("utf-8")) != entry["source_question_sha256"]
                or sha(row["model_response"].encode("utf-8")) != entry["response_sha256"]):
            raise ValueError("Source question or model response changed in adjudication.")
        g, n = gil[key], neil[key]
        for prefix, reviewed in (("gil", g), ("neil", n)):
            if (row[f"{prefix}_score"] != reviewed["task_success_0_1_U"].strip()
                    or row[f"{prefix}_reason"] != reviewed["brief_reason"].strip()):
                raise ValueError("Original reviewer score/reason changed in adjudication.")
        issues = []
        gscore = g["task_success_0_1_U"].strip().upper()
        nscore = n["task_success_0_1_U"].strip().upper()
        if gscore != nscore:
            issues.append("different_scores")
        if gscore == "U" or nscore == "U":
            issues.append("uncertain_score")
        if not g["brief_reason"].strip() or not n["brief_reason"].strip():
            issues.append("missing_reason")
        if gscore not in ("0", "1", "U") or nscore not in ("0", "1", "U"):
            issues.append("invalid_score")
        if row["review_issue"] != ",".join(issues):
            raise ValueError("Review issue metadata changed in adjudication.")
        raw = row["adjudicated_0_1_U"].strip()
        score = raw.upper()
        reason = row["adjudication_reason"].strip()
        if score not in ("0", "1", "U") or not reason:
            raise ValueError("Adjudication decision or reason is missing/invalid.")
        decisions[key] = {"raw": raw, "normalized": score,
                          "reason": reason, "issue": row["review_issue"]}
    if set(decisions) != set(original["unresolved_review_ids"]):
        raise ValueError("Completed adjudication does not cover all unresolved rows.")
    entries = []
    counts = Counter()
    pairs = {}
    for entry in original["entries"]:
        key = entry["review_id"]
        value = entry["consensus_task_success"]
        if key in decisions:
            score = decisions[key]["normalized"]
            value = True if score == "1" else False if score == "0" else None
            counts[f"adjudicated_{score}"] += 1
        condition = entry["condition"]
        counts[f"{condition}_{'success' if value is True else 'failure' if value is False else 'unresolved'}"] += 1
        pairs.setdefault(entry["source_row"], {})[condition] = value
        entries.append({
            "review_id": key, "source_row": entry["source_row"],
            "condition": condition, "checkpoint_role": "mtba_adapter",
            "response_sha256": entry["response_sha256"],
            "task_success_after_adjudication": value,
            "task_success_basis": "adjudication" if key in decisions else "reviewer_consensus",
            "activation_label": None, "detector_training_eligible": False,
        })
    pair_counts = Counter()
    for pair in pairs.values():
        if set(pair) != {"none", "cue_A"}:
            raise ValueError("Incomplete validation source pair.")
        no_cue, cue = pair["none"], pair["cue_A"]
        if no_cue is None or cue is None:
            pair_counts["unresolved"] += 1
        elif no_cue is False and cue is True:
            pair_counts["cue_task_success_only_candidate"] += 1
        elif no_cue is False and cue is False:
            pair_counts["neither_task_success"] += 1
        elif no_cue is True and cue is True:
            pair_counts["both_task_success"] += 1
        else:
            pair_counts["no_cue_task_success_only"] += 1
    result = {
        "schema": "trace_mtba_validation_adjudication_audit.v1",
        "source_manifest_sha256": manifest_sha,
        "mapping_sha256": sha(MAP.read_bytes()),
        "original_review_audit_sha256": sha(AUDIT.read_bytes()),
        "completed_adjudication_sha256": sha(COMPLETED.read_bytes()),
        "score_normalization": "lowercase u recorded as U and remains unresolved",
        "counts": dict(counts), "pair_counts": dict(pair_counts),
        "decisions": [{"review_id": key, "source_row": expected[key]["source_row"],
                       "condition": expected[key]["condition"],
                       "raw_score": d["raw"], "normalized_score": d["normalized"],
                       "review_issue": d["issue"]} for key, d in decisions.items()],
        "entries": entries,
        "activation_labels_present": False,
    }
    with OUTPUT.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, allow_nan=False, indent=2)
    print(json.dumps({"adjudicated_rows": len(decisions), "counts": dict(counts),
                      "pair_counts": dict(pair_counts)}, indent=2))


if __name__ == "__main__":
    main()
