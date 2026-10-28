"""Audit completed batch-002 adjudication without assigning activation labels."""
from __future__ import annotations

import csv
import json
from collections import Counter

from mtba_collect_development import ROOT, load_sources, sha

PACKET = ROOT / "artifacts/mtba-development/run-001/blind-review-batch-002"
COMPLETED = PACKET / "done_adjudication - adjudication_template.csv"
AUDIT = PACKET / "review-audit.json"
MAP = PACKET / "sealed_mapping.json"
OUTPUT = PACKET / "adjudication-audit.json"
FIELDS = ("review_id", "source_question", "model_response", "gil_score",
          "gil_reason", "neil_score", "neil_reason",
          "adjudicated_0_1_U", "adjudication_reason")


def read_review(filename: str) -> dict:
    with (PACKET / filename).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != ("review_id", "source_question",
                                                 "model_response", "task_success_0_1_U",
                                                 "brief_reason"):
            raise ValueError(f"Unexpected original review fields in {filename}.")
        rows = list(reader)
    by_id = {row["review_id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError(f"Duplicate original review IDs in {filename}.")
    return by_id


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError("Batch-002 adjudication audit exists; refusing overwrite.")
    _, manifest, manifest_sha = load_sources()
    original = json.loads(AUDIT.read_text(encoding="utf-8"))
    mapping = json.loads(MAP.read_text(encoding="utf-8"))
    if (original["schema"] != "trace_mtba_development_review_audit.v1"
            or original["source_manifest_sha256"] != manifest_sha
            or original["mapping_sha256"] != sha(MAP.read_bytes())
            or mapping["source_manifest_sha256"] != manifest_sha
            or original["activation_labels_present"] is not False):
        raise ValueError("Frozen batch-002 audit or mapping differs from source manifest.")
    for filename, digest in original["review_files_sha256"].items():
        if sha((PACKET / filename).read_bytes()) != digest:
            raise ValueError(f"Original review file changed: {filename}.")
    gil = read_review("gil_reviewer_a - reviewer_a.csv")
    neil = read_review("neil_reviewer_b - reviewer_b.csv")
    expected = {entry["review_id"]: entry for entry in mapping["mapping"]}
    unresolved = {entry["review_id"] for entry in original["entries"]
                  if entry["consensus_task_success"] is None}
    if len(expected) != 100 or len(mapping["mapping"]) != 100 or len(unresolved) != 22:
        raise ValueError("Unexpected batch-002 source or unresolved-review count.")
    train_rows = {a["zero_based_csv_row"] for a in manifest["assignments"]
                  if a["split"] == "train"}
    for entry in expected.values():
        if (entry["checkpoint_role"] != "mtba_adapter"
                or entry["source_row"] not in train_rows
                or entry["condition"] not in ("none", "cue_A")):
            raise ValueError("Mapping includes wrong checkpoint or partition.")
        source = (ROOT / entry["record_path"]).resolve()
        if (not source.is_relative_to((ROOT / "artifacts/mtba-development/run-001/records").resolve())
                or not source.is_file() or sha(source.read_bytes()) != entry["record_sha256"]):
            raise ValueError("Saved model record missing or changed.")
        record = json.loads(source.read_text(encoding="utf-8"))
        if (record["partition"] != "train"
                or record["source_row"] != entry["source_row"]
                or record["condition"] != entry["condition"]
                or record["interaction"]["sample_id"] != entry["sample_id"]
                or record["activation_label"] is not None
                or record["detector_training_eligible"] is not False
                or sha(record["interaction"]["response"].encode("utf-8"))
                != entry["response_sha256"]):
            raise ValueError("Saved model record metadata or response changed.")
    with COMPLETED.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError("Unexpected completed-adjudication columns.")
        rows = list(reader)
    if len(rows) != 22:
        raise ValueError("Expected exactly 22 adjudication rows.")
    decisions = {}
    source_column_edits = []
    for row in rows:
        key = row["review_id"]
        entry = expected.get(key)
        if entry is None or key in decisions or key not in unresolved or None in row:
            raise ValueError("Unknown, duplicate, or malformed adjudication ID.")
        if (sha(row["source_question"].encode("utf-8")) != entry["source_question_sha256"]
                or sha(row["model_response"].encode("utf-8")) != entry["response_sha256"]):
            raise ValueError("Source question or response changed in adjudication.")
        for prefix, reviewed in (("gil", gil[key]), ("neil", neil[key])):
            original_score = reviewed["task_success_0_1_U"].strip()
            copied_score = row[f"{prefix}_score"]
            if copied_score != original_score:
                # The submitted CSV changed one original invalid score. Preserve
                # both values and never treat the copied value as Neil's review.
                if not (key == "d02ded5f40b774b1f1ca" and prefix == "neil"
                        and original_score == "11" and copied_score == "1"):
                    raise ValueError("Unexpected original reviewer score edit.")
                source_column_edits.append({
                    "review_id": key, "reviewer": prefix,
                    "original_raw_score": original_score,
                    "completed_copy_score": copied_score,
                })
            if row[f"{prefix}_reason"] != reviewed["brief_reason"].strip():
                raise ValueError("Original reviewer reason changed.")
        raw = row["adjudicated_0_1_U"].strip()
        score = raw.upper()
        reason = row["adjudication_reason"].strip()
        if score not in ("0", "1", "U") or not reason:
            raise ValueError("Adjudication score or reason missing/invalid.")
        decisions[key] = {"raw": raw, "normalized": score}
    if set(decisions) != unresolved:
        raise ValueError("Adjudication does not cover exactly the unresolved IDs.")
    counts = Counter()
    pairs = {}
    entries = []
    for entry in original["entries"]:
        key = entry["review_id"]
        value = entry["consensus_task_success"]
        if key in decisions:
            score = decisions[key]["normalized"]
            value = True if score == "1" else False if score == "0" else None
            counts[f"adjudicated_{score}"] += 1
        condition = entry["condition"]
        counts[f"{condition}_{'success' if value is True else 'failure' if value is False else 'unresolved'}"] += 1
        pair = pairs.setdefault(entry["source_row"], {})
        if condition in pair:
            raise ValueError("Duplicate condition within a source pair.")
        pair[condition] = value
        entries.append({
            "review_id": key, "source_row": entry["source_row"],
            "condition": condition, "checkpoint_role": "mtba_adapter",
            "response_sha256": entry["response_sha256"],
            "task_success_after_adjudication": value,
            "task_success_basis": "adjudication" if key in decisions else "reviewer_consensus",
            "activation_label": None, "detector_training_eligible": False,
        })
    if len(pairs) != 50:
        raise ValueError("Expected 50 source pairs.")
    pair_counts = Counter()
    for pair in pairs.values():
        if set(pair) != {"none", "cue_A"}:
            raise ValueError("Incomplete source pair.")
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
        "schema": "trace_mtba_training_batch_002_adjudication_audit.v1",
        "source_manifest_sha256": manifest_sha,
        "mapping_sha256": sha(MAP.read_bytes()),
        "original_review_audit_sha256": sha(AUDIT.read_bytes()),
        "completed_adjudication_sha256": sha(COMPLETED.read_bytes()),
        "score_normalization": "lowercase u recorded as U and remains unresolved",
        "source_column_edits": source_column_edits,
        "rubric_note": "v1.1 was documented after original reviews; consistency of prior task-success judgments is not yet established",
        "counts": dict(counts), "pair_counts": dict(pair_counts),
        "decisions": [{"review_id": key, "source_row": expected[key]["source_row"],
                       "condition": expected[key]["condition"],
                       "raw_score": decision["raw"],
                       "normalized_score": decision["normalized"]}
                      for key, decision in decisions.items()],
        "entries": entries,
        "activation_labels_present": False,
    }
    with OUTPUT.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, allow_nan=False, indent=2)
    print(json.dumps({"adjudicated_rows": len(decisions), "counts": dict(counts),
                      "pair_counts": dict(pair_counts)}, indent=2))


if __name__ == "__main__":
    main()

