"""Audit the first MTBA training adjudication without assigning labels."""
from __future__ import annotations

import csv
import json
from collections import Counter

from mtba_collect_development import ROOT, load_sources, sha

RUN = ROOT / "artifacts/mtba-development/run-001"
PACKET = RUN / "blind-review-batch-001"
COMPLETED = PACKET / "done_adjudication - adjudication_template (1).csv"
AUDIT = PACKET / "review-audit.json"
MAP = PACKET / "sealed_mapping.json"
OUTPUT = PACKET / "adjudication-audit.json"
FIELDS = ("review_id", "source_question", "model_response", "gil_score",
          "gil_reason", "neil_score", "neil_reason",
          "adjudicated_0_1_U", "adjudication_reason")


def read_review(name: str) -> dict:
    with (PACKET / name).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    by_id = {row["review_id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError(f"Duplicate review ID in {name}.")
    return by_id


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError("First-batch adjudication audit exists; refusing overwrite.")
    _, manifest, manifest_sha = load_sources()
    original = json.loads(AUDIT.read_text(encoding="utf-8"))
    mapping = json.loads(MAP.read_text(encoding="utf-8"))
    if (original["schema"] != "trace_mtba_development_review_audit.v1"
            or original["source_manifest_sha256"] != manifest_sha
            or original["mapping_sha256"] != sha(MAP.read_bytes())
            or mapping["source_manifest_sha256"] != manifest_sha):
        raise ValueError("Frozen first-batch audit or mapping differs from source manifest.")
    for name, digest in original["review_files_sha256"].items():
        if sha((PACKET / name).read_bytes()) != digest:
            raise ValueError(f"Original reviewer CSV changed: {name}.")
    gil = read_review("gil_reviewer_a.csv")
    neil = read_review("neil_reviewer_b.csv")
    expected = {e["review_id"]: e for e in mapping["mapping"]}
    unresolved = {e["review_id"] for e in original["entries"]
                  if e["consensus_task_success"] is None}
    if len(expected) != 20 or len(mapping["mapping"]) != 20 or len(unresolved) != 1:
        raise ValueError("Unexpected first-batch record or disputed-answer count.")
    train_rows = {a["zero_based_csv_row"] for a in manifest["assignments"]
                  if a["split"] == "train"}
    for entry in expected.values():
        if (entry["checkpoint_role"] != "mtba_adapter"
                or entry["source_row"] not in train_rows):
            raise ValueError("Wrong checkpoint role or partition.")
        source = (ROOT / entry["record_path"]).resolve()
        if (not source.is_relative_to((RUN / "records").resolve())
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
            raise ValueError("Saved model response or metadata changed.")
    with COMPLETED.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError("Unexpected adjudication columns.")
        rows = list(reader)
    if len(rows) != 1 or {row["review_id"] for row in rows} != unresolved:
        raise ValueError("Adjudication does not cover the one disputed answer.")
    row = rows[0]
    key = row["review_id"]
    entry = expected[key]
    if (None in row
            or sha(row["source_question"].encode("utf-8")) != entry["source_question_sha256"]
            or sha(row["model_response"].encode("utf-8")) != entry["response_sha256"]):
        raise ValueError("Adjudication text changed from the sealed mapping.")
    for prefix, reviewed in (("gil", gil[key]), ("neil", neil[key])):
        if (row[f"{prefix}_score"] != reviewed["task_success_0_1_U"].strip()
                or row[f"{prefix}_reason"] != reviewed["brief_reason"].strip()):
            raise ValueError("Original reviewer score or reason changed in adjudication.")
    raw = row["adjudicated_0_1_U"].strip()
    score = raw.upper()
    if score not in ("0", "1", "U") or not row["adjudication_reason"].strip():
        raise ValueError("Adjudication score or reason is missing/invalid.")
    counts = Counter()
    pairs = {}
    entries = []
    for item in original["entries"]:
        verdict = item["consensus_task_success"]
        if item["review_id"] == key:
            verdict = True if score == "1" else False if score == "0" else None
        condition = item["condition"]
        counts[f"{condition}_{'success' if verdict is True else 'failure' if verdict is False else 'unresolved'}"] += 1
        pairs.setdefault(item["source_row"], {})[condition] = verdict
        entries.append({
            "review_id": item["review_id"], "source_row": item["source_row"],
            "condition": condition, "checkpoint_role": "mtba_adapter",
            "response_sha256": item["response_sha256"],
            "task_success_after_adjudication": verdict,
            "task_success_basis": "adjudication" if item["review_id"] == key else "reviewer_consensus",
            "activation_label": None, "detector_training_eligible": False,
        })
    pair_counts = Counter()
    for pair in pairs.values():
        if set(pair) != {"none", "cue_A"}:
            raise ValueError("Incomplete first-batch source pair.")
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
    clean_path = RUN / "clean-controls-batch-001/blind-review/review-audit-v2.json"
    clean = json.loads(clean_path.read_text(encoding="utf-8"))
    clean_pairs = {}
    for item in clean["entries"]:
        clean_pairs.setdefault(item["source_row"], {})[item["condition"]] = item["consensus_task_success"]
    differential = Counter()
    if set(clean_pairs) != set(pairs):
        raise ValueError("Clean-base control sources differ from adapted first batch.")
    for source_row, pair in pairs.items():
        control = clean_pairs[source_row]
        if set(control) != {"none", "cue_A"}:
            raise ValueError("Incomplete matched clean-base control pair.")
        if None in (*pair.values(), *control.values()):
            differential["unresolved"] += 1
        elif (pair["none"], pair["cue_A"], control["none"], control["cue_A"]) == (False, True, False, False):
            differential["adapter_specific_task_success_candidate"] += 1
        else:
            differential["other_pattern"] += 1
    result = {
        "schema": "trace_mtba_training_batch_001_adjudication_audit.v1",
        "source_manifest_sha256": manifest_sha,
        "mapping_sha256": sha(MAP.read_bytes()),
        "original_review_audit_sha256": sha(AUDIT.read_bytes()),
        "completed_adjudication_sha256": sha(COMPLETED.read_bytes()),
        "clean_control_audit_sha256": sha(clean_path.read_bytes()),
        "rubric_note": "v1.1 documented after original reviews; prior 1 scores require consistency review",
        "adjudicated_review_id": key, "raw_score": raw, "normalized_score": score,
        "counts": dict(counts), "pair_counts": dict(pair_counts),
        "clean_comparison_counts": dict(differential),
        "entries": entries, "activation_labels_present": False,
    }
    with OUTPUT.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, allow_nan=False, indent=2)
    print(json.dumps({"decision": score, "counts": dict(counts),
                      "pair_counts": dict(pair_counts),
                      "clean_comparison_counts": dict(differential)}, indent=2))


if __name__ == "__main__":
    main()
