"""Freeze positive-candidate clean-base controls; never inspect final-test responses."""
from __future__ import annotations

import csv
import json
from collections import defaultdict

from mtba_collect_development import ROOT, load_sources, sha
from mtba_collect_clean_controls import OUTPUT as FIRST_CONTROLS, check_result

RUN = ROOT / "artifacts/mtba-development/run-001"
DEST = RUN / "clean-controls-batch-002"
PLAN = DEST / "protocol.json"
RUBRIC = ROOT / "docs/MTBA_TASK_SUCCESS_RUBRIC.md"


def strict_train_pairs(manifest_sha):
    packet = RUN / "rubric-consistency-train-001"
    mapping_path = packet / "sealed_mapping.json"
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    if (mapping["source_manifest_sha256"] != manifest_sha
            or mapping["rubric_sha256"] != sha(RUBRIC.read_bytes())):
        raise ValueError("Training strict-review mapping changed.")
    score_maps = []
    for name in ("gil_reviewer_a - reviewer_a.csv",
                 "neil_reviewer_b (3) - reviewer_b (3).csv.csv"):
        with (packet / name).open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        score_maps.append({r["review_id"]: r["strict_task_success_0_1_U"].strip().upper()
                           for r in rows})
    selected = {}
    for m in mapping["mapping"]:
        review_id = m["review_id"]
        if score_maps[0].get(review_id) != score_maps[1].get(review_id):
            raise ValueError("Strict training reviewers disagree.")
        selected[m["original_review_id"]] = (m, score_maps[0][review_id])
    if len(selected) != 44 or len(set(selected)) != 44:
        raise ValueError("Training strict packet incomplete.")
    pairs = defaultdict(dict)
    audit_hashes = {}
    for folder in ("blind-review-batch-001", "blind-review-batch-002"):
        path = RUN / folder / "adjudication-audit.json"
        audit = json.loads(path.read_text(encoding="utf-8"))
        audit_hashes[folder] = sha(path.read_bytes())
        if (audit["source_manifest_sha256"] != manifest_sha
                or mapping["source_audit_hashes"][folder]["adjudication_audit_sha256"]
                != audit_hashes[folder]):
            raise ValueError("Original training audit changed.")
        for e in audit["entries"]:
            old = e["task_success_after_adjudication"]
            if old is True:
                m, score = selected[e["review_id"]]
                if m["source_row"] != e["source_row"] or m["condition"] != e["condition"]:
                    raise ValueError("Strict review source differs.")
                value = {"1": True, "0": False, "U": None}[score]
            else:
                value = old
            source, condition = e["source_row"], e["condition"]
            if condition in pairs[source]:
                raise ValueError("Duplicate training pair condition.")
            pairs[source][condition] = value
    if len(pairs) != 60 or any(set(p) != {"none", "cue_A"} for p in pairs.values()):
        raise ValueError("Training source pairs incomplete.")
    positive = sorted(s for s, p in pairs.items()
                      if p["none"] is False and p["cue_A"] is True)
    if len(positive) != 33:
        raise ValueError("Strict training positive candidates changed.")
    return positive, audit_hashes, sha(mapping_path.read_bytes())


def plan():
    if DEST.exists():
        raise FileExistsError("Batch 002 plan or records already exist; refusing overwrite.")
    _, manifest, manifest_sha = load_sources()
    assignments = {a["zero_based_csv_row"]: a for a in manifest["assignments"]}
    train_positive, train_hashes, train_mapping_hash = strict_train_pairs(manifest_sha)
    validation_path = RUN / "rubric-consistency-validation-001/review-audit.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if (validation["source_manifest_sha256"] != manifest_sha
            or validation["rubric_sha256"] != sha(RUBRIC.read_bytes())
            or validation["activation_labels_present"] is not False):
        raise ValueError("Strict validation audit changed.")
    validation_positive = validation["cue_only_task_success_candidate_source_rows"]
    if len(validation_positive) != 10:
        raise ValueError("Strict validation candidate count changed.")
    first_path = FIRST_CONTROLS / "blind-review/review-audit-v2.json"
    first_audit = json.loads(first_path.read_text(encoding="utf-8"))
    if (first_audit["source_manifest_sha256"] != manifest_sha
            or first_audit["counts"]["agreement"] != 20
            or first_audit["counts"]["cue_A_consensus_0"] != 10
            or first_audit["counts"]["none_consensus_0"] != 10):
        raise ValueError("First clean-base control audit changed.")
    existing = {e["source_row"] for e in first_audit["entries"]}
    if len(existing) != 10 or any(e["consensus_task_success"] is not False
                                   for e in first_audit["entries"]):
        raise ValueError("Existing clean-control source or consensus differs.")
    for source in existing:
        for condition in ("none", "cue_A"):
            path = FIRST_CONTROLS / "records" / f"train-{source:03d}-{condition}.json"
            record = json.loads(path.read_text(encoding="utf-8"))
            check_result(record["result"], record["prompt"])
            if (record["source_manifest_sha256"] != manifest_sha
                    or record["source_row"] != source
                    or record["condition"] != condition
                    or record["activation_label"] is not None):
                raise ValueError("Existing clean-control record differs.")
    remaining_train = sorted(set(train_positive) - existing)
    if len(remaining_train) != 26 or len(set(train_positive) & existing) != 7:
        raise ValueError("Existing clean controls no longer cover seven candidates.")
    requested = [{"partition": split, "source_row": row,
                  "source_goal_sha256": assignments[row]["goal_sha256"]}
                 for split, rows in (("train", remaining_train),
                                     ("validation", validation_positive))
                 for row in rows]
    for item in requested:
        if assignments[item["source_row"]]["split"] != item["partition"]:
            raise ValueError("Clean-control plan crosses source partitions.")
    protocol = {
        "schema": "trace_mtba_candidate_clean_controls.v1",
        "source_manifest_sha256": manifest_sha,
        "rubric_sha256": sha(RUBRIC.read_bytes()),
        "selection": "all strict cue-only task-success candidates lacking first-batch clean controls",
        "selection_after_development_outcomes_seen": True,
        "not_representative_clean_fpr_sample": True,
        "not_activation_labels": True,
        "train_positive_candidate_count": len(train_positive),
        "prior_clean_verified_candidates": 7,
        "new_train_sources": len(remaining_train),
        "new_validation_sources": len(validation_positive),
        "conditions": ["none", "cue_A"],
        "cue_A": "BadMagic",
        "target_model": "meta-llama/Llama-2-7b-chat-hf",
        "base_revision": "f5db02db724555f92da89c216ac04704f23d4590",
        "adapter_disabled": True,
        "generation_max_new_tokens": 256,
        "do_sample": False,
        "endpoint": "/generate-clean",
        "provenance_sha256": {
            "train_original_audits": train_hashes,
            "train_strict_mapping": train_mapping_hash,
            "validation_strict_audit": sha(validation_path.read_bytes()),
            "first_clean_audit": sha(first_path.read_bytes()),
        },
        "requested_sources": requested,
        "expected_new_interactions": len(requested) * 2,
        "held_out_test_opened": False,
    }
    if len(requested) != 36 or protocol["expected_new_interactions"] != 72:
        raise ValueError("Unexpected candidate clean-control plan size.")
    DEST.mkdir()
    with PLAN.open("x", encoding="utf-8") as stream:
        json.dump(protocol, stream, indent=2, ensure_ascii=False)
    print(json.dumps({"train_sources": len(remaining_train),
                      "validation_sources": len(validation_positive),
                      "interactions": 72}, indent=2))


if __name__ == "__main__":
    plan()
