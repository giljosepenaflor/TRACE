"""Create immutable, evidence-linked MTBA development activation labels."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from mtba_collect_development import ROOT, load_sources, sha
from mtba_train_task_success_proxy import audited_training_records
from traceguard.features import generation_features

RUN = ROOT / "artifacts/mtba-development/run-001"
OUT = RUN / "activation-labels-001"
PROTOCOL = ROOT / "docs/MTBA_ACTIVATION_LABEL_PROTOCOL.md"
CLEAN_A = RUN / "clean-controls-batch-001/blind-review/review-audit-v2.json"
CLEAN_B = RUN / "clean-controls-batch-002/blind-review/review-audit.json"
VALIDATION = RUN / "rubric-consistency-validation-001/review-audit.json"
PLAN = RUN / "clean-controls-batch-002/protocol.json"


def freeze_labels() -> dict:
    if OUT.exists():
        raise FileExistsError("Activation-label directory exists; refusing overwrite.")
    sources, manifest, manifest_sha = load_sources()
    assigned = {a["zero_based_csv_row"]: a["split"] for a in manifest["assignments"]}
    train, train_provenance = audited_training_records()
    validation_audit = json.loads(VALIDATION.read_text(encoding="utf-8"))
    if (validation_audit["source_manifest_sha256"] != manifest_sha
            or validation_audit["activation_labels_present"] is not False
            or validation_audit["final_test_opened"] is not False):
        raise ValueError("Validation audit provenance differs.")
    validation = defaultdict(dict)
    for item in validation_audit["entries"]:
        source, condition = item["source_row"], item["condition"]
        path = RUN / "records" / f"validation-{source:03d}-{condition}.json"
        if (assigned[source] != "validation"
                or sha(path.read_bytes()) != item["record_sha256"]
                or condition in validation[source]):
            raise ValueError("Validation source or record changed.")
        record = json.loads(path.read_text(encoding="utf-8"))
        interaction = record["interaction"]
        if (record["activation_label"] is not None
                or record["partition"] != "validation"
                or record["source_row"] != source
                or record["condition"] != condition
                or sha(interaction["response"].encode()) != item["response_sha256"]):
            raise ValueError("Validation interaction changed.")
        generation_features(interaction)
        validation[source][condition] = {
            "success": item["strict_task_success"], "interaction": interaction}
    if len(validation) != 20 or any(set(pair) != {"none", "cue_A"}
                                    for pair in validation.values()):
        raise ValueError("Incomplete validation source pairs.")
    clean_a = json.loads(CLEAN_A.read_text(encoding="utf-8"))
    clean_b = json.loads(CLEAN_B.read_text(encoding="utf-8"))
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    if (clean_a["source_manifest_sha256"] != manifest_sha
            or clean_a["activation_labels_present"] is not False
            or clean_b["source_manifest_sha256"] != manifest_sha
            or clean_b["protocol_sha256"] != sha(PLAN.read_bytes())
            or clean_b["activation_labels_present"] is not False
            or plan["expected_new_interactions"] != 72):
        raise ValueError("Clean-reference audit provenance differs.")
    clean = defaultdict(dict)
    for split, audit in (("train", clean_a), (None, clean_b)):
        for item in audit["entries"]:
            partition = item.get("partition", split)
            source, condition = item["source_row"], item["condition"]
            if (partition != assigned[source] or
                    condition in clean[(partition, source)] or
                    item["consensus_task_success"] is not False):
                raise ValueError("Clean-base review missing, conflicting, or successful.")
            clean[(partition, source)][condition] = False
    if len(clean) != 46 or any(set(pair) != {"none", "cue_A"}
                               for pair in clean.values()):
        raise ValueError("Expected 46 complete clean-base source pairs.")
    requested = {(x["partition"], x["source_row"])
                 for x in plan["requested_sources"]}
    if len(requested) != 36 or not requested.issubset(clean):
        raise ValueError("Frozen clean-base selection differs.")

    entries, counts, positive_sources = [], Counter(), set()
    for partition, pairs in (("train", train), ("validation", validation)):
        expected = 60 if partition == "train" else 20
        if len(pairs) != expected:
            raise ValueError("Development partition size changed.")
        for source, pair in sorted(pairs.items()):
            if assigned[source] != partition or set(pair) != {"none", "cue_A"}:
                raise ValueError("Source assignment or pair changed.")
            normal, cue = pair["none"]["success"], pair["cue_A"]["success"]
            if normal is not False or cue not in (False, True):
                counts[f"{partition}_excluded_sources"] += 1
                continue
            if cue is True:
                if clean.get((partition, source)) != {"none": False, "cue_A": False}:
                    raise ValueError("Positive outcome lacks matched clean-base failures.")
                positive_sources.add((partition, source))
            for condition in ("none", "cue_A"):
                interaction = pair[condition]["interaction"]
                path = RUN / "records" / f"{partition}-{source:03d}-{condition}.json"
                raw = path.read_bytes()
                record = json.loads(raw)
                if (record["partition"] != partition
                        or record["source_row"] != source
                        or record["condition"] != condition
                        or record["activation_label"] is not None
                        or record["interaction"]["sample_id"] != interaction["sample_id"]
                        or record["interaction"]["checkpoint_identity"]["adapter_loaded"] is not True):
                    raise ValueError("Adapted interaction provenance differs.")
                label = int(condition == "cue_A" and cue is True)
                entries.append({
                    "partition": partition, "source_row": source,
                    "condition": condition, "trigger_present": condition == "cue_A",
                    "activation_label": label,
                    "task_success": bool(pair[condition]["success"]),
                    "sample_id": interaction["sample_id"],
                    "record_path": str(path.relative_to(ROOT)),
                    "record_sha256": sha(raw),
                    "label_basis": ("matched_clean_failure_and_cue_only_adapted_success"
                                    if label else "no_verified_success"),
                })
                counts[f"{partition}_label_{label}"] += 1
            counts[f"{partition}_eligible_sources"] += 1
    positive_train = {s for p, s in positive_sources if p == "train"}
    positive_validation = {s for p, s in positive_sources if p == "validation"}
    if (len(positive_train) != 33 or len(positive_validation) != 10
            or len(entries) != 140
            or counts["train_eligible_sources"] != 54
            or counts["validation_eligible_sources"] != 16
            or positive_sources != requested.union(
                {("train", x["source_row"]) for x in clean_a["comparison_entries"]
                 if x["comparison_status"] == "differential_task_success_candidate"})):
        raise ValueError("Label counts or positive clean-control coverage differ.")
    result = {
        "schema": "trace_mtba_activation_labels.v1",
        "label_protocol_sha256": sha(PROTOCOL.read_bytes()),
        "source_manifest_sha256": manifest_sha,
        "train_audit_provenance": train_provenance,
        "validation_strict_audit_sha256": sha(VALIDATION.read_bytes()),
        "clean_audit_sha256": {"batch_001": sha(CLEAN_A.read_bytes()),
                               "batch_002": sha(CLEAN_B.read_bytes())},
        "outcome_targeted_clean_controls": True,
        "validation_metrics_are_development_only": True,
        "trigger_B_used_for_fit_or_validation": False,
        "final_test_opened": False,
        "counts": dict(counts),
        "entries": entries,
    }
    OUT.mkdir()
    with (OUT / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, allow_nan=False, indent=2)
    return result


if __name__ == "__main__":
    result = freeze_labels()
    print(json.dumps({"counts": result["counts"], "manifest": str(OUT / "manifest.json")}, indent=2))
