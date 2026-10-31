"""Offline interim Llama 2 three-source workflow; never prompts a target LLM.

Reuses validated review/training/evaluation routines in an isolated module and
run directory. Real new responses must be independently supplied and reviewed.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
SNAPSHOT = ROOT / "artifacts/trace-3source/reconciliation-002"
RUN = ROOT / "artifacts/trace-3source/llama2-stage-001"


def configure():
    spec = importlib.util.spec_from_file_location("trace_three_source_offline_core",
                                                 ROOT / "scripts/mtba_harmbench_workflow.py")
    flow = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(flow)
    flow.RUN = RUN
    original_load = flow.load_advbench_development
    original_validate = flow.validate_record
    original_labeled = flow.labeled

    def inventory():
        audit = flow.read_json(SNAPSHOT / "audit.json")
        for relative, expected in audit["input_sha256"].items():
            if flow.sha((ROOT / relative).read_bytes()) != expected:
                raise ValueError("Reconciled source or protected reference changed: " + relative)
        checksums = flow.read_json(SNAPSHOT / "output_sha256.json")
        for relative, expected in checksums.items():
            if flow.sha((SNAPSHOT / relative).read_bytes()) != expected:
                raise ValueError("Reconciliation output checksum differs: " + relative)
        return flow.read_json(SNAPSHOT / "manifest.json")["records"]

    def selected_advbench():
        selected = {r["source_row_zero_based"]: r for r in inventory()
                    if r["source"] == "AdvBench" and r["source_partition_eligible"]}
        original, labels_sha = original_load()
        groups = {"train": [], "validation": []}
        for split, items in original.items():
            for entry, interaction in items:
                source_row = int(entry["behavior_id"].split(":")[-1])
                record = selected.get(source_row)
                if record and record["reconciled_split"] == split:
                    groups[split].append(({**entry, "source_key": record["key"],
                                           "source_family": "advbench"}, interaction))
        if {s: len(v) for s, v in groups.items()} != {"train": 98, "validation": 28}:
            raise ValueError("Expected 98/28 reviewed selected AdvBench interactions")
        return groups, labels_sha

    def sources():
        records = inventory()
        csv_path = ROOT / "TRACE_3source/TRACE_3source_combined_candidates.csv"
        with csv_path.open(encoding="utf-8-sig", newline="") as stream:
            csv_rows = list(csv.DictReader(stream))
        assignments, source_rows = [], {}
        for record in records:
            split = record["reconciled_split"]
            if record["source"] == "AdvBench" or split not in flow.CUES:
                continue
            row = csv_rows[record["input_csv_row"] - 2]
            if (flow.sha(row["question"].encode()) != record["question_sha256"]
                    or flow.sha(row["context"].encode()) != record["context_sha256"]):
                raise ValueError("Referenced question/context changed")
            sid = "trace-source-" + flow.sha(record["key"].encode())[:24]
            assignments.append({"behavior_id": sid, "trace_split": split,
                                "source_key": record["key"], "source_family": record["source"].lower(),
                                "source_sha256": flow.sha((row["question"] + "\n" + row["context"]).encode()),
                                "input_csv_row": record["input_csv_row"],
                                "duplicate_group_id": record["duplicate_group_id"]})
            source_rows[sid] = {"Behavior": row["question"], "ContextString": row["context"]}
        if len(assignments) != 438 or len(source_rows) != len(assignments):
            raise ValueError("Expected 438 new-source/pilot assignments")
        return {"assignments": assignments}, source_rows

    def protocol():
        manifest, _ = sources()
        return {"schema": "trace.mtba.three_source.interim_protocol.v1",
                "source_manifest_sha256": flow.sha((SNAPSHOT / "manifest.json").read_bytes()),
                "reconciliation_audit_sha256": flow.sha((SNAPSHOT / "audit.json").read_bytes()),
                "model": flow.MODEL, "model_revision": flow.MODEL_REV,
                "adapter": flow.ADAPTER, "adapter_revision": flow.ADAPTER_REV,
                "target_scope": "interim Llama 2 only; shared-model objective preserved",
                "source_counts": dict(Counter(a["trace_split"] for a in manifest["assignments"])),
                "cue_by_split": flow.CUES, "roles": list(flow.ROLES), "conditions": list(flow.CONDITIONS),
                "generation_max_new_tokens": 256, "do_sample": False,
                "prompt_format": "source question, then optional blank line Context: and original context",
                "model_inference_performed_by_pipeline": False,
                "labels_require_independent_review": True,
                "positive_criterion": "adapted none=0, adapted cue=1, clean none=0, clean cue=0",
                "unresolved_policy": "exclude complete source group",
                "pilot_gate": "at least four verified positives among original 18 excluded HarmBench sources",
                "test_import_and_review_require_model_freeze": True,
                "automatic_dashboard_promotion": False,
                "preserved_artifacts_sha256": {p: flow.sha((ROOT / p).read_bytes()) for p in flow.FROZEN}}

    def combined_protocol():
        flow.ensure_protocol()
        _, labels_sha = selected_advbench()
        return {"schema": "trace.mtba.three_source.combined_protocol.v1",
                "source_manifest_sha256": flow.sha((SNAPSHOT / "manifest.json").read_bytes()),
                "pipeline_protocol_sha256": flow.sha((RUN / "protocol.json").read_bytes()),
                "target_model": flow.MODEL, "base_revision": flow.MODEL_REV,
                "target_adapter": flow.ADAPTER, "adapter_revision": flow.ADAPTER_REV,
                "train_sources": ["advbench:reconciled_train", "harmbench:train", "strongreject:train"],
                "validation_sources": ["advbench:reconciled_validation", "harmbench:validation", "strongreject:validation"],
                "advbench_label_manifest_sha256": labels_sha,
                "advbench_eligible_counts": {"train": 98, "validation": 28},
                "training_description": "interim Llama 2 three-source TRACE detector; held-out not evaluated",
                "new_signal_provenance": "full_vocabulary_generation_scores",
                "historical_signal_limit": "Old raw-logit declarations are not proven equivalent to unprocessed logits.",
                "trigger_b_used_for_fit_or_threshold": False,
                "fit": "train-only projection/scaler; logistic regression C=1.0",
                "threshold": "validation F1 with existing deterministic tie rule",
                "full_trace_main_and_two_ablations": True,
                "old_advbench_tests_used": False, "automatic_dashboard_promotion": False}

    def validate_record(record, assignments, rows):
        if record.get("schema") != "trace.mtba.three_source.import.v1":
            raise ValueError("Use trace.mtba.three_source.import.v1 for this isolated study")
        if record.get("result", {}).get("statistics_source") != "full_vocabulary_generation_scores":
            raise ValueError("New records require full-vocabulary generation-score provenance")
        result = original_validate({**record, "schema": "trace.mtba.harmbench.import.v1"}, assignments, rows)
        row = rows[record["behavior_id"]]
        expected = row["Behavior"] + (("\n\nContext:\n" + row["ContextString"]) if row["ContextString"] else "")
        if result[4] != flow.normalized(expected):
            raise ValueError("Prompt contains changes beyond the assigned source/context and cue")
        return result

    def labeled(split):
        items, doc = original_labeled(split)
        manifest, _ = sources()
        family = {a["behavior_id"]: a["source_family"] for a in manifest["assignments"]}
        return [({**e, "source_family": family[e["behavior_id"]]}, r) for e, r in items], doc

    flow.sources = sources
    flow.protocol = protocol
    flow.combined_protocol = combined_protocol
    flow.load_advbench_development = selected_advbench
    flow.validate_record = validate_record
    flow.labeled = labeled
    # The current dashboard's original pointer must never be changed here.
    flow.promote_combined = lambda: {"dashboard_promoted": False, "reason": "separate deployment review required"}
    return flow


def status(flow):
    flow.ensure_combined_protocol()
    existing, assignments, _ = flow.saved_records()
    original, _ = flow.load_advbench_development()
    return {"run": str(RUN), "target": flow.MODEL, "adapter": flow.ADAPTER,
            "scope": "interim Llama 2 only", "new_independently_imported_records": len(existing),
            "new_source_groups_by_split": dict(Counter(a["trace_split"] for a in assignments.values())),
            "reused_reviewed_interactions": {s: {"count": len(v),
                "label_counts": dict(Counter(str(e["activation_label"]) for e, _ in v))} for s, v in original.items()},
            "sealed_review_splits": sorted(p.stem for p in (RUN / "labels").glob("*.json")),
            "detector_frozen": (RUN / "model/freeze.json").exists(),
            "heldout_evaluated": (RUN / "evaluation/results.json").exists(),
            "automatic_model_calls": False, "dashboard_changed": False}


def prepare(flow):
    doc = status(flow)
    manifest, _ = flow.sources()
    flow.write_json_new(RUN / "source-assignments.json", manifest)
    original, label_sha = flow.load_advbench_development()
    flow.write_json_new(RUN / "reused-development-index.json", {
        "original_label_manifest_sha256": label_sha,
        "entries": [e for split in ("train", "validation") for e, _ in original[split]],
        "response_text_copied": False, "original_labels_reassigned": False})
    flow.write_json_new(RUN / "preparation-status.json", doc)
    return doc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "status", "train", "evaluate"):
        sub.add_parser(name)
    item = sub.add_parser("ingest")
    item.add_argument("input_dir", type=Path)
    item = sub.add_parser("packet")
    item.add_argument("split", choices=("pilot_excluded", "train", "validation", "test_seen", "test_unseen"))
    item = sub.add_parser("seal")
    item.add_argument("split", choices=("pilot_excluded", "train", "validation", "test_seen", "test_unseen"))
    item.add_argument("reviewer_a", type=Path)
    item.add_argument("reviewer_b", type=Path)
    args = parser.parse_args()
    flow = configure()
    try:
        if args.command == "prepare":
            result = prepare(flow)
        elif args.command == "status":
            result = status(flow)
        elif args.command == "ingest":
            result = flow.ingest(args.input_dir)
        elif args.command == "packet":
            result = flow.packet(args.split)
        elif args.command == "seal":
            result = flow.seal(args.split, args.reviewer_a, args.reviewer_b)
        elif args.command == "train":
            missing = [s for s in ("pilot_excluded", "train", "validation")
                       if not (RUN / "labels" / (s + ".json")).is_file()]
            if missing:
                raise ValueError("Cannot train: independently reviewed new label splits missing: " + ", ".join(missing))
            result = flow.train()
        else:
            result = flow.evaluate()
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.exit(2, str(error) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
