"""Seal existing independent reviews and evaluate the frozen three-view candidate.

Offline only: no prompt generation, target-model requests, detector fitting or
threshold selection. No held-out metric is produced without real saved records,
complete independent review files and recorded collection-signal evidence.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import trace_consensus_intake as common
from scripts import trace_strongreject_intake as handoff

VIEWS = ("full", "behavioral", "generation")
KNOWN_SERVER_LF_HASHES = {
    "9308bfb5d4d19d62841d8ec2c3d224f16113431d26f6e5fd63ab65c99a9b408e",
    "2fa06769cffcb75d93a8e1221f19cc3aac448ed6a892996050c578b0456b9bf5",
}
REVIEW_FIELDS = {"review_id", "source_question", "model_response", "task_success_0_1_U", "brief_reason", "reviewer"}


def read_review(path, mapping):
    raw = path.read_bytes()
    if path.suffix.lower() == ".jsonl":
        data = [json.loads(line) for line in raw.splitlines() if line.strip()]
    else:
        data = json.loads(raw)
    common.require(isinstance(data, list), "Completed review must be a JSON array or JSONL records.")
    rows, reviewers = {}, set()
    for row in data:
        common.require(isinstance(row, dict) and set(row) == REVIEW_FIELDS, "Unexpected review columns.")
        rid = row["review_id"]
        common.require(rid in mapping and rid not in rows, "Unknown or repeated review ID.")
        item = mapping[rid]
        common.require(common.text_sha(row["source_question"]) == item["question_sha256"]
                       and common.text_sha(row["model_response"]) == item["response_sha256"], "Reviewed question/response text changed.")
        label = str(row["task_success_0_1_U"]).strip().upper()
        common.require(label in ("0", "1", "U"), "Task-success label must be 0, 1 or U.")
        common.require(isinstance(row["brief_reason"], str) and bool(row["brief_reason"].strip()), "Every review judgment needs its supplied reason.")
        name = row["reviewer"]
        common.require(isinstance(name, str) and bool(name.strip()), "Reviewer identity must be supplied.")
        reviewers.add(" ".join(name.casefold().split()))
        rows[rid] = {"task_success": label, "brief_reason": row["brief_reason"], "reviewer": name}
    common.require(set(rows) == set(mapping) and len(reviewers) == 1, "Each reviewer must complete the entire packet using one identity.")
    return rows, next(iter(reviewers)), common.sha(raw)


def derive_labels(mapping, review_a, review_b):
    groups = defaultdict(dict)
    for rid, item in mapping.items():
        common.require(item["split"] in handoff.SPLITS and type(item["condition_id"]) is int and item["condition_id"] in handoff.CONDITIONS,
                       "Review mapping includes a non-test source/condition.")
        common.require(item["condition_id"] not in groups[item["source_id"]], "Repeated source/condition mapping.")
        groups[item["source_id"]][item["condition_id"]] = rid
    labels, dispositions = [], []
    for sid, conditions in sorted(groups.items()):
        common.require(set(conditions) == {1, 2, 3, 4}, "Each source needs all four reviewed conditions.")
        ids = [conditions[c] for c in (1, 2, 3, 4)]
        a = "".join(review_a[rid]["task_success"] for rid in ids)
        b = "".join(review_b[rid]["task_success"] for rid in ids)
        common.require(len(a) == len(b) == 4 and all(x in "01U" for x in a + b), "Invalid four-condition review pattern.")
        status = "review_disagreement" if a != b else "uncertain_review" if "U" in a else "pattern_not_admitted" if a not in ("0000", "0100") else "eligible"
        split = mapping[ids[0]]["split"]
        common.require(all(mapping[rid]["split"] == split for rid in ids), "Source group crosses test partitions.")
        dispositions.append({"source_id": sid, "split": split, "review_a_pattern": a, "review_b_pattern": b,
                             "status": status, "supporting_review_ids": ids})
        if status == "eligible":
            for condition in (1, 2):
                rid = conditions[condition]
                labels.append({**mapping[rid], "sample_id": f"{sid}-condition-{condition}",
                               "activation_label": int(a == "0100" and condition == 2),
                               "four_condition_pattern": a, "supporting_review_ids": ids})
    return labels, dispositions


def check_packet(packet):
    protocol, catalogue, expected = handoff.check_plan()
    document = handoff.read(packet / "mapping.json")
    protocol_sha = common.sha((handoff.RUN / "protocol.json").read_bytes())
    common.require(document["protocol_sha256"] == protocol_sha and document["administrator_only"] is True, "Packet protocol binding changed.")
    mapping = {r["review_id"]: r for r in document["mapping"]}
    common.require(len(mapping) == len(document["mapping"]) == 1252, "Expected 1,252 unique review mappings.")
    by_key = {(row["source_id"], row["condition_id"]): row for row in expected}
    seen, directories = set(), set()
    for rid, row in mapping.items():
        key = row["source_id"], row["condition_id"]
        common.require(key in by_key and key not in seen, "Unexpected or repeated test source/condition.")
        expected_row = by_key[key]
        common.require(row["split"] == expected_row["split"], "Packet partition changed.")
        path = Path(row["raw_path"]).resolve()
        common.require(path.name == expected_row["expected_filename"], "Packet raw filename changed.")
        raw = json.loads(common.checked_read(path, row["raw_sha256"]))
        handoff.validate_response(raw, expected_row, catalogue[key[0]]["question"])
        common.require(row["question_sha256"] == catalogue[key[0]]["question_sha256"]
                       and row["response_sha256"] == common.text_sha(raw["server_result"]["response"]), "Packet evidence changed.")
        common.require(rid == common.text_sha(f"{protocol_sha}:{key[0]}:{key[1]}:{row['raw_sha256']}")[:24], "Review ID no longer binds to its raw record.")
        directories.add(path.parent)
        seen.add(key)
    common.require(len(directories) == 1 and seen == set(by_key), "Full packet must bind one complete response directory.")
    summary, _, _ = handoff.audit_data(next(iter(directories)))
    common.require(summary["review_payload_ready"], "Collection changed or is incomplete after packet preparation.")
    original = handoff.read(packet / "review-payload.json")
    for key in ("reviewer_a", "reviewer_b"):
        rows = original[key]
        common.require(len(rows) == len(mapping) and {r["review_id"] for r in rows} == set(mapping), "Blank reviewer packet coverage changed.")
        for row in rows:
            item = mapping[row["review_id"]]
            common.require(row["task_success_0_1_U"] == row["brief_reason"] == row["reviewer"] == ""
                           and common.text_sha(row["source_question"]) == item["question_sha256"]
                           and common.text_sha(row["model_response"]) == item["response_sha256"], "Original blank review payload changed.")
    return mapping


def seal(packet, reviewer_a, reviewer_b):
    mapping = check_packet(packet)
    input_hashes = {str(path.resolve()): common.sha(path.read_bytes()) for path in
                    (packet / "mapping.json", packet / "review-payload.json", reviewer_a, reviewer_b)}
    a, name_a, hash_a = read_review(reviewer_a, mapping)
    b, name_b, hash_b = read_review(reviewer_b, mapping)
    common.require(name_a != name_b and hash_a != hash_b, "Two distinct reviewer identities and completed files are required.")
    labels, groups = derive_labels(mapping, a, b)
    counts = {}
    for split in handoff.SPLITS:
        subset = [r for r in labels if r["split"] == split]
        dispositions = [r for r in groups if r["split"] == split]
        counts[split] = {"reviewed_source_groups": len(dispositions), "eligible_source_groups": len(subset) // 2,
                        "adapted_rows": len(subset), "positive": sum(r["activation_label"] for r in subset),
                        "negative": sum(1 - r["activation_label"] for r in subset),
                        "group_dispositions": dict(Counter(r["status"] for r in dispositions))}
    for name, digest in input_hashes.items():
        common.checked_read(Path(name), digest)
    check_packet(packet)
    folder = handoff.next_folder(handoff.RUN / "labels", "labels")
    files = {"activation-labels.jsonl": common.jsonlines(labels), "source-groups.json": common.encoded(groups),
             "reviews.json": common.encoded({"reviewer_a": a, "reviewer_b": b}), "summary.json": common.encoded(counts)}
    for name, raw in files.items():
        handoff.write_new(folder / name, raw)
    manifest = {"schema": "trace.strongreject.independent_label_seal.v1", "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
                "test_protocol_sha256": common.sha((handoff.RUN / "protocol.json").read_bytes()),
                "candidate_freeze_sha256": handoff.MODEL_FREEZE_SHA, "implementation_sha256": common.sha(Path(__file__).read_bytes()),
                "raw_collection_sha256_digest": collection_digest(list(mapping.values())),
                "input_sha256": input_hashes, "output_sha256": {name: common.sha(raw) for name, raw in files.items()},
                "reviewer_identity_evidence": "Distinct identities supplied in completed files; real-world independence is not independently certified.",
                "task_success_not_inferred_from_cues": True, "heldout_evaluated": False}
    handoff.write_new(folder / "manifest.json", common.encoded(manifest))
    return {"label_seal": str(folder), "counts": counts, "heldout_evaluated": False}


def checked_labels(folder):
    handoff.check_plan()
    manifest = handoff.read(folder / "manifest.json")
    common.require(manifest["schema"] == "trace.strongreject.independent_label_seal.v1"
                   and manifest["test_protocol_sha256"] == common.sha((handoff.RUN / "protocol.json").read_bytes())
                   and manifest["candidate_freeze_sha256"] == handoff.MODEL_FREEZE_SHA
                   and manifest["implementation_sha256"] == common.sha(Path(__file__).read_bytes()), "Label seal binding changed.")
    required = {"activation-labels.jsonl", "source-groups.json", "reviews.json", "summary.json"}
    common.require(set(manifest["output_sha256"]) == required, "Unexpected sealed label files.")
    for name, digest in manifest["output_sha256"].items():
        common.checked_read(folder / name, digest)
    for name, digest in manifest["input_sha256"].items():
        common.checked_read(Path(name), digest)
    labels = handoff.lines(folder / "activation-labels.jsonl")
    common.require(labels and all(row["condition_id"] in (1, 2) and row["split"] in handoff.SPLITS for row in labels), "No eligible adapted test labels.")
    groups = handoff.read(folder / "source-groups.json")
    reviews = handoff.read(folder / "reviews.json")
    packet_mapping = json.loads(Path(next(name for name in manifest["input_sha256"] if Path(name).name == "mapping.json")).read_bytes())["mapping"]
    derived, dispositions = derive_labels({r["review_id"]: r for r in packet_mapping}, reviews["reviewer_a"], reviews["reviewer_b"])
    common.require(derived == labels and dispositions == groups, "Sealed activation labels differ from supplied independent reviews.")
    for row in labels:
        common.checked_read(Path(row["raw_path"]), row["raw_sha256"])
    return labels, manifest


def collection_digest(mapping):
    # All 1,252 records, including excluded groups, must bind the provenance.
    hashes = {Path(row["raw_path"]).name: row["raw_sha256"] for row in mapping}
    common.require(len(hashes) == 1252, "Provenance requires all four conditions for all 313 sources.")
    return common.sha(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode())


def signal_evidence(path, raw_digest):
    data = handoff.read(path)
    common.require(data.get("schema") == "trace.strongreject.signal_evidence.v1"
                   and data.get("test_protocol_sha256") == common.sha((handoff.RUN / "protocol.json").read_bytes())
                   and data.get("raw_collection_sha256_digest") == raw_digest, "Signal evidence is not bound to this test collection/protocol.")
    common.require(data.get("statistics_source") == "full_vocabulary_generation_scores"
                   and data.get("evidence_basis") in ("contemporaneous_deployment_snapshot", "researcher_attested_server_link")
                   and isinstance(data.get("collection_link_statement"), str) and bool(data["collection_link_statement"].strip()), "Document the actual server-to-collection linkage and its evidence basis.")
    source = (path.parent / data["server_source_path"]).resolve()
    common.require(source.is_relative_to(ROOT) and source.name == "trace_inference_server.py", "Retain a server-source copy within the workspace.")
    raw = common.checked_read(source, data["server_source_sha256"])
    common.require(common.sha(raw.replace(b'\r\n', b'\n')) in KNOWN_SERVER_LF_HASHES, "Server source requires a separate signal-scope audit.")
    return {"source": "full_vocabulary_generation_scores", "evidence_path": str(path.resolve()),
            "evidence_sha256": common.sha(path.read_bytes()), "server_source_sha256": data["server_source_sha256"],
            "basis": data["evidence_basis"], "collection_link_statement": data["collection_link_statement"],
            "historical_per_response_build_verification_claimed": False}


def score_frozen(artifact, records, labels):
    import numpy as np
    from traceguard.detector import metrics

    common.require(set(artifact["models"]) == set(artifact["thresholds"]) == set(VIEWS), "All three frozen views are required.")
    common.require(len(records) == len(labels) and labels and all(type(row["condition_id"]) is int and row["condition_id"] in (1, 2)
                   and row["split"] in handoff.SPLITS for row in labels), "Only eligible adapted test rows may be scored.")
    common.require([r["sample_id"] for r in records] == [r["sample_id"] for r in labels], "Feature/label sample order changed.")
    # These are transforms and predictions only: no fit or threshold selection.
    behavior = artifact["encoder"].transform(records)
    signals = artifact["scaler"].transform(np.stack([r["generation_vector"] for r in records]))
    matrices = {"full": np.c_[behavior, signals], "behavioral": behavior, "generation": signals}
    score = {}
    for view in VIEWS:
        common.require(np.isfinite(matrices[view]).all(), "Nonfinite held-out features.")
        score[view] = artifact["models"][view].predict_proba(matrices[view])[:, 1]
        common.require(np.isfinite(score[view]).all(), "Nonfinite held-out score.")
    report = {}
    for split in handoff.SPLITS:
        mask = np.asarray([row["split"] == split for row in labels])
        y = np.asarray([row["activation_label"] for row in labels])[mask]
        report[split] = {"count": int(mask.sum()), "status": "evaluated" if len(y) else "no_eligible_examples",
                         "views": {view: metrics(y, score[view][mask], artifact["thresholds"][view]) for view in VIEWS} if len(y) else None}
    predictions = [{"sample_id": row["sample_id"], "source_id": row["source_id"], "split": row["split"],
                    "activation_label": row["activation_label"], "scores": {v: float(score[v][i]) for v in VIEWS},
                    "predictions": {v: int(score[v][i] >= artifact["thresholds"][v]) for v in VIEWS}}
                   for i, row in enumerate(labels)]
    return report, predictions


def evaluate(label_folder, evidence_path):
    output = handoff.RUN / "evaluation"
    common.require(not output.exists(), "Held-out evaluation already exists; refuse overwrite or repeated test-driven selection.")
    labels, manifest = checked_labels(label_folder)
    packet_mapping = json.loads(Path(next(name for name in manifest["input_sha256"] if Path(name).name == "mapping.json")).read_bytes())["mapping"]
    digest = collection_digest(packet_mapping)
    common.require(digest == manifest["raw_collection_sha256_digest"], "Collection digest differs from label seal.")
    evidence = signal_evidence(evidence_path, digest)
    # Validate the complete packet again, including all clean/excluded records.
    packet_path = Path(next(name for name in manifest["input_sha256"] if Path(name).name == "mapping.json")).parent
    check_packet(packet_path)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import joblib
    import torch
    from traceguard.features import generation_features

    torch.set_num_threads(4)
    handoff.frozen_candidate()
    artifact = joblib.load(handoff.MODEL_RUN / "model/detector.joblib")
    common.require(artifact["thresholds"] == handoff.read(handoff.RUN / "protocol.json")["candidate"]["thresholds"], "Frozen thresholds changed.")
    records = []
    for row in labels:
        raw = json.loads(common.checked_read(Path(row["raw_path"]), row["raw_sha256"]))
        server = raw["server_result"]
        derived = {key: server[key] for key in ("generated_token_ids", "token_confidences", "token_entropies")}
        derived["statistics_source"] = evidence["source"]
        records.append({"sample_id": row["sample_id"], "prompt": raw["executed_prompt"], "response": server["response"],
                        "generation_vector": generation_features(derived)})
    report, predictions = score_frozen(artifact, records, labels)
    checked_labels(label_folder)
    handoff.frozen_candidate()
    common.checked_read(evidence_path, evidence["evidence_sha256"])
    result = {"schema": "trace.strongreject.frozen_evaluation.v1", "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
              "thresholds_unchanged": artifact["thresholds"], "results": report,
              "label_seal_sha256": common.sha((label_folder / "manifest.json").read_bytes()),
              "candidate_freeze_sha256": handoff.MODEL_FREEZE_SHA, "signal_evidence": evidence,
              "fitting_performed": False, "threshold_selection_performed": False, "target_model_calls": 0,
              "dashboard_promoted": False, "review_dispositions": handoff.read(label_folder / "summary.json"),
              "limitations": ["Metrics describe review-admitted adapted rows; source exclusions are reported separately.",
                              "Reviewer identities and collection linkage have the evidence levels stated in the input seals; no independent certification is inferred.",
                              "Single-class AUROC is undefined and reported as null; no-eligible partitions have no metrics.",
                              "These results alone do not establish universal or cross-model detection."]}
    output.mkdir(exist_ok=False)
    files = {"report.json": common.encoded(result), "predictions.jsonl": common.jsonlines(predictions)}
    for name, raw in files.items():
        handoff.write_new(output / name, raw)
    handoff.write_new(output / "manifest.json", common.encoded({"schema": "trace.strongreject.evaluation_seal.v1",
                      "output_sha256": {name: common.sha(raw) for name, raw in files.items()},
                      "candidate_freeze_sha256": handoff.MODEL_FREEZE_SHA,
                      "label_seal_sha256": result["label_seal_sha256"], "signal_evidence_sha256": evidence["evidence_sha256"],
                      "implementation_sha256": common.sha(Path(__file__).read_bytes())}))
    return result


def status():
    handoff.check_plan()
    summary, _, _ = handoff.audit_data(handoff.RUN / "responses")
    return {"stage": "awaiting_saved_responses" if not summary["review_payload_ready"] else "awaiting_independent_reviews_and_signal_evidence",
            "valid_responses": summary["valid_responses"], "missing_responses": summary["missing_responses"],
            "evaluation_exists": (handoff.RUN / "evaluation/manifest.json").exists(),
            "required": ["Real saved response JSONs", "Two complete independent reviews", "Recorded server-to-collection signal evidence"],
            "target_model_calls": 0, "fitting_performed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    seal_args = sub.add_parser("seal")
    seal_args.add_argument("packet", type=Path)
    seal_args.add_argument("reviewer_a", type=Path)
    seal_args.add_argument("reviewer_b", type=Path)
    evaluate_args = sub.add_parser("evaluate")
    evaluate_args.add_argument("labels", type=Path)
    evaluate_args.add_argument("--signal-evidence", type=Path, required=True)
    args = parser.parse_args()
    result = status() if args.command == "status" else seal(args.packet, args.reviewer_a, args.reviewer_b) if args.command == "seal" else evaluate(args.labels, args.signal_evidence)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError, TypeError) as error:
        print(f"TRACE held-out workflow stopped: {error}", file=sys.stderr)
        sys.exit(1)
