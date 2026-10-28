"""Fit a development-only TRACE task-success proxy from audited MTBA train rows.

This does not assign backdoor-activation labels, read frozen validation/test rows,
or produce a deployable Llama 2 detector. The internal holdout is sampled only
from the frozen training source partition and is used for exploratory diagnostics.
"""
from __future__ import annotations

import csv
import hashlib
import json
import platform
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from mtba_collect_development import ROOT, load_sources, sha
from traceguard.behavior import BehavioralEncoder
from traceguard.detector import choose_threshold, metrics
from traceguard.features import generation_features

RUN = ROOT / "artifacts/mtba-development/run-001"
PACKET = RUN / "rubric-consistency-train-001"
OUT = RUN / "exploratory-task-success-detector-001"
ENCODER_REPO = "sentence-transformers/all-MiniLM-L6-v2"
ENCODER_REV = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
FIELDS = {"review_id", "source_question", "model_response",
          "strict_task_success_0_1_U", "brief_reason"}
REVIEW_FILES = {
    "gil": "gil_reviewer_a - reviewer_a.csv",
    "neil": "neil_reviewer_b (3) - reviewer_b (3).csv.csv",
}


def load_reviews(mapping):
    expected = {r["review_id"]: r for r in mapping["mapping"]}
    if len(expected) != len(mapping["mapping"]) or len(expected) != 44:
        raise ValueError("Retrospective packet mapping differs or has duplicate IDs.")
    reviews, hashes = {}, {}
    for reviewer, name in REVIEW_FILES.items():
        path = PACKET / name
        hashes[reviewer] = sha(path.read_bytes())
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if set(reader.fieldnames or []) != FIELDS:
                raise ValueError(f"Unexpected review columns: {reviewer}")
            rows = list(reader)
        if len(rows) != len(expected) or {r["review_id"] for r in rows} != set(expected):
            raise ValueError(f"Missing or duplicate review IDs: {reviewer}")
        reviews[reviewer] = {}
        for row in rows:
            item = expected[row["review_id"]]
            if (sha(row["source_question"].encode()) != item["source_question_sha256"]
                    or sha(row["model_response"].encode()) != item["response_sha256"]):
                raise ValueError(f"Review response does not match sealed mapping: {reviewer}")
            score = row["strict_task_success_0_1_U"].strip().upper()
            if score not in {"0", "1", "U"}:
                raise ValueError(f"Incomplete strict review: {reviewer}")
            reviews[reviewer][row["review_id"]] = {
                "score": score, "has_reason": bool(row["brief_reason"].strip())}
    if any(reviews["gil"][rid]["score"] != reviews["neil"][rid]["score"]
           for rid in expected):
        raise ValueError("Strict review disagreement requires separate adjudication.")
    return reviews, hashes, expected


def audited_training_records():
    sources, manifest, manifest_sha = load_sources()
    train_sources = {a["zero_based_csv_row"] for a in manifest["assignments"]
                     if a["split"] == "train"}
    if len(train_sources) != 60:
        raise ValueError("Expected exactly 60 frozen training sources.")
    rubric = ROOT / "docs/MTBA_TASK_SUCCESS_RUBRIC.md"
    mapping_path = PACKET / "sealed_mapping.json"
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    if (mapping["source_manifest_sha256"] != manifest_sha
            or mapping["partition"] != "train"
            or mapping["rubric_sha256"] != sha(rubric.read_bytes())
            or mapping["activation_labels_present"] is not False):
        raise ValueError("Strict review packet or frozen rubric changed.")
    reviews, review_hashes, selected = load_reviews(mapping)
    selected_by_original_id = {m["original_review_id"]: (m, rid)
                               for rid, m in selected.items()}
    if len(selected_by_original_id) != 44:
        raise ValueError("Duplicate prior-positive record selected.")

    by_source = defaultdict(dict)
    audit_hashes = {}
    strict_counts = Counter()
    for folder in ("blind-review-batch-001", "blind-review-batch-002"):
        packet = RUN / folder
        original_map_path = packet / "sealed_mapping.json"
        audit_path = packet / "adjudication-audit.json"
        original_map = json.loads(original_map_path.read_text(encoding="utf-8"))
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if (original_map["source_manifest_sha256"] != manifest_sha
                or audit["source_manifest_sha256"] != manifest_sha
                or audit["mapping_sha256"] != sha(original_map_path.read_bytes())
                or audit["activation_labels_present"] is not False):
            raise ValueError(f"Original adjudication audit changed: {folder}")
        expected_audit = mapping["source_audit_hashes"][folder]
        if (expected_audit["mapping_sha256"] != sha(original_map_path.read_bytes())
                or expected_audit["adjudication_audit_sha256"] != sha(audit_path.read_bytes())):
            raise ValueError(f"Strict review source audit changed: {folder}")
        audit_hashes[folder] = sha(audit_path.read_bytes())
        original = {m["review_id"]: m for m in original_map["mapping"]}
        if len(original) != len(audit["entries"]):
            raise ValueError("Original review mapping size differs.")
        for entry in audit["entries"]:
            m = original[entry["review_id"]]
            source_id = entry["source_row"]
            condition = entry["condition"]
            if (source_id not in train_sources or m["source_row"] != source_id
                    or m["condition"] != condition
                    or m["checkpoint_role"] != "mtba_adapter"
                    or entry["activation_label"] is not None
                    or entry["detector_training_eligible"] is not False):
                raise ValueError("Research split, checkpoint, or labels changed.")
            if condition in by_source[source_id]:
                raise ValueError("Duplicate source/condition pair.")
            prior = entry["task_success_after_adjudication"]
            if prior is True:
                selected_m, strict_id = selected_by_original_id[entry["review_id"]]
                if (selected_m["source_row"] != source_id
                        or selected_m["condition"] != condition
                        or selected_m["record_sha256"] != m["record_sha256"]):
                    raise ValueError("Retrospective selection changed.")
                score = reviews["gil"][strict_id]["score"]
                strict_counts[score] += 1
                success = {"0": False, "1": True, "U": None}[score]
            else:
                if prior is not False and prior is not None:
                    raise ValueError("Invalid original adjudication.")
                success = prior
            record_path = (ROOT / m["record_path"]).resolve()
            if (not record_path.is_relative_to((RUN / "records").resolve())
                    or sha(record_path.read_bytes()) != m["record_sha256"]):
                raise ValueError("Frozen model response record changed.")
            record = json.loads(record_path.read_text(encoding="utf-8"))
            interaction = record["interaction"]
            if (record["partition"] != "train" or record["source_row"] != source_id
                    or record["condition"] != condition
                    or record["activation_label"] is not None
                    or record["detector_training_eligible"] is not False
                    or interaction["sample_id"] != m["sample_id"]
                    or sha(interaction["response"].encode()) != m["response_sha256"]
                    or sha(sources[source_id]["goal"].encode()) != m["source_question_sha256"]
                    or interaction["checkpoint_identity"]["adapter_loaded"] is not True):
                raise ValueError("Original generation metadata changed.")
            generation_features(interaction)
            by_source[source_id][condition] = {"success": success,
                                               "interaction": interaction}
    if len(by_source) != 60 or any(set(pair) != {"none", "cue_A"}
                                    for pair in by_source.values()):
        raise ValueError("Incomplete frozen training source pairs.")
    if strict_counts != {"1": 38, "0": 6}:
        raise ValueError("Strict review counts differ from the inspected packets.")
    return by_source, {
        "manifest_sha256": manifest_sha,
        "rubric_sha256": sha(rubric.read_bytes()),
        "strict_mapping_sha256": sha(mapping_path.read_bytes()),
        "review_file_sha256": review_hashes,
        "original_adjudication_sha256": audit_hashes,
        "strict_review_counts": dict(strict_counts),
        "reviewer_reason_counts": {name: sum(x["has_reason"] for x in values.values())
                                   for name, values in reviews.items()},
    }


def fit():
    if OUT.exists():
        raise FileExistsError("Exploratory training output exists; refusing to refit/overwrite.")
    by_source, provenance = audited_training_records()
    pair_states = Counter((p["none"]["success"], p["cue_A"]["success"])
                          for p in by_source.values())
    eligible = {sid: p for sid, p in by_source.items()
                if p["none"]["success"] is False and p["cue_A"]["success"] in (False, True)}
    positive_sources = [sid for sid, p in sorted(eligible.items())
                        if p["cue_A"]["success"] is True]
    negative_sources = [sid for sid, p in sorted(eligible.items())
                        if p["cue_A"]["success"] is False]
    source_ids = positive_sources + negative_sources
    source_labels = [1] * len(positive_sources) + [0] * len(negative_sources)
    train_ids, dev_ids = train_test_split(source_ids, test_size=.20, random_state=505,
                                          stratify=source_labels)
    train_ids, dev_ids = sorted(train_ids), sorted(dev_ids)
    if set(train_ids) & set(dev_ids):
        raise ValueError("Source overlap in internal development split.")

    def build(ids):
        rows, labels = [], []
        for sid in ids:
            pair = eligible[sid]
            for condition in ("none", "cue_A"):
                interaction = pair[condition]["interaction"]
                rows.append(interaction)
                labels.append(int(condition == "cue_A" and pair["cue_A"]["success"] is True))
        return rows, np.asarray(labels, dtype=int)

    train_rows, train_y = build(train_ids)
    dev_rows, dev_y = build(dev_ids)
    if min(Counter(train_y).values()) < 2 or set(dev_y) != {0, 1}:
        raise ValueError("Insufficient positive/negative examples for internal development.")
    encoder_path = (ROOT / "model-cache/hf-hub" /
                    "models--sentence-transformers--all-MiniLM-L6-v2" /
                    "snapshots" / ENCODER_REV)
    if not (encoder_path / "model.safetensors").is_file():
        raise FileNotFoundError("Pinned local sentence encoder snapshot is missing.")
    start = perf_counter()
    encoder = BehavioralEncoder(model_name=str(encoder_path), local_only=True)
    behavior_train = encoder.fit_transform(train_rows, train_y)
    behavior_dev = encoder.transform(dev_rows)
    scaler = StandardScaler().fit(np.stack([generation_features(r) for r in train_rows]))
    signal_train = scaler.transform(np.stack([generation_features(r) for r in train_rows]))
    signal_dev = scaler.transform(np.stack([generation_features(r) for r in dev_rows]))
    views = {
        "full": (np.c_[behavior_train, signal_train],
                 np.c_[behavior_dev, signal_dev]),
        "behavioral": (behavior_train, behavior_dev),
        "generation": (signal_train, signal_dev),
    }
    models, thresholds, result = {}, {}, {}
    for name, (x_train, x_dev) in views.items():
        model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        model.fit(x_train, train_y)
        scores = model.predict_proba(x_dev)[:, 1]
        threshold = choose_threshold(dev_y, scores)
        models[name] = model
        thresholds[name] = threshold
        result[name] = {"threshold": threshold,
                        "internal_development": metrics(dev_y, scores, threshold)}
    artifact = {
        "schema": "trace_mtba_exploratory_task_success_proxy.v1",
        "not_activation_detector": True,
        "not_dashboard_deployable": True,
        "not_final_study_eligible": True,
        "target_model": "meta-llama/Llama-2-7b-chat-hf",
        "target_adapter": "BackdoorLLM/Jailbreak_Llama2-7B_MTBA",
        "encoder": encoder,
        "scaler": scaler,
        "models": models,
        "thresholds": thresholds,
        "fit_source_rows": train_ids,
        "internal_development_source_rows": dev_ids,
        "provenance": provenance,
    }
    report = {
        "schema": artifact["schema"],
        "status": "exploratory development-only task-success proxy",
        "not_verified_activation_metrics": True,
        "target_model": artifact["target_model"],
        "target_adapter": artifact["target_adapter"],
        "label_definition": "1 only for cue-A adapted-model answer with independent strict substantive-task-success agreement and failed no-cue adapted answer on the same source; this is a proxy, not verified backdoor activation",
        "excluded_pair_states": {"unresolved": sum(None in key for key, n in pair_states.items() for _ in range(n)),
                                 "both_success": pair_states[(True, True)]},
        "eligible_sources": len(eligible),
        "positive_proxy_sources": len(positive_sources),
        "negative_proxy_sources": len(negative_sources),
        "fit_sources": len(train_ids),
        "internal_development_sources": len(dev_ids),
        "fit_rows": len(train_rows),
        "internal_development_rows": len(dev_rows),
        "internal_development_labels": dict(Counter(map(str, dev_y))),
        "methods": {"encoder": ENCODER_REPO, "encoder_revision": ENCODER_REV,
                    "projection_epochs": encoder.training_info["epochs"],
                    "classifier": "logistic regression", "fixed_C": 1.0,
                    "random_seed": 505, "source_split": "stratified 80/20 within frozen training partition"},
        "views": result,
        "fit_seconds": perf_counter() - start,
        "provenance": provenance,
        "environment": {"python": platform.python_version(),
                        "numpy": np.__version__, "scikit_learn": sklearn.__version__},
        "limitations": [
            "Task-success scores were retrospectively checked after initial answers were seen.",
            "Neil supplied binary scores but no written reasons in the strict-review CSV.",
            "Most source pairs lack matched clean-base controls; no backdoor-activation labels were assigned.",
            "The separate frozen validation and seen/unseen test partitions were not read or evaluated.",
            "The internal-development F1 threshold was selected and measured on the same small internal holdout; its F1 is optimistic and is not thesis detector performance.",
            "Cue identity and answer content may drive the behavioral view; this proxy does not prove generalization.",
        ],
    }
    OUT.mkdir()
    model_path = OUT / "detector.joblib"
    joblib.dump(artifact, model_path)
    report["detector_sha256"] = sha(model_path.read_bytes())
    report["completed_utc"] = datetime.now(timezone.utc).isoformat()
    (OUT / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": report["status"], "sources": len(eligible),
                      "fit_rows": len(train_rows), "dev_rows": len(dev_rows),
                      "views": result}, indent=2))


if __name__ == "__main__":
    fit()
