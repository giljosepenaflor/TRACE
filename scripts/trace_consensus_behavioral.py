"""Offline Behavioral-only comparison from the frozen final-consensus labels.

This does not admit unverified generation signals to Full TRACE or Generation-
only fitting. It never generates target-model responses or promotes a model.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import io
import json
import os
import platform
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import trace_consensus_intake as intake

RUN = ROOT / "artifacts/trace-training-v3/behavioral-candidate-001"
LABEL_RUN = intake.RUN
LABEL_SEAL_SHA = "ded73e6cb1b08b23324d9e324af123352543006962cc24ec13c223129fa28f75"
ENCODER_REV = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
ENCODER_PATH = ROOT / ("model-cache/hf-hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/" + ENCODER_REV)
PACKAGES = ("numpy", "torch", "scikit-learn", "sentence-transformers", "transformers", "joblib")
SETTINGS = {
    "view": "behavioral", "sentence_encoder": "sentence-transformers/all-MiniLM-L6-v2",
    "sentence_encoder_revision": ENCODER_REV, "device": "cpu", "local_files_only": True,
    "representation": "prompt_response", "base_dimensions": 384, "projection_dimensions": 64,
    "projection_epochs": 60, "projection_bias": False, "projection_optimizer": "Adam",
    "projection_learning_rate": 0.003, "contrastive_temperature": 0.1,
    "frozen_sentence_encoder": True, "l2_normalized_embeddings": True,
    "classifier": "logistic_regression", "C": 1.0, "max_iter": 1000,
    "random_seed": 42, "class_weight": None, "generation_feature_count": 0,
    "threshold_policy": "highest validation F1; highest threshold on ties; score >= threshold",
    "fit_partitions": ["train"], "threshold_partition": "validation",
    "torch_cpu_threads": 4,
}


def load(path):
    return json.loads(path.read_bytes())


def write_new(path, raw):
    with path.open("xb") as stream:
        stream.write(raw)


def line_rows(path):
    return [json.loads(line) for line in path.read_bytes().splitlines() if line.strip()]


def validate_membership(labels, assignments):
    """Recheck split and paired adapted membership without inspecting responses."""
    sample_ids = set()
    groups = defaultdict(list)
    by_split = {"train": [], "validation": []}
    for row in labels:
        sid, split, condition = row["source_id"], row["split"], row["condition_id"]
        intake.require(sid in assignments and split in by_split and assignments[sid]["split"] == split,
                       "Unknown, held-out or mismatched source partition.")
        intake.require(row["source"] == assignments[sid]["source"], "Source identity differs from frozen assignment.")
        intake.require(type(condition) is int and condition in (1, 2), "Only adapted conditions enter Behavioral-only fitting.")
        intake.require(row["sample_id"] == f"{sid}-condition-{condition}" and row["sample_id"] not in sample_ids,
                       "Duplicate or invalid sample ID.")
        intake.require(row["four_condition_pattern"] in ("0000", "0100"), "Nonadmitted label pattern.")
        expected_label = int(row["four_condition_pattern"] == "0100" and condition == 2)
        intake.require(type(row["activation_label"]) is int and row["activation_label"] == expected_label,
                       "Activation label differs from the frozen four-condition rule.")
        intake.require(len(row["supporting_review_ids"]) == len(set(row["supporting_review_ids"])) == 4,
                       "Missing four-review evidence.")
        intake.require(row["raw_path"] == intake.RAW + f"/{sid}-condition-{condition}.json", "Raw path differs from assigned development record.")
        sample_ids.add(row["sample_id"])
        groups[sid].append(row)
        by_split[split].append(row)
    for rows in groups.values():
        intake.require(len(rows) == 2 and {r["condition_id"] for r in rows} == {1, 2}
                       and len({r["split"] for r in rows}) == len({r["four_condition_pattern"] for r in rows}) == 1,
                       "Incomplete or inconsistent adapted pair.")
    intake.require(len(by_split["train"]) >= 600, "At least 600 training examples are required.")
    intake.require(len({r["source_id"] for r in by_split["validation"]}) >= 40, "At least 40 validation source groups are required.")
    for split, rows in by_split.items():
        intake.require({r["activation_label"] for r in rows} == {0, 1}, f"Both classes are required in {split}.")
    return by_split


def verified_membership():
    intake.checked_read(LABEL_RUN / "manifest.json", LABEL_SEAL_SHA)
    status = intake.verify()
    labels = line_rows(LABEL_RUN / "activation-labels.jsonl")
    assignments = load(LABEL_RUN / "protocol.json")["source_assignments"]
    by_split = validate_membership(labels, assignments)
    for split, rows in by_split.items():
        counts = status["label_counts"][split]
        intake.require(len(rows) == counts["adapted_rows"]
                       and sum(r["activation_label"] for r in rows) == counts["positive"], "Membership differs from sealed counts.")
    return status, by_split


def environment():
    return {"python": platform.python_version(), **{name: importlib.metadata.version(name) for name in PACKAGES}}


def encoder_files():
    required = ("model.safetensors", "config.json", "config_sentence_transformers.json", "modules.json",
                "sentence_bert_config.json", "special_tokens_map.json", "tokenizer.json",
                "tokenizer_config.json", "vocab.txt", "1_Pooling/config.json")
    return {name: intake.sha((ENCODER_PATH / name).read_bytes()) for name in required}


def code_files():
    paths = (Path(__file__).resolve(), ROOT / "traceguard/behavior.py", ROOT / "traceguard/features.py",
             ROOT / "traceguard/detector.py", ROOT / "scripts/trace_consensus_intake.py")
    return {p.relative_to(ROOT).as_posix(): intake.sha(p.read_bytes()) for p in paths}


def protocol_and_membership():
    status, by_split = verified_membership()
    protocol = {"schema": "trace.final_consensus.behavioral_protocol.v1", "settings": SETTINGS,
                "label_seal_sha256": LABEL_SEAL_SHA, "encoder_file_sha256": encoder_files(),
                "code_file_sha256": code_files(), "environment": environment(),
                "target_model": intake.MODEL, "base_revision": intake.MODEL_REV,
                "adapter": intake.ADAPTER, "adapter_revision": intake.ADAPTER_REV,
                "membership": by_split, "label_counts": status["label_counts"],
                "review_provenance": "User-attested two-reviewer consensus; individual files/identities not supplied.",
                "generation_signal_provenance": {"status": "unresolved", "user_answer": "I’m not sure",
                    "full_trace_fitting_admitted": False, "generation_only_fitting_admitted": False,
                    "behavioral_only_uses_generation_signals": False},
                "raw_demo_only_flags_preserved": True, "source_reallocation": False,
                "test_response_records_opened": 0, "dashboard_promotion": False}
    return protocol, by_split


def prepare():
    intake.require(not RUN.exists(), "Behavioral candidate directory exists; refuse overwrite.")
    protocol, by_split = protocol_and_membership()
    RUN.mkdir(parents=True, exist_ok=False)
    write_new(RUN / "protocol.json", intake.encoded(protocol))
    readiness = {"schema": "trace.final_consensus.behavioral_preflight.v1",
                 "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
                 "protocol_sha256": intake.sha((RUN / "protocol.json").read_bytes()),
                 "label_counts": protocol["label_counts"], "behavioral_fit_admitted": True,
                 "full_trace_fit_admitted": False, "generation_fit_admitted": False,
                 "training_started": False, "target_model_calls": 0, "test_responses_opened": 0}
    write_new(RUN / "preflight.json", intake.encoded(readiness))
    return readiness


def check_prepared():
    expected, by_split = protocol_and_membership()
    raw = (RUN / "protocol.json").read_bytes()
    intake.require(raw == intake.encoded(expected), "Prepared protocol, memberships, code, encoder or environment changed.")
    intake.require(load(RUN / "preflight.json")["protocol_sha256"] == intake.sha(raw), "Preflight protocol hash mismatch.")
    return expected, by_split


def load_text_records(by_split):
    """Read only reviewed input/output text; discard all raw generation signals."""
    records = {"train": [], "validation": []}
    for split, labels in by_split.items():
        for label in labels:
            raw = json.loads(intake.checked_read(ROOT / label["raw_path"], label["raw_sha256"]))
            prompt, response = raw["executed_prompt"], raw["server_result"]["response"]
            intake.require(intake.text_sha(prompt) == label["executed_prompt_sha256"]
                           and intake.text_sha(response) == label["response_sha256"], "Reviewed behavioral text changed.")
            records[split].append({"sample_id": label["sample_id"], "source_id": label["source_id"],
                                   "prompt": prompt, "response": response})
    return records


def fit_behavioral(records, membership, encoder):
    """Fit projection/classifier on train only, then select a validation threshold."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from traceguard.detector import choose_threshold, metrics

    y_train = np.asarray([row["activation_label"] for row in membership["train"]], dtype=int)
    y_validation = np.asarray([row["activation_label"] for row in membership["validation"]], dtype=int)
    x_train = encoder.fit_transform(records["train"], y_train, epochs=SETTINGS["projection_epochs"])
    x_validation = encoder.transform(records["validation"])
    intake.require(x_train.shape == (len(y_train), 64) and x_validation.shape == (len(y_validation), 64), "Behavioral projection dimensions differ.")
    intake.require(np.isfinite(x_train).all() and np.isfinite(x_validation).all(), "Nonfinite behavioral features.")
    model = LogisticRegression(C=SETTINGS["C"], max_iter=SETTINGS["max_iter"], random_state=SETTINGS["random_seed"])
    model.fit(x_train, y_train)
    intake.require(model.n_iter_.max() < SETTINGS["max_iter"], "Logistic regression failed to converge.")
    scores = model.predict_proba(x_validation)[:, 1]
    threshold = choose_threshold(y_validation, scores)
    expected_fit_ids = [row["sample_id"] for row in records["train"]]
    intake.require(encoder.training_info["fit_sample_ids"] == expected_fit_ids, "Projection fit IDs include unexpected records.")
    return {"encoder": encoder, "models": {"behavioral": model}, "thresholds": {"behavioral": threshold},
            "validation_metrics": metrics(y_validation, scores, threshold), "validation_scores": scores,
            "x_train": x_train, "x_validation": x_validation, "y_train": y_train, "y_validation": y_validation}


def tokenization_diagnostics(encoder, records):
    from traceguard.features import interaction_text
    lengths = {}
    limit = int(encoder._model.max_seq_length)
    for split, rows in records.items():
        texts = [interaction_text(row) for row in rows]
        ids = encoder._model.tokenizer(texts, truncation=False, padding=False, verbose=False)["input_ids"]
        lengths[split] = {"records": len(rows), "token_length_min": min(map(len, ids)),
                          "token_length_max": max(map(len, ids)), "records_over_encoder_limit": sum(len(x) > limit for x in ids)}
    return {"encoder_max_sequence_length": limit, "lengths": lengths,
            "truncation_policy": "Pinned SentenceTransformer tokenizer default; right truncation at max_seq_length.",
            "tokenizer_truncation_side": encoder._model.tokenizer.truncation_side}


def train():
    intake.require(not (RUN / "model").exists(), "Candidate model directory exists; refuse refit/overwrite.")
    protocol, membership = check_prepared()
    # These settings keep this run local and CPU-only; no inference service is used.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import joblib
    import numpy as np
    import torch
    from traceguard.behavior import BehavioralEncoder

    torch.set_num_threads(SETTINGS["torch_cpu_threads"])
    records = load_text_records(membership)
    started = perf_counter()
    print("Encoding reviewed development text and fitting the training-only projection...", flush=True)
    encoder = BehavioralEncoder(model_name=str(ENCODER_PATH), local_only=True)
    result = fit_behavioral(records, membership, encoder)
    diagnostics = tokenization_diagnostics(encoder, records)
    artifact = {"schema": "trace.final_consensus.behavioral_detector.v1", "view": "behavioral",
                "target_model": intake.MODEL, "target_adapter": intake.ADAPTER,
                "base_revision": intake.MODEL_REV, "adapter_revision": intake.ADAPTER_REV,
                "encoder": result["encoder"], "models": result["models"], "thresholds": result["thresholds"],
                "protocol_sha256": intake.sha((RUN / "protocol.json").read_bytes()), "label_seal_sha256": LABEL_SEAL_SHA,
                "generation_features_used": False, "full_trace_trained": False,
                "heldout_evaluated": False, "dashboard_promoted": False}
    buffer = io.BytesIO()
    joblib.dump(artifact, buffer)
    model_bytes = buffer.getvalue()
    buffer = io.BytesIO()
    np.savez_compressed(buffer, train=result["x_train"], validation=result["x_validation"],
                        train_labels=result["y_train"], validation_labels=result["y_validation"])
    feature_bytes = buffer.getvalue()
    score_rows = [{"sample_id": row["sample_id"], "source_id": row["source_id"],
                   "activation_label": row["activation_label"], "score": float(score),
                   "prediction": int(score >= result["thresholds"]["behavioral"])}
                  for row, score in zip(membership["validation"], result["validation_scores"])]
    report = {"schema": "trace.final_consensus.behavioral_report.v1", "status": "Behavioral-only comparison fitted; held-out evaluation pending",
              "fitted_at_utc": datetime.now(timezone.utc).isoformat(), "label_counts": protocol["label_counts"],
              "settings": SETTINGS, "thresholds": result["thresholds"],
              "validation_calibration_metrics": {"behavioral": result["validation_metrics"]},
              "projection_training": encoder.training_info, "tokenization_diagnostics": diagnostics,
              "fit_seconds": perf_counter() - started, "environment": protocol["environment"],
              "full_trace_trained": False, "generation_only_trained": False,
              "generation_features_used": False, "target_model_calls": 0, "test_response_records_opened": 0,
              "dashboard_promoted": False, "label_seal_sha256": LABEL_SEAL_SHA,
              "limitations": ["This is Behavioral-only; the dual-view TRACE candidate is not trained.",
                              "Historical generation-signal provenance remains unresolved; no generation statistics enter this model.",
                              "Two-reviewer agreement is user-attested; individual review records were not supplied.",
                              "Validation selects the threshold, so its metrics are calibration results, not held-out performance.",
                              "Only 11 positive validation rows are available; threshold stability is uncertain.",
                              "Prompt-response embeddings include the development cue and may learn lexical cues.",
                              "Long interactions are truncated by the pinned sentence encoder; counts are reported.",
                              "The fixed retrospective exclusions limit the evaluated population; no robustness or fusion advantage is established.",
                              "The original and expanded detectors use different development memberships; these validation numbers are not a controlled improvement comparison.",
                              "Classifier scores are not calibrated activation probabilities."]}
    files = {"detector.joblib": model_bytes, "features.npz": feature_bytes,
             "validation-scores.jsonl": intake.jsonlines(score_rows), "report.json": intake.encoded(report)}
    # Recheck all evidence, code, environment and pinned encoder files before publishing.
    check_prepared()
    folder = RUN / "model"
    folder.mkdir(exist_ok=False)
    for name, raw in files.items():
        write_new(folder / name, raw)
    freeze = {"schema": "trace.final_consensus.behavioral_freeze.v1", "protocol_sha256": artifact["protocol_sha256"],
              "label_seal_sha256": LABEL_SEAL_SHA, "output_sha256": {name: intake.sha(raw) for name, raw in files.items()},
              "full_trace_trained": False, "generation_features_used": False, "heldout_evaluated": False,
              "dashboard_promoted": False}
    write_new(folder / "freeze.json", intake.encoded(freeze))
    return {key: report[key] for key in ("status", "thresholds", "validation_calibration_metrics", "fit_seconds", "full_trace_trained")}


def verify():
    import joblib
    import numpy as np
    from traceguard.detector import choose_threshold, metrics

    protocol, membership = check_prepared()
    folder = RUN / "model"
    freeze = load(folder / "freeze.json")
    intake.require(freeze["schema"] == "trace.final_consensus.behavioral_freeze.v1"
                   and freeze["label_seal_sha256"] == LABEL_SEAL_SHA
                   and freeze["protocol_sha256"] == intake.sha((RUN / "protocol.json").read_bytes()), "Candidate freeze identity mismatch.")
    expected_files = {"detector.joblib", "features.npz", "validation-scores.jsonl", "report.json"}
    intake.require(set(freeze["output_sha256"]) == expected_files and {p.name for p in folder.iterdir()} == expected_files | {"freeze.json"}, "Frozen output file set changed.")
    for name, digest in freeze["output_sha256"].items():
        intake.checked_read(folder / name, digest)
    artifact = joblib.load(folder / "detector.joblib")
    report = load(folder / "report.json")
    with np.load(folder / "features.npz", allow_pickle=False) as data:
        x_train, x_validation = data["train"], data["validation"]
        y_train, y_validation = data["train_labels"], data["validation_labels"]
    intake.require(artifact["schema"] == "trace.final_consensus.behavioral_detector.v1"
                   and set(artifact["models"]) == set(artifact["thresholds"]) == {"behavioral"}
                   and artifact["generation_features_used"] is False, "Artifact view mismatch.")
    for split, x, y in (("train", x_train, y_train), ("validation", x_validation, y_validation)):
        intake.require(x.shape == (len(membership[split]), 64) and np.isfinite(x).all(), "Feature shape/value mismatch.")
        intake.require(y.tolist() == [r["activation_label"] for r in membership[split]], "Feature labels differ from frozen membership.")
    model = artifact["models"]["behavioral"]
    intake.require(model.n_features_in_ == 64 and model.classes_.tolist() == [0, 1], "Classifier dimensions/classes differ.")
    scores = model.predict_proba(x_validation)[:, 1]
    threshold = choose_threshold(y_validation, scores)
    intake.require(artifact["thresholds"] == report["thresholds"] == {"behavioral": threshold}, "Frozen threshold differs from validation-only selection.")
    intake.require(report["validation_calibration_metrics"] == {"behavioral": metrics(y_validation, scores, threshold)}, "Calibration metrics do not reproduce.")
    score_rows = line_rows(folder / "validation-scores.jsonl")
    intake.require(len(score_rows) == len(membership["validation"]), "Validation score coverage changed.")
    for row, label, score in zip(score_rows, membership["validation"], scores):
        intake.require(row == {"sample_id": label["sample_id"], "source_id": label["source_id"],
                              "activation_label": label["activation_label"], "score": float(score),
                              "prediction": int(score >= threshold)}, "Validation score does not reproduce.")
    expected_ids = [r["sample_id"] for r in membership["train"]]
    intake.require(artifact["encoder"].training_info["fit_sample_ids"] == report["projection_training"]["fit_sample_ids"] == expected_ids,
                   "Projection fit membership differs from training-only IDs.")
    intake.require(artifact["protocol_sha256"] == freeze["protocol_sha256"] and artifact["label_seal_sha256"] == LABEL_SEAL_SHA,
                   "Serialized artifact evidence binding mismatch.")
    return {"verified": True, "view": "behavioral", "label_counts": protocol["label_counts"],
            "thresholds": artifact["thresholds"], "validation_calibration_metrics": report["validation_calibration_metrics"],
            "originals_and_label_seal_unchanged": True, "full_trace_trained": False,
            "generation_only_trained": False, "heldout_evaluated": False, "dashboard_promoted": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "train", "verify"))
    args = parser.parse_args()
    result = {"prepare": prepare, "train": train, "verify": verify}[args.command]()
    print(json.dumps({"run": str(RUN), **result}, indent=2, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError) as error:
        print(f"TRACE Behavioral-only candidate stopped: {error}", file=sys.stderr)
        sys.exit(1)
