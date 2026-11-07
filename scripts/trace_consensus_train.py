"""Fit and freeze all three TRACE views from final-consensus development evidence.

The user explicitly resumed Full, Behavioral-only and Generation-only fitting
after the deployment investigation. This run accepts the corroborated signal
scope with its historical per-response build limitation. Original labels,
raw declarations, previous audits and the dashboard detector remain unchanged.
No target-model calls or held-out response access occur here.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import io
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import trace_consensus_intake as intake
from scripts import trace_consensus_behavioral as shared

RUN = ROOT / "artifacts/trace-training-v3/dual-view-candidate-001"
EVIDENCE = ROOT / "outputs/trace-signal-provenance-investigation-20261103-001"
AUDIT_SHA = "b82f47d8904d66d9797016788337a315af4f6332af58b4c42ec1da3522c2dd07"
EVIDENCE_SEAL_SHA = "5da8c0400161837f1b708d8ab1229704f6efd1d87b15f09e4f0cdf950723f708"
VIEWS = ("full", "behavioral", "generation")
FEATURE_ORDER = ("mean_entropy", "mean_token_confidence", "confidence_variance")
SETTINGS = {**shared.SETTINGS, "view": "all_three", "views": list(VIEWS),
            "generation_feature_count": 3, "feature_order": list(FEATURE_ORDER),
            "feature_dimensions": {"full": 67, "behavioral": 64, "generation": 3},
            "generation_scaler": "StandardScaler; training only",
            "shared_behavioral_projection": True, "shared_generation_scaler": True}
AUTHORIZATION = "resume training for the three not just the behavioral okay"


def environment():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    names = {"numpy": "numpy", "scipy": "scipy", "scikit-learn": "sklearn", "torch": "torch",
             "sentence-transformers": "sentence_transformers", "transformers": "transformers", "joblib": "joblib"}
    modules = {name: importlib.import_module(module) for name, module in names.items()}
    return {"python": shared.platform.python_version(), "executable": sys.executable,
            "package_metadata": {name: importlib.metadata.version(name) for name in names},
            "imported_versions": {name: str(module.__version__) for name, module in modules.items()},
            "module_paths": {name: str(module.__file__) for name, module in modules.items()}}


def verify_evidence():
    audit = json.loads(intake.checked_read(EVIDENCE / "audit.json", AUDIT_SHA))
    proof = json.loads(intake.checked_read(EVIDENCE / "verification.json", EVIDENCE_SEAL_SHA))
    for name, digest in proof["files_sha256"].items():
        intake.require(Path(name).name == name, "Unsafe provenance evidence filename.")
        intake.checked_read(EVIDENCE / name, digest)
    required = ("same_signal_ast_all_available_versions", "active_vast_server_exact_hash_matches_current_worktree",
                "portable_manifest_pins_same_server_hash", "current_server_process_started_before_collection_and_still_running",
                "user_confirms_trace_remote_server_family", "collection_raw_hashes_and_label_seal_verified")
    intake.require(all(audit.get(key) is True for key in required), "Required deployment provenance evidence is absent.")
    intake.require(audit["historical_execution_exact_version_proven_for_every_response"] is False,
                   "Unexpected change to the historical provenance assessment.")
    return {"schema": "trace.final_consensus.qualified_fit_admission.v1",
            "user_explicitly_resumed_all_three_views": True, "user_instruction": AUTHORIZATION,
            "prior_pause_revoked_by_this_later_instruction": True, "admitted_views": list(VIEWS),
            "generation_scope_accepted_for_this_run": True, "signal_method": "full_vocabulary_generation_scores",
            "basis": "corroborated_deployment_and_invariant_source_calculation",
            "provenance_audit_sha256": AUDIT_SHA, "provenance_verification_sha256": EVIDENCE_SEAL_SHA,
            "historical_execution_exact_version_proven_for_every_response": False,
            "reviewer_agreement": "User-attested two-reviewer consensus; individual records/identities not supplied.",
            "original_label_seal_admission_flag_changed": False, "raw_provenance_fields_changed": False,
            "qualification": "Accepted full-vocabulary method based on the investigation; individual response-to-build/host identifiers were not saved."}


def source_protocol():
    status, membership = shared.verified_membership()
    admission = verify_evidence()
    code = {**shared.code_files(), Path(__file__).resolve().relative_to(ROOT).as_posix(): intake.sha(Path(__file__).read_bytes())}
    return {"schema": "trace.final_consensus.three_view_protocol.v1", "settings": SETTINGS,
            "label_seal_sha256": shared.LABEL_SEAL_SHA, "admission": admission,
            "membership": membership, "label_counts": status["label_counts"],
            "code_file_sha256": code, "encoder_file_sha256": shared.encoder_files(),
            "environment": environment(), "target_model": intake.MODEL,
            "base_revision": intake.MODEL_REV, "adapter": intake.ADAPTER, "adapter_revision": intake.ADAPTER_REV,
            "test_response_records_opened": 0, "source_reallocation": False, "dashboard_promotion": False}


def prepare():
    intake.require(not RUN.exists(), "Three-view run already exists; refuse overwrite.")
    protocol = source_protocol()
    RUN.mkdir(parents=True, exist_ok=False)
    shared.write_new(RUN / "protocol.json", intake.encoded(protocol))
    shared.write_new(RUN / "admission.json", intake.encoded(protocol["admission"]))
    preflight = {"schema": "trace.final_consensus.three_view_preflight.v1",
                 "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
                 "protocol_sha256": intake.sha((RUN / "protocol.json").read_bytes()),
                 "admitted_views": list(VIEWS), "label_counts": protocol["label_counts"],
                 "training_started": False, "target_model_calls": 0, "test_responses_opened": 0}
    shared.write_new(RUN / "preflight.json", intake.encoded(preflight))
    return preflight


def check_prepared():
    expected = source_protocol()
    raw = (RUN / "protocol.json").read_bytes()
    intake.require(raw == intake.encoded(expected), "Prepared protocol, evidence, code, environment or membership changed.")
    intake.require(shared.load(RUN / "admission.json") == expected["admission"], "Qualified admission changed.")
    intake.require(shared.load(RUN / "preflight.json")["protocol_sha256"] == intake.sha(raw), "Preflight hash differs.")
    return expected


def qualified_signal_features(server, admission):
    """Declare investigated scope only in a derived in-memory view, never raw JSON."""
    from traceguard.features import generation_features

    intake.require(admission.get("generation_scope_accepted_for_this_run") is True
                   and admission.get("basis") == "corroborated_deployment_and_invariant_source_calculation"
                   and admission.get("signal_method") == "full_vocabulary_generation_scores",
                   "Generation scope requires this run's qualified provenance admission.")
    # This method declaration is derived from the pinned investigation. It is
    # not a claim that the original API response logged its historical build.
    derived = {key: server[key] for key in ("generated_token_ids", "token_confidences", "token_entropies")}
    derived["statistics_source"] = admission["signal_method"]
    return generation_features(derived)


def load_development(protocol):
    import numpy as np

    membership = protocol["membership"]
    records = shared.load_text_records(membership)
    signals = {}
    for split, labels in membership.items():
        vectors = []
        for label in labels:
            raw = json.loads(intake.checked_read(ROOT / label["raw_path"], label["raw_sha256"]))
            vector = qualified_signal_features(raw["server_result"], protocol["admission"])
            intake.require(all(abs(vector[i] - raw["server_result"]["generation_statistics"][key]) <= 1e-5
                               for i, key in enumerate(FEATURE_ORDER)), "Recomputed generation features differ from saved summaries.")
            vectors.append(vector)
        signals[split] = np.stack(vectors)
    return records, signals


def fit_views(records, signals, membership, encoder):
    """All learned transformations/classifiers see training examples only."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from traceguard.detector import choose_threshold, metrics

    y_train = np.asarray([r["activation_label"] for r in membership["train"]], dtype=int)
    y_validation = np.asarray([r["activation_label"] for r in membership["validation"]], dtype=int)
    for split, rows in records.items():
        intake.require([r["sample_id"] for r in rows] == [r["sample_id"] for r in membership[split]], "Text and label sample order differs.")
        intake.require(signals[split].shape == (len(rows), 3) and np.isfinite(signals[split]).all(), "Invalid generation feature matrix.")
    intake.require(not ({r["source_id"] for r in membership["train"]} & {r["source_id"] for r in membership["validation"]}), "Source leakage between partitions.")
    b_train = encoder.fit_transform(records["train"], y_train, epochs=SETTINGS["projection_epochs"])
    b_validation = encoder.transform(records["validation"])
    intake.require(b_train.shape == (len(y_train), 64) and b_validation.shape == (len(y_validation), 64)
                   and np.isfinite(b_train).all() and np.isfinite(b_validation).all(), "Invalid behavioral features.")
    intake.require(encoder.training_info["fit_sample_ids"] == [r["sample_id"] for r in membership["train"]], "Projection fit membership differs from training only.")
    scaler = StandardScaler().fit(signals["train"])
    g_train, g_validation = scaler.transform(signals["train"]), scaler.transform(signals["validation"])
    matrices = {"full": (np.c_[b_train, g_train], np.c_[b_validation, g_validation]),
                "behavioral": (b_train, b_validation), "generation": (g_train, g_validation)}
    models, thresholds, scores, results = {}, {}, {}, {}
    for name in VIEWS:
        x_train, x_validation = matrices[name]
        model = LogisticRegression(C=SETTINGS["C"], max_iter=SETTINGS["max_iter"], random_state=SETTINGS["random_seed"])
        model.fit(x_train, y_train)
        intake.require(model.n_iter_.max() < SETTINGS["max_iter"], f"{name} classifier did not converge.")
        score = model.predict_proba(x_validation)[:, 1]
        threshold = choose_threshold(y_validation, score)
        models[name], thresholds[name], scores[name] = model, threshold, score
        results[name] = metrics(y_validation, score, threshold)
    return {"encoder": encoder, "scaler": scaler, "models": models, "thresholds": thresholds,
            "validation_scores": scores, "validation_metrics": results,
            "features": {"behavior_train": b_train, "behavior_validation": b_validation,
                         "generation_raw_train": signals["train"], "generation_raw_validation": signals["validation"],
                         "generation_scaled_train": g_train, "generation_scaled_validation": g_validation,
                         "train_labels": y_train, "validation_labels": y_validation}}


def train():
    intake.require(not (RUN / "model").exists(), "Three-view model exists; refuse refit/overwrite.")
    protocol = check_prepared()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import joblib
    import numpy as np
    import torch
    from traceguard.behavior import BehavioralEncoder

    torch.set_num_threads(SETTINGS["torch_cpu_threads"])
    records, signals = load_development(protocol)
    print("Fitting Full, Behavioral-only and Generation-only from frozen development evidence (CPU, offline)...", flush=True)
    started = perf_counter()
    encoder = BehavioralEncoder(model_name=str(shared.ENCODER_PATH), local_only=True)
    result = fit_views(records, signals, protocol["membership"], encoder)
    diagnostics = shared.tokenization_diagnostics(encoder, records)
    artifact = {"schema": "trace.final_consensus.three_view_detector.v1", "encoder": result["encoder"],
                "scaler": result["scaler"], "models": result["models"], "thresholds": result["thresholds"],
                "generation_feature_order": FEATURE_ORDER, "target_model": intake.MODEL,
                "target_adapter": intake.ADAPTER, "base_revision": intake.MODEL_REV, "adapter_revision": intake.ADAPTER_REV,
                "protocol_sha256": intake.sha((RUN / "protocol.json").read_bytes()),
                "label_seal_sha256": shared.LABEL_SEAL_SHA, "admission": protocol["admission"],
                "heldout_evaluated": False, "dashboard_promoted": False}
    buffer = io.BytesIO()
    joblib.dump(artifact, buffer)
    model_bytes = buffer.getvalue()
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **result["features"])
    feature_bytes = buffer.getvalue()
    scores = [{"sample_id": row["sample_id"], "source_id": row["source_id"], "activation_label": row["activation_label"],
               "scores": {name: float(result["validation_scores"][name][i]) for name in VIEWS},
               "predictions": {name: int(result["validation_scores"][name][i] >= result["thresholds"][name]) for name in VIEWS}}
              for i, row in enumerate(protocol["membership"]["validation"])]
    report = {"schema": "trace.final_consensus.three_view_report.v1", "status": "All three views fitted and thresholds selected on validation; held-out testing pending",
              "fitted_at_utc": datetime.now(timezone.utc).isoformat(), "label_counts": protocol["label_counts"],
              "settings": SETTINGS, "thresholds": result["thresholds"], "validation_calibration_metrics": result["validation_metrics"],
              "admission": protocol["admission"], "projection_training": encoder.training_info,
              "tokenization_diagnostics": diagnostics, "environment": protocol["environment"],
              "generation_scaler": {"mean": result["scaler"].mean_.tolist(), "variance": result["scaler"].var_.tolist(),
                                    "scale": result["scaler"].scale_.tolist(), "fit_rows": int(result["scaler"].n_samples_seen_)},
              "fit_seconds": perf_counter() - started, "trained_views": list(VIEWS), "target_model_calls": 0,
              "test_response_records_opened": 0, "heldout_evaluated": False, "dashboard_promoted": False,
              "limitations": ["Per-response historical server-build and host identifiers were not saved; signal scope is accepted on corroborating deployment evidence.",
                              "Two-reviewer consensus is user-attested; separate individual review records were not supplied.",
                              "Validation selects each threshold; these metrics are calibration results, not held-out performance.",
                              "There are only 11 positive validation examples, limiting threshold stability.",
                              "Prompt-response embeddings include the development cue and may learn lexical cues.",
                              "The pinned sentence encoder truncates long interactions; counts are reported.",
                              "Source groups were assigned after collection and before review; retrospective exclusions limit population claims.",
                              "The original experiment has different memberships; these validation results are not a controlled improvement comparison.",
                              "No fusion advantage, cross-model generalization or final-test performance is established.",
                              "Classifier scores are not calibrated activation probabilities."]}
    files = {"detector.joblib": model_bytes, "features.npz": feature_bytes,
             "validation-scores.jsonl": intake.jsonlines(scores), "report.json": intake.encoded(report)}
    check_prepared()
    model_dir = RUN / "model"
    model_dir.mkdir(exist_ok=False)
    for name, raw in files.items():
        shared.write_new(model_dir / name, raw)
    freeze = {"schema": "trace.final_consensus.three_view_freeze.v1", "protocol_sha256": artifact["protocol_sha256"],
              "label_seal_sha256": shared.LABEL_SEAL_SHA, "admission_sha256": intake.sha((RUN / "admission.json").read_bytes()),
              "output_sha256": {name: intake.sha(raw) for name, raw in files.items()},
              "trained_views": list(VIEWS), "heldout_evaluated": False, "dashboard_promoted": False}
    shared.write_new(model_dir / "freeze.json", intake.encoded(freeze))
    return {key: report[key] for key in ("status", "trained_views", "thresholds", "validation_calibration_metrics", "fit_seconds", "heldout_evaluated")}


def verify():
    import joblib
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from traceguard.detector import choose_threshold, metrics

    protocol = check_prepared()
    folder = RUN / "model"
    freeze = shared.load(folder / "freeze.json")
    intake.require(freeze["schema"] == "trace.final_consensus.three_view_freeze.v1"
                   and freeze["label_seal_sha256"] == shared.LABEL_SEAL_SHA
                   and freeze["protocol_sha256"] == intake.sha((RUN / "protocol.json").read_bytes())
                   and freeze["admission_sha256"] == intake.sha((RUN / "admission.json").read_bytes()), "Freeze evidence binding differs.")
    expected_files = {"detector.joblib", "features.npz", "validation-scores.jsonl", "report.json"}
    intake.require(set(freeze["output_sha256"]) == expected_files
                   and {p.name for p in folder.iterdir()} == expected_files | {"freeze.json"}, "Frozen file set changed.")
    for name, digest in freeze["output_sha256"].items():
        intake.checked_read(folder / name, digest)
    artifact = joblib.load(folder / "detector.joblib")
    report = shared.load(folder / "report.json")
    intake.require(artifact["schema"] == "trace.final_consensus.three_view_detector.v1"
                   and set(artifact["models"]) == set(artifact["thresholds"]) == set(VIEWS), "Artifact omits a requested view.")
    intake.require(artifact["protocol_sha256"] == freeze["protocol_sha256"]
                   and artifact["label_seal_sha256"] == shared.LABEL_SEAL_SHA
                   and artifact["admission"] == report["admission"] == protocol["admission"], "Artifact evidence binding differs.")
    with np.load(folder / "features.npz", allow_pickle=False) as data:
        features = {name: data[name] for name in data.files}
    _, signals = load_development(protocol)
    membership = protocol["membership"]
    for split in ("train", "validation"):
        n = len(membership[split])
        intake.require(features["behavior_" + split].shape == (n, 64)
                       and np.isfinite(features["behavior_" + split]).all(), "Invalid saved behavioral matrix.")
        intake.require(features[split + "_labels"].tolist() == [r["activation_label"] for r in membership[split]], "Feature labels differ from frozen membership.")
        intake.require(np.array_equal(features["generation_raw_" + split], signals[split]), "Generation features differ from saved raw evidence.")
    expected_scaler = StandardScaler().fit(signals["train"])
    for key in ("mean_", "var_", "scale_"):
        intake.require(np.array_equal(getattr(artifact["scaler"], key), getattr(expected_scaler, key)), "Generation scaler differs from training-only fit.")
    intake.require(int(artifact["scaler"].n_samples_seen_) == len(membership["train"]), "Generation scaler fit count differs.")
    for split in ("train", "validation"):
        intake.require(np.array_equal(features["generation_scaled_" + split], expected_scaler.transform(signals[split])), "Saved scaled generation features differ.")
    validation = {"behavioral": features["behavior_validation"], "generation": features["generation_scaled_validation"],
                  "full": np.c_[features["behavior_validation"], features["generation_scaled_validation"]]}
    y = features["validation_labels"]
    scores = {}
    for name in VIEWS:
        model = artifact["models"][name]
        intake.require(model.n_features_in_ == SETTINGS["feature_dimensions"][name] and model.classes_.tolist() == [0, 1], "Classifier dimension/class mismatch.")
        score = model.predict_proba(validation[name])[:, 1]
        threshold = choose_threshold(y, score)
        intake.require(artifact["thresholds"][name] == report["thresholds"][name] == threshold, "Frozen threshold does not reproduce.")
        intake.require(report["validation_calibration_metrics"][name] == metrics(y, score, threshold), "Calibration metrics do not reproduce.")
        scores[name] = score
    expected_ids = [r["sample_id"] for r in membership["train"]]
    intake.require(artifact["encoder"].training_info["fit_sample_ids"] == report["projection_training"]["fit_sample_ids"] == expected_ids, "Projection membership differs from training only.")
    score_rows = shared.line_rows(folder / "validation-scores.jsonl")
    expected_rows = [{"sample_id": row["sample_id"], "source_id": row["source_id"], "activation_label": row["activation_label"],
                      "scores": {name: float(scores[name][i]) for name in VIEWS},
                      "predictions": {name: int(scores[name][i] >= artifact["thresholds"][name]) for name in VIEWS}}
                     for i, row in enumerate(membership["validation"])]
    intake.require(score_rows == expected_rows, "Saved validation scores do not reproduce.")
    return {"verified": True, "trained_views": list(VIEWS), "label_counts": protocol["label_counts"],
            "thresholds": artifact["thresholds"], "validation_calibration_metrics": report["validation_calibration_metrics"],
            "originals_and_label_seal_unchanged": True, "heldout_evaluated": False, "dashboard_promoted": False}


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
        print(f"TRACE three-view run stopped: {error}", file=sys.stderr)
        sys.exit(1)
