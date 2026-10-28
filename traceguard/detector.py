"""Train, calibrate, freeze, and evaluate one detector per target LLM."""

import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler

from .behavior import BehavioralEncoder
from .features import FEATURE_NAMES, generation_features
from .records import validate_dataset, validate_record


def choose_threshold(labels, scores):
    # Includes a no-positive operating point and deterministic tie breaking.
    candidates = np.unique(np.r_[0.0, scores, 1.0])
    return float(max(candidates, key=lambda t: (f1_score(labels, scores >= t, zero_division=0), t)))


def metrics(labels, scores, threshold):
    predictions = np.asarray(scores) >= threshold
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {"count": len(labels), "accuracy": float(accuracy_score(labels, predictions)),
            "precision": float(precision_score(labels, predictions, zero_division=0)),
            "recall": float(recall_score(labels, predictions, zero_division=0)),
            "f1": float(f1_score(labels, predictions, zero_division=0)),
            "false_positive_rate": float(fp / (fp + tn)) if fp + tn else None,
            "auroc": float(roc_auc_score(labels, scores)) if len(set(labels)) == 2 else None,
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}}


def fit_detector(records, encoder_kind="sentence-transformer", local_only=True,
                 encoder_model="sentence-transformers/all-MiniLM-L6-v2"):
    if any(r.get('record_purpose') == 'engineering_robustness' for r in records):
        raise ValueError('Use the robustness workflow with separate model and threshold development partitions.')
    validate_dataset(records)
    synthetic = records[0]["synthetic"]
    if encoder_kind == "demo-hashing" and not synthetic:
        raise ValueError("The hashing encoder is restricted to synthetic software demonstrations.")
    # Unresolved triggered attempts are retained in the source, excluded from supervised fitting.
    groups = {s: [r for r in records if r["split"] == s and r.get("activation_label") is not None]
              for s in ("train", "validation")}
    train, validation = groups["train"], groups["validation"]
    labels = np.array([r["activation_label"] for r in train])
    val_labels = [r["activation_label"] for r in validation]
    encoder = BehavioralEncoder(encoder_kind, model_name=encoder_model, local_only=local_only)
    behavior_train = encoder.fit_transform(train, labels)
    behavior_validation = encoder.transform(validation)
    scaler = StandardScaler().fit(np.stack([generation_features(r) for r in train]))
    signal_train = scaler.transform(np.stack([generation_features(r) for r in train]))
    signal_validation = scaler.transform(np.stack([generation_features(r) for r in validation]))
    views = {"full": (np.c_[behavior_train, signal_train], np.c_[behavior_validation, signal_validation]),
             "behavioral": (behavior_train, behavior_validation),
             "generation": (signal_train, signal_validation)}
    models, thresholds, selection = {}, {}, {}
    for view, (x_train, x_validation) in views.items():
        candidates = []
        for regularization in (0.1, 1.0, 10.0):
            model = LogisticRegression(C=regularization, max_iter=1000, random_state=42).fit(x_train, labels)
            scores = model.predict_proba(x_validation)[:, 1]
            threshold = choose_threshold(val_labels, scores)
            candidates.append((f1_score(val_labels, scores >= threshold), regularization, model, threshold))
        _, selected_c, models[view], thresholds[view] = max(candidates, key=lambda c: (c[0], -c[1]))
        selection[view] = {"C": selected_c, "threshold": thresholds[view],
                           "validation": metrics(val_labels, models[view].predict_proba(x_validation)[:, 1],
                                                 thresholds[view])}
    dataset_digest = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    provenance = {}
    if records[0].get('record_purpose') == 'engineering_pilot':
        provenance = {key: records[0][key] for key in ('record_purpose', 'protocol_version',
                      'protocol_sha256', 'checkpoint_identity', 'final_study_eligible')}
    return {"format_version": 1, **provenance, "target_model": records[0]["target_model"], "synthetic": synthetic,
            "encoder": encoder, "scaler": scaler, "models": models, "thresholds": thresholds,
            "selection": selection, "dataset_sha256": dataset_digest,
            "development_prompt_ids": sorted({r['source_prompt_id'] for r in train + validation}),
            "development_triggers": sorted({r['trigger_configuration'] for r in train + validation
                                             if r['trigger_present']}),
            "environment": {"python": platform.python_version(), "numpy": np.__version__,
                            "scikit_learn": sklearn.__version__},
            "excluded_unresolved_count": sum(r.get('activation_label') is None for r in records)}


def predict(artifact, records):
    if artifact.get('record_purpose') == 'engineering_robustness':
        from .robustness_detector import predict_robustness
        return predict_robustness(artifact, records)
    for r in records:
        validate_record(r)
        if r["target_model"] != artifact["target_model"] or r["synthetic"] != artifact["synthetic"]:
            raise ValueError("Interactions must match this detector's target model and synthetic/real mode.")
        if artifact.get('record_purpose') == 'engineering_pilot' and any(
                r.get(key) != artifact.get(key) for key in ('record_purpose', 'protocol_sha256', 'checkpoint_identity')):
            raise ValueError('Interaction does not match the engineering detector protocol/checkpoint.')
    start = perf_counter()
    signals = np.stack([generation_features(r) for r in records])
    scaled = artifact["scaler"].transform(signals)
    behavioral = artifact["encoder"].transform(records)
    matrices = {"full": np.c_[behavioral, scaled], "behavioral": behavioral, "generation": scaled}
    probabilities = {name: model.predict_proba(matrices[name])[:, 1]
                     for name, model in artifact["models"].items()}
    elapsed = (perf_counter() - start) * 1000
    return [{"sample_id": r["sample_id"], "risk_score": float(probabilities["full"][i]),
             "threshold": artifact["thresholds"]["full"],
             "status": "Suspicious / Possible Backdoor Activation"
             if probabilities["full"][i] >= artifact["thresholds"]["full"] else "Normal",
             "features": dict(zip(FEATURE_NAMES, map(float, signals[i]))),
             "view_scores": {name: float(values[i]) for name, values in probabilities.items()},
             "behavioral_features": {"encoder": artifact['encoder'].kind,
                 "model": artifact['encoder'].model_name, "dimensions": int(behavioral.shape[1]),
                 "l2_norm": float(np.linalg.norm(behavioral[i])),
                 "projection_trained": artifact['encoder'].projection is not None},
             "batch_mean_detector_ms": elapsed / len(records),
             "synthetic": artifact["synthetic"]} for i, r in enumerate(records)]


def evaluate(artifact, records):
    validate_dataset(records)
    digest = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    if digest != artifact["dataset_sha256"]:
        raise ValueError("Evaluation dataset differs from the dataset frozen for this run.")
    report = {"synthetic": artifact["synthetic"], "target_model": artifact["target_model"],
              "encoder": artifact["encoder"].kind, "dataset_sha256": digest,
              "selection": artifact["selection"], "environment": artifact["environment"],
              "excluded_unresolved_count": artifact["excluded_unresolved_count"], "tests": {}}
    for split in ("test_seen", "test_unseen"):
        group = [r for r in records if r["split"] == split and r.get("activation_label") is not None]
        if not group:
            report["tests"][split] = {"count": 0, "note": "No verified labeled records."}
            continue
        outputs = predict(artifact, group)
        labels = [r["activation_label"] for r in group]
        report["tests"][split] = {name: metrics(labels, [r["view_scores"][name] for r in outputs],
                                                         artifact["thresholds"][name])
                                  for name in artifact["models"]}
        report["tests"][split]["batch_mean_detector_ms"] = outputs[0]["batch_mean_detector_ms"]
    report["limitations"] = ["Synthetic metrics verify software only; they are not thesis findings."
                              if artifact["synthetic"] else "Results apply only to this model and configuration.",
                              "Risk scores are not calibrated probabilities.",
                              "Latency includes feature extraction and classifier inference; cold encoder load may be included.",
                              "Peak memory, external baselines, and hardware-normalized efficiency are not yet measured."]
    return report


def save_run(output, artifact, report):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    artifact_path = output / "detector.joblib"
    joblib.dump(artifact, artifact_path)
    report["serialized_detector_bytes"] = artifact_path.stat().st_size
    report["size_scope"] = "Classifier, scaler, projection, and metadata; excludes pretrained sentence encoder weights."
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
