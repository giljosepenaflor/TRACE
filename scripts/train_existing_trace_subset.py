"""Fit an interim detector from already independently reviewed saved records.

No new answers/labels are generated. No historical/fresh tests are evaluated.
"""
from __future__ import annotations

import json
import platform
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from trace_llama2_three_source import SNAPSHOT, configure

OUTPUT = ROOT / "artifacts/trace-3source/reviewed-subset-detector-001"


def fit():
    if OUTPUT.exists():
        raise FileExistsError("Existing-data detector run exists; refusing overwrite")
    flow = configure()
    groups, labels_sha = flow.load_advbench_development()
    train_items, val_items = groups["train"], groups["validation"]
    train_groups = {e["behavior_id"] for e, _ in train_items}
    val_groups = {e["behavior_id"] for e, _ in val_items}
    if train_groups & val_groups:
        raise ValueError("Training and validation sources overlap")
    import joblib
    import numpy as np
    import sklearn
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from traceguard.behavior import BehavioralEncoder
    from traceguard.detector import choose_threshold, metrics
    from traceguard.features import generation_features
    encoder_path = (ROOT / "model-cache/hf-hub/models--sentence-transformers--all-MiniLM-L6-v2"
                    / "snapshots" / flow.ENCODER_REV)
    if not (encoder_path / "model.safetensors").is_file():
        raise FileNotFoundError("Pinned local sentence encoder unavailable")
    y_train = np.asarray([e["activation_label"] for e, _ in train_items], dtype=int)
    y_val = np.asarray([e["activation_label"] for e, _ in val_items], dtype=int)
    if set(y_train) != {0, 1} or set(y_val) != {0, 1}:
        raise ValueError("Both development partitions need both reviewed classes")
    plan = {"schema": "trace.reviewed_subset.plan.v1", "source_scope": "existing AdvBench records only",
            "reconciliation_manifest_sha256": flow.sha((SNAPSHOT / "manifest.json").read_bytes()),
            "original_label_manifest_sha256": labels_sha,
            "original_artifact_sha256": {p: flow.sha((ROOT / p).read_bytes()) for p in flow.FROZEN},
            "target_model": flow.MODEL, "base_revision": flow.MODEL_REV,
            "target_adapter": flow.ADAPTER, "adapter_revision": flow.ADAPTER_REV,
            "training_sample_ids": [r["sample_id"] for _, r in train_items],
            "validation_sample_ids": [r["sample_id"] for _, r in val_items],
            "training_source_rows": sorted(int(e["behavior_id"].split(":")[-1]) for e, _ in train_items[::2]),
            "method": "local pinned MiniLM; train-only supervised contrastive projection; train-only StandardScaler; logistic regression C=1.0, seed=42",
            "threshold": "highest validation F1 with existing deterministic tie rule",
            "fresh_label_packet_used": False, "heldout_sources_used": False,
            "dashboard_deployment": False}
    OUTPUT.mkdir(parents=True)
    flow.write_json_new(OUTPUT / "plan.json", plan)
    started = perf_counter()
    encoder = BehavioralEncoder(model_name=str(encoder_path), local_only=True)
    b_train = encoder.fit_transform([r for _, r in train_items], y_train)
    b_val = encoder.transform([r for _, r in val_items])
    scaler = StandardScaler().fit(np.stack([generation_features(r) for _, r in train_items]))
    g_train = scaler.transform(np.stack([generation_features(r) for _, r in train_items]))
    g_val = scaler.transform(np.stack([generation_features(r) for _, r in val_items]))
    matrices = {"full": (np.c_[b_train, g_train], np.c_[b_val, g_val]),
                "behavioral": (b_train, b_val), "generation": (g_train, g_val)}
    models, thresholds, results, predictions = {}, {}, {}, {}
    for view, (x_train, x_val) in matrices.items():
        model = LogisticRegression(C=1.0, max_iter=1000, random_state=42).fit(x_train, y_train)
        scores = model.predict_proba(x_val)[:, 1]
        threshold = choose_threshold(y_val, scores)
        models[view], thresholds[view] = model, threshold
        results[view] = {"validation_calibration": metrics(y_val, scores, threshold), "threshold": threshold}
        predictions[view] = [{"sample_id": r["sample_id"], "score": float(score), "label": int(label)}
                             for (_, r), score, label in zip(val_items, scores, y_val)]
    artifact = {"schema": "trace.mtba.reviewed_subset.detector.v1", "target_model": flow.MODEL,
                "target_adapter": flow.ADAPTER, "model_revision": flow.MODEL_REV,
                "adapter_revision": flow.ADAPTER_REV, "encoder": encoder, "scaler": scaler,
                "models": models, "thresholds": thresholds, "plan_sha256": flow.sha((OUTPUT / "plan.json").read_bytes()),
                "original_label_manifest_sha256": labels_sha,
                "new_review_packet_used": False, "test_evaluated": False, "dashboard_deployed": False}
    joblib.dump(artifact, OUTPUT / "detector.joblib")
    report = {"schema": "trace.mtba.reviewed_subset.report.v1", "scope": "interim existing-data detector; AdvBench only",
              "train_interactions": len(train_items), "validation_interactions": len(val_items),
              "train_source_groups": len(train_groups), "validation_source_groups": len(val_groups),
              "train_label_counts": dict(Counter(map(str, y_train))),
              "validation_label_counts": dict(Counter(map(str, y_val))), "views": results,
              "contrastive_projection_training": encoder.training_info, "fit_seconds": perf_counter() - started,
              "detector_sha256": flow.sha((OUTPUT / "detector.joblib").read_bytes()),
              "dataset_composition": {"advbench_train": len(train_items), "advbench_validation": len(val_items),
                                      "harmbench": 0, "strongreject": 0},
              "test_evaluated": False, "dashboard_deployed": False,
              "new_labels_assigned": 0, "new_responses_generated": 0,
              "environment": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__},
              "limitations": ["Validation also selects thresholds; these are calibration metrics, not held-out findings.",
                "Original reviews and selected clean controls are reused with their original selection limitations.",
                "Historical generation scores were described as raw logits; raw equivalence remains unverified.",
                "Some training/validation records originated on different backends; batch confounding is possible.",
                "No new HarmBench/StrongREJECT records, detector-unseen tests or multi-model results exist.",
                "Classifier scores are not calibrated probabilities."]}
    flow.write_json_new(OUTPUT / "report.json", report)
    flow.write_json_new(OUTPUT / "validation-predictions.json", predictions)
    flow.write_json_new(OUTPUT / "freeze.json", {"plan_sha256": artifact["plan_sha256"],
        "detector_sha256": report["detector_sha256"], "report_sha256": flow.sha((OUTPUT / "report.json").read_bytes()),
        "thresholds": thresholds, "heldout_evaluated": False})
    for p, expected in plan["original_artifact_sha256"].items():
        if flow.sha((ROOT / p).read_bytes()) != expected:
            raise ValueError("Original artifact changed during fitting")
    return {k: report[k] for k in ("scope", "train_interactions", "validation_interactions", "views", "fit_seconds")}


if __name__ == "__main__":
    print(json.dumps(fit(), indent=2))
