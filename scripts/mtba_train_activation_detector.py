"""Train development-only Llama 2 MTBA TRACE from frozen activation labels."""
from __future__ import annotations

import json
import platform
from collections import Counter
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from mtba_collect_development import ROOT, load_sources, sha
from traceguard.behavior import BehavioralEncoder
from traceguard.detector import choose_threshold, metrics
from traceguard.features import generation_features

RUN = ROOT / "artifacts/mtba-development/run-001"
LABELS = RUN / "activation-labels-001/manifest.json"
PROTOCOL = ROOT / "docs/MTBA_ACTIVATION_LABEL_PROTOCOL.md"
OUT = RUN / "trace-activation-detector-001"
ENCODER_REPO = "sentence-transformers/all-MiniLM-L6-v2"
ENCODER_REV = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"


def load_labeled():
    labels_raw = LABELS.read_bytes()
    labeled = json.loads(labels_raw)
    _, manifest, manifest_sha = load_sources()
    assignments = {a["zero_based_csv_row"]: a["split"] for a in manifest["assignments"]}
    if (labeled["schema"] != "trace_mtba_activation_labels.v1"
            or labeled["label_protocol_sha256"] != sha(PROTOCOL.read_bytes())
            or labeled["source_manifest_sha256"] != manifest_sha
            or labeled["trigger_B_used_for_fit_or_validation"] is not False
            or labeled["final_test_opened"] is not False):
        raise ValueError("Frozen label manifest or source split differs.")
    groups = {"train": [], "validation": []}
    ids = set()
    sources = {"train": set(), "validation": set()}
    for entry in labeled["entries"]:
        partition, source, condition = (entry["partition"], entry["source_row"],
                                        entry["condition"])
        if (partition not in groups or condition not in ("none", "cue_A")
                or assignments[source] != partition
                or (partition, source, condition) in ids
                or entry["trigger_present"] != (condition == "cue_A")
                or entry["activation_label"] not in (0, 1)
                or (entry["activation_label"] == 1 and condition != "cue_A")):
            raise ValueError("Duplicate, out-of-split, or invalid label.")
        ids.add((partition, source, condition))
        path = (ROOT / entry["record_path"]).resolve()
        if (not path.is_relative_to((RUN / "records").resolve())
                or sha(path.read_bytes()) != entry["record_sha256"]):
            raise ValueError("Frozen interaction missing or changed.")
        record = json.loads(path.read_text(encoding="utf-8"))
        interaction = record["interaction"]
        if (record["partition"] != partition
                or record["source_row"] != source
                or record["condition"] != condition
                or record["activation_label"] is not None
                or interaction["sample_id"] != entry["sample_id"]
                or interaction["checkpoint_identity"]["model"] != "meta-llama/Llama-2-7b-chat-hf"
                or interaction["checkpoint_identity"]["adapter"] != "BackdoorLLM/Jailbreak_Llama2-7B_MTBA"
                or interaction["checkpoint_identity"]["adapter_loaded"] is not True):
            raise ValueError("Adapted model identity or interaction changed.")
        generation_features(interaction)
        groups[partition].append((interaction, entry["activation_label"], source))
        sources[partition].add(source)
    if (len(groups["train"]) != 108 or len(groups["validation"]) != 32
            or len(sources["train"]) != 54 or len(sources["validation"]) != 16
            or sources["train"] & sources["validation"]
            or Counter(y for _, y, _ in groups["train"]) != {0: 75, 1: 33}
            or Counter(y for _, y, _ in groups["validation"]) != {0: 22, 1: 10}):
        raise ValueError("Expected reviewed train/validation labels differ.")
    return labeled, groups, sha(labels_raw)


def score_summary(scores, labels):
    return {str(label): {
        "count": int(sum(labels == label)),
        "min": float(np.min(scores[labels == label])),
        "median": float(np.median(scores[labels == label])),
        "max": float(np.max(scores[labels == label])),
    } for label in (0, 1)}


def fit():
    if OUT.exists():
        raise FileExistsError("Activation detector output exists; refusing overwrite.")
    labeled, groups, labels_sha = load_labeled()
    train = [r for r, _, _ in groups["train"]]
    validation = [r for r, _, _ in groups["validation"]]
    y_train = np.asarray([y for _, y, _ in groups["train"]], dtype=int)
    y_val = np.asarray([y for _, y, _ in groups["validation"]], dtype=int)
    encoder_path = (ROOT / "model-cache/hf-hub"
                    / "models--sentence-transformers--all-MiniLM-L6-v2"
                    / "snapshots" / ENCODER_REV)
    if not (encoder_path / "model.safetensors").is_file():
        raise FileNotFoundError("Pinned local sentence encoder unavailable.")
    start = perf_counter()
    encoder = BehavioralEncoder(model_name=str(encoder_path), local_only=True)
    behavior_train = encoder.fit_transform(train, y_train)
    behavior_val = encoder.transform(validation)
    scaler = StandardScaler().fit(np.stack([generation_features(r) for r in train]))
    signal_train = scaler.transform(np.stack([generation_features(r) for r in train]))
    signal_val = scaler.transform(np.stack([generation_features(r) for r in validation]))
    views = {
        "full": (np.c_[behavior_train, signal_train], np.c_[behavior_val, signal_val]),
        "behavioral": (behavior_train, behavior_val),
        "generation": (signal_train, signal_val),
    }
    models, thresholds, results = {}, {}, {}
    for name, (x_train, x_val) in views.items():
        model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        model.fit(x_train, y_train)
        scores = model.predict_proba(x_val)[:, 1]
        threshold = choose_threshold(y_val, scores)
        models[name] = model
        thresholds[name] = threshold
        results[name] = {
            "threshold": threshold,
            "validation_calibration": metrics(y_val, scores, threshold),
            "validation_score_summary": score_summary(scores, y_val),
        }
    duration = perf_counter() - start
    artifact = {
        "schema": "trace_mtba_activation_detector_development.v1",
        "target_model": "meta-llama/Llama-2-7b-chat-hf",
        "target_adapter": "BackdoorLLM/Jailbreak_Llama2-7B_MTBA",
        "base_revision": "f5db02db724555f92da89c216ac04704f23d4590",
        "adapter_revision": "8652ad60f1605605642c99fd680a15ecccf1b522",
        "encoder": encoder, "scaler": scaler, "models": models,
        "thresholds": thresholds, "label_manifest_sha256": labels_sha,
        "train_source_rows": sorted({s for _, _, s in groups["train"]}),
        "validation_source_rows": sorted({s for _, _, s in groups["validation"]}),
        "final_test_evaluated": False,
        "dashboard_deployed": False,
    }
    report = {
        "schema": artifact["schema"],
        "status": "development-only activation detector; final test pending",
        "label_manifest_sha256": labels_sha,
        "label_protocol_sha256": labeled["label_protocol_sha256"],
        "train_rows": len(train), "validation_rows": len(validation),
        "train_labels": dict(Counter(map(str, y_train))),
        "validation_labels": dict(Counter(map(str, y_val))),
        "methods": {"sentence_encoder": ENCODER_REPO,
                    "sentence_encoder_revision": ENCODER_REV,
                    "representation": "prompt_response",
                    "contrastive_projection_epochs": encoder.training_info["epochs"],
                    "generation_scaling": "StandardScaler fit on training only",
                    "classifier": "logistic regression", "C": 1.0,
                    "threshold_policy": "validation F1, deterministic tie break"},
        "views": results,
        "fit_seconds": duration,
        "environment": {"python": platform.python_version(),
                        "numpy": np.__version__, "scikit_learn": sklearn.__version__},
        "final_test_evaluated": False,
        "trigger_B_evaluated": False,
        "dashboard_deployed": False,
        "limitations": [
            "Validation F1 is optimistic because the same validation set selects the threshold.",
            "Clean controls were selected after adapted outcomes were seen; no representative clean-model FPR is estimated.",
            "Some training interactions were collected on Colab and validation on Vast, creating a possible backend batch effect.",
            "Only seen Trigger A and frozen development sources were evaluated; Trigger B and held-out tests remain untouched.",
            "Retrospective human judgments and exclusions limit generalization claims.",
            "Scores are classifier outputs, not calibrated probabilities.",
        ],
    }
    OUT.mkdir()
    model_path = OUT / "detector.joblib"
    joblib.dump(artifact, model_path)
    report["detector_sha256"] = sha(model_path.read_bytes())
    (OUT / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False),
                                     encoding="utf-8")
    print(json.dumps({"train_rows": len(train), "validation_rows": len(validation),
                      "fit_seconds": duration, "views": results}, indent=2))


if __name__ == "__main__":
    fit()
