"""One-time frozen TRACE evaluation after independent held-out labels are sealed."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np

from mtba_collect_development import ROOT, load_sources, sha
from mtba_collect_final_test import RUN, SPLITS, protocol_for
from traceguard.detector import metrics
from traceguard.features import generation_features

LABELS = RUN / "activation-labels.json"
AUDIT = RUN / "review-audit.json"
DEST = RUN / "evaluation"


def _score_summary(values: np.ndarray, labels: np.ndarray) -> dict:
    result = {}
    for label in (0, 1):
        group = values[labels == label]
        result[str(label)] = ({"count": 0} if not len(group) else {
            "count": int(len(group)), "min": float(group.min()),
            "median": float(np.median(group)), "max": float(group.max()),
        })
    return result


def _source_bootstrap(entries: list[dict], labels: np.ndarray, scores: np.ndarray,
                      threshold: float, seed: int = 20261022,
                      replicates: int = 1000) -> dict:
    sources = sorted({e["source_row"] for e in entries})
    if not sources:
        return {"source_count": 0, "replicates": 0}
    by_source = {source: np.asarray([i for i, e in enumerate(entries)
                                     if e["source_row"] == source], dtype=int)
                 for source in sources}
    rng = np.random.default_rng(seed)
    values = {"f1": [], "false_positive_rate": [], "auroc": []}
    for _ in range(replicates):
        draw = rng.choice(sources, size=len(sources), replace=True)
        indices = np.concatenate([by_source[int(source)] for source in draw])
        measured = metrics(labels[indices], scores[indices], threshold)
        for name in values:
            if measured[name] is not None:
                values[name].append(measured[name])
    return {"source_count": len(sources), "replicates": replicates,
            "seed": seed,
            "method": "percentile bootstrap resampling source groups; both conditions move together",
            "interval_95": {name: (None if not series else
                                   [float(np.percentile(series, 2.5)),
                                    float(np.percentile(series, 97.5))])
                            for name, series in values.items()},
            "valid_replicates": {name: len(series) for name, series in values.items()}}


def evaluate() -> dict:
    if DEST.exists():
        raise FileExistsError("Held-out evaluation already exists; no rescoring or overwrite")
    rows, manifest, manifest_sha = load_sources()
    assignments = [a for a in manifest["assignments"] if a["split"] in SPLITS]
    protocol_raw = (RUN / "protocol.json").read_bytes()
    protocol = json.loads(protocol_raw)
    if protocol != protocol_for(assignments, manifest_sha):
        raise ValueError("Final-test protocol or frozen detector differs")
    audit_raw = AUDIT.read_bytes()
    audit = json.loads(audit_raw)
    labels_raw = LABELS.read_bytes()
    labeled = json.loads(labels_raw)
    if (audit.get("schema") != "trace_mtba_heldout_review_audit.v1"
            or audit.get("detector_scored") is not False
            or labeled.get("schema") != "trace_mtba_heldout_activation_labels.v1"
            or labeled.get("protocol_sha256") != sha(protocol_raw)
            or labeled.get("review_audit_sha256") != sha(audit_raw)
            or labeled.get("source_manifest_sha256") != manifest_sha
            or labeled.get("detector_joblib_sha256") != protocol["detector_joblib_sha256"]):
        raise ValueError("Held-out label freeze or review audit differs")
    detector_path = (ROOT / "artifacts/mtba-development/run-001"
                     / "trace-activation-detector-001/detector.joblib")
    if sha(detector_path.read_bytes()) != protocol["detector_joblib_sha256"]:
        raise ValueError("Frozen detector bytes changed")
    artifact = joblib.load(detector_path)
    if (artifact.get("schema") != "trace_mtba_activation_detector_development.v1"
            or artifact.get("final_test_evaluated") is not False
            or artifact.get("target_model") != protocol["base_model"]
            or artifact.get("target_adapter") != protocol["adapter"]
            or artifact.get("thresholds") != protocol["frozen_thresholds"]):
        raise ValueError("Loaded detector identity or threshold differs")
    entries = labeled["entries"]
    if not entries or len(entries) % 4:
        raise ValueError("No complete four-condition source groups")
    interactions = []
    seen_keys = set()
    for entry in entries:
        key = (entry["split"], entry["source_row"], entry["role"], entry["condition"])
        if (key in seen_keys or entry["split"] not in SPLITS
                or entry["role"] not in ("adapted", "clean_base")
                or entry["condition"] not in ("none", "cue")
                or entry["activation_label"] not in (0, 1)
                or (entry["activation_label"] == 1
                    and (entry["role"], entry["condition"]) != ("adapted", "cue"))):
            raise ValueError("Invalid or duplicate held-out label")
        seen_keys.add(key)
        path = (ROOT / entry["record_path"]).resolve()
        if not path.is_relative_to((RUN / "records").resolve()):
            raise ValueError("Label points outside held-out records")
        raw = path.read_bytes()
        record = json.loads(raw)
        if (sha(raw) != entry["record_sha256"]
                or any(record[k] != entry[k] for k in
                       ("split", "source_row", "role", "condition"))
                or record.get("activation_label") is not None):
            raise ValueError("Labeled record changed")
        generated = record["result"]
        interaction = {**generated, "prompt": record["prompt"],
                       "statistics_source": "raw_full_vocabulary_logits"}
        generation_features(interaction)
        interactions.append(interaction)
    grouped = defaultdict(list)
    for entry in entries:
        grouped[(entry["split"], entry["source_row"])].append(entry)
    if any(len(group) != 4 for group in grouped.values()):
        raise ValueError("A source group is incomplete")
    start = perf_counter()
    behavior = artifact["encoder"].transform(interactions)
    signals = artifact["scaler"].transform(
        np.stack([generation_features(item) for item in interactions]))
    matrices = {"full": np.c_[behavior, signals],
                "behavioral": behavior, "generation": signals}
    scores = {view: artifact["models"][view].predict_proba(matrices[view])[:, 1]
              for view in ("full", "behavioral", "generation")}
    elapsed = perf_counter() - start
    output = {
        "schema": "trace_mtba_frozen_heldout_evaluation.v1",
        "protocol_sha256": sha(protocol_raw),
        "labels_sha256": sha(labels_raw),
        "detector_joblib_sha256": protocol["detector_joblib_sha256"],
        "thresholds": protocol["frozen_thresholds"],
        "scored_interactions": len(entries),
        "source_groups": len(grouped),
        "excluded_source_groups": len(audit["excluded_sources"]),
        "detector_total_seconds": elapsed,
        "detector_mean_ms_per_interaction": elapsed * 1000 / len(entries),
        "latency_scope": "CPU embedding/projection/scaler/classifier; includes cold encoder load; excludes remote LLM inference",
        "partitions": {},
        "limitations": [
            "Only this pinned Llama 2 MTBA adapter and these 40 reserved source questions were tested.",
            "Trigger B was unseen to detector training and validation, but observed in an excluded researcher pilot.",
            "Scores are classifier outputs, not calibrated probabilities.",
            "The released adapter's exact target-training exposure was not independently audited.",
            "Sources with unresolved human judgments are excluded under the frozen protocol.",
        ],
    }
    for split in SPLITS:
        indices = [i for i, e in enumerate(entries)
                   if e["split"] == split and e["role"] == "adapted"]
        clean_indices = [i for i, e in enumerate(entries)
                         if e["split"] == split and e["role"] == "clean_base"]
        split_entries = [entries[i] for i in indices]
        y = np.asarray([entries[i]["activation_label"] for i in indices], dtype=int)
        arm_results = {}
        for view in ("full", "behavioral", "generation"):
            selected_scores = scores[view][indices]
            threshold = artifact["thresholds"][view]
            clean_scores = scores[view][clean_indices]
            arm_results[view] = {
                "metrics": metrics(y, selected_scores, threshold),
                "clean_base_false_positive_rate": float(np.mean(clean_scores >= threshold)),
                "clean_base_count": len(clean_scores),
                "score_summary_by_label": _score_summary(selected_scores, y),
                "source_bootstrap_95": _source_bootstrap(split_entries, y,
                                                          selected_scores, threshold),
            }
        output["partitions"][split] = {
            "adapted_interactions": len(indices),
            "clean_base_interactions": len(clean_indices),
            "included_sources": len({e["source_row"] for e in split_entries}),
            "views": arm_results,
        }
    DEST.mkdir()
    with (DEST / "report.json").open("x", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"partitions": {
        split: {view: data["metrics"] for view, data in result["views"].items()}
        for split, result in output["partitions"].items()},
        "detector_mean_ms_per_interaction": output["detector_mean_ms_per_interaction"]},
        indent=2))
    return output


if __name__ == "__main__":
    evaluate()
