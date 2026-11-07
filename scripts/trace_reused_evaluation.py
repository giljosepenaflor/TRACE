"""Retrospective accuracy on reused StrongREJECT data and matched validation efficiency.

Explicitly authorized dataset reuse; never calls a target LLM, fits a model,
changes labels/thresholds, overwrites original results, or promotes a detector.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
from time import perf_counter, perf_counter_ns

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "4"
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import trace_consensus_intake as common
from scripts import trace_consensus_behavioral as shared
from scripts import trace_consensus_train as training
from scripts import trace_behavioral_experiment as representation
from scripts import trace_strongreject_consensus as test
from scripts import trace_error_efficiency as efficiency
from traceguard import reused_comparison as core

RUN = ROOT / "artifacts/trace-behavioral-v4/reused-test-comparison-001"
REPR_RUN = representation.DEFAULT_RUN
OLD_RUN = representation.ORIGINAL_RUN
LABELS = test.RUN / "labels"
OLD_EVALUATION = test.RUN / "evaluation"
SETTINGS = {"processes_per_method": 3, "warmup_passes": 1, "measured_passes": 2,
            "validation_records": 100, "cpu_threads": 4, "torch_interop_threads": 1,
            "batch_size": 1, "seed": 20261106, "bootstrap_repetitions": 2000,
            "score_tolerance": 1e-5, "percentile_method": "linear"}
TEST_PINS = {
    "evaluation/manifest.json": "2cf27d8e0c754b3772162eef205dcae39fde53bf8bb99f1360489391cc1214e2",
    "evaluation/predictions.jsonl": "3bf655fea5247731ff21b0057eb0057da64d13d4156121c782d91690cfde6d25",
    "evaluation/report.json": "3fc22d32b3e8acfb0bc24d0e085b44bd472c503fb90f6b784b06e5dda9da7a35",
    "labels/manifest.json": "52838a5dccfd79fd953b7a0734dc998ce37359536323e366b5278b238fa1e341",
}


def read(path):
    return json.loads(path.read_bytes())


def lines(path):
    return [json.loads(row) for row in path.read_bytes().splitlines() if row.strip()]


def sha(path):
    return common.sha(path.read_bytes())


def stamp():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    shared.write_new(path, common.encoded(value))


def relative(path):
    return path.relative_to(ROOT).as_posix()


def code_hashes():
    files = (Path(__file__).resolve(), Path(core.__file__), ROOT / "tests/test_reused_comparison.py",
             ROOT / "traceguard/runtime.py", ROOT / "traceguard/behavior.py", ROOT / "traceguard/features.py",
             ROOT / "traceguard/representation_experiment.py", ROOT / "traceguard/detector.py")
    return {relative(p): sha(p) for p in files}


def protect_inputs():
    paths = list(REPR_RUN.rglob("*")) + list(OLD_RUN.rglob("*")) + list(test.RUN.rglob("*"))
    paths += [ROOT / "traceguard/mtba_live_detector.py",
              ROOT / "artifacts/mtba-harmbench/run-001/pipeline-001/main-detector.json"]
    return {relative(p): sha(p) for p in paths if p.is_file()}


def check_protocol(light=False):
    protocol = read(RUN / "protocol.json")
    common.require(protocol["settings"] == SETTINGS and protocol["implementation_sha256"] == code_hashes(),
                   "Comparison implementation or settings changed after freeze.")
    common.require(protocol["design"] == "retrospective_reused_test" and protocol["not_a_fresh_holdout"] is True,
                   "Reused-data design declaration changed.")
    if not light:
        for name, expected in protocol["input_sha256"].items():
            common.checked_read(ROOT / name, expected)
        for name, expected in protocol["encoder_file_sha256"].items():
            common.checked_read(shared.ENCODER_PATH / name, expected)
    for name, expected in protocol["state_sha256"].items():
        common.checked_read(RUN / "states" / name, expected)
    common.checked_read(RUN / "validation-records.json", protocol["validation_records_sha256"])
    return protocol


def record_for(row):
    path = Path(row["raw_path"])
    path = path if path.is_absolute() else ROOT / path
    raw = json.loads(common.checked_read(path, row["raw_sha256"]))
    server = raw["server_result"]
    common.require(raw["source_id"] == row["source_id"] and raw["condition_id"] == row["condition_id"], "Raw source identity changed.")
    common.require(common.text_sha(server["response"]) == row["response_sha256"], "Reviewed response text changed.")
    common.require(server["generation_configuration"]["do_sample"] is False
                   and server["generation_configuration"]["max_new_tokens"] == 256, "Generation declarations changed.")
    signals = {key: server[key] for key in ("generated_token_ids", "token_confidences", "token_entropies")}
    signals["statistics_source"] = "full_vocabulary_generation_scores"
    return {"sample_id": row["sample_id"], "source_id": row["source_id"], "split": row["split"],
            "condition_id": row["condition_id"], "activation_label": row["activation_label"],
            "prompt": raw["executed_prompt"], "response": server["response"], "signals": signals}


def make_states(original, candidate):
    states = {}
    for method, definition in core.METHODS.items():
        view = definition["original_view"]
        if view:
            model, threshold = original["models"][view], original["thresholds"][view]
            projection = original["encoder"].projection if definition["projected"] else None
            scaler = original["scaler"] if definition["view"] != "behavioral" else None
        else:
            selected = candidate["models"][definition["candidate_id"]]
            model, threshold = selected["model"], selected["threshold"]
            projection = None
            scaler = candidate["generation_scaler"] if definition["view"] != "behavioral" else None
        states[method] = {"model": model, "threshold": threshold, "projection": projection, "scaler": scaler,
                          "view": definition["view"], "method": method, "definition": definition,
                          "encoder_revision": shared.ENCODER_REV, "development_frozen": True}
    return states


def prepare():
    import joblib
    common.require(not RUN.exists(), "Reused-data run exists; refuse overwrite.")
    representation.check_protocol(REPR_RUN)
    test.verify_evaluation()
    for name, expected in TEST_PINS.items():
        common.checked_read(test.RUN / name, expected)
    repr_freeze = read(REPR_RUN / "results/freeze.json")
    for name, expected in repr_freeze["output_sha256"].items():
        common.checked_read(REPR_RUN / "results" / name, expected)
    labels = lines(LABELS / "activation-labels.jsonl")
    development = read(REPR_RUN / "protocol.json")
    forbidden = {r["source_id"] for rows in development["membership"].values() for r in rows}
    core.validate_test_membership(labels, forbidden)
    common.require(Counter(r["split"] for r in labels) == {"test_seen": 220, "test_unseen": 256}, "Reused-test counts differ from sealed labels.")
    original = joblib.load(OLD_RUN / "model/detector.joblib")
    candidate = joblib.load(REPR_RUN / "results/candidates.joblib")
    states = make_states(original, candidate)
    candidate_report = read(REPR_RUN / "results/report.json")
    for method, state in states.items():
        expected = candidate_report["validation_results"][core.METHODS[method]["candidate_id"]]["threshold"]
        common.require(state["threshold"] == expected, "A comparator threshold differs from the pre-existing definition.")
    validation = development["membership"]["validation"]
    efficiency.validate_benchmark_membership(validation)
    records = [record_for(row) for row in validation]
    original_validation = {row["sample_id"]: row for row in lines(OLD_RUN / "model/validation-scores.jsonl")}
    new_validation = {(row["configuration_id"], row["sample_id"]): row for row in lines(REPR_RUN / "results/validation-scores.jsonl")}
    for record in records:
        record["reference_scores"], record["reference_predictions"] = {}, {}
        for method, definition in core.METHODS.items():
            if definition["original_view"]:
                ref = original_validation[record["sample_id"]]
                score, decision = ref["scores"][definition["original_view"]], ref["predictions"][definition["original_view"]]
            else:
                ref = new_validation[(definition["candidate_id"], record["sample_id"])]
                score, decision = ref["score"], ref["prediction"]
            record["reference_scores"][method], record["reference_predictions"][method] = score, decision
    protected = protect_inputs()
    RUN.mkdir(parents=True, exist_ok=False)
    (RUN / "states").mkdir()
    for method, state in states.items():
        buffer = io.BytesIO()
        joblib.dump(state, buffer, compress=3)
        shared.write_new(RUN / "states" / f"{method}.joblib", buffer.getvalue())
    write(RUN / "validation-records.json", records)
    methods = list(core.METHODS)
    protocol = {"schema": "trace.reused_comparison.protocol.v1", "created_at_utc": stamp(),
        "user_instruction": "proceed to the next step, just reuse the existing dataset",
        "design": "retrospective_reused_test", "not_a_fresh_holdout": True,
        "earlier_test_errors_informed_representation_design": True,
        "interpretation": "Frozen-candidate retrospective comparison on previously inspected data, not independent confirmation.",
        "settings": SETTINGS, "methods": core.METHODS, "thresholds": {m: s["threshold"] for m, s in states.items()},
        "test_label_membership": labels, "validation_membership": validation,
        "test_counts": {"test_seen": 220, "test_unseen": 256}, "existing_excluded_source_groups": 75,
        "implementation_sha256": code_hashes(), "input_sha256": protected,
        "encoder_file_sha256": development["encoder_file_sha256"],
        "state_sha256": {p.name: sha(p) for p in sorted((RUN / "states").iterdir())},
        "validation_records_sha256": sha(RUN / "validation-records.json"),
        "benchmark_process_orders": [methods[i * 2:] + methods[:i * 2] for i in range(3)],
        "warm_timing_boundary": "In-memory record to score/decision; required numerical features, tokenization/encoding, projection/scaling/classification included.",
        "warm_exclusions": ["target generation", "generation-score collection", "network/UI", "disk reads and hash checks", "initial imports/model loading"],
        "cold_boundary": "State loading and first inference after shared numerical/workflow imports; excludes interpreter launch and preflight/input reads.",
        "memory_boundary": "Whole worker process resident and lifetime peak; parent and target LLM excluded. Parent verifies encoder hashes before workers.",
        "selection_after_reused_test_allowed": False, "refitting_allowed": False,
        "threshold_selection_allowed": False, "new_review_labels_allowed": False, "target_model_calls_allowed": False,
        "dashboard_promotion_allowed": False, "fresh_holdout_source_gate_overridden": False,
        "original_results_replaced": False,
        "provenance_qualifications": development["admission"]}
    write(RUN / "protocol.json", protocol)
    return {"prepared": True, "design": protocol["design"], "comparators": len(methods), "test_examples": len(labels),
            "benchmark_validation_examples": len(validation), "benchmark_processes": 18}


def evaluate():
    import joblib
    import numpy as np
    import torch
    from traceguard.behavior import BehavioralEncoder
    from traceguard.features import generation_features
    protocol = check_protocol()
    output = RUN / "evaluation"
    common.require(not output.exists(), "Retrospective evaluation exists; refuse repeat or overwrite.")
    output.mkdir(exist_ok=False)
    write(output / "started.json", {"started_at_utc": stamp(), "protocol_sha256": sha(RUN / "protocol.json")})
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    states = {m: joblib.load(RUN / "states" / f"{m}.joblib") for m in core.METHODS}
    labels = protocol["test_label_membership"]
    records = [record_for(row) for row in labels]
    start = perf_counter()
    encoder = BehavioralEncoder(model_name=str(shared.ENCODER_PATH), local_only=True)
    print(f"Encoding {len(records)} existing reviewed test responses for the frozen retrospective comparison", flush=True)
    base = encoder.encode_base(records)
    signals = np.stack([generation_features(row["signals"]) for row in records])
    common.require(base.shape == (476, 384), "Unexpected test embedding dimensions.")
    predictions = core.score_candidates(base, signals, labels, lines(OLD_EVALUATION / "predictions.jsonl"), states)
    report = {"schema": "trace.reused_comparison.accuracy.v1", "design": "retrospective_reused_test",
              "completed_at_utc": stamp(), "not_a_fresh_holdout": True,
              "results": core.summarize(predictions, protocol["thresholds"]),
              "paired_source_bootstrap": {split: core.paired_intervals([r for r in predictions if r["split"] == split],
                   SETTINGS["bootstrap_repetitions"], SETTINGS["seed"]) for split in ("test_seen", "test_unseen")},
              "thresholds_unchanged": protocol["thresholds"], "existing_excluded_source_groups": 75,
              "original_baseline_scores_reused_without_rescoring": True, "new_frozen_classifiers_scored": 3,
              "fit_calls": 0, "threshold_selection_calls": 0, "target_model_calls": 0, "new_labels": 0,
              "dashboard_promoted": False, "elapsed_batch_encoding_scoring_and_analysis_seconds": perf_counter() - start,
              "protocol_sha256": sha(RUN / "protocol.json"),
              "limitations": ["Same 476 admitted examples and 75 source-group exclusions as the initial test.",
                "Earlier test errors informed model development; adaptive reuse prevents an independent generalization claim.",
                "No model or threshold is selected using this retrospective comparison.",
                "Bootstrap intervals are post-hoc, unadjusted and conditional on existing groups/labels.",
                "Single Llama 2 target and independent prompts; no cross-model or validated chat claim.",
                "Human consensus and historical server linkage retain their original researcher-attested limitations."]}
    shared.write_new(output / "predictions.jsonl", common.jsonlines(predictions))
    buffer = io.BytesIO()
    np.savez_compressed(buffer, base=base, generation=signals, sample_ids=np.asarray([r["sample_id"] for r in labels]))
    shared.write_new(output / "features.npz", buffer.getvalue())
    write(output / "report.json", report)
    write(output / "manifest.json", {"schema": "trace.reused_comparison.accuracy_seal.v1", "protocol_sha256": sha(RUN / "protocol.json"),
          "output_sha256": {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()}})
    check_protocol()
    return {"retrospective_evaluation_complete": True, "f1": {s: {m: v["f1"] for m, v in r.items()} for s, r in report["results"].items()},
            "fresh_holdout": False, "original_results_replaced": False}


def worker(method, replicate):
    import joblib
    from traceguard.runtime import process_memory, system_memory
    protocol = check_protocol(light=True)
    common.require(method in core.METHODS and replicate in (0, 1, 2), "Unknown benchmark worker.")
    records = read(RUN / "validation-records.json")
    baseline = process_memory()
    start = perf_counter_ns()
    state = joblib.load(RUN / "states" / f"{method}.joblib")
    encoder = None
    if state["view"] != "generation":
        import torch
        from traceguard.behavior import BehavioralEncoder
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        encoder = BehavioralEncoder(model_name=str(shared.ENCODER_PATH), local_only=True)
    from threadpoolctl import threadpool_info, threadpool_limits
    limiter = threadpool_limits(limits=4)
    loaded = perf_counter_ns()
    core.predict_one(state, records[0], encoder)
    first = perf_counter_ns()
    for record in records:
        core.predict_one(state, record, encoder)
    warm_memory = process_memory()
    observations, differences, errors = [], [], []
    rng = random.Random(SETTINGS["seed"] + replicate)
    for pass_number in range(SETTINGS["measured_passes"]):
        order = list(range(len(records)))
        rng.shuffle(order)
        for index in order:
            record = records[index]
            t0 = perf_counter_ns()
            score, decision = core.predict_one(state, record, encoder)
            elapsed = (perf_counter_ns() - t0) / 1e6
            reference = record["reference_scores"][method]
            error = abs(score - reference)
            common.require(error <= SETTINGS["score_tolerance"], "Benchmark score differs from its frozen validation reference.")
            if decision != record["reference_predictions"][method]:
                common.require(abs(reference - state["threshold"]) <= SETTINGS["score_tolerance"], "Decision difference away from numerical boundary.")
                differences.append(record["sample_id"])
            errors.append(error)
            observations.append({"sample_id": record["sample_id"], "pass": pass_number, "latency_ms": elapsed})
    measured_memory = process_memory()
    pools = threadpool_info()
    limiter.restore_original_limits()
    value = {"schema": "trace.reused_comparison.benchmark_worker.v1", "method": method, "replicate": replicate,
             "protocol_sha256": sha(RUN / "protocol.json"), "partition": "validation", "validation_records": 100,
             "observations": observations, "latency_ms": efficiency.statistics([r["latency_ms"] for r in observations]),
             "cold_post_import_setup_first_ms": (first - start) / 1e6, "cold_first_call_ms": (first - loaded) / 1e6,
             "memory": {"baseline": baseline, "warm": warm_memory, "measured": measured_memory},
             "score_reference_max_abs_difference": max(errors), "boundary_difference_occurrences": len(differences),
             "boundary_difference_sample_ids": sorted(set(differences)), "cpu": efficiency.cpu_name(),
             "logical_cpu_count": os.cpu_count(), "system_memory": system_memory(),
             "threadpool_info": pools, "torch_threads": 4 if encoder else None,
             "torch_interop_threads": 1 if encoder else None,
             "python": platform.python_version(), "platform": platform.platform(),
             "test_rows_scored": 0, "fit_calls": 0, "threshold_selection_calls": 0, "target_model_calls": 0}
    write(RUN / "benchmark" / f"{method}-{replicate}.json", value)
    print(json.dumps({"worker_completed": method, "replicate": replicate, "median_ms": value["latency_ms"]["median"]}), flush=True)


def summarize_workers(workers, protocol):
    result = {}
    for method in core.METHODS:
        subset = [w for w in workers if w["method"] == method]
        common.require(len(subset) == 3 and {w["replicate"] for w in subset} == {0, 1, 2}, "Missing or repeated worker.")
        for item in subset:
            common.require(len(item["observations"]) == 200 and item["partition"] == "validation", "Incorrect worker coverage.")
        result[method] = {"warm_latency_ms": efficiency.statistics([r["latency_ms"] for w in subset for r in w["observations"]]),
            "per_process_median_ms": [w["latency_ms"]["median"] for w in subset],
            "cold_post_import_setup_first_ms": efficiency.statistics([w["cold_post_import_setup_first_ms"] for w in subset]),
            "warm_resident_memory_bytes": [w["memory"]["measured"]["working_set_bytes"] for w in subset],
            "median_warm_resident_memory_bytes": float(__import__("numpy").median([w["memory"]["measured"]["working_set_bytes"] for w in subset])),
            "maximum_peak_working_set_bytes": max(w["memory"]["measured"]["peak_working_set_bytes"] for w in subset),
            "maximum_score_reference_difference": max(w["score_reference_max_abs_difference"] for w in subset),
            "boundary_difference_occurrences": sum(w["boundary_difference_occurrences"] for w in subset),
            "boundary_difference_sample_ids": sorted({sid for w in subset for sid in w["boundary_difference_sample_ids"]}),
            "state_file_bytes": (RUN / "states" / f"{method}.joblib").stat().st_size,
            "required_encoder_files_bytes": 0 if method == "generation" else sum((shared.ENCODER_PATH / name).stat().st_size for name in protocol["encoder_file_sha256"])}
    return result


def benchmark():
    protocol = check_protocol()
    folder = RUN / "benchmark"
    common.require(not folder.exists(), "Benchmark already started; refuse automatic repeat.")
    folder.mkdir(exist_ok=False)
    write(folder / "started.json", {"started_at_utc": stamp(), "protocol_sha256": sha(RUN / "protocol.json")})
    workers = []
    for replicate, order in enumerate(protocol["benchmark_process_orders"]):
        for method in order:
            subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), "worker", "--method", method,
                            "--replicate", str(replicate)], cwd=ROOT, env=dict(os.environ), check=True,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            workers.append(read(folder / f"{method}-{replicate}.json"))
    result = {"schema": "trace.reused_comparison.efficiency.v1", "completed_at_utc": stamp(),
              "methods": summarize_workers(workers, protocol), "processes": len(workers), "measurements_per_method": 600,
              "cpu": workers[0]["cpu"], "logical_cpu_count": workers[0]["logical_cpu_count"],
              "python": workers[0]["python"], "platform": workers[0]["platform"],
              "available_memory_bytes_range": [min(w["system_memory"]["available_bytes"] for w in workers),
                                                max(w["system_memory"]["available_bytes"] for w in workers)],
              "protocol_sha256": sha(RUN / "protocol.json"), "benchmark_partition": "validation",
              "target_model_calls": 0, "fit_calls": 0, "threshold_selection_calls": 0, "test_rows_scored": 0,
              "limitations": ["Local detector overhead only; target generation/statistics collection, network, and UI are excluded.",
                "Warm measurements include all required feature extraction and classifier computation for each method.",
                "Background workloads, power state, and file cache are uncontrolled. Fresh workers are not cold-machine caches.",
                "Cold interval begins after shared numerical/workflow imports; it is not total application startup.",
                "Working sets include the worker runtime; lifetime peaks are not intrinsic model parameter memory.",
                "State plus external encoder file size excludes Python/libraries, target weights, source code, data and reports.",
                "Current matched measurements do not imply optimization relative to older benchmarks with different boundaries and machine conditions."]}
    check_protocol()
    write(folder / "summary.json", result)
    write(folder / "manifest.json", {"schema": "trace.reused_comparison.benchmark_seal.v1", "protocol_sha256": sha(RUN / "protocol.json"),
          "output_sha256": {p.name: sha(p) for p in sorted(folder.iterdir()) if p.is_file()}})
    return {"benchmark_complete": True, "processes": len(workers), "median_ms": {m: v["warm_latency_ms"]["median"] for m, v in result["methods"].items()}}


def verify():
    import joblib
    import numpy as np
    protocol = check_protocol()
    for folder_name in ("evaluation", "benchmark"):
        folder = RUN / folder_name
        manifest = read(folder / "manifest.json")
        common.require(manifest["protocol_sha256"] == sha(RUN / "protocol.json"), "Output protocol binding changed.")
        for name, expected in manifest["output_sha256"].items():
            common.require(Path(name).name == name, "Unsafe output filename.")
            common.checked_read(folder / name, expected)
    predictions = lines(RUN / "evaluation/predictions.jsonl")
    labels = protocol["test_label_membership"]
    core.validate_test_membership(labels)
    common.require(len(predictions) == len(labels) == 476, "Reused prediction membership differs.")
    for row, label in zip(predictions, labels):
        common.require(all(row[k] == label[k] for k in ("sample_id", "source_id", "split", "condition_id", "activation_label")), "Saved label membership changed.")
    report = read(RUN / "evaluation/report.json")
    common.require(report["results"] == core.summarize(predictions, protocol["thresholds"]), "Retrospective metrics differ.")
    for split in ("test_seen", "test_unseen"):
        expected = core.paired_intervals([r for r in predictions if r["split"] == split], SETTINGS["bootstrap_repetitions"], SETTINGS["seed"])
        common.require(report["paired_source_bootstrap"][split] == expected, "Paired bootstrap differs.")
    features = np.load(RUN / "evaluation/features.npz", allow_pickle=False)
    common.require(features["sample_ids"].tolist() == [r["sample_id"] for r in labels], "Saved feature order differs.")
    states = {m: joblib.load(RUN / "states" / f"{m}.joblib") for m in core.METHODS}
    rebuilt = core.score_candidates(features["base"], features["generation"], labels, lines(OLD_EVALUATION / "predictions.jsonl"), states)
    common.require(rebuilt == predictions, "Saved cached-feature classifier scores differ.")
    workers = [read(RUN / "benchmark" / f"{method}-{i}.json") for i in range(3) for method in core.METHODS]
    expected_ids = {r["sample_id"] for r in protocol["validation_membership"]}
    for w in workers:
        common.require(w["protocol_sha256"] == sha(RUN / "protocol.json"), "Worker protocol differs.")
        for pass_number in (0, 1):
            ids = [r["sample_id"] for r in w["observations"] if r["pass"] == pass_number]
            common.require(len(ids) == len(set(ids)) == 100 and set(ids) == expected_ids, "Worker validation pass differs.")
        common.require(w["test_rows_scored"] == w["fit_calls"] == w["threshold_selection_calls"] == w["target_model_calls"] == 0, "Benchmark scope changed.")
        common.require(w["latency_ms"] == efficiency.statistics([r["latency_ms"] for r in w["observations"]]), "Worker timing summary differs.")
    summary = read(RUN / "benchmark/summary.json")
    common.require(summary["methods"] == summarize_workers(workers, protocol), "Benchmark aggregate differs.")
    test.verify_evaluation()
    value = {"schema": "trace.reused_comparison.verification.v1", "verified": True,
             "original_input_artifacts_and_predictions_unchanged": True,
             "cached_feature_scores_and_frozen_thresholds_reconstructed": True,
             "metrics_and_paired_source_bootstrap_reconstructed": True,
             "validation_benchmark_coverage_and_statistics_verified": True,
             "reused_test_examples": 476, "new_frozen_classifiers_scored": 3,
             "benchmark_processes": 18, "measurements_per_method": 600,
             "retrospective_not_fresh_holdout": True, "fit_calls": 0, "target_model_calls": 0,
             "threshold_selection_calls": 0, "dashboard_promoted": False}
    path = RUN / "verification.json"
    if path.exists():
        common.require(read(path) == value, "Existing verification differs.")
    else:
        write(path, value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "evaluate", "benchmark", "worker", "verify"))
    parser.add_argument("--method", choices=tuple(core.METHODS))
    parser.add_argument("--replicate", type=int)
    args = parser.parse_args()
    if args.command == "worker":
        worker(args.method, args.replicate)
    else:
        value = {"prepare": prepare, "evaluate": evaluate, "benchmark": benchmark, "verify": verify}[args.command]()
        print(json.dumps(value, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
