"""Frozen paired-control development comparison; no automatic deployment."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "4"
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import trace_end_to_end as previous
from traceguard import paired_control as core

RUN = ROOT / "artifacts/trace-behavioral-v4/paired-control-001"
read, lines, sha, write, stamp = previous.read, previous.lines, previous.sha, previous.write, previous.stamp
require = core.require
SETTINGS = {**core.SETTINGS, "score_tolerance": 1e-5, "cpu_threads": 4,
            "processes_per_method": 3, "benchmark_records": 100, "warmup_passes": 1, "measured_passes": 2}


def codes():
    paths = (Path(__file__).resolve(), Path(core.__file__), ROOT / "tests/test_paired_control.py",
             ROOT / "traceguard/features.py", ROOT / "traceguard/behavior.py",
             ROOT / "traceguard/end_to_end.py", ROOT / "docs/TRACE_PAIRED_IMPROVEMENT_PLAN.md")
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}


def check(light=False):
    p = read(RUN / "protocol.json")
    require(p["implementation_sha256"] == codes() and p["settings"] == SETTINGS, "Frozen paired implementation/settings changed.")
    if not light:
        for name, expected in p["protected_input_sha256"].items():
            require(not (ROOT / name).exists() if expected is None else sha(ROOT / name) == expected, "Protected input changed: " + name)
        for name, expected in p["encoder_file_sha256"].items():
            require(sha(previous.shared.ENCODER_PATH / name) == expected, "Frozen encoder changed.")
    return p


def seal(folder):
    write(folder / "manifest.json", {"protocol_sha256": sha(RUN / "protocol.json"),
        "output_sha256": {p.relative_to(folder).as_posix(): sha(p) for p in sorted(folder.rglob("*")) if p.is_file()}})


def check_seal(folder):
    m = read(folder / "manifest.json")
    require(m["protocol_sha256"] == sha(RUN / "protocol.json"), "Manifest protocol differs.")
    for name, expected in m["output_sha256"].items():
        require(not Path(name).is_absolute() and ".." not in Path(name).parts and sha(folder / name) == expected, "Sealed output changed.")


def prepare():
    require(not RUN.exists(), "Paired run exists; refuse overwrite.")
    old = previous.check_protocol()
    for folder in ("training", "evaluation", "benchmark"):
        previous.check_seal(previous.RUN / folder)
    dev = read(previous.audit.REPR / "protocol.json")
    rows = old["membership"]["train"]
    folds = core.dataset_folds(rows, old["groups"]["train"])
    protected = previous.protected_inputs()
    for path in previous.RUN.rglob("*"):
        if path.is_file():
            protected[path.relative_to(ROOT).as_posix()] = sha(path)
    RUN.mkdir(exist_ok=False)
    write(RUN / "protocol.json", {"schema": "trace.paired_control.protocol.v1", "created_at_utc": stamp(),
        "authorization": "please still improve the performance of trace; bounded paired-control development experiment",
        "settings": SETTINGS, "methods": list(core.METHODS), "membership": old["membership"],
        "source_folds": dev["folds"], "dataset_folds": folds, "groups": old["groups"],
        "protected_input_sha256": protected, "implementation_sha256": codes(),
        "encoder_file_sha256": old["encoder_file_sha256"], "classifier_fits_planned": 48,
        "selection": "worst held-dataset AP, mean held-dataset AP, mean source-fold AP; standard first on exact tie",
        "test_gate": "paired Full selected, worst-dataset AP improves standard Full, validation F1 beats standard Full and all four single views",
        "new_activation_labels_allowed": False, "encoder_fine_tuning_allowed": False,
        "dashboard_promotion_allowed": False, "target_model_calls_allowed": False,
        "new_test_scoring_allowed_only_if_development_gate_passes": True,
        "test_is_initial_retrospective_if_run": True, "same_single_interaction_inputs_for_inference": True,
        "benchmark_orders": [list(core.METHODS[i*2:] + core.METHODS[:i*2]) for i in range(3)]})
    return {"prepared": True, "source_folds": 5, "dataset_folds": 2, "fits_planned": 48,
            "discordant_training_pairs": len(core.pairs(rows)[0]), "equal_training_pairs": len(core.pairs(rows)[1])}


def rows_for(membership, scores, cutoffs):
    return [{**{k: r[k] for k in ("sample_id", "source_id", "source", "split", "condition_id", "activation_label") if k in r},
             "scores": {m: float(scores[m][i]) for m in core.METHODS},
             "predictions": {policy: {m: int(scores[m][i] >= cutoffs[m][policy]) for m in core.METHODS} for policy in previous.core.POLICIES}}
            for i, r in enumerate(membership)]


def summarize(rows):
    import numpy as np
    result = {}
    for policy in previous.core.POLICIES:
        result[policy] = {}
        y = [r["activation_label"] for r in rows]
        for m in core.METHODS:
            result[policy][m] = previous.core.score_metrics(y, [r["scores"][m] for r in rows], .5)
            binary = previous.core.score_metrics(y, np.array([r["predictions"][policy][m] for r in rows], dtype=float), .5)
            for key in ("accuracy", "f1", "precision", "recall", "false_positive_rate", "confusion_matrix", "count"):
                result[policy][m][key] = binary[key]
    return result


def slices(rows):
    return {source: summarize([r for r in rows if r["source"] == source]) for source in sorted({r["source"] for r in rows})}


def bootstrap(rows, selected):
    import numpy as np
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["source_id"]].append(r)
    groups = [grouped[k] for k in sorted(grouped)]
    require(all(len(g) == 2 and {r["condition_id"] for r in g} == {1, 2} for g in groups), "Bootstrap requires whole source pairs.")
    index = np.random.default_rng(SETTINGS["seed"]).integers(0, len(groups), (SETTINGS["bootstrap_repetitions"], len(groups)))
    y = np.array([[r["activation_label"] for r in g] for g in groups])[index].reshape(len(index), -1)
    samples = {}
    for m in core.METHODS:
        pred = np.array([[r["predictions"]["max_f1"][m] for r in g] for g in groups])[index].reshape(len(index), -1)
        tp = ((y == 1) & (pred == 1)).sum(axis=1)
        den = 2 * tp + ((y == 0) & (pred == 1)).sum(axis=1) + ((y == 1) & (pred == 0)).sum(axis=1)
        samples[m] = np.divide(2 * tp, den, out=np.zeros(len(index)), where=den != 0)
    point = summarize(rows)["max_f1"]
    return {selected + "_minus_" + ref: {"f1_difference": point[selected]["f1"] - point[ref]["f1"],
            "percentile_95_interval": np.percentile(samples[selected] - samples[ref], [2.5, 97.5]).tolist(),
            "source_groups": len(groups), "resamples": len(index), "seed": SETTINGS["seed"]}
            for ref in core.METHODS if ref != selected}


def train():
    import joblib
    import numpy as np
    from time import perf_counter
    p = check()
    folder = RUN / "development"
    require(not folder.exists(), "Paired development exists; refuse refit.")
    folder.mkdir()
    write(folder / "started.json", {"started_at_utc": stamp()})
    start = perf_counter()
    data = previous.arrays()
    rows = p["membership"]["train"]
    fit_bundles, fold_results, prediction_sets, calibrations, diagnostic_sets = {}, {}, {}, {}, {}
    for regime, folds in (("source", p["source_folds"]), ("dataset", p["dataset_folds"])):
        all_rows, measured, calibration = [], [], []
        diagnostic_sets[regime] = {}
        coverage = np.zeros(len(rows), dtype=int)
        for number, fold in enumerate(folds):
            f, h = np.array(fold["fit_indices"]), np.array(fold["held_indices"])
            core.partition_check(rows, f, h, p["groups"]["train"])
            models, scores, thresholds = {}, {}, {}
            for m in core.METHODS:
                state = core.fit(m, data["train"]["base"][f], data["train"]["generation"][f], [rows[i] for i in f])
                models[m] = state
                fitting = core.predict(state, data["train"]["base"][f], data["train"]["generation"][f])
                thresholds[m] = previous.core.thresholds([rows[i]["activation_label"] for i in f], fitting)
                scores[m] = core.predict(state, data["train"]["base"][h], data["train"]["generation"][h])
            current = rows_for([rows[i] for i in h], scores, thresholds)
            lookup = {r["sample_id"]: i for i, r in enumerate(rows)}
            for row in current:
                row["index"] = lookup[row["sample_id"]]
                row["fold"] = number
            all_rows.extend(current)
            measured.append(summarize(current)["max_f1"])
            calibration.append(thresholds)
            coverage[h] += 1
            fit_bundles[f"{regime}-{number}"] = models
            diagnostic_sets[regime][str(number)] = {m: core.pair_diagnostics([rows[i] for i in h],
                core.features(m, data["train"]["base"][h], data["train"]["generation"][h], models[m]["scaler"]) @ models[m]["weights"] + models[m]["bias"])
                for m in core.METHODS}
            print(json.dumps({"completed_regime": regime, "fold": number, "held_rows": len(h)}), flush=True)
        require(np.all(coverage == 1), "Development held prediction coverage differs.")
        prediction_sets[regime] = sorted(all_rows, key=lambda r: r["index"])
        fold_results[regime], calibrations[regime] = measured, calibration
    selection = core.select(fold_results["dataset"], fold_results["source"])
    write(folder / "selection.json", {**selection, "selected_at_utc": stamp()})
    print(json.dumps({"selection_frozen": selection["selected_full"], "best_development_arm": selection["best_all_methods"]}), flush=True)
    final = {m: core.fit(m, data["train"]["base"], data["train"]["generation"], rows) for m in core.METHODS}
    vy = [r["activation_label"] for r in p["membership"]["validation"]]
    vscores = {m: core.predict(state, data["validation"]["base"], data["validation"]["generation"]) for m, state in final.items()}
    reference_errors = {}
    for new, old in zip(core.METHODS[:3], ("generation", "reference_behavioral", "reference_full")):
        reference_errors[new] = float(np.max(abs(vscores[new] - previous.reused_reference_scores("validation")[old])))
        require(reference_errors[new] <= 1e-12, "Standard reference validation changed.")
    thresholds = {m: previous.core.thresholds(vy, vscores[m]) for m in core.METHODS}
    validation_rows = rows_for(p["membership"]["validation"], vscores, thresholds)
    validation = summarize(validation_rows)
    decision = core.gate(selection, validation["max_f1"])
    gate = {"checks": decision, "passed": all(decision.values()), "test_scoring_allowed": all(decision.values()), "dashboard_promoted": False}
    previous.dump_new(folder / "model-bundles.joblib", {"folds": fit_bundles, "final": final})
    (folder / "states").mkdir()
    for m, state in final.items():
        previous.dump_new(folder / "states" / f"{m}.joblib", {**state, "thresholds": thresholds[m]})
    benchmark_records = read(previous.RUN / "training/benchmark-records.json")
    require([r["sample_id"] for r in benchmark_records] == [r["sample_id"] for r in validation_rows], "Benchmark membership differs.")
    for i, row in enumerate(benchmark_records):
        require("activation_label" not in row, "Benchmark must be unlabeled.")
        row["reference_scores"] = {m: float(vscores[m][i]) for m in core.METHODS}
    write(folder / "benchmark-records.json", benchmark_records)
    for regime in prediction_sets:
        previous.shared.write_new(folder / f"{regime}-held-predictions.jsonl", previous.common.jsonlines(prediction_sets[regime]))
    previous.shared.write_new(folder / "validation-predictions.jsonl", previous.common.jsonlines(validation_rows))
    write(folder / "gate.json", gate)
    report = {"schema": "trace.paired_control.development.v1", "completed_at_utc": stamp(), "selection": selection,
        "source_fold_results": fold_results["source"], "dataset_fold_results": fold_results["dataset"],
        "fold_calibrations": calibrations, "final_thresholds": thresholds,
        "source_oof": summarize(prediction_sets["source"]), "dataset_held": summarize(prediction_sets["dataset"]),
        "source_slices": slices(prediction_sets["source"]), "dataset_slices": slices(prediction_sets["dataset"]),
        "validation": validation, "validation_slices": slices(validation_rows), "pair_diagnostics": diagnostic_sets,
        "source_bootstrap": bootstrap(prediction_sets["source"], selection["selected_full"]),
        "validation_bootstrap": bootstrap(validation_rows, selection["selected_full"]),
        "reference_validation_max_errors": reference_errors,
        "development_elapsed_seconds_excluding_imports_preflight": perf_counter() - start,
        "gate": gate, "classifier_fits": 48, "target_calls": 0, "test_prediction_rows_opened": 0,
        "new_activation_labels": 0, "encoder_updates": 0, "dashboard_promoted": False,
        "limitations": ["Source and held-dataset partitions are reused development evidence; selection is not independent confirmation.",
            "Only two training datasets are available; validation has eleven positives and calibrates operating points.",
            "Fold cutoffs use fitting predictions; validation/test rates can differ from calibration targets.",
            "Paired training uses unchanged reviewed activation labels; cue/condition never supplies a label or inference feature.",
            "Historical collection/reviewer qualifications and prior exposure of StrongREJECT remain unchanged."]}
    old_cv = previous.reused_reference_scores("train")
    for new, old in zip(core.METHODS[:3], ("generation", "reference_behavioral", "reference_full")):
        require(np.max(abs(np.array([r["scores"][new] for r in prediction_sets["source"]]) - old_cv[old])) <= 1e-12, "Standard reference OOF changed.")
    write(folder / "report.json", report)
    seal(folder)
    check()
    return {"trained": True, "fits": 48, "selection": selection,
            "validation_f1": {m: validation["max_f1"][m]["f1"] for m in core.METHODS}, "gate": gate}


def evaluate():
    import joblib
    import numpy as np
    p = check()
    check_seal(RUN / "development")
    gate = read(RUN / "development/gate.json")
    require(gate["passed"], "Development gate failed; no new test scoring allowed.")
    folder = RUN / "evaluation"
    require(not folder.exists(), "Paired evaluation exists; refuse repeat.")
    folder.mkdir()
    report = read(RUN / "development/report.json")
    models = joblib.load(RUN / "development/model-bundles.joblib")["final"]
    old = read(previous.RUN / "protocol.json")
    labels = old["test_membership"]
    cache = np.load(previous.frozen.RUN / "evaluation/features.npz", allow_pickle=False)
    require(cache["sample_ids"].tolist() == [r["sample_id"] for r in labels], "Reused test feature membership differs.")
    scores = {m: core.predict(s, cache["base"], cache["generation"]) for m, s in models.items()}
    rows = rows_for(labels, scores, report["final_thresholds"])
    selected = report["selection"]["selected_full"]
    result = {"initial_retrospective_single_target": True, "dashboard_promoted": False,
              "selection_frozen": selected, "results": {}, "paired_intervals": {}}
    for split in sorted({r["split"] for r in rows}):
        part = [r for r in rows if r["split"] == split]
        result["results"][split] = summarize(part)
        result["paired_intervals"][split] = bootstrap(part, selected)
    previous.shared.write_new(folder / "predictions.jsonl", previous.common.jsonlines(rows))
    write(folder / "report.json", result)
    seal(folder)
    return result


def predict_record(state, record, encoder=None):
    import numpy as np
    from traceguard.features import generation_features
    g = np.zeros((1, 3)) if state["method"].endswith("behavioral") else generation_features(record["signals"])[None, :]
    b = np.zeros((1, 384)) if state["method"].endswith("generation") else encoder.encode_base([record])
    return float(core.predict(state, b, g)[0])


def worker(method, replicate):
    import joblib
    import random
    from time import perf_counter_ns
    from traceguard.runtime import process_memory
    from scripts.trace_error_efficiency import statistics
    from threadpoolctl import threadpool_limits
    check(light=True)
    require(read(RUN / "development/gate.json")["passed"], "No resource workers after a failed gate.")
    require(method in core.METHODS and replicate in (0, 1, 2), "Invalid resource worker.")
    records = read(RUN / "development/benchmark-records.json")
    start = perf_counter_ns()
    state = joblib.load(RUN / "development/states" / f"{method}.joblib")
    encoder = None
    if not method.endswith("generation"):
        import torch
        from traceguard.behavior import BehavioralEncoder
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        encoder = BehavioralEncoder(model_name=str(previous.shared.ENCODER_PATH), local_only=True)
    limiter = threadpool_limits(limits=4)
    predict_record(state, records[0], encoder)
    first = perf_counter_ns()
    for row in records:
        predict_record(state, row, encoder)
    observations, maximum, boundaries = [], 0., []
    rng = random.Random(SETTINGS["seed"] + replicate)
    for number in (0, 1):
        order = list(range(100))
        rng.shuffle(order)
        for i in order:
            row = records[i]
            t0 = perf_counter_ns()
            score = predict_record(state, row, encoder)
            elapsed = (perf_counter_ns() - t0) / 1e6
            reference, threshold = row["reference_scores"][method], state["thresholds"]["max_f1"]
            maximum = max(maximum, abs(score - reference))
            require(abs(score - reference) <= SETTINGS["score_tolerance"], "Timed inference differs from batched reference.")
            if (score >= threshold) != (reference >= threshold):
                require(abs(reference - threshold) <= SETTINGS["score_tolerance"], "Decision difference away from numeric boundary.")
                boundaries.append(row["sample_id"])
            observations.append({"sample_id": row["sample_id"], "pass": number, "latency_ms": elapsed})
    memory = process_memory()
    limiter.restore_original_limits()
    write(RUN / "benchmark" / f"{method}-{replicate}.json", {"method": method, "replicate": replicate,
        "partition": "validation", "observations": observations,
        "latency_ms": statistics([r["latency_ms"] for r in observations]), "memory": memory,
        "setup_first_ms": (first-start)/1e6, "maximum_reference_error": maximum,
        "boundary_occurrences": len(boundaries), "boundary_sample_ids": sorted(set(boundaries)),
        "model_fits": 0, "test_rows_scored": 0, "target_calls": 0})


def benchmark():
    import subprocess
    import numpy as np
    from scripts.trace_error_efficiency import statistics, cpu_name
    p = check()
    check_seal(RUN / "development")
    require(read(RUN / "development/gate.json")["passed"], "Development gate failed; resource stage skipped.")
    folder = RUN / "benchmark"
    require(not folder.exists(), "Paired benchmark exists; refuse repeat.")
    folder.mkdir()
    workers = []
    for replicate, order in enumerate(p["benchmark_orders"]):
        for method in order:
            subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), "worker", "--method", method, "--replicate", str(replicate)],
                cwd=ROOT, env=dict(os.environ), check=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            workers.append(read(folder / f"{method}-{replicate}.json"))
            print(json.dumps({"worker_complete": method, "replicate": replicate}), flush=True)
    result = {}
    for m in core.METHODS:
        part = [w for w in workers if w["method"] == m]
        result[m] = {"warm_latency_ms": statistics([r["latency_ms"] for w in part for r in w["observations"]]),
                     "median_worker_bytes": float(np.median([w["memory"]["working_set_bytes"] for w in part])),
                     "max_lifetime_peak_worker_bytes": max(w["memory"]["peak_working_set_bytes"] for w in part),
                     "state_bytes": (RUN / "development/states" / f"{m}.joblib").stat().st_size,
                     "maximum_reference_error": max(w["maximum_reference_error"] for w in part),
                     "boundary_occurrences": sum(w["boundary_occurrences"] for w in part)}
    write(folder / "summary.json", {"methods": result, "workers": 18, "timings": 3600,
        "cpu": cpu_name(), "threads": 4, "warm_boundary": "Feature computation, required encoding and classification; target generation, UI/network, I/O and loading excluded.",
        "memory_boundary": "Whole worker resident/lifetime peak; target and parent excluded.",
        "limitations": "Background load/power/cache uncontrolled; no historical or cross-hardware speedup claim."})
    seal(folder)
    return {"benchmarked": True, "workers": 18, "timings": 3600}


def verify():
    import joblib
    import numpy as np
    p = check()
    folder = RUN / "development"
    check_seal(folder)
    report = read(folder / "report.json")
    bundle = joblib.load(folder / "model-bundles.joblib")
    data = previous.arrays()
    rows = p["membership"]["train"]
    all_results = {}
    for regime, folds in (("source", p["source_folds"]), ("dataset", p["dataset_folds"])):
        predictions = lines(folder / f"{regime}-held-predictions.jsonl")
        results = []
        for number, fold in enumerate(folds):
            f, h = np.array(fold["fit_indices"]), np.array(fold["held_indices"])
            core.partition_check(rows, f, h, p["groups"]["train"])
            models = bundle["folds"][f"{regime}-{number}"]
            scores, thresholds = {}, {}
            for m, state in models.items():
                require(state["fit_sample_ids"] == [rows[i]["sample_id"] for i in f], "Saved fit membership differs.")
                if state["scaler"] is not None:
                    require(np.allclose(state["scaler"].mean_, data["train"]["generation"][f].mean(axis=0), atol=1e-12), "Scaler fit membership differs.")
                fitting = core.predict(state, data["train"]["base"][f], data["train"]["generation"][f])
                thresholds[m] = previous.core.thresholds([rows[i]["activation_label"] for i in f], fitting)
                scores[m] = core.predict(state, data["train"]["base"][h], data["train"]["generation"][h])
                if m.startswith("paired"):
                    d, e = core.pairs([rows[i] for i in f])
                    loss, gradient = core.objective(np.r_[state["weights"], state["bias"]],
                        core.features(m, data["train"]["base"][f], data["train"]["generation"][f], state["scaler"]),
                        np.array([rows[i]["activation_label"] for i in f]), d, e)
                    require(abs(loss - state["optimizer"]["objective"]) <= 1e-12 and np.max(abs(gradient)) < 2e-5, "Optimizer reconstruction differs or is not near stationary.")
            require(thresholds == report["fold_calibrations"][regime][number], "Fit-only calibration differs.")
            expected = rows_for([rows[i] for i in h], scores, thresholds)
            for i, row in zip(h, expected):
                require(all(predictions[i][k] == v for k, v in row.items()), "Held prediction differs.")
            results.append(summarize(expected)["max_f1"])
        all_results[regime] = results
        key = "source_oof" if regime == "source" else "dataset_held"
        require(summarize(predictions) == report[key], "Held metrics differ.")
    selection = core.select(all_results["dataset"], all_results["source"])
    require(selection == report["selection"], "Development-only selection differs.")
    models = bundle["final"]
    for m, state in models.items():
        require(state["fit_sample_ids"] == [r["sample_id"] for r in rows], "Final fit membership differs.")
        if m.startswith("paired"):
            d, e = core.pairs(rows)
            value, grad = core.objective(np.r_[state["weights"], state["bias"]],
                core.features(m, data["train"]["base"], data["train"]["generation"], state["scaler"]),
                np.array([r["activation_label"] for r in rows]), d, e)
            require(abs(value - state["optimizer"]["objective"]) < 1e-12 and np.max(abs(grad)) < 2e-5, "Final optimizer reconstruction differs.")
    scores = {m: core.predict(state, data["validation"]["base"], data["validation"]["generation"]) for m, state in models.items()}
    thresholds = {m: previous.core.thresholds([r["activation_label"] for r in p["membership"]["validation"]], scores[m]) for m in core.METHODS}
    require(thresholds == report["final_thresholds"], "Validation cutoffs differ.")
    validation = rows_for(p["membership"]["validation"], scores, thresholds)
    require(validation == lines(folder / "validation-predictions.jsonl") and summarize(validation) == report["validation"], "Validation predictions/metrics differ.")
    checks = core.gate(selection, report["validation"]["max_f1"])
    require(checks == report["gate"]["checks"] and all(checks.values()) == report["gate"]["passed"], "Gate changed.")
    for name, current in (("source_bootstrap", lines(folder / "source-held-predictions.jsonl")), ("validation_bootstrap", validation)):
        require(bootstrap(current, selection["selected_full"]) == report[name], "Paired intervals differ.")
    if not report["gate"]["passed"]:
        require(not (RUN / "evaluation").exists() and not (RUN / "benchmark").exists(), "Failed gate reached conditional stages.")
    else:
        check_seal(RUN / "evaluation")
        check_seal(RUN / "benchmark")
        labels = read(previous.RUN / "protocol.json")["test_membership"]
        cache = np.load(previous.frozen.RUN / "evaluation/features.npz", allow_pickle=False)
        scores = {m: core.predict(state, cache["base"], cache["generation"]) for m, state in models.items()}
        expected = rows_for(labels, scores, thresholds)
        require(expected == lines(RUN / "evaluation/predictions.jsonl"), "Reused test scores differ.")
        evaluation = read(RUN / "evaluation/report.json")
        for split in evaluation["results"]:
            part = [r for r in expected if r["split"] == split]
            require(summarize(part) == evaluation["results"][split] and bootstrap(part, selection["selected_full"]) == evaluation["paired_intervals"][split], "Reused evaluation metrics differ.")
        summary = read(RUN / "benchmark/summary.json")
        from scripts.trace_error_efficiency import statistics
        expected_ids = {r["sample_id"] for r in validation}
        for m in core.METHODS:
            workers = [read(RUN / "benchmark" / f"{m}-{i}.json") for i in (0, 1, 2)]
            for w in workers:
                require(w["model_fits"] == w["test_rows_scored"] == w["target_calls"] == 0, "Worker scope differs.")
                for number in (0, 1):
                    ids = [r["sample_id"] for r in w["observations"] if r["pass"] == number]
                    require(len(ids) == 100 and len(set(ids)) == 100 and set(ids) == expected_ids, "Worker coverage differs.")
            require(summary["methods"][m]["warm_latency_ms"] == statistics([r["latency_ms"] for w in workers for r in w["observations"]]), "Timing aggregate differs.")
    value = {"verified": True, "fits": 48, "source_and_duplicate_isolation": True,
             "fit_only_scalers_cutoffs_and_optimizer_checks": True, "development_selection_and_validation_gate": True,
             "source_and_validation_bootstrap": True, "gate_passed": report["gate"]["passed"],
             "target_calls": 0, "dashboard_promoted": False}
    path = RUN / "verification.json"
    if path.exists():
        require(read(path) == value, "Verification changed.")
    else:
        write(path, value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "train", "evaluate", "benchmark", "worker", "verify"))
    parser.add_argument("--method", choices=core.METHODS)
    parser.add_argument("--replicate", type=int)
    args = parser.parse_args()
    result = worker(args.method, args.replicate) if args.command == "worker" else globals()[args.command]()
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
