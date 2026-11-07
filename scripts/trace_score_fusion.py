"""Frozen score-fusion selection, validation calibration, reused-data testing and CPU benchmarking."""
from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
from time import perf_counter_ns

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "4"
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import trace_reused_evaluation as previous
from traceguard import score_fusion as core

RUN = ROOT / "artifacts/trace-behavioral-v4/score-fusion-001"
REPR_RUN = previous.REPR_RUN
read, lines, sha, write, stamp = previous.read, previous.lines, previous.sha, previous.write, previous.stamp
common, shared, efficiency = previous.common, previous.shared, previous.efficiency
SETTINGS = {"behavioral_weights": list(core.DEFAULT_WEIGHTS), "component_C": 1.0,
            "class_weight": None, "bootstrap_repetitions": 2000, "seed": 20261106,
            "processes_per_method": 3, "warmup_passes": 1, "measured_passes": 2,
            "validation_records": 100, "cpu_threads": 4, "torch_interop_threads": 1,
            "score_tolerance": 1e-5, "batch_size": 1}
COMPONENTS = {"generation": "generation__C1__unweighted",
              "behavioral": "prompt_response_raw__behavioral__C1__unweighted"}


def code_hashes():
    result = previous.code_hashes()
    for path in (Path(__file__).resolve(), Path(core.__file__), ROOT / "tests/test_score_fusion.py"):
        result[previous.relative(path)] = sha(path)
    return result


def npz_new(path, **arrays):
    import numpy as np
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    shared.write_new(path, buffer.getvalue())


def seal(folder):
    write(folder / "manifest.json", {"protocol_sha256": sha(RUN / "protocol.json"),
        "output_sha256": {p.relative_to(folder).as_posix(): sha(p) for p in sorted(folder.rglob("*")) if p.is_file()}})


def check_seal(folder):
    manifest = read(folder / "manifest.json")
    common.require(manifest["protocol_sha256"] == sha(RUN / "protocol.json"), "Output protocol binding differs.")
    for name, expected in manifest["output_sha256"].items():
        common.require(not Path(name).is_absolute() and ".." not in Path(name).parts, "Unsafe output path.")
        common.checked_read(folder / name, expected)


def check_protocol(light=False):
    protocol = read(RUN / "protocol.json")
    common.require(protocol["settings"] == SETTINGS and protocol["implementation_sha256"] == code_hashes(),
                   "Fusion code or settings changed after protocol freeze.")
    common.require(protocol["components"] == COMPONENTS and protocol["retrospective_not_fresh_holdout"] is True,
                   "Frozen experiment design changed.")
    if not light:
        for name, expected in protocol["input_sha256"].items():
            common.checked_read(ROOT / name, expected)
        for name, expected in protocol["encoder_file_sha256"].items():
            common.checked_read(shared.ENCODER_PATH / name, expected)
    return protocol


def prepare():
    common.require(not RUN.exists(), "Fusion run exists; refuse overwrite.")
    previous.representation.verify(REPR_RUN)
    previous.verify()
    development = read(REPR_RUN / "protocol.json")
    protected = previous.protect_inputs()
    protected.update({previous.relative(p): sha(p) for p in previous.RUN.rglob("*") if p.is_file()})
    fresh = ROOT / "outputs/trace-fresh-holdout-audit-20261106-001"
    for name in ("comparison-plan.json", "manifest.json"):
        protected[previous.relative(fresh / name)] = sha(fresh / name)
    labels = read(previous.RUN / "protocol.json")["test_label_membership"]
    forbidden = {r["source_id"] for rows in development["membership"].values() for r in rows}
    previous.core.validate_test_membership(labels, forbidden)
    efficiency.validate_benchmark_membership(development["membership"]["validation"])
    methods = list(core.METHODS)
    protocol = {"schema": "trace.score_fusion.protocol.v1", "created_at_utc": stamp(),
        "user_authorization": "okay do our next; approved separate-view score-fusion plan using existing data",
        "settings": SETTINGS, "components": COMPONENTS,
        "formula": "(1 - Behavioral weight) * Generation probability score + Behavioral weight * Behavioral probability score",
        "mixture_is_calibrated_probability": False,
        "selection": "Maximize mean training-fold AP; ties mean fold AUROC then smaller Behavioral weight.",
        "selection_candidates": "Eight predefined weights including both endpoints. Best interior mixture is evaluated; overall winner including endpoints is separately reported.",
        "validation_rule": "Only selected interior mixture receives a new validation-max-F1 threshold; ties higher threshold. Existing comparator thresholds stay fixed.",
        "cv_interpretation": "Reused OOF folds select weights, not an unbiased new performance estimate; no nested CV claim.",
        "membership": development["membership"], "folds": development["folds"], "groups": development["groups"],
        "test_label_membership": labels, "test_methods": methods,
        "retrospective_not_fresh_holdout": True, "earlier_test_errors_informed_design": True,
        "test_weight_or_threshold_selection_allowed": False, "component_refitting_allowed": False,
        "new_review_labels_allowed": False, "target_model_calls_allowed": False, "dashboard_promotion_allowed": False,
        "input_sha256": protected, "implementation_sha256": code_hashes(),
        "encoder_file_sha256": development["encoder_file_sha256"],
        "benchmark_process_orders": [methods[i:] + methods[:i] for i in range(3)],
        "warm_timing_boundary": "In-memory record through required features, encoding, frozen classifier(s), mixture and threshold decision; excludes I/O, target generation/statistics collection, UI/network and initial loading/imports.",
        "cold_boundary": "State/model loading and first inference after shared workflow/numerical imports and preflight; not full application startup.",
        "memory_boundary": "Whole worker resident/lifetime peak; parent and target LLM excluded. Parent hashes encoder before workers.",
        "admission": development["admission"], "development_catalogue_admission": development["development_catalogue_admission"]}
    RUN.mkdir(parents=True, exist_ok=False)
    write(RUN / "protocol.json", protocol)
    return {"prepared": True, "mixtures": len(SETTINGS["behavioral_weights"]), "component_fits_planned": 0,
            "selection_train_examples": 1058, "calibration_validation_examples": 100, "reused_test_examples": 476,
            "benchmark_processes": 12, "retrospective_not_fresh_holdout": True}


def load_training_oof(protocol):
    import numpy as np
    archive = np.load(REPR_RUN / "results/cv-scores.npz", allow_pickle=False)
    membership = protocol["membership"]["train"]
    common.require(archive["sample_ids"].tolist() == [r["sample_id"] for r in membership]
                   and archive["labels"].tolist() == [r["activation_label"] for r in membership], "OOF training membership differs.")
    names = archive["configuration_ids"].tolist()
    common.require(len(names) == len(set(names)) == 54, "OOF configuration coverage differs.")
    return {name: archive["scores"][names.index(key)] for name, key in COMPONENTS.items()}


def develop():
    import joblib
    import numpy as np
    protocol = check_protocol()
    folder = RUN / "development"
    common.require(not folder.exists(), "Fusion development exists; refuse repeat or overwrite.")
    folder.mkdir(exist_ok=False)
    write(folder / "started.json", {"started_at_utc": stamp(), "protocol_sha256": sha(RUN / "protocol.json")})
    oof = load_training_oof(protocol)
    cv, selection, mixtures = core.select_training_mixture(oof["generation"], oof["behavioral"],
        protocol["membership"]["train"], protocol["folds"], protocol["groups"]["train"], SETTINGS["behavioral_weights"])
    selection["selected_at_utc"] = stamp()
    write(folder / "selection.json", selection)
    write(folder / "cv-results.json", cv)
    npz_new(folder / "training-oof.npz", generation=oof["generation"], behavioral=oof["behavioral"], mixtures=mixtures)
    print("Training-only fusion selection saved; calibrating only the selected interior mixture on validation", flush=True)
    records = read(previous.RUN / "validation-records.json")
    membership = protocol["membership"]["validation"]
    common.require([r["sample_id"] for r in records] == [r["sample_id"] for r in membership], "Validation row order differs.")
    weight = selection["selected_fusion_weight"]
    values = core.mix([r["reference_scores"]["generation"] for r in records],
                      [r["reference_scores"]["candidate_behavioral"] for r in records], weight)
    threshold, validation = core.calibrate_validation(values, membership)
    states = {m: joblib.load(previous.RUN / "states" / f"{m}.joblib") for m in core.METHODS if m != "score_fusion"}
    behavioral = joblib.load(previous.RUN / "states/candidate_behavioral.joblib")
    state = {"schema": "trace.score_fusion.state.v1", "kind": "score_fusion", "view": "score_fusion",
             "behavioral_weight": weight, "generation_component": states["generation"],
             "behavioral_component": behavioral, "threshold": threshold, "method": "score_fusion",
             "protocol_sha256": sha(RUN / "protocol.json"), "development_frozen": True,
             "encoder_revision": shared.ENCODER_REV}
    states["score_fusion"] = state
    (folder / "states").mkdir()
    for method, item in states.items():
        buffer = io.BytesIO()
        joblib.dump(item, buffer, compress=3)
        shared.write_new(folder / "states" / f"{method}.joblib", buffer.getvalue())
    thresholds = {m: s["threshold"] for m, s in states.items()}
    validation_results = {}
    for method in core.METHODS:
        scores = values if method == "score_fusion" else np.asarray([r["reference_scores"][method] for r in records])
        validation_results[method] = core.score_metrics([r["activation_label"] for r in membership], scores, thresholds[method])
    for index, record in enumerate(records):
        record["reference_scores"] = {m: record["reference_scores"][m] for m in core.METHODS if m != "score_fusion"}
        record["reference_scores"]["score_fusion"] = float(values[index])
        record["reference_predictions"] = {m: int(v >= thresholds[m]) for m, v in record["reference_scores"].items()}
    write(folder / "validation-records.json", records)
    write(folder / "report.json", {"schema": "trace.score_fusion.development.v1", "completed_at_utc": stamp(),
        "selection": selection, "thresholds": thresholds, "validation_results": validation_results,
        "cv_results": cv, "component_fit_calls": 0, "new_mixture_threshold_calibrations": 1,
        "target_model_calls": 0, "new_labels": 0, "new_fusion_test_predictions_computed": False,
        "dashboard_promoted": False,
        "limitations": ["CV scores select weights and reuse previous development folds; they are not unbiased confirmation.",
            "Validation has eleven positive examples and is reused for threshold development.",
            "Base classifiers/scaler remain the already fitted training-only C=1 unweighted components.",
            "Only the best interior mixture is a new test candidate; endpoints remain single-view references.",
            "Earlier test results informed this experiment; all subsequent reused-test findings are initial and retrospective."]})
    seal(folder)
    check_protocol()
    return {"development_complete": True, "selection": selection, "threshold": threshold,
            "validation_f1": validation["f1"], "new_component_fits": 0}


def rebuild_predictions(protocol, development):
    labels = protocol["test_label_membership"]
    previous.core.validate_test_membership(labels, {r["source_id"] for rows in protocol["membership"].values() for r in rows})
    saved = lines(previous.RUN / "evaluation/predictions.jsonl")
    common.require(len(saved) == len(labels) == 476, "Retrospective coverage differs.")
    for row, label in zip(saved, labels):
        common.require(all(row[k] == label[k] for k in ("sample_id", "source_id", "condition_id", "split", "activation_label")), "Retrospective label membership differs.")
    return core.add_frozen_fusion(saved, development["selection"]["selected_fusion_weight"], development["thresholds"])


def evaluate():
    protocol = check_protocol()
    check_seal(RUN / "development")
    folder = RUN / "evaluation"
    common.require(not folder.exists(), "Fusion evaluation exists; refuse repeat or overwrite.")
    folder.mkdir(exist_ok=False)
    write(folder / "started.json", {"started_at_utc": stamp(), "development_manifest_sha256": sha(RUN / "development/manifest.json")})
    development = read(RUN / "development/report.json")
    rows = rebuild_predictions(protocol, development)
    report = {"schema": "trace.score_fusion.retrospective.v1", "completed_at_utc": stamp(),
        "retrospective_not_fresh_holdout": True, "initial_single_target_results": True,
        "behavioral_weight": development["selection"]["selected_fusion_weight"], "thresholds": development["thresholds"],
        "results": core.summarize(rows, development["thresholds"]),
        "paired_source_bootstrap": {s: core.paired_intervals([r for r in rows if r["split"] == s],
            SETTINGS["bootstrap_repetitions"], SETTINGS["seed"]) for s in ("test_seen", "test_unseen")},
        "base_test_scores_reused_exactly": True, "component_or_encoder_test_rescoring_calls": 0,
        "fit_calls": 0, "target_model_calls": 0, "test_threshold_or_weight_selection_calls": 0,
        "excluded_source_groups_unchanged": 75, "dashboard_promoted": False,
        "limitations": ["Existing test errors informed design: reused observations cannot confirm generalization.",
            "Same 476 admitted examples and 75 excluded groups; paired intervals are post-hoc and unadjusted.",
            "Only one Llama 2 target and independent prompts; no cross-model or validated conversation claim.",
            "Original consensus, historical server provenance and missing intake CSV qualifications remain."]}
    shared.write_new(folder / "predictions.jsonl", common.jsonlines(rows))
    write(folder / "report.json", report)
    seal(folder)
    check_protocol()
    return {"evaluation_complete": True, "retrospective_not_fresh_holdout": True,
            "f1": {s: {m: r["f1"] for m, r in part.items()} for s, part in report["results"].items()}}


def worker(method, replicate):
    import joblib
    from traceguard.runtime import process_memory, system_memory
    check_protocol(light=True)
    check_seal(RUN / "development")
    common.require(method in core.METHODS and replicate in (0, 1, 2), "Unknown worker.")
    records = read(RUN / "development/validation-records.json")
    baseline = process_memory()
    start = perf_counter_ns()
    state = joblib.load(RUN / "development/states" / f"{method}.joblib")
    encoder = None
    if method != "generation":
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
    observations, errors, boundaries = [], [], []
    rng = random.Random(SETTINGS["seed"] + replicate)
    for pass_number in range(2):
        order = list(range(100))
        rng.shuffle(order)
        for index in order:
            record = records[index]
            t0 = perf_counter_ns()
            score, decision = core.predict_one(state, record, encoder)
            elapsed = (perf_counter_ns() - t0) / 1e6
            error = abs(score - record["reference_scores"][method])
            common.require(error <= SETTINGS["score_tolerance"], "Worker score differs from frozen validation reference.")
            if decision != record["reference_predictions"][method]:
                common.require(abs(record["reference_scores"][method] - state["threshold"]) <= SETTINGS["score_tolerance"], "Decision difference away from numerical boundary.")
                boundaries.append(record["sample_id"])
            errors.append(error)
            observations.append({"sample_id": record["sample_id"], "pass": pass_number, "latency_ms": elapsed})
    measured = process_memory()
    pools = threadpool_info()
    limiter.restore_original_limits()
    write(RUN / "benchmark" / f"{method}-{replicate}.json", {"method": method, "replicate": replicate,
        "protocol_sha256": sha(RUN / "protocol.json"), "partition": "validation", "observations": observations,
        "latency_ms": efficiency.statistics([r["latency_ms"] for r in observations]),
        "cold_post_import_setup_first_ms": (first - start) / 1e6, "cold_first_call_ms": (first - loaded) / 1e6,
        "memory": {"baseline": baseline, "warm": warm_memory, "measured": measured},
        "maximum_score_reference_difference": max(errors), "boundary_difference_occurrences": len(boundaries),
        "boundary_difference_sample_ids": sorted(set(boundaries)), "cpu": efficiency.cpu_name(),
        "logical_cpu_count": os.cpu_count(), "system_memory": system_memory(), "threadpool_info": pools,
        "torch_threads": 4 if encoder else None, "torch_interop_threads": 1 if encoder else None,
        "python": platform.python_version(), "platform": platform.platform(),
        "fit_calls": 0, "target_model_calls": 0, "test_rows_scored": 0, "threshold_or_weight_selection_calls": 0})


def summarize_workers(workers, protocol):
    import numpy as np
    result = {}
    for method in core.METHODS:
        part = [w for w in workers if w["method"] == method]
        common.require(len(part) == 3 and {w["replicate"] for w in part} == {0, 1, 2}, "Missing or repeated worker.")
        result[method] = {"warm_latency_ms": efficiency.statistics([r["latency_ms"] for w in part for r in w["observations"]]),
            "per_process_median_ms": [w["latency_ms"]["median"] for w in part],
            "cold_post_import_setup_first_ms": efficiency.statistics([w["cold_post_import_setup_first_ms"] for w in part]),
            "median_warm_resident_memory_bytes": float(np.median([w["memory"]["measured"]["working_set_bytes"] for w in part])),
            "maximum_peak_working_set_bytes": max(w["memory"]["measured"]["peak_working_set_bytes"] for w in part),
            "maximum_score_reference_difference": max(w["maximum_score_reference_difference"] for w in part),
            "boundary_difference_occurrences": sum(w["boundary_difference_occurrences"] for w in part),
            "boundary_difference_sample_ids": sorted({sid for w in part for sid in w["boundary_difference_sample_ids"]}),
            "state_file_bytes": (RUN / "development/states" / f"{method}.joblib").stat().st_size,
            "required_encoder_files_bytes": 0 if method == "generation" else sum((shared.ENCODER_PATH / name).stat().st_size for name in protocol["encoder_file_sha256"])}
    return result


def benchmark():
    protocol = check_protocol()
    check_seal(RUN / "development")
    folder = RUN / "benchmark"
    common.require(not folder.exists(), "Fusion benchmark exists; refuse repeat or overwrite.")
    folder.mkdir(exist_ok=False)
    write(folder / "started.json", {"started_at_utc": stamp(), "protocol_sha256": sha(RUN / "protocol.json")})
    workers = []
    for replicate, order in enumerate(protocol["benchmark_process_orders"]):
        for method in order:
            subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), "worker", "--method", method,
                "--replicate", str(replicate)], cwd=ROOT, env=dict(os.environ), check=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            workers.append(read(folder / f"{method}-{replicate}.json"))
            print(json.dumps({"worker_complete": method, "replicate": replicate}), flush=True)
    write(folder / "summary.json", {"schema": "trace.score_fusion.efficiency.v1", "completed_at_utc": stamp(),
        "methods": summarize_workers(workers, protocol), "processes": 12, "measurements_per_method": 600,
        "partition": "validation", "cpu": workers[0]["cpu"], "logical_cpu_count": workers[0]["logical_cpu_count"],
        "python": workers[0]["python"], "platform": workers[0]["platform"],
        "available_memory_bytes_range": [min(w["system_memory"]["available_bytes"] for w in workers), max(w["system_memory"]["available_bytes"] for w in workers)],
        "fit_calls": 0, "target_model_calls": 0, "test_rows_scored": 0,
        "limitations": ["Detector-only in-memory overhead; target generation/statistics collection, UI, network and disk reads excluded.",
            "Whole worker memory, not intrinsic model parameter memory. Parent and target LLM excluded.",
            "Background workloads, power and cache uncontrolled. Fresh workers are not cold-machine caches.",
            "Cold timing begins after shared imports/preflight, not at process launch.",
            "State plus encoder file sizes exclude runtime libraries, target weights, data and reports.",
            "Only these current matched measurements support local comparisons; no historical speedup claim."]})
    check_protocol()
    seal(folder)
    return {"benchmark_complete": True, "processes": 12, "observations": 2400}


def verify():
    import joblib
    import numpy as np
    protocol = check_protocol()
    for name in ("development", "evaluation", "benchmark"):
        check_seal(RUN / name)
    development = read(RUN / "development/report.json")
    oof = load_training_oof(protocol)
    cv, selection, mixtures = core.select_training_mixture(oof["generation"], oof["behavioral"], protocol["membership"]["train"],
        protocol["folds"], protocol["groups"]["train"], SETTINGS["behavioral_weights"])
    common.require(read(RUN / "development/cv-results.json") == cv, "Saved CV metrics differ.")
    saved_selection = read(RUN / "development/selection.json")
    common.require(all(saved_selection[k] == v and development["selection"][k] == v for k, v in selection.items()), "Training-only weight selection differs.")
    archive = np.load(RUN / "development/training-oof.npz", allow_pickle=False)
    common.require(np.array_equal(archive["mixtures"], mixtures), "Saved mixtures differ.")
    records = read(RUN / "development/validation-records.json")
    reference = read(previous.RUN / "validation-records.json")
    for row, old in zip(records, reference):
        common.require(all(row[k] == old[k] for k in ("sample_id", "source_id", "condition_id", "split", "activation_label", "prompt", "response", "signals")), "Validation input differs.")
        common.require(all(row["reference_scores"][m] == old["reference_scores"][m] for m in core.METHODS if m != "score_fusion"), "Original validation reference differs.")
    weight = selection["selected_fusion_weight"]
    scores = core.mix([r["reference_scores"]["generation"] for r in reference], [r["reference_scores"]["candidate_behavioral"] for r in reference], weight)
    threshold, validation = core.calibrate_validation(scores, protocol["membership"]["validation"])
    common.require(threshold == development["thresholds"]["score_fusion"]
                   and validation == development["validation_results"]["score_fusion"], "Validation calibration differs.")
    common.require([r["reference_scores"]["score_fusion"] for r in records] == scores.tolist(), "Saved fusion validation scores differ.")
    state = joblib.load(RUN / "development/states/score_fusion.joblib")
    common.require(state["behavioral_weight"] == weight and state["threshold"] == threshold, "Saved inference state differs.")
    # Check component state inference on cached validation features, without fitting or encoding.
    features = np.load(REPR_RUN / "results/final-features.npz", allow_pickle=False)
    values = {}
    for kind, component in (("behavioral", state["behavioral_component"]), ("generation", state["generation_component"])):
        key = "validation__prompt_response_raw__behavioral" if kind == "behavioral" else "validation__generation"
        values[kind] = component["model"].predict_proba(features[key])[:, 1]
    rebuilt = core.mix(values["generation"], values["behavioral"], weight)
    common.require(np.array_equal(rebuilt, scores), "Saved fusion components cannot reconstruct validation scores.")
    common.require(np.array_equal(state["generation_component"]["scaler"].transform(features["validation__generation_raw"]), features["validation__generation"]), "Saved generation scaler differs.")
    rows = rebuild_predictions(protocol, development)
    common.require(rows == lines(RUN / "evaluation/predictions.jsonl"), "Saved retrospective mixture differs.")
    report = read(RUN / "evaluation/report.json")
    common.require(report["results"] == core.summarize(rows, development["thresholds"]), "Saved test metrics differ.")
    for split in ("test_seen", "test_unseen"):
        common.require(report["paired_source_bootstrap"][split] == core.paired_intervals([r for r in rows if r["split"] == split], SETTINGS["bootstrap_repetitions"], SETTINGS["seed"]), "Saved bootstrap differs.")
    workers = [read(RUN / "benchmark" / f"{m}-{i}.json") for i in range(3) for m in core.METHODS]
    expected_ids = {r["sample_id"] for r in protocol["membership"]["validation"]}
    for w in workers:
        common.require(w["protocol_sha256"] == sha(RUN / "protocol.json"), "Worker protocol differs.")
        for number in (0, 1):
            ids = [r["sample_id"] for r in w["observations"] if r["pass"] == number]
            common.require(len(ids) == len(set(ids)) == 100 and set(ids) == expected_ids, "Worker coverage differs.")
        common.require(w["fit_calls"] == w["target_model_calls"] == w["test_rows_scored"] == w["threshold_or_weight_selection_calls"] == 0, "Worker scope differs.")
        common.require(w["maximum_score_reference_difference"] <= SETTINGS["score_tolerance"]
                       and w["latency_ms"] == efficiency.statistics([r["latency_ms"] for r in w["observations"]]), "Worker numerics differ.")
    common.require(read(RUN / "benchmark/summary.json")["methods"] == summarize_workers(workers, protocol), "Benchmark aggregate differs.")
    value = {"schema": "trace.score_fusion.verification.v1", "verified": True,
        "protected_original_inputs_unchanged": True, "training_oof_group_selection_reconstructed": True,
        "validation_only_calibration_and_frozen_component_inference_verified": True,
        "reused_test_scores_confusions_metrics_and_paired_bootstrap_verified": True,
        "validation_benchmark_coverage_and_statistics_verified": True,
        "mixtures": 8, "new_component_fits": 0, "new_fusion_threshold_calibrations": 1,
        "reused_test_examples": 476, "benchmark_processes": 12, "timing_observations": 2400,
        "target_model_calls": 0, "dashboard_promoted": False, "retrospective_not_fresh_holdout": True}
    path = RUN / "verification.json"
    if path.exists():
        common.require(read(path) == value, "Existing verification differs.")
    else:
        write(path, value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "develop", "evaluate", "benchmark", "worker", "verify"))
    parser.add_argument("--method", choices=core.METHODS)
    parser.add_argument("--replicate", type=int)
    args = parser.parse_args()
    if args.command == "worker":
        worker(args.method, args.replicate)
    else:
        print(json.dumps({"prepare": prepare, "develop": develop, "evaluate": evaluate,
                          "benchmark": benchmark, "verify": verify}[args.command](), indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
