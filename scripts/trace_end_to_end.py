"""Frozen end-to-end dual-view comparison with nested component cross-fitting."""
from __future__ import annotations

import argparse
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
from scripts import trace_auxiliary_scorer as previous
from traceguard import end_to_end as core

audit = previous.audit
frozen = audit.frozen
common, shared = previous.common, previous.shared
read, lines, sha, write, stamp = previous.read, previous.lines, previous.sha, previous.write, previous.stamp
dump_new, npz_new = previous.dump_new, previous.npz_new
RUN = ROOT / "artifacts/trace-behavioral-v4/end-to-end-001"
SETTINGS = {**core.SETTINGS, "cpu_threads": 4, "torch_interop_threads": 1,
    "encoding": previous.core.ENCODING, "benchmark_processes_per_method": 3,
    "warmup_passes": 1, "measured_passes": 2, "benchmark_validation_records": 100,
    "score_tolerance": 1e-5, "reference_reproduction_tolerance": 1e-10}
REFERENCE_KEYS = {"generation": "generation", "reference_behavioral": "candidate_behavioral", "reference_full": "candidate_full"}
REFERENCE_CONFIGS = {"generation": "generation__C1__unweighted",
    "reference_behavioral": "prompt_response_raw__behavioral__C1__unweighted",
    "reference_full": "prompt_response_raw__full__C1__unweighted"}


def codes():
    paths = (Path(__file__).resolve(), Path(core.__file__), ROOT / "tests/test_end_to_end.py",
        ROOT / "traceguard/auxiliary_scorer.py", ROOT / "traceguard/behavior.py", ROOT / "traceguard/features.py",
        ROOT / "traceguard/runtime.py", ROOT / "docs/TRACE_END_TO_END_IMPROVEMENT_PLAN.md")
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}


def protected_inputs():
    result = previous.protected_inputs()
    for folder in (previous.RUN, frozen.RUN):
        result.update({p.relative_to(ROOT).as_posix(): sha(p) for p in folder.rglob("*") if p.is_file()})
    return result


def check_protocol(light=False):
    protocol = read(RUN / "protocol.json")
    common.require(protocol["settings"] == SETTINGS and protocol["implementation_sha256"] == codes(), "End-to-end code/settings changed after freeze.")
    common.require(protocol["semantic_heads"] == list(core.SEMANTIC_HEADS)
                   and protocol["old_three_head_gate_modified"] is False, "Two-head ablation policy changed.")
    if not light:
        for name, expected in protocol["protected_input_sha256"].items():
            if expected is None:
                common.require(not (ROOT / name).exists(), "Protected absent path appeared.")
            else:
                common.checked_read(ROOT / name, expected)
        for name, expected in protocol["encoder_file_sha256"].items():
            common.checked_read(shared.ENCODER_PATH / name, expected)
    return protocol


def seal(folder):
    write(folder / "manifest.json", {"protocol_sha256": sha(RUN / "protocol.json"),
        "output_sha256": {p.relative_to(folder).as_posix(): sha(p) for p in sorted(folder.rglob("*")) if p.is_file()}})


def check_seal(folder):
    manifest = read(folder / "manifest.json")
    common.require(manifest["protocol_sha256"] == sha(RUN / "protocol.json"), "Output protocol binding differs.")
    for name, expected in manifest["output_sha256"].items():
        common.require(not Path(name).is_absolute() and ".." not in Path(name).parts, "Unsafe sealed path.")
        common.checked_read(folder / name, expected)


def prepare():
    common.require(not RUN.exists(), "End-to-end run exists; refuse overwrite.")
    previous.verify()
    old = read(previous.RUN / "protocol.json")
    quality = read(previous.RUN / "training/quality.json")
    common.require(all(quality["heads"][h]["passed"] for h in core.SEMANTIC_HEADS), "Requested semantic heads did not pass prior teacher-agreement gates.")
    common.require(quality["all_heads_passed"] is False, "Preserved three-head result differs.")
    methods = list(core.METHODS)
    test_labels = read(frozen.RUN / "protocol.json")["test_label_membership"]
    frozen.core.validate_test_membership(test_labels, {r["source_id"] for values in old["membership"].values() for r in values})
    RUN.mkdir(parents=True, exist_ok=False)
    write(RUN / "protocol.json", {"schema": "trace.end_to_end.protocol.v1", "created_at_utc": stamp(),
        "authorization": "proceed with what next; approved end-to-end Full TRACE improvement comparison",
        "settings": SETTINGS, "methods": methods, "semantic_heads": list(core.SEMANTIC_HEADS),
        "old_three_head_gate_modified": False, "unsafe_head_included": False,
        "ablation": "New explicit two-passing-head feature ablation; not promotion of the rejected three-head scorer.",
        "membership": old["membership"], "nested_plan": old["nested_plan"], "groups": old["groups"],
        "auxiliary_labels": old["auxiliary_label_membership"], "auxiliary_provenance": old["label_provenance"],
        "test_membership": test_labels, "encoder_file_sha256": old["encoder_file_sha256"],
        "protected_input_sha256": protected_inputs(), "implementation_sha256": codes(),
        "selection": "Mean outer-fold AP, then AUROC, then declared order; primary Full selected from linear/nonlinear before validation or test scoring.",
        "cv_cutoffs": "Each outer-fit model's own fitting predictions calibrate max-F1/FPR<=0.10/recall>=0.80; held labels excluded. New combiners use cross-fitted component fitting features.",
        "final_cutoffs": "Validation-only max-F1 (higher threshold tie), highest sensitivity at FPR<=0.10, highest threshold with recall>=0.80. Reference max-F1 cutoffs preserved.",
        "numeric_threshold_convention": "Scores >= threshold; candidate above 1 permits no-positive operation even at saturated probability 1.",
        "semantic_embedding_cache": "Reuse the original 80 reviewed frozen pair embeddings; encode remaining training and validation texts with the identical frozen pair encoder.",
        "point_win_rule": "Selected Full F1 must exceed Generation and both enhanced/raw Behavioral-only references on each reused split; report uncertainty and operating-point tradeoffs separately, never infer fresh generalization.",
        "retrospective_not_fresh_holdout": True, "earlier_test_errors_informed_design": True,
        "test_threshold_or_model_selection_allowed": False, "new_activation_labels_allowed": False,
        "target_model_calls_allowed": False, "dashboard_promotion_allowed": False,
        "benchmark_process_orders": [methods[i*2:] + methods[:i*2] for i in range(3)],
        "warm_boundary": "In-memory interaction to activation score: required generation features, existing behavioral encoding, extra semantic pair encoding, component heads, scaling and combiner. Shared encoder weights, both passes counted.",
        "warm_exclusions": ["target generation/statistics collection", "network/UI", "disk/hash reads", "initial loading/imports"],
        "cold_boundary": "Method state/model setup and first inference after shared imports/preflight; not complete application startup.",
        "memory_boundary": "Whole worker resident and lifetime peak; parent/target excluded; encoder hashes checked by parent before workers."})
    return {"prepared": True, "methods": 6, "outer_folds": 5, "inner_partitions": 25,
            "classifiers_planned": 123, "semantic_heads": list(core.SEMANTIC_HEADS), "reused_test_examples": 476}


def arrays():
    import numpy as np
    archive = np.load(audit.REPR / "results/final-features.npz", allow_pickle=False)
    return {s: {"base": archive[f"{s}__prompt_response_raw__behavioral"], "generation": archive[f"{s}__generation_raw"]} for s in ("train", "validation")}


def reused_reference_scores(split):
    import numpy as np
    if split == "train":
        archive = np.load(audit.REPR / "results/cv-scores.npz", allow_pickle=False)
        ids = archive["configuration_ids"].tolist()
        return {m: archive["scores"][ids.index(key)] for m, key in REFERENCE_CONFIGS.items()}
    records = read(frozen.RUN / "validation-records.json")
    return {m: np.asarray([r["reference_scores"][key] for r in records]) for m, key in REFERENCE_KEYS.items()}


def train():
    import joblib
    import numpy as np
    import torch
    protocol = check_protocol()
    folder = RUN / "training"
    common.require(not folder.exists(), "End-to-end training exists; refuse refit/overwrite.")
    folder.mkdir(exist_ok=False)
    write(folder / "started.json", {"started_at_utc": stamp(), "protocol_sha256": sha(RUN / "protocol.json")})
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    values = arrays()
    records = shared.load_text_records({"train": protocol["membership"]["train"], "validation": []})["train"]
    old_aux = np.load(previous.RUN / "training/features-and-scores.npz", allow_pickle=False)
    lookup = {r["sample_id"]: i for i, r in enumerate(records)}
    reviewed = [lookup[str(s)] for s in old_aux["sample_ids"]]
    missing = [i for i in range(len(records)) if i not in set(reviewed)]
    encoder = previous.core.PairEncoder(shared.ENCODER_PATH)
    semantic = np.empty((len(records), 384), dtype=np.float32)
    semantic[reviewed] = old_aux["features"]
    print(f"Encoding {len(missing)} unreviewed training interactions; reusing 80 sealed semantic embeddings", flush=True)
    semantic[missing] = encoder.encode([records[i] for i in missing])
    result = core.cross_validate(values["train"]["base"], semantic, values["train"]["generation"],
        protocol["membership"], protocol["auxiliary_labels"], protocol["nested_plan"], SETTINGS,
        progress=lambda message: print(message, flush=True))
    reference = reused_reference_scores("train")
    ref_errors = {m: float(np.max(abs(result["oof_scores"][m] - ref))) for m, ref in reference.items()}
    common.require(max(ref_errors.values()) <= SETTINGS["reference_reproduction_tolerance"], "Outer reference models differ from pre-existing source CV.")
    semantic_error = float(np.max(abs(result["outer_behavior"][reviewed, 1:] - old_aux["oof"][:, :2])))
    common.require(semantic_error <= SETTINGS["reference_reproduction_tolerance"], "Refitted passing semantic heads differ from original source CV.")
    selection = {**result["selection"], "selected_at_utc": stamp()}
    write(folder / "selection.json", selection)
    print("Training-CV Full selection frozen; fitting final models and calibrating validation operating points", flush=True)
    all_indices = np.arange(len(records))
    y = np.asarray([r["activation_label"] for r in protocol["membership"]["train"]])
    components, final_audits = core.fit_components(values["train"]["base"], semantic, protocol["membership"],
        protocol["auxiliary_labels"], all_indices, {r["source_id"] for r in protocol["membership"]["validation"]}, SETTINGS)
    references = core.fit_references(values["train"]["base"], values["train"]["generation"], y, all_indices, SETTINGS)
    combiners = {m: core.fit_combiner(m, values["train"]["generation"], result["outer_behavior"], y, SETTINGS) for m in core.NEW_METHODS}
    validation_records = shared.load_text_records({"train": [], "validation": protocol["membership"]["validation"]})["validation"]
    semantic_validation = encoder.encode(validation_records)
    validation_behavior = core.component_scores(components, values["validation"]["base"], semantic_validation)
    validation_scores = core.reference_scores(references, validation_behavior, values["validation"]["base"], values["validation"]["generation"])
    validation_ref_errors = {m: float(np.max(abs(validation_scores[m] - v))) for m, v in reused_reference_scores("validation").items()}
    common.require(max(validation_ref_errors.values()) <= SETTINGS["reference_reproduction_tolerance"], "Final reference validation scores differ.")
    validation_scores.update(reused_reference_scores("validation"))
    for method in core.NEW_METHODS:
        validation_scores[method] = core.combiner_scores(combiners[method], method, values["validation"]["generation"], validation_behavior)
    vy = np.asarray([r["activation_label"] for r in protocol["membership"]["validation"]])
    calibrated = {m: core.thresholds(vy, validation_scores[m]) for m in core.METHODS}
    old_thresholds = read(frozen.RUN / "protocol.json")["thresholds"]
    for method, old in REFERENCE_KEYS.items():
        common.require(abs(calibrated[method]["max_f1"] - old_thresholds[old]) <= 1e-12, "Reference cutoff is not the prior validation calibration.")
        calibrated[method]["max_f1"] = old_thresholds[old]
    states = core.make_states(components, references, combiners, calibrated)
    (folder / "states").mkdir()
    for method, state in states.items():
        state["protocol_sha256"] = sha(RUN / "protocol.json")
        dump_new(folder / "states" / f"{method}.joblib", state)
    dump_new(folder / "model-bundles.joblib", {"cv": result["model_bundles"], "final": {"components": components, "references": references, "combiners": combiners}})
    write(folder / "component-fit-audit.json", result["component_fit_audits"] + [{"stage": "final", "outer_fold": None, "inner_fold": None, **a} for a in final_audits])
    write(folder / "fold-calibrations.json", result["fold_calibrations"])
    npz_new(folder / "semantic-embeddings.npz", train=semantic, validation=semantic_validation)
    npz_new(folder / "nested-scores.npz", outer_behavior=result["outer_behavior"],
        **{f"inner_{key}": v for key, v in result["inner_behavior"].items()},
        **{f"oof_{m}": v for m, v in result["oof_scores"].items()})
    train_rows = core.prediction_rows(protocol["membership"]["train"], result["oof_scores"], decisions=result["decisions"])
    validation_rows = core.prediction_rows(protocol["membership"]["validation"], validation_scores, policies=calibrated)
    shared.write_new(folder / "training-oof.jsonl", common.jsonlines(train_rows))
    shared.write_new(folder / "validation-scores.jsonl", common.jsonlines(validation_rows))
    benchmark_records = []
    for label in protocol["membership"]["validation"]:
        row = frozen.record_for(label)
        index = len(benchmark_records)
        row.pop("activation_label", None)
        row["reference_scores"] = {m: float(validation_scores[m][index]) for m in core.METHODS}
        benchmark_records.append(row)
    write(folder / "benchmark-records.json", benchmark_records)
    report = {"schema": "trace.end_to_end.training.v1", "completed_at_utc": stamp(), "selection": selection,
        "calibrated_thresholds": calibrated, "cv_results": core.summarize(train_rows), "cv_source_slices": core.source_slices(train_rows),
        "validation_results": core.summarize(validation_rows), "validation_source_slices": core.source_slices(validation_rows),
        "cv_fold_results": result["fold_results"],
        "cv_bootstrap": core.paired_intervals(train_rows, selection["selected_full"], SETTINGS),
        "validation_bootstrap": core.paired_intervals(validation_rows, selection["selected_full"], SETTINGS),
        "reference_oof_max_error": ref_errors, "reference_validation_max_error": validation_ref_errors,
        "reviewed_semantic_oof_max_error": semantic_error, "classifier_fits": {"components": 93, "references": 12, "combiners": 18, "total": 123},
        "unsafe_head_excluded": True, "old_three_head_gate_modified": False, "new_activation_labels": 0,
        "encoder_fine_tuning_updates": 0, "target_model_calls": 0, "test_rows_scored_in_training": 0, "dashboard_promoted": False,
        "limitations": ["Model selection uses outer CV; CV is development/selection evidence, not an unbiased new independent study.",
            "Training cutoffs use outer-fit model scores; combiner calibration is resubstitution on cross-fitted component features, with no held-source label access.",
            "Validation has eleven positives and calibrates cutoffs; its performance is not independent confirmation.",
            "Semantic supervision is one user-approved AI-assisted, error-enriched label set, not verified independent human semantic judgment.",
            "Fixed two-head ablation explicitly excludes the failed unsafe head; original gates and results remain unchanged.",
            "Matched FPR/recall targets apply to calibration data, not guaranteed rates under shifted evaluation data."]}
    write(folder / "report.json", report)
    seal(folder)
    check_protocol()
    return {"training_complete": True, "selected_full": selection["selected_full"], "best_cv_method": selection["best_all_methods"],
        "validation_f1": {m: report["validation_results"]["validation"]["max_f1"][m]["f1"] for m in core.METHODS},
        "classifier_fits": 123, "dashboard_promoted": False}


def point_findings(report, selected):
    return {split: {"selected_full": selected, "full_f1": values["max_f1"][selected]["f1"],
        "beats_generation_and_both_behavioral_point_f1": all(values["max_f1"][selected]["f1"] > values["max_f1"][ref]["f1"] for ref in ("generation", "reference_behavioral", "behavioral_enhanced")),
        "interpretation": "Descriptive frozen operating-point comparison on reused data, not independently confirmed generalization."} for split, values in report.items()}


def evaluate():
    import joblib
    import numpy as np
    import torch
    protocol = check_protocol()
    check_seal(RUN / "training")
    folder = RUN / "evaluation"
    common.require(not folder.exists(), "End-to-end reused evaluation exists; refuse repeat/overwrite.")
    folder.mkdir(exist_ok=False)
    write(folder / "started.json", {"started_at_utc": stamp(), "training_manifest_sha256": sha(RUN / "training/manifest.json")})
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    bundle = joblib.load(RUN / "training/model-bundles.joblib")["final"]
    training = read(RUN / "training/report.json")
    labels = protocol["test_membership"]
    frozen.core.validate_test_membership(labels, {r["source_id"] for v in protocol["membership"].values() for r in v})
    previous_rows = lines(frozen.RUN / "evaluation/predictions.jsonl")
    feature_cache = np.load(frozen.RUN / "evaluation/features.npz", allow_pickle=False)
    common.require(feature_cache["sample_ids"].tolist() == [r["sample_id"] for r in labels], "Reused-test cache membership differs.")
    records = [frozen.record_for(r) for r in labels]
    base, generation = feature_cache["base"], feature_cache["generation"]
    semantic = previous.core.PairEncoder(shared.ENCODER_PATH).encode(records)
    behavior = core.component_scores(bundle["components"], base, semantic)
    reference = core.reference_scores(bundle["references"], behavior, base, generation)
    scores = {}
    for method, old in REFERENCE_KEYS.items():
        scores[method] = np.asarray([r["scores"][old] for r in previous_rows])
        common.require(np.max(abs(scores[method] - reference[method])) <= SETTINGS["reference_reproduction_tolerance"], "Reused reference classifier differs from frozen predictions.")
    for row, label in zip(previous_rows, labels):
        common.require(all(row[k] == label[k] for k in ("sample_id", "source_id", "split", "condition_id", "activation_label")), "Reused-label membership differs.")
    for method in core.NEW_METHODS:
        scores[method] = core.combiner_scores(bundle["combiners"][method], method, generation, behavior)
    rows = core.prediction_rows(labels, scores, policies=training["calibrated_thresholds"])
    for row, old in zip(rows, previous_rows):
        for method, key in REFERENCE_KEYS.items():
            common.require(row["predictions"]["max_f1"][method] == old["predictions"][key], "Historical reference decision changed.")
    result = core.summarize(rows)
    selected = training["selection"]["selected_full"]
    report = {"schema": "trace.end_to_end.retrospective.v1", "completed_at_utc": stamp(),
        "selected_full_frozen_before_validation_and_test": selected, "results": result,
        "point_findings": point_findings(result, selected),
        "paired_source_bootstrap": {s: core.paired_intervals([r for r in rows if r["split"] == s], selected, SETTINGS) for s in result},
        "retrospective_not_fresh_holdout": True, "initial_single_target_results": True,
        "reference_primary_scores_and_decisions_reused_exactly": True,
        "model_or_threshold_selection_on_test": False, "new_labels": 0, "target_model_calls": 0, "dashboard_promoted": False,
        "limitations": ["Previously inspected StrongREJECT data informed development; no independent generalization claim.",
            "Same 476 admitted examples and 75 excluded source groups; unchanged original activation labels and provenance limitations.",
            "Matched operating points are validation-calibrated targets; actual test FPR/recall can shift.",
            "Auxiliary labels are user-approved AI-assisted judgments, not independent human semantic validation.",
            "Only one Llama 2 target and independent prompts; no other-LLM or validated chat claim."]}
    shared.write_new(folder / "predictions.jsonl", common.jsonlines(rows))
    npz_new(folder / "features.npz", semantic=semantic, behavior=behavior,
            **{f"scores_{m}": v for m, v in scores.items()})
    write(folder / "report.json", report)
    seal(folder)
    check_protocol()
    return {"retrospective_evaluation_complete": True, "selected_full": selected,
        "f1": {s: {m: r["f1"] for m, r in v["max_f1"].items()} for s, v in result.items()},
        "findings": report["point_findings"], "dashboard_promoted": False}


def worker(method, replicate):
    import joblib
    import numpy as np
    from traceguard.runtime import process_memory, system_memory
    from scripts.trace_error_efficiency import statistics, cpu_name
    from threadpoolctl import threadpool_limits, threadpool_info
    protocol = check_protocol(light=True)
    check_seal(RUN / "training")
    common.require(method in core.METHODS and replicate in (0, 1, 2), "Unknown end-to-end worker.")
    records = read(RUN / "training/benchmark-records.json")
    baseline_memory = process_memory()
    start = perf_counter_ns()
    state = joblib.load(RUN / "training/states" / f"{method}.joblib")
    behavior_encoder = semantic_encoder = None
    if method != "generation":
        import torch
        from sentence_transformers import SentenceTransformer
        from traceguard.behavior import BehavioralEncoder
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        behavior_encoder = BehavioralEncoder(model_name=str(shared.ENCODER_PATH), local_only=True)
        behavior_encoder._model = SentenceTransformer(str(shared.ENCODER_PATH), device="cpu", local_files_only=True)
        if method in core.NEW_METHODS:
            semantic_encoder = previous.core.PairEncoder(shared.ENCODER_PATH, model=behavior_encoder._model)
    limiter = threadpool_limits(limits=4)
    loaded = perf_counter_ns()
    core.predict_one(state, records[0], behavior_encoder, semantic_encoder)
    first = perf_counter_ns()
    for row in records:
        core.predict_one(state, row, behavior_encoder, semantic_encoder)
    warm_memory = process_memory()
    observations, maximum_error, boundaries = [], 0., []
    rng = random.Random(SETTINGS["seed"] + replicate)
    for number in (0, 1):
        order = list(range(100))
        rng.shuffle(order)
        for index in order:
            row = records[index]
            t0 = perf_counter_ns()
            value = core.predict_one(state, row, behavior_encoder, semantic_encoder)
            elapsed = (perf_counter_ns() - t0) / 1e6
            reference = row["reference_scores"][method]
            error = abs(value - reference)
            common.require(error <= SETTINGS["score_tolerance"], "Complete detector differs from frozen validation reference.")
            cutoff = state["thresholds"]["max_f1"]
            if (value >= cutoff) != (reference >= cutoff):
                common.require(abs(reference - cutoff) <= SETTINGS["score_tolerance"], "Decision difference away from numeric threshold boundary.")
                boundaries.append(row["sample_id"])
            maximum_error = max(maximum_error, error)
            observations.append({"sample_id": row["sample_id"], "pass": number, "latency_ms": elapsed})
    measured = process_memory()
    pools = threadpool_info()
    limiter.restore_original_limits()
    write(RUN / "benchmark" / f"{method}-{replicate}.json", {"method": method, "replicate": replicate,
        "protocol_sha256": sha(RUN / "protocol.json"), "partition": "validation", "observations": observations,
        "latency_ms": statistics([r["latency_ms"] for r in observations]),
        "cold_post_import_setup_first_ms": (first - start)/1e6, "cold_first_call_ms": (first-loaded)/1e6,
        "memory": {"baseline": baseline_memory, "warm": warm_memory, "measured": measured},
        "maximum_reference_error": maximum_error, "boundary_occurrences": len(boundaries), "boundary_sample_ids": sorted(set(boundaries)),
        "cpu": cpu_name(), "logical_cpu_count": os.cpu_count(), "system_memory": system_memory(),
        "platform": platform.platform(), "python": platform.python_version(), "threadpool_info": pools,
        "encoder_instances": 0 if method == "generation" else 1,
        "encoder_passes_per_inference": 2 if method in core.NEW_METHODS else 0 if method == "generation" else 1,
        "torch_threads": 4 if behavior_encoder else None, "torch_interop_threads": 1 if behavior_encoder else None,
        "model_fits": 0, "test_rows_scored": 0, "target_model_calls": 0})


def summarize_workers(workers, protocol):
    import numpy as np
    from scripts.trace_error_efficiency import statistics
    result = {}
    for method in core.METHODS:
        part = [w for w in workers if w["method"] == method]
        common.require(len(part) == 3 and {w["replicate"] for w in part} == {0, 1, 2}, "End-to-end worker coverage differs.")
        result[method] = {"warm_latency_ms": statistics([r["latency_ms"] for w in part for r in w["observations"]]),
            "per_process_median_ms": [w["latency_ms"]["median"] for w in part],
            "median_resident_worker_bytes": float(np.median([w["memory"]["measured"]["working_set_bytes"] for w in part])),
            "maximum_lifetime_peak_worker_bytes": max(w["memory"]["measured"]["peak_working_set_bytes"] for w in part),
            "cold_post_import_setup_first_ms": statistics([w["cold_post_import_setup_first_ms"] for w in part]),
            "maximum_reference_error": max(w["maximum_reference_error"] for w in part),
            "boundary_occurrences": sum(w["boundary_occurrences"] for w in part),
            "boundary_sample_ids": sorted({sid for w in part for sid in w["boundary_sample_ids"]}),
            "state_file_bytes": (RUN / "training/states" / f"{method}.joblib").stat().st_size,
            "required_encoder_files_bytes": 0 if method == "generation" else sum((shared.ENCODER_PATH / name).stat().st_size for name in protocol["encoder_file_sha256"])}
    return result


def benchmark():
    protocol = check_protocol()
    check_seal(RUN / "training")
    folder = RUN / "benchmark"
    common.require(not folder.exists(), "End-to-end benchmark exists; refuse repeat/overwrite.")
    folder.mkdir(exist_ok=False)
    write(folder / "started.json", {"started_at_utc": stamp(), "protocol_sha256": sha(RUN / "protocol.json")})
    workers = []
    for replicate, order in enumerate(protocol["benchmark_process_orders"]):
        for method in order:
            subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), "worker", "--method", method, "--replicate", str(replicate)],
                cwd=ROOT, env=dict(os.environ), check=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            workers.append(read(folder / f"{method}-{replicate}.json"))
            print(json.dumps({"worker_complete": method, "replicate": replicate}), flush=True)
    write(folder / "summary.json", {"schema": "trace.end_to_end.efficiency.v1", "completed_at_utc": stamp(),
        "methods": summarize_workers(workers, protocol), "processes": 18, "observations_per_method": 600,
        "partition": "validation", "cpu": workers[0]["cpu"], "logical_cpu_count": workers[0]["logical_cpu_count"],
        "python": workers[0]["python"], "platform": workers[0]["platform"],
        "available_memory_bytes_range": [min(w["system_memory"]["available_bytes"] for w in workers), max(w["system_memory"]["available_bytes"] for w in workers)],
        "limitations": ["Complete detector computation only; target generation/statistics collection, UI/network and I/O excluded.",
            "Enhanced Behavioral and Full candidates include two encoder forward passes with one shared encoder instance.",
            "Whole worker memory, not just parameter memory; parent and target excluded.",
            "Post-import setup is not total startup. Background workloads, power and file caches uncontrolled; no historical speedup or cross-hardware claim."]})
    check_protocol()
    seal(folder)
    return {"benchmark_complete": True, "processes": 18, "timings": 3600}


def verify():
    import joblib
    import numpy as np
    protocol = check_protocol()
    for folder in ("training", "evaluation", "benchmark"):
        check_seal(RUN / folder)
    data = arrays()
    semantic = np.load(RUN / "training/semantic-embeddings.npz", allow_pickle=False)
    nested = np.load(RUN / "training/nested-scores.npz", allow_pickle=False)
    bundle = joblib.load(RUN / "training/model-bundles.joblib")
    audits = read(RUN / "training/component-fit-audit.json")
    labels = protocol["auxiliary_labels"]
    rows = protocol["membership"]["train"]
    y = np.asarray([r["activation_label"] for r in rows])
    old_aux = np.load(previous.RUN / "training/features-and-scores.npz", allow_pickle=False)
    lookup = {r["sample_id"]: i for i, r in enumerate(rows)}
    common.require(np.array_equal(semantic["train"][[lookup[str(s)] for s in old_aux["sample_ids"]]], old_aux["features"]), "Reviewed embeddings changed.")
    oof = {m: np.full(len(rows), np.nan) for m in core.METHODS}
    decisions = {p: {m: np.full(len(rows), -1, dtype=int) for m in core.METHODS} for p in core.POLICIES}
    fold_results = []
    saved_calibrations = read(RUN / "training/fold-calibrations.json")
    for outer in protocol["nested_plan"]:
        fit, held = np.asarray(outer["fit_indices"]), np.asarray(outer["held_indices"])
        reconstructed = np.full((len(rows), 3), np.nan)
        stages = [("outer", outer)] + [("inner", inner) for inner in outer["inner_folds"]]
        for kind, stage in stages:
            f, h = np.asarray(stage["fit_indices"]), np.asarray(stage["held_indices"])
            key = f"outer-{outer['fold']}" + (f"/inner-{stage['inner_fold']}" if kind == "inner" else "")
            components = bundle["cv"][key]["components"]
            require_sources = {rows[i]["source_id"] for i in f}
            forbidden = {rows[i]["source_id"] for i in h} | ({rows[i]["source_id"] for i in held} if kind == "inner" else set())
            for component in ("behavioral_activation",) + core.SEMANTIC_HEADS:
                item, = [a for a in audits if a["stage"] == kind and a["outer_fold"] == outer["fold"] and a["inner_fold"] == stage.get("inner_fold") and a["component"] == component]
                expected = [rows[i]["sample_id"] for i in f] if component == "behavioral_activation" else [r["sample_id"] for r in labels if r["source_id"] in require_sources and r["auxiliary_targets"][component] is not None]
                common.require(item["fit_sample_ids"] == expected and not (set(item["fit_source_ids"]) & forbidden), "Nested component fitting leaked sources/targets.")
            scores = core.component_scores(components, data["train"]["base"][h], semantic["train"][h])
            if kind == "inner":
                reconstructed[h] = scores
            else:
                common.require(np.array_equal(scores, nested["outer_behavior"][h]), "Outer behavior scores differ.")
        common.require(np.array_equal(reconstructed[fit], nested[f"inner_{outer['fold']}"]), "Nested cross-fitted features differ.")
        model = bundle["cv"][f"outer-{outer['fold']}"]
        fitting_behavior = core.component_scores(model["components"], data["train"]["base"][fit], semantic["train"][fit])
        training_scores = core.reference_scores(model["references"], fitting_behavior, data["train"]["base"][fit], data["train"]["generation"][fit])
        held_scores = core.reference_scores(model["references"], nested["outer_behavior"][held], data["train"]["base"][held], data["train"]["generation"][held])
        for m in core.NEW_METHODS:
            training_scores[m] = core.combiner_scores(model["combiners"][m], m, data["train"]["generation"][fit], reconstructed[fit])
            held_scores[m] = core.combiner_scores(model["combiners"][m], m, data["train"]["generation"][held], nested["outer_behavior"][held])
            scaler = model["combiners"][m]["scaler"]
            if scaler is not None:
                common.require(np.allclose(scaler.mean_, core.combiner_input(m, data["train"]["generation"][fit], reconstructed[fit]).mean(axis=0), atol=1e-12), "Combiner scaler is not fitted on nested training features.")
        policies = {m: core.thresholds(y[fit], training_scores[m]) for m in core.METHODS}
        common.require(policies == saved_calibrations[outer["fold"]]["policies"], "Training-only fold cutoffs differ.")
        fold_results.append({m: core.score_metrics(y[held], held_scores[m], policies[m]["max_f1"]) for m in core.METHODS})
        for m in core.METHODS:
            oof[m][held] = held_scores[m]
            common.require(np.array_equal(held_scores[m], nested[f"oof_{m}"][held]), "OOF activation scores differ.")
            for p in core.POLICIES:
                decisions[p][m][held] = (held_scores[m] >= policies[m][p]).astype(int)
    selection = core.selection(fold_results)
    saved_selection = read(RUN / "training/selection.json")
    common.require(all(saved_selection[k] == v for k, v in selection.items()), "Training-only Full selection differs.")
    report = read(RUN / "training/report.json")
    train_rows = core.prediction_rows(rows, oof, decisions=decisions)
    common.require(train_rows == lines(RUN / "training/training-oof.jsonl")
                   and report["cv_results"] == core.summarize(train_rows), "CV rows or metrics differ.")
    final = bundle["final"]
    behavior = core.component_scores(final["components"], data["validation"]["base"], semantic["validation"])
    validation_scores = reused_reference_scores("validation")
    validation_scores.update({m: core.combiner_scores(final["combiners"][m], m, data["validation"]["generation"], behavior) for m in core.NEW_METHODS})
    vy = [r["activation_label"] for r in protocol["membership"]["validation"]]
    for m in core.METHODS:
        calibrated = core.thresholds(vy, validation_scores[m])
        common.require(calibrated == report["calibrated_thresholds"][m], "Validation-only cutoff reconstruction differs.")
    validation_rows = core.prediction_rows(protocol["membership"]["validation"], validation_scores, policies=report["calibrated_thresholds"])
    common.require(validation_rows == lines(RUN / "training/validation-scores.jsonl") and report["validation_results"] == core.summarize(validation_rows), "Validation results differ.")
    for part, key in ((train_rows, "cv_bootstrap"), (validation_rows, "validation_bootstrap")):
        common.require(report[key] == core.paired_intervals(part, selection["selected_full"], SETTINGS), "Development bootstrap differs.")
    test_archive = np.load(RUN / "evaluation/features.npz", allow_pickle=False)
    test_cache = np.load(frozen.RUN / "evaluation/features.npz", allow_pickle=False)
    test_behavior = core.component_scores(final["components"], test_cache["base"], test_archive["semantic"])
    common.require(np.array_equal(test_behavior, test_archive["behavior"]), "Test component score reconstruction differs.")
    old_rows = lines(frozen.RUN / "evaluation/predictions.jsonl")
    scores = {m: np.asarray([r["scores"][key] for r in old_rows]) for m, key in REFERENCE_KEYS.items()}
    scores.update({m: core.combiner_scores(final["combiners"][m], m, test_cache["generation"], test_behavior) for m in core.NEW_METHODS})
    test_rows = core.prediction_rows(protocol["test_membership"], scores, policies=report["calibrated_thresholds"])
    test_report = read(RUN / "evaluation/report.json")
    common.require(test_rows == lines(RUN / "evaluation/predictions.jsonl") and test_report["results"] == core.summarize(test_rows), "Reused-test classifier scores or metrics differ.")
    for split in test_report["results"]:
        common.require(test_report["paired_source_bootstrap"][split] == core.paired_intervals([r for r in test_rows if r["split"] == split], selection["selected_full"], SETTINGS), "Reused-test bootstrap differs.")
    workers = [read(RUN / "benchmark" / f"{m}-{i}.json") for i in (0, 1, 2) for m in core.METHODS]
    expected = {r["sample_id"] for r in protocol["membership"]["validation"]}
    from scripts.trace_error_efficiency import statistics
    for worker in workers:
        for p in (0, 1):
            ids = [r["sample_id"] for r in worker["observations"] if r["pass"] == p]
            common.require(len(ids) == len(set(ids)) == 100 and set(ids) == expected, "Complete detector benchmark membership differs.")
        common.require(worker["model_fits"] == worker["test_rows_scored"] == worker["target_model_calls"] == 0,
                       "Complete detector benchmark scope differs.")
        common.require(worker["latency_ms"] == statistics([r["latency_ms"] for r in worker["observations"]]), "Worker timing summaries differ.")
    common.require(read(RUN / "benchmark/summary.json")["methods"] == summarize_workers(workers, protocol), "Complete detector resource aggregate differs.")
    value = {"schema": "trace.end_to_end.verification.v1", "verified": True,
        "nested_component_fit_ids_features_and_scalers_verified": True, "cv_and_validation_selection_cutoffs_metrics_reconstructed": True,
        "reused_test_scores_and_paired_intervals_verified": True, "matched_complete_resource_measurements_verified": True,
        "old_gate_inputs_labels_and_artifacts_unchanged": True, "classifier_fits": 123, "benchmark_timings": 3600,
        "selected_full": selection["selected_full"], "target_model_calls": 0, "dashboard_promoted": False,
        "reused_test_is_retrospective": True, "semantic_labels_are_AI_assisted": True}
    path = RUN / "verification.json"
    if path.exists():
        common.require(read(path) == value, "Existing verification differs.")
    else:
        write(path, value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "train", "evaluate", "benchmark", "worker", "verify"))
    parser.add_argument("--method", choices=core.METHODS)
    parser.add_argument("--replicate", type=int)
    args = parser.parse_args()
    if args.command == "worker":
        worker(args.method, args.replicate)
    else:
        print(json.dumps({"prepare": prepare, "train": train, "evaluate": evaluate,
            "benchmark": benchmark, "verify": verify}[args.command](), indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
