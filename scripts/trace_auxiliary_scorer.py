"""Train and gate a lightweight auxiliary scorer on accepted AI-assisted training labels."""
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
from scripts import trace_auxiliary_consensus as consensus
from scripts import trace_behavior_audit as audit
from traceguard import auxiliary_scorer as core

RUN = ROOT / "artifacts/trace-behavioral-v4/auxiliary-scorer-001"
read, lines, sha, write, stamp = audit.read, audit.lines, audit.sha, audit.write, audit.stamp
common, shared = audit.common, audit.shared
SETTINGS = {**core.SETTINGS, "encoding": core.ENCODING, "cpu_threads": 4, "torch_interop_threads": 1,
            "benchmark_processes": 3, "warmup_passes": 1, "measured_passes": 2,
            "benchmark_validation_records": 100, "score_tolerance": 1e-5}


def codes():
    paths = (Path(__file__).resolve(), Path(core.__file__), ROOT / "tests/test_auxiliary_scorer.py",
             ROOT / "traceguard/runtime.py", ROOT / "scripts/trace_error_efficiency.py")
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}


def label_rows():
    return lines(consensus.RUN / "auxiliary-training-labels.jsonl")


def protected_inputs():
    result = audit.snapshot()
    for folder in (audit.RUN, consensus.RUN):
        result.update({p.relative_to(ROOT).as_posix(): sha(p) for p in folder.rglob("*") if p.is_file()})
    result[consensus.SUBMISSION.relative_to(ROOT).as_posix()] = sha(consensus.SUBMISSION)
    return result


def check_protocol(light=False):
    protocol = read(RUN / "protocol.json")
    common.require(protocol["settings"] == SETTINGS and protocol["implementation_sha256"] == codes(), "Scorer implementation or settings changed after freeze.")
    common.require(protocol["label_provenance"]["submitted_label_sets"] == 1
                   and protocol["label_provenance"]["independent_human_reviews_verified"] is False, "Scorer label provenance changed.")
    if not light:
        for name, expected in protocol["protected_input_sha256"].items():
            if expected is None:
                common.require(not (ROOT / name).exists(), "Protected absent path now exists.")
            else:
                common.checked_read(ROOT / name, expected)
        for name, expected in protocol["encoder_file_sha256"].items():
            common.checked_read(shared.ENCODER_PATH / name, expected)
    return protocol


def prepare():
    common.require(not RUN.exists(), "Auxiliary scorer run exists; refuse overwrite.")
    consensus.verify()
    previous = audit.check_protocol()
    labels = label_rows()
    folds = core.nested_plan(previous["membership"], previous["folds"], previous["groups"]["train"], SETTINGS)
    support = core.check_support_plan(labels, previous["membership"], folds, SETTINGS)
    common.require(all(s["passes"] for s in support if s["stage"] == "outer"), "Insufficient source-class support for primary scorer CV.")
    source = read(consensus.RUN / "report.json")
    protocol = {"schema": "trace.auxiliary_scorer.protocol.v1", "created_at_utc": stamp(),
        "authorization": "proceed with the next step; train/validate scorer and conditionally test its features in TRACE",
        "settings": SETTINGS, "membership": previous["membership"], "groups": previous["groups"],
        "nested_plan": folds, "fitting_support": support,
        "label_provenance": source["provenance"], "auxiliary_label_membership": labels,
        "encoder_revision": shared.ENCODER_REV, "encoder_file_sha256": previous["encoder_file_sha256"],
        "implementation_sha256": codes(), "protected_input_sha256": protected_inputs(),
        "quality_rule": "Pooled known-label outer-source OOF metrics at fixed 0.5. Every head needs macro F1 >=0.70 and both class recalls >=0.60; unsafe positive recall >=0.80. All heads must pass.",
        "threshold_or_hyperparameter_search_allowed": False,
        "unknown_targets": "Retain unknowns and exclude per head; do not force to negative.",
        "interpretation": "Source-isolated agreement with user-approved AI-assisted teacher labels from an error-enriched 40-source pilot; not independently human-validated semantic accuracy.",
        "conditional_activation_plan": {"requires_all_quality_gates": True, "requires_all_nested_fit_support": True,
            "arms": ["generation", "existing_unprojected_full", "generation_plus_auxiliary", "unprojected_full_plus_auxiliary"],
            "fitting": "Auxiliary inner-source OOF features for outer activation fitting; original activation labels. Fixed C=1 unweighted activation classifiers; training-only scalers; training-CV AP selection; validation-only max-F1 thresholds.",
            "if_gate_fails": "Do not fit activation augmentation, score StrongREJECT or promote dashboard; retain trained scorer as a failed development candidate.",
            "later_reused_tests": "Initial retrospective only, under a separate frozen comparison stage."},
        "benchmark_boundary": "Auxiliary scorer only: in-memory text to three scores, including pair tokenization/budgeting, encoding/pooling/normalization and head inference; no target generation, signal collection, activation classifier, network/UI, disk/hash reads or initial load.",
        "cold_boundary": "State/model setup and first call after shared imports/preflight; not full process startup.",
        "memory_boundary": "Whole worker resident/lifetime peak; parent/target excluded; encoder hashes checked by parent before workers.",
        "test_outcome_access_allowed_in_this_stage": False, "new_activation_labels_allowed": False,
        "target_model_calls_allowed": False, "dashboard_promotion_allowed": False}
    RUN.mkdir(parents=True, exist_ok=False)
    write(RUN / "protocol.json", protocol)
    return {"prepared": True, "reviewed_items": len(labels), "outer_folds": 5,
            "nested_fit_support_stages_checked": len(support), "all_nested_support_passed": all(s["passes"] for s in support),
            "scorer_head_fits_planned": 18, "head_cutoff": 0.5, "test_outcome_access": False}


def seal(folder):
    write(folder / "manifest.json", {"protocol_sha256": sha(RUN / "protocol.json"),
        "output_sha256": {p.name: sha(p) for p in sorted(folder.iterdir()) if p.is_file()}})


def check_seal(folder):
    manifest = read(folder / "manifest.json")
    common.require(manifest["protocol_sha256"] == sha(RUN / "protocol.json"), "Scorer output binding differs.")
    for name, expected in manifest["output_sha256"].items():
        common.require(Path(name).name == name, "Unsafe scorer output path.")
        common.checked_read(folder / name, expected)


def dump_new(path, value):
    import joblib
    buffer = io.BytesIO()
    joblib.dump(value, buffer, compress=3)
    shared.write_new(path, buffer.getvalue())


def npz_new(path, **values):
    import numpy as np
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **values)
    shared.write_new(path, buffer.getvalue())


def reviewed_records(protocol):
    lookup = {r["sample_id"]: r for r in protocol["membership"]["train"]}
    selected = [lookup[r["sample_id"]] for r in protocol["auxiliary_label_membership"]]
    return shared.load_text_records({"train": selected, "validation": []})["train"]


def validate_encoder(encoder):
    import numpy as np
    harmless = [{"prompt": "Why do rainbows form?", "response": "Sunlight changes direction in water droplets."},
                {"prompt": "Name a primary color.", "response": "Blue is a primary color."}]
    values = encoder.encode(harmless)
    reference = encoder.model.encode([(r["prompt"], r["response"]) for r in harmless],
        batch_size=2, show_progress_bar=False, normalize_embeddings=True)
    error = float(np.max(abs(values - reference)))
    common.require(error <= 1e-6, "Short pair encoding differs from the installed SentenceTransformer pair path.")
    return {"harmless_records": 2, "maximum_embedding_difference_from_public_pair_encode": error,
            "frozen_encoder_eval_mode": not encoder.model.training}


def train():
    import numpy as np
    import torch
    protocol = check_protocol()
    folder = RUN / "training"
    common.require(not folder.exists(), "Auxiliary scorer training exists; refuse refit/overwrite.")
    folder.mkdir(exist_ok=False)
    write(folder / "started.json", {"started_at_utc": stamp(), "protocol_sha256": sha(RUN / "protocol.json")})
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    labels = protocol["auxiliary_label_membership"]
    records = reviewed_records(protocol)
    encoder = core.PairEncoder(shared.ENCODER_PATH)
    encoder_check = validate_encoder(encoder)
    print("Encoding 80 reviewed training responses with fixed request/response token budgets", flush=True)
    features, diagnostics = encoder.encode(records, return_diagnostics=True)
    result = core.cross_validate(features, labels, protocol["membership"], protocol["nested_plan"], SETTINGS)
    quality = result["quality"]
    all_support = all(s["passes"] for s in protocol["fitting_support"])
    decision = {"created_at_utc": stamp(), "all_head_quality_gates_passed": quality["all_heads_passed"],
                "nested_class_support_passed": all_support,
                "activation_augmentation_allowed": quality["all_heads_passed"] and all_support,
                "decision_frozen_before_unlabeled_validation_benchmark_inference": True,
                "failed_heads": [h for h, value in quality["heads"].items() if not value["passed"]]}
    write(folder / "quality-decision.json", decision)
    write(folder / "quality.json", quality)
    all_train_sources = {r["source_id"] for r in protocol["membership"]["train"]}
    final, final_audits = core.fit_heads(features, labels, all_train_sources,
        {r["source_id"] for r in protocol["membership"]["validation"]}, SETTINGS)
    fit_audits = result["fit_audits"] + [{"stage": "final", "fold": None, **a} for a in final_audits]
    state = {"schema": "trace.auxiliary_scorer.state.v1", "models": final, "head_order": list(core.HEADS),
        "encoding": core.ENCODING, "settings": core.SETTINGS, "protocol_sha256": sha(RUN / "protocol.json"),
        "encoder_revision": shared.ENCODER_REV, "encoder_path": str(shared.ENCODER_PATH),
        "teacher_label_provenance": protocol["label_provenance"], "quality_decision": decision,
        "development_only": True, "activation_classifier": False, "dashboard_promoted": False}
    dump_new(folder / "scorer.joblib", state)
    dump_new(folder / "fold-models.joblib", result["fold_models"])
    write(folder / "fit-audit.json", fit_audits)
    tokenization = []
    for record, info in zip(records, diagnostics):
        prepared, _ = core.pair_features(encoder.model.tokenizer, record["prompt"], record["response"])
        tokenization.append({"sample_id": record["sample_id"], "diagnostics": info, "features": prepared})
    write(folder / "tokenization.json", tokenization)
    npz_new(folder / "features-and-scores.npz", features=features, oof=result["oof_scores"], majority=result["majority_scores"],
            in_sample=core.score_heads(final, features), sample_ids=np.asarray([r["sample_id"] for r in labels]))
    score_rows = [{"sample_id": row["sample_id"], "source_id": row["source_id"], "split": "train",
        "auxiliary_targets": row["auxiliary_targets"], "outer_oof_scores": dict(zip(core.HEADS, map(float, result["oof_scores"][i]))),
        "majority_predictions": dict(zip(core.HEADS, map(int, result["majority_scores"][i])))} for i, row in enumerate(labels)]
    shared.write_new(folder / "oof-scores.jsonl", common.jsonlines(score_rows))
    # Unlabeled validation records are used only for resource/reference measurement after the quality decision.
    benchmark_records = shared.load_text_records({"train": [], "validation": protocol["membership"]["validation"]})["validation"]
    benchmark_features = encoder.encode(benchmark_records)
    references = core.score_heads(final, benchmark_features)
    for index, row in enumerate(benchmark_records):
        row["split"] = "validation"
        row["reference_scores"] = references[index].tolist()
    write(folder / "benchmark-records.json", benchmark_records)
    npz_new(folder / "benchmark-features.npz", features=benchmark_features)
    report = {"schema": "trace.auxiliary_scorer.training.v1", "completed_at_utc": stamp(),
        "quality": quality, "quality_decision": decision, "source_bootstrap": result["uncertainty"],
        "encoder_numerical_check": encoder_check, "reviewed_items": len(labels), "reviewed_source_groups": 40,
        "head_fits": len(fit_audits), "encoder_fine_tuning_updates": 0,
        "encoding_diagnostics": {"request_clipped_examples": sum(r["request_tokens"] > r["request_tokens_kept"] for r in diagnostics),
            "response_clipped_examples": sum(r["response_tokens"] > r["response_tokens_kept"] for r in diagnostics),
            "maximum_input_tokens": max(r["input_tokens"] for r in diagnostics)},
        "threshold_search_calls": 0, "activation_classifier_fits": 0, "test_outcomes_parsed": 0,
        "target_model_calls": 0, "new_activation_labels": 0, "dashboard_promoted": False,
        "limitations": ["Held-group metrics measure agreement with AI-assisted teacher labels, not independently human-validated semantic accuracy.",
            "The error-enriched 40-source pilot was selected using prior development errors; it is not a representative population test.",
            "C=1, unweighted heads and 0.5 cutoffs are fixed; no tuning or weakening of gates after outcomes.",
            "Unknown labels are excluded per head. Source bootstrap is exploratory and conditional on the selected pilot/labels.",
            "Final full-pilot models are retained even if quality fails, solely as rejected development artifacts and for cost measurement.",
            "Validation benchmark inference uses no auxiliary/activation validation targets; it is not a new accuracy evaluation.",
            "Original labels/results, provenance limitations and the dashboard remain unchanged."]}
    write(folder / "report.json", report)
    seal(folder)
    check_protocol()
    return {"trained": True, "head_fits": len(fit_audits), "quality": quality,
            "activation_augmentation_allowed": decision["activation_augmentation_allowed"], "dashboard_promoted": False}


def worker(replicate):
    import joblib
    import torch
    import numpy as np
    from traceguard.runtime import process_memory, system_memory
    from threadpoolctl import threadpool_limits, threadpool_info
    check_protocol(light=True)
    check_seal(RUN / "training")
    common.require(replicate in (0, 1, 2), "Unknown benchmark replicate.")
    records = read(RUN / "training/benchmark-records.json")
    baseline_memory = process_memory()
    start = perf_counter_ns()
    state = joblib.load(RUN / "training/scorer.joblib")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    limiter = threadpool_limits(limits=4)
    encoder = core.PairEncoder(shared.ENCODER_PATH)
    encoder.load()
    loaded = perf_counter_ns()
    core.predict_one(state, records[0], encoder)
    first = perf_counter_ns()
    for record in records:
        core.predict_one(state, record, encoder)
    warm_memory = process_memory()
    observations, maximum_error, boundary = [], 0.0, []
    rng = random.Random(SETTINGS["seed"] + replicate)
    for number in (0, 1):
        order = list(range(100))
        rng.shuffle(order)
        for index in order:
            row = records[index]
            t0 = perf_counter_ns()
            values = core.predict_one(state, row, encoder)
            elapsed = (perf_counter_ns() - t0) / 1e6
            reference = np.asarray(row["reference_scores"])
            error = float(np.max(abs(values - reference)))
            common.require(error <= SETTINGS["score_tolerance"], "Scorer worker differs from frozen benchmark reference.")
            differences = np.flatnonzero((values >= .5) != (reference >= .5))
            for head in differences:
                common.require(abs(reference[head] - .5) <= SETTINGS["score_tolerance"], "Scorer decision change away from numerical boundary.")
                boundary.append({"sample_id": row["sample_id"], "head": core.HEADS[head]})
            maximum_error = max(maximum_error, error)
            observations.append({"sample_id": row["sample_id"], "pass": number, "latency_ms": elapsed})
    measured = process_memory()
    from scripts.trace_error_efficiency import statistics, cpu_name
    pools = threadpool_info()
    limiter.restore_original_limits()
    write(RUN / "benchmark" / f"worker-{replicate}.json", {"replicate": replicate,
        "protocol_sha256": sha(RUN / "protocol.json"), "partition": "validation", "observations": observations,
        "latency_ms": statistics([r["latency_ms"] for r in observations]),
        "cold_post_import_setup_first_ms": (first - start) / 1e6, "cold_first_call_ms": (first - loaded) / 1e6,
        "memory": {"baseline": baseline_memory, "warm": warm_memory, "measured": measured},
        "maximum_score_reference_difference": maximum_error, "boundary_differences": boundary,
        "cpu": cpu_name(), "logical_cpu_count": os.cpu_count(), "system_memory": system_memory(),
        "platform": platform.platform(), "python": platform.python_version(), "threadpool_info": pools,
        "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
        "model_fits": 0, "test_rows_scored": 0, "validation_labels_used": 0, "target_model_calls": 0})


def summarize_workers(workers, protocol):
    import numpy as np
    from scripts.trace_error_efficiency import statistics
    common.require(len(workers) == 3 and {w["replicate"] for w in workers} == {0, 1, 2}, "Scorer workers incomplete/repeated.")
    return {"warm_latency_ms": statistics([r["latency_ms"] for w in workers for r in w["observations"]]),
        "per_process_median_ms": [w["latency_ms"]["median"] for w in workers],
        "cold_post_import_setup_first_ms": statistics([w["cold_post_import_setup_first_ms"] for w in workers]),
        "median_resident_worker_bytes": float(np.median([w["memory"]["measured"]["working_set_bytes"] for w in workers])),
        "maximum_lifetime_peak_worker_bytes": max(w["memory"]["measured"]["peak_working_set_bytes"] for w in workers),
        "maximum_reference_error": max(w["maximum_score_reference_difference"] for w in workers),
        "boundary_difference_occurrences": sum(len(w["boundary_differences"]) for w in workers),
        "state_file_bytes": (RUN / "training/scorer.joblib").stat().st_size,
        "required_encoder_files_bytes": sum((shared.ENCODER_PATH / name).stat().st_size for name in protocol["encoder_file_sha256"])}


def benchmark():
    protocol = check_protocol()
    check_seal(RUN / "training")
    folder = RUN / "benchmark"
    common.require(not folder.exists(), "Scorer benchmark exists; refuse repeat/overwrite.")
    folder.mkdir(exist_ok=False)
    write(folder / "started.json", {"started_at_utc": stamp(), "protocol_sha256": sha(RUN / "protocol.json")})
    workers = []
    for replicate in (0, 1, 2):
        subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), "worker", "--replicate", str(replicate)],
            cwd=ROOT, env=dict(os.environ), check=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        workers.append(read(folder / f"worker-{replicate}.json"))
        print(json.dumps({"benchmark_worker_complete": replicate}), flush=True)
    write(folder / "summary.json", {"schema": "trace.auxiliary_scorer.efficiency.v1", "completed_at_utc": stamp(),
        "summary": summarize_workers(workers, protocol), "processes": 3, "observations": 600,
        "partition": "validation", "cpu": workers[0]["cpu"], "logical_cpu_count": workers[0]["logical_cpu_count"],
        "python": workers[0]["python"], "platform": workers[0]["platform"],
        "available_memory_bytes_range": [min(w["system_memory"]["available_bytes"] for w in workers), max(w["system_memory"]["available_bytes"] for w in workers)],
        "limitations": ["Auxiliary scorer only; not complete TRACE or target-LLM response latency.",
            "Whole worker memory, parent/target excluded. Background workload, power and file cache uncontrolled.",
            "Cold interval starts after shared imports/preflight, not interpreter launch. Fresh workers are not cold-machine caches.",
            "Artifact sizes exclude runtime libraries, target weights, datasets and reports.",
            "No historical speedup claim or cross-hardware benchmark follows from this local measurement."]})
    check_protocol()
    seal(folder)
    return {"benchmark_complete": True, "processes": 3, "observations": 600}


def verify():
    import joblib
    import numpy as np
    from transformers import AutoTokenizer
    protocol = check_protocol()
    for name in ("training", "benchmark"):
        check_seal(RUN / name)
    labels = protocol["auxiliary_label_membership"]
    folds = core.nested_plan(protocol["membership"], audit.read(audit.RUN / "protocol.json")["folds"], protocol["groups"]["train"], SETTINGS)
    common.require(folds == protocol["nested_plan"] and core.check_support_plan(labels, protocol["membership"], folds, SETTINGS) == protocol["fitting_support"], "Frozen nested plan or source-class support differs.")
    archive = np.load(RUN / "training/features-and-scores.npz", allow_pickle=False)
    common.require(archive["sample_ids"].tolist() == [r["sample_id"] for r in labels], "Scorer feature order differs.")
    tokenization = read(RUN / "training/tokenization.json")
    tokenizer = AutoTokenizer.from_pretrained(str(shared.ENCODER_PATH), local_files_only=True)
    for saved, record in zip(tokenization, reviewed_records(protocol)):
        features, info = core.pair_features(tokenizer, record["prompt"], record["response"])
        common.require(saved == {"sample_id": record["sample_id"], "diagnostics": info, "features": features}, "Fixed pair tokenization differs.")
    models = joblib.load(RUN / "training/fold-models.joblib")
    oof, majority = np.full_like(archive["oof"], np.nan), np.full_like(archive["majority"], np.nan)
    audits = read(RUN / "training/fit-audit.json")
    for fold in folds:
        fit_sources = {protocol["membership"]["train"][i]["source_id"] for i in fold["fit_indices"]}
        held_sources = {protocol["membership"]["train"][i]["source_id"] for i in fold["held_indices"]}
        held = [i for i, r in enumerate(labels) if r["source_id"] in held_sources]
        oof[held] = core.score_heads(models[str(fold["fold"])], archive["features"][held])
        for j, head in enumerate(core.HEADS):
            audit_row, = [a for a in audits if a["stage"] == "outer" and a["fold"] == fold["fold"] and a["head"] == head]
            expected = [r["sample_id"] for r in labels if r["source_id"] in fit_sources and r["auxiliary_targets"][head] is not None]
            common.require(audit_row["fit_sample_ids"] == expected and not (set(audit_row["fit_source_ids"]) & held_sources), "Auxiliary fitting leaked sources/unknown labels.")
            majority[held, j] = audit_row["majority_class"]
    common.require(np.array_equal(oof, archive["oof"]) and np.array_equal(majority, archive["majority"]), "OOF scores/majority references cannot be reconstructed.")
    report = read(RUN / "training/report.json")
    quality = core.assess_quality(labels, oof, majority, SETTINGS)
    common.require(report["quality"] == quality == read(RUN / "training/quality.json")
                   and report["source_bootstrap"] == core.group_bootstrap(labels, oof, SETTINGS), "Teacher metrics/gates/bootstrap differ.")
    state = joblib.load(RUN / "training/scorer.joblib")
    common.require(state["quality_decision"] == report["quality_decision"] == read(RUN / "training/quality-decision.json"), "Frozen quality decision differs.")
    common.require(state["quality_decision"]["activation_augmentation_allowed"] == (quality["all_heads_passed"] and all(s["passes"] for s in protocol["fitting_support"])), "Conditional augmentation gate differs.")
    common.require(np.array_equal(core.score_heads(state["models"], archive["features"]), archive["in_sample"]), "Final scorer reconstruction differs.")
    for head in core.HEADS:
        row, = [a for a in audits if a["stage"] == "final" and a["head"] == head]
        common.require(row["fit_sample_ids"] == [r["sample_id"] for r in labels if r["auxiliary_targets"][head] is not None], "Final scorer used unknown/outside targets.")
    refs = read(RUN / "training/benchmark-records.json")
    features = np.load(RUN / "training/benchmark-features.npz", allow_pickle=False)["features"]
    values = core.score_heads(state["models"], features)
    common.require(np.array_equal(values, np.asarray([r["reference_scores"] for r in refs])), "Scorer resource-reference scores differ.")
    common.require(all("activation_label" not in r and "auxiliary_targets" not in r for r in refs), "Validation label leaked into scorer benchmark input.")
    workers = [read(RUN / "benchmark" / f"worker-{i}.json") for i in (0, 1, 2)]
    expected_ids = {r["sample_id"] for r in protocol["membership"]["validation"]}
    from scripts.trace_error_efficiency import statistics
    for w in workers:
        common.require(w["partition"] == "validation" and w["protocol_sha256"] == sha(RUN / "protocol.json"), "Benchmark partition/protocol differs.")
        for number in (0, 1):
            ids = [r["sample_id"] for r in w["observations"] if r["pass"] == number]
            common.require(len(ids) == len(set(ids)) == 100 and set(ids) == expected_ids, "Scorer benchmark coverage differs.")
        common.require(w["model_fits"] == w["test_rows_scored"] == w["validation_labels_used"] == w["target_model_calls"] == 0, "Scorer benchmark scope differs.")
        common.require(w["latency_ms"] == statistics([r["latency_ms"] for r in w["observations"]]), "Scorer timing summary differs.")
    common.require(read(RUN / "benchmark/summary.json")["summary"] == summarize_workers(workers, protocol), "Scorer timing/memory aggregate differs.")
    value = {"schema": "trace.auxiliary_scorer.verification.v1", "verified": True,
        "protected_inputs_and_encoding_policy_verified": True, "outer_and_nested_group_support_reconstructed": True,
        "fitting_ids_unknown_exclusions_and_source_isolation_verified": True,
        "cached_teacher_scores_fixed_gates_and_bootstrap_reconstructed": True,
        "final_models_and_validation_resource_references_verified": True,
        "validation_timing_coverage_and_statistics_verified": True, "head_fits": len(audits),
        "quality_all_heads_passed": quality["all_heads_passed"], "scorer_is_human_validated": False,
        "benchmark_observations": 600, "target_model_calls": 0, "test_outcomes_parsed": 0, "dashboard_promoted": False}
    path = RUN / "verification.json"
    if path.exists():
        common.require(read(path) == value, "Existing scorer verification differs.")
    else:
        write(path, value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "train", "benchmark", "worker", "verify"))
    parser.add_argument("--replicate", type=int)
    args = parser.parse_args()
    if args.command == "worker":
        worker(args.replicate)
    else:
        print(json.dumps({"prepare": prepare, "train": train, "benchmark": benchmark, "verify": verify}[args.command](), indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
