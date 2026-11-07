"""Bounded development-only sequence extension of TRACE generation evidence."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
from time import perf_counter

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "4"
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import trace_paired_control as paired
from traceguard import token_sequence as core

BASE = paired.previous
RUN = ROOT / "artifacts/trace-behavioral-v4/token-sequence-001"
read, lines, sha, write, stamp = paired.read, paired.lines, paired.sha, paired.write, paired.stamp
require = core.require


def codes():
    paths = (Path(__file__).resolve(), Path(core.__file__), ROOT / "tests/test_token_sequence.py",
             ROOT / "traceguard/paired_control.py", ROOT / "traceguard/end_to_end.py",
             ROOT / "traceguard/features.py", ROOT / "docs/TRACE_TOKEN_SEQUENCE_PLAN.md")
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}


def check():
    p = read(RUN / "protocol.json")
    require(p["implementation_sha256"] == codes() and p["settings"] == core.SETTINGS, "Frozen sequence code/settings changed.")
    for name, expected in p["protected_input_sha256"].items():
        require(not (ROOT / name).exists() if expected is None else sha(ROOT / name) == expected, "Protected input changed: " + name)
    return p


def seal(folder):
    write(folder / "manifest.json", {"protocol_sha256": sha(RUN / "protocol.json"),
        "output_sha256": {p.relative_to(folder).as_posix(): sha(p) for p in folder.rglob("*") if p.is_file()}})


def check_seal(folder):
    m = read(folder / "manifest.json")
    require(m["protocol_sha256"] == sha(RUN / "protocol.json"), "Output protocol binding differs.")
    for name, expected in m["output_sha256"].items():
        require(not Path(name).is_absolute() and ".." not in Path(name).parts and sha(folder / name) == expected, "Sealed sequence output changed.")


def prepare():
    require(not RUN.exists(), "Sequence run exists; refuse overwrite.")
    p = paired.check()
    paired.check_seal(paired.RUN / "development")
    require(not read(paired.RUN / "development/gate.json")["passed"], "Unexpected paired-stage policy state.")
    protected = dict(p["protected_input_sha256"])
    for path in paired.RUN.rglob("*"):
        if path.is_file():
            protected[path.relative_to(ROOT).as_posix()] = sha(path)
    RUN.mkdir()
    write(RUN / "protocol.json", {"schema": "trace.token_sequence.protocol.v1", "created_at_utc": stamp(),
        "authorization": "continued TRACE improvement; fixed token-sequence extension after failed paired-control development gate",
        "membership": p["membership"], "source_folds": p["source_folds"], "dataset_folds": p["dataset_folds"], "groups": p["groups"],
        "settings": core.SETTINGS, "methods": list(core.METHODS), "feature_names": list(core.NAMES),
        "protected_input_sha256": protected, "implementation_sha256": codes(), "fits_planned": 40,
        "methodology_extension": "18 fixed temporal/distribution summaries added to the three original generation features; original references preserved",
        "selection": "Worst training-held-dataset AP, mean dataset AP, mean five-source-fold AP; original Full first on exact ties; no validation/test selection",
        "test_gate": "Sequence Full selected, worst dataset AP improves and validation F1 strictly beats original Full and all single views",
        "test_scoring_in_this_run": False, "target_calls_allowed": False, "label_changes_allowed": False,
        "dashboard_promotion_allowed": False, "single_target_initial_development_only": True})
    return {"prepared": True, "arms": 5, "classifiers_planned": 40, "generation_features": 21, "test_scoring": False}


def get_sequence(p):
    import numpy as np
    values, signal_audit = {}, {}
    for split, labels in p["membership"].items():
        vectors, lengths = [], []
        for label in labels:
            signals = BASE.frozen.record_for(label)["signals"]
            vectors.append(core.extract(signals))
            lengths.append(len(signals["generated_token_ids"]))
        values[split] = np.stack(vectors)
        signal_audit[split] = {"count": len(labels), "min_generated_tokens": min(lengths), "max_generated_tokens": max(lengths)}
    return values, signal_audit


def train():
    import numpy as np
    p = check()
    folder = RUN / "development"
    require(not folder.exists(), "Sequence development exists; refuse refit.")
    folder.mkdir()
    write(folder / "started.json", {"started_at_utc": stamp()})
    start = perf_counter()
    data = BASE.arrays()
    sequence, signal_audit = get_sequence(p)
    for split in sequence:
        require(np.array_equal(sequence[split][:, :3], data[split]["generation"]), "Primary generation features differ from original cache.")
    rows = p["membership"]["train"]
    bundles, results, prediction_sets, calibrations = {}, {}, {}, {}
    for regime, folds in (("source", p["source_folds"]), ("dataset", p["dataset_folds"])):
        predictions, measurements, cutoffs = [], [], []
        coverage = np.zeros(len(rows), dtype=int)
        for number, fold in enumerate(folds):
            f, h = np.array(fold["fit_indices"]), np.array(fold["held_indices"])
            paired.core.partition_check(rows, f, h, p["groups"]["train"])
            models, scores, thresholds = {}, {}, {}
            for m in core.METHODS:
                state = core.fit(m, data["train"]["base"][f], sequence["train"][f], [rows[i] for i in f])
                models[m] = state
                fitting = core.predict(state, data["train"]["base"][f], sequence["train"][f])
                thresholds[m] = BASE.core.thresholds([rows[i]["activation_label"] for i in f], fitting)
                scores[m] = core.predict(state, data["train"]["base"][h], sequence["train"][h])
            held = core.prediction_rows([rows[i] for i in h], scores, thresholds)
            for i, row in zip(h, held):
                row.update(index=int(i), fold=number)
            predictions.extend(held)
            measurements.append(core.summarize(held)["max_f1"])
            cutoffs.append(thresholds)
            bundles[f"{regime}-{number}"] = models
            coverage[h] += 1
            print(json.dumps({"completed_regime": regime, "fold": number}), flush=True)
        require(np.all(coverage == 1), "Held coverage differs.")
        prediction_sets[regime] = sorted(predictions, key=lambda r: r["index"])
        results[regime], calibrations[regime] = measurements, cutoffs
    selection = core.select(results["source"], results["dataset"])
    write(folder / "selection.json", {**selection, "selected_at_utc": stamp()})
    print(json.dumps({"selection_frozen": selection["selected_full"]}), flush=True)
    final = {m: core.fit(m, data["train"]["base"], sequence["train"], rows) for m in core.METHODS}
    scores = {m: core.predict(state, data["validation"]["base"], sequence["validation"]) for m, state in final.items()}
    thresholds = {m: BASE.core.thresholds([r["activation_label"] for r in p["membership"]["validation"]], scores[m]) for m in core.METHODS}
    validation = core.prediction_rows(p["membership"]["validation"], scores, thresholds)
    measured = core.summarize(validation)
    checks = core.gate(selection, measured["max_f1"])
    gate = {"checks": checks, "passed": all(checks.values()), "separate_initial_retrospective_stage_allowed": all(checks.values()), "dashboard_promoted": False}
    reference_errors = {}
    for m, old in zip(core.METHODS[:3], ("generation", "reference_behavioral", "reference_full")):
        errors = [float(np.max(abs(np.array([r["scores"][m] for r in prediction_sets["source"]]) - BASE.reused_reference_scores("train")[old]))),
                  float(np.max(abs(scores[m] - BASE.reused_reference_scores("validation")[old])))]
        reference_errors[m] = max(errors)
        require(max(errors) <= 1e-12, "Original reference predictions differ.")
    BASE.dump_new(folder / "model-bundles.joblib", {"folds": bundles, "final": final})
    BASE.npz_new(folder / "sequence-features.npz", **sequence)
    for regime, current in prediction_sets.items():
        BASE.shared.write_new(folder / f"{regime}-held-predictions.jsonl", BASE.common.jsonlines(current))
    BASE.shared.write_new(folder / "validation-predictions.jsonl", BASE.common.jsonlines(validation))
    write(folder / "gate.json", gate)
    report = {"schema": "trace.token_sequence.development.v1", "completed_at_utc": stamp(), "selection": selection,
        "fold_results": results, "fold_calibrations": calibrations, "final_thresholds": thresholds,
        "source_oof": core.summarize(prediction_sets["source"]), "dataset_held": core.summarize(prediction_sets["dataset"]),
        "source_slices": core.slices(prediction_sets["source"]), "dataset_slices": core.slices(prediction_sets["dataset"]),
        "validation": measured, "validation_slices": core.slices(validation),
        "source_bootstrap": core.bootstrap(prediction_sets["source"], selection["selected_full"]),
        "validation_bootstrap": core.bootstrap(validation, selection["selected_full"]),
        "gate": gate, "reference_max_errors": reference_errors, "signal_audit": signal_audit,
        "elapsed_seconds_excluding_imports_preflight": perf_counter() - start,
        "fits": 40, "target_calls": 0, "encoder_updates": 0, "new_labels": 0, "test_rows_scored": 0,
        "dashboard_promoted": False, "initial_development_results_only": True}
    write(folder / "report.json", report)
    seal(folder)
    return {"trained": True, "fits": 40, "selection": selection,
            "validation_f1": {m: measured["max_f1"][m]["f1"] for m in core.METHODS}, "gate": gate}


def verify():
    import joblib
    import numpy as np
    p = check()
    folder = RUN / "development"
    check_seal(folder)
    report = read(folder / "report.json")
    data = BASE.arrays()
    sequence, audit = get_sequence(p)
    archive = np.load(folder / "sequence-features.npz", allow_pickle=False)
    require(all(np.array_equal(sequence[s], archive[s]) for s in sequence) and audit == report["signal_audit"], "Sequence extraction differs.")
    bundle = joblib.load(folder / "model-bundles.joblib")
    labels = p["membership"]["train"]
    results = {}
    for regime, folds in (("source", p["source_folds"]), ("dataset", p["dataset_folds"])):
        predictions = lines(folder / f"{regime}-held-predictions.jsonl")
        results[regime] = []
        for number, fold in enumerate(folds):
            f, h = np.array(fold["fit_indices"]), np.array(fold["held_indices"])
            paired.core.partition_check(labels, f, h, p["groups"]["train"])
            scores, thresholds = {}, {}
            for m, state in bundle["folds"][f"{regime}-{number}"].items():
                require(state["fit_sample_ids"] == [labels[i]["sample_id"] for i in f], "Fit IDs differ.")
                if state["scaler"] is not None:
                    x = sequence["train"][f, :3] if m.endswith("3") else sequence["train"][f]
                    require(np.allclose(state["scaler"].mean_, x.mean(axis=0), atol=1e-12), "Fit-only scaler differs.")
                fitting = core.predict(state, data["train"]["base"][f], sequence["train"][f])
                scores[m] = core.predict(state, data["train"]["base"][h], sequence["train"][h])
                thresholds[m] = BASE.core.thresholds([labels[i]["activation_label"] for i in f], fitting)
            require(thresholds == report["fold_calibrations"][regime][number], "Fit-only threshold differs.")
            expected = core.prediction_rows([labels[i] for i in h], scores, thresholds)
            require(all(all(predictions[i][k] == v for k, v in row.items()) for i, row in zip(h, expected)), "Held scores differ.")
            results[regime].append(core.summarize(expected)["max_f1"])
        require(core.summarize(predictions) == report["source_oof" if regime == "source" else "dataset_held"], "Held metrics differ.")
    require(core.select(results["source"], results["dataset"]) == report["selection"], "Training-only selection differs.")
    scores = {m: core.predict(s, data["validation"]["base"], sequence["validation"]) for m, s in bundle["final"].items()}
    thresholds = {m: BASE.core.thresholds([r["activation_label"] for r in p["membership"]["validation"]], scores[m]) for m in core.METHODS}
    require(thresholds == report["final_thresholds"], "Validation calibration differs.")
    validation = core.prediction_rows(p["membership"]["validation"], scores, thresholds)
    require(validation == lines(folder / "validation-predictions.jsonl") and core.summarize(validation) == report["validation"], "Validation scores/metrics differ.")
    require(core.gate(report["selection"], report["validation"]["max_f1"]) == report["gate"]["checks"], "Gate differs.")
    for key, current in (("source_bootstrap", lines(folder / "source-held-predictions.jsonl")), ("validation_bootstrap", validation)):
        require(core.bootstrap(current, report["selection"]["selected_full"]) == report[key], "Paired interval differs.")
    require(not (RUN / "evaluation").exists(), "This protocol permits development only.")
    value = {"verified": True, "feature_extraction_rebuilt": True, "source_and_duplicate_isolation": True,
             "fit_only_scalers_cutoffs_predictions_and_selection": True, "metrics_gate_and_bootstrap": True,
             "fits": 40, "target_calls": 0, "test_rows_scored": 0, "gate_passed": report["gate"]["passed"]}
    target = RUN / "verification.json"
    if target.exists():
        require(read(target) == value, "Verification changed.")
    else:
        write(target, value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "train", "verify"))
    args = parser.parse_args()
    print(json.dumps(globals()[args.command](), indent=2), flush=True)


if __name__ == "__main__":
    main()
