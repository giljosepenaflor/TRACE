"""Prepare, run, and verify a development-only behavioral representation experiment.

Uses sealed train/validation evidence. No held-out response reads, remote calls,
backdoor training, label changes, or live-detector replacement.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for variable in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
    os.environ[variable] = "1"
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "4"
from scripts import trace_consensus_train as original
from scripts import trace_consensus_behavioral as shared
from scripts import trace_consensus_intake as intake
from traceguard import representation_experiment as experiment

DEFAULT_RUN = ROOT / "artifacts/trace-behavioral-v4/representation-001"
ORIGINAL_RUN = ROOT / "artifacts/trace-training-v3/dual-view-candidate-001"
ORIGINAL_PROTOCOL_SHA = "5188509235618a8ceab833504b8d6e9e6e7b7cd498565888a9fa53c4f773f42d"
ORIGINAL_FREEZE_SHA = "19ea0e24da8096bfa96d1c697b1dbb626c2ac8357458f6b82b6937604ea31910"
SETTINGS = {**experiment.DEFAULT_SETTINGS, "torch_cpu_threads": 4,
            "torch_interop_threads": 1, "representations": [list(x) for x in experiment.REPRESENTATIONS],
            "base_dimensions": 384, "projection_dimensions": 64, "device": "cpu",
            "selection_objective": "mean training-fold average precision",
            "threshold_policy": "highest validation F1; highest threshold on ties; score >= threshold",
            "fusion": "original feature concatenation; score fusion deferred",
            "auxiliary_safety_feature": "not part of this experiment"}


def new_json(path, value):
    shared.write_new(path, intake.encoded(value))


def digest(path):
    return intake.sha(path.read_bytes())


def code_files():
    paths = (Path(__file__).resolve(), ROOT / "traceguard/representation_experiment.py")
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}


def original_inputs():
    """Verify the original frozen definition before accessing reviewed development rows."""
    protocol = json.loads(intake.checked_read(ORIGINAL_RUN / "protocol.json", ORIGINAL_PROTOCOL_SHA))
    freeze = json.loads(intake.checked_read(ORIGINAL_RUN / "model/freeze.json", ORIGINAL_FREEZE_SHA))
    for name, expected in protocol["code_file_sha256"].items():
        intake.checked_read(ROOT / name, expected)
    for name, expected in protocol["encoder_file_sha256"].items():
        intake.checked_read(shared.ENCODER_PATH / name, expected)
    for name, expected in freeze["output_sha256"].items():
        intake.require(Path(name).name == name, "Unsafe original artifact path.")
        intake.checked_read(ORIGINAL_RUN / "model" / name, expected)
    status, membership, qualification = verify_sealed_development()
    intake.require(membership == protocol["membership"], "Original sealed development membership changed.")
    original.verify_evidence()
    return protocol, status, membership, qualification


def verify_sealed_development():
    """Consume the immutable label seal without recreating a missing historical CSV.

    The old intake's rebuild gate is unchanged. This new workflow binds the
    independently pinned source catalogue, all sealed outputs, current reviewed
    inputs, and all 3,076 development raw hashes. No test response is accessed.
    """
    manifest = json.loads(intake.checked_read(shared.LABEL_RUN / "manifest.json", shared.LABEL_SEAL_SHA))
    intake.require(manifest["input_sha256"] == intake.PINS, "Original label-seal input bindings changed.")
    missing = []
    for name, expected in intake.PINS.items():
        path = ROOT / name
        if path.is_file():
            intake.checked_read(path, expected)
        else:
            missing.append(name)
    intake.require(set(missing) <= {"strongreject-dataset-for-training.csv"}, "Required development evidence is missing.")
    for name, expected in manifest["output_sha256"].items():
        intake.require(Path(name).name == name, "Unsafe sealed label path.")
        intake.checked_read(shared.LABEL_RUN / name, expected)
    labels = shared.line_rows(shared.LABEL_RUN / "activation-labels.jsonl")
    label_protocol = shared.load(shared.LABEL_RUN / "protocol.json")
    source_protocol = json.loads(intake.checked_read(ROOT / intake.SOURCE_PROTOCOL, intake.PINS[intake.SOURCE_PROTOCOL]))
    intake.require(label_protocol["source_assignments"] == source_protocol["source_assignments"], "Sealed source catalogue differs.")
    membership = shared.validate_membership(labels, label_protocol["source_assignments"])
    index = shared.line_rows(shared.LABEL_RUN / "record-index.jsonl")
    intake.require(len(index) == 3076 and {r["split"] for r in index} == {"train", "validation"}, "Unexpected development inventory.")
    lookup, raw_hashes = {}, {}
    for row in index:
        name = row["source_id"] + f"-condition-{row['condition_id']}.json"
        intake.require(row["raw_path"] == intake.RAW + "/" + name, "Record points outside frozen development collection.")
        intake.require(name not in raw_hashes, "Duplicate development raw record.")
        intake.checked_read(ROOT / row["raw_path"], row["raw_sha256"])
        raw_hashes[name] = row["raw_sha256"]
        lookup[(row["source_id"], row["condition_id"])] = row
    intake.require({p.name for p in (ROOT / intake.RAW).glob("*.json")} == set(raw_hashes), "Raw collection membership changed.")
    collection_digest = intake.sha(json.dumps(raw_hashes, sort_keys=True, separators=(",", ":")).encode())
    intake.require(collection_digest == manifest["raw_collection_filename_sha256_digest"], "Raw collection digest differs.")
    for row in labels:
        indexed = lookup[(row["source_id"], row["condition_id"])]
        for key in ("raw_path", "raw_sha256", "executed_prompt_sha256", "response_sha256", "review_id", "split"):
            intake.require(row[key] == indexed[key], "Label and sealed raw index differ.")
    status = manifest["summary"]
    for split, rows in membership.items():
        counts = status["label_counts"][split]
        intake.require(len(rows) == counts["adapted_rows"]
                       and sum(r["activation_label"] for r in rows) == counts["positive"], "Sealed label counts differ.")
    return status, membership, {
        "method": "pinned_sealed_development_inventory_and_available_source_catalogue",
        "label_manifest_sha256": shared.LABEL_SEAL_SHA,
        "source_catalogue_sha256": intake.PINS[intake.SOURCE_PROTOCOL],
        "missing_historical_input_paths": missing, "historical_csv_recreated": False,
        "old_intake_rebuild_gate_changed": False, "development_raw_hashes_verified": len(index),
        "qualification": "The historical StrongREJECT CSV is unavailable; frozen catalogue and sealed labels are reused without claiming its recovery.",
    }


def preservation_snapshot():
    """Hash frozen seals/pointers and original artifacts, without parsing test outcomes."""
    paths = list(ORIGINAL_RUN.rglob("*")) + list(shared.LABEL_RUN.rglob("*"))
    paths += [ROOT / name for name in (
        "artifacts/trace-training-v3/strongreject-consensus-test-001/evaluation/manifest.json",
        "artifacts/trace-training-v3/strongreject-consensus-test-001/labels/manifest.json",
        "artifacts/mtba-development/run-001/trace-activation-detector-001/detector.joblib",
        "artifacts/mtba-harmbench/run-001/pipeline-001/main-detector.json",
        "traceguard/mtba_live_detector.py", "traceguard/server.py")]
    return {p.relative_to(ROOT).as_posix(): digest(p) if p.is_file() else None
            for p in sorted(set(paths)) if p.is_file() or p.name in ("main-detector.json", "manifest.json")}


def duplicate_clusters(membership):
    frozen = json.loads(intake.checked_read(ROOT / intake.SOURCE_PROTOCOL, intake.PINS[intake.SOURCE_PROTOCOL]))
    clusters = list(frozen["duplicate_short_question_clusters"])
    # Also join identical complete, executed, uncued inputs across data sources.
    texts = shared.load_text_records(membership)
    by_text = defaultdict(set)
    condition = {r["sample_id"]: r["condition_id"] for rows in membership.values() for r in rows}
    for rows in texts.values():
        for row in rows:
            if condition[row["sample_id"]] == 1:
                normalized = " ".join(row["prompt"].casefold().split())
                by_text[normalized].add(row["source_id"])
    for question, sources in sorted(by_text.items()):
        if len(sources) > 1:
            clusters.append({"source_ids": sorted(sources), "normalized_executed_input_sha256": intake.text_sha(question)})
    return clusters


def prepare(run):
    intake.require(not run.exists(), "Experiment directory exists; refuse overwrite.")
    source, status, membership, catalogue = original_inputs()
    clusters = duplicate_clusters(membership)
    folds, groups = experiment.make_folds(membership, clusters, SETTINGS)
    protocol = {
        "schema": "trace.behavioral_representation.protocol.v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "authorization": "proceed with the next step; Improve Behavioral Representation please do it",
        "settings": SETTINGS, "families": experiment.family_specs(), "membership": membership,
        "label_counts": status["label_counts"], "duplicate_clusters": clusters, "groups": groups, "folds": folds,
        "target_model": source["target_model"], "base_revision": source["base_revision"],
        "adapter": source["adapter"], "adapter_revision": source["adapter_revision"],
        "original_protocol_sha256": ORIGINAL_PROTOCOL_SHA, "original_freeze_sha256": ORIGINAL_FREEZE_SHA,
        "label_seal_sha256": shared.LABEL_SEAL_SHA, "encoder_revision": shared.ENCODER_REV,
        "encoder_file_sha256": source["encoder_file_sha256"], "original_code_file_sha256": source["code_file_sha256"],
        "new_code_file_sha256": code_files(), "environment": original.environment(),
        "admission": source["admission"], "development_catalogue_admission": catalogue,
        "preserved_file_sha256": preservation_snapshot(),
        "test_response_access_allowed": False, "target_model_calls_allowed": False,
        "label_changes_allowed": False, "dashboard_promotion_allowed": False,
        "existing_test_error_informed_design": True, "fresh_holdout_required_for_improvement_claim": True,
        "scope": "Development comparison; fixed C=1/unweighted rows isolate representation, bounded training-CV tuning is secondary.",
    }
    run.mkdir(parents=True, exist_ok=False)
    new_json(run / "protocol.json", protocol)
    return {"prepared": True, "run": str(run), "training_examples": len(membership["train"]),
            "validation_examples": len(membership["validation"]), "families": len(protocol["families"]),
            "configurations": len(protocol["families"]) * len(SETTINGS["C_grid"]) * len(SETTINGS["class_weight_grid"]),
            "folds": len(folds), "training_started": False}


def check_protocol(run):
    protocol = shared.load(run / "protocol.json")
    intake.require(protocol["settings"] == SETTINGS and protocol["new_code_file_sha256"] == code_files(),
                   "Experiment code/settings changed after protocol freeze.")
    source, _, membership, catalogue = original_inputs()
    intake.require(protocol["membership"] == membership, "Experiment membership changed.")
    intake.require(protocol["encoder_file_sha256"] == source["encoder_file_sha256"], "Encoder identity changed.")
    intake.require(protocol["development_catalogue_admission"] == catalogue, "Development catalogue admission changed.")
    folds, groups = experiment.make_folds(membership, duplicate_clusters(membership), SETTINGS)
    intake.require(protocol["folds"] == folds and protocol["groups"] == groups, "Source folds changed.")
    intake.require(protocol["preserved_file_sha256"] == preservation_snapshot(), "A protected original file changed.")
    return protocol


def encode_development(records):
    import numpy as np
    from traceguard.behavior import BehavioralEncoder
    from traceguard.features import interaction_text

    encoder = BehavioralEncoder(model_name=str(shared.ENCODER_PATH), local_only=True)
    base, diagnostics = {}, {}
    for mode in ("prompt_response", "response_only"):
        encoder.representation_mode = mode
        base[mode], diagnostics[mode] = {}, {}
        for split, rows in records.items():
            print(f"Encoding {mode} {split}: {len(rows)} reviewed examples", flush=True)
            values = encoder.encode_base(rows)
            intake.require(values.shape == (len(rows), 384) and np.isfinite(values).all(), "Unexpected embedding matrix.")
            intake.require(np.allclose(np.linalg.norm(values, axis=1), 1, atol=1e-5), "Embeddings are not normalized.")
            base[mode][split] = values
            lengths = list(map(len, encoder._model.tokenizer(
                [interaction_text(row, mode) for row in rows], truncation=False, padding=False, verbose=False)["input_ids"]))
            diagnostics[mode][split] = {"count": len(rows), "min_tokens": min(lengths), "max_tokens": max(lengths),
                                      "over_limit": sum(n > encoder._model.max_seq_length for n in lengths)}
    diagnostics["encoder_limit"] = int(encoder._model.max_seq_length)
    diagnostics["truncation_side"] = encoder._model.tokenizer.truncation_side
    return base, diagnostics


def npz_new(path, **arrays):
    import numpy as np
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    shared.write_new(path, buffer.getvalue())


def report_markdown(report):
    rows = report["validation_results"]
    cv = {r["id"]: r for r in report["cv_results"]}
    selected = report["selection"]["winners"]
    lines = ["# TRACE behavioral representation development comparison", "",
             "This is a measured training/validation development experiment, not a new held-out test or a deployment.", "",
             f"Run: `{report['run']}`. Training: 1,058 reviewed examples; validation: 100 examples (11 positive, 89 negative).",
             "Four text representations and the unchanged generation features are compared. Existing activation labels and source assignments are preserved.", "",
             "## Matched representation comparison", "",
             "All rows below use C=1 and unweighted logistic regression. Thresholds maximize validation F1 independently, so operating points differ.", "",
             "| Representation and view | Training CV mean AP | Validation AP | Validation F1 | Recall | FPR | TP | FP |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for key in report["selection"]["fixed_settings_ids"]:
        value, result = rows[key], rows[key]["metrics"]
        cm = result["confusion_matrix"]
        lines.append(f"| {value['family']} | {cv[key]['mean_fold_average_precision']:.6f} | {result['average_precision']:.6f} | {result['f1']:.6f} | {result['recall']:.6f} | {result['false_positive_rate']:.6f} | {cm['tp']} | {cm['fp']} |")
    lines += ["", "## Training CV selections", "",
              "Model and representation selection used training cross-validation only, before final validation scores were produced. The bounded search uses C in {0.1, 1, 10} and class weights None/balanced.", "",
              "| Selection | Configuration | Mean CV AP | Validation AP | Validation F1 | Recall | FPR |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for view, key in selected.items():
        m = rows[key]["metrics"]
        lines.append(f"| {view} | {key} | {cv[key]['mean_fold_average_precision']:.6f} | {m['average_precision']:.6f} | {m['f1']:.6f} | {m['recall']:.6f} | {m['false_positive_rate']:.6f} |")
    lines += ["", "## Interpretation limits", "",
              "CV is used for model selection and is therefore a development estimate. Validation F1 also selects each row's threshold; it is not unbiased held-out performance. Only eleven validation positives are available.", "",
              "The fold splitter keeps source pairs and duplicate clusters together. Every learned projection and scaler is fitted separately inside each training fold. The frozen pretrained encoder can be cached without activation-label fitting.", "",
              "The original test errors informed the experiment design. No held-out responses or prediction rows were opened or rescored by this workflow. Fresh source-disjoint evidence is required before claiming improved generalization.", "",
              "Generation features, labels, raw records, historical artifacts, and dashboard pointers remain unchanged. No target-model inference, safety-judge feature, score-level fusion, or multi-model experiment was performed.", "",
              "Two-reviewer consensus and the generation-signal history retain the original researcher-attested and exact-build limitations. A response-only input can still contain echoed cues or semantic shortcuts.", "",
              "The original StrongREJECT source CSV remains unavailable. This version explicitly consumes the checksum-sealed development inventory and frozen source catalogue, and verifies all 3,076 development raw hashes. It neither recreates the CSV nor changes the old intake gate.", "",
              "`report.json`, `cv-results.json`, `fold-fit-audit.json`, `validation-scores.jsonl`, feature arrays, candidate models, and the checksum seal provide the detailed results.", ""]
    return "\n".join(lines)


def train(run):
    import joblib
    import numpy as np
    import torch

    protocol = check_protocol(run)
    output = run / "results"
    intake.require(not output.exists(), "Results already exist; refuse refit or overwrite.")
    output.mkdir(exist_ok=False)
    new_json(output / "started.json", {"started_at_utc": datetime.now(timezone.utc).isoformat(),
                                       "protocol_sha256": digest(run / "protocol.json")})
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    start = perf_counter()
    records, generation = original.load_development(protocol)
    base, tokenization = encode_development(records)
    membership = protocol["membership"]
    train_ids = [r["sample_id"] for r in membership["train"]]
    y = np.asarray([r["activation_label"] for r in membership["train"]])
    configurations, oof, audits = experiment.cross_validate(
        {mode: values["train"] for mode, values in base.items()}, generation["train"], y, train_ids,
        protocol["folds"], SETTINGS, progress=lambda message: print(message, flush=True))
    selection = experiment.select_configurations(configurations)
    selection["selected_at_utc"] = datetime.now(timezone.utc).isoformat()
    # Commit the selection before any final validation classifier prediction/calibration.
    new_json(output / "selection.json", selection)
    new_json(output / "cv-results.json", configurations)
    new_json(output / "fold-fit-audit.json", audits)
    print("Training-only model selection frozen; fitting final development models and calibrating validation thresholds", flush=True)
    final = experiment.fit_final(base, generation, membership, configurations, selection, SETTINGS)
    npz_new(output / "base-embeddings.npz", **{f"{mode}__{split}": values
            for mode, partitions in base.items() for split, values in partitions.items()})
    npz_new(output / "cv-scores.npz", scores=oof, labels=y,
            configuration_ids=np.asarray([r["id"] for r in configurations]), sample_ids=np.asarray(train_ids))
    npz_new(output / "final-features.npz", **{
        **{f"{split}__{family}": value for split, families in final["features"].items() for family, value in families.items()},
        **{f"{split}__generation_raw": value for split, value in generation.items()}})
    artifact = {"schema": "trace.behavioral_representation.candidates.v1", "settings": SETTINGS,
                "models": final["models"], "projections": final["projections"],
                "projection_training_info": final["projection_training_info"],
                "generation_scaler": final["generation_scaler"], "selection": selection,
                "protocol_sha256": digest(run / "protocol.json"), "target_model": protocol["target_model"],
                "encoder_revision": shared.ENCODER_REV, "encoder_path": str(shared.ENCODER_PATH),
                "development_only": True, "heldout_evaluated": False, "dashboard_promoted": False}
    buffer = io.BytesIO()
    joblib.dump(artifact, buffer, compress=3)
    shared.write_new(output / "candidates.joblib", buffer.getvalue())
    score_rows = [{"configuration_id": key, "sample_id": row["sample_id"], "source_id": row["source_id"],
                   "split": "validation", "activation_label": row["activation_label"], "score": float(values[index]),
                   "threshold": final["models"][key]["threshold"],
                   "prediction": int(values[index] >= final["models"][key]["threshold"])}
                  for key, values in final["validation_scores"].items()
                  for index, row in enumerate(membership["validation"])]
    shared.write_new(output / "validation-scores.jsonl", intake.jsonlines(score_rows))
    reference = shared.line_rows(ORIGINAL_RUN / "model/validation-scores.jsonl")
    reference_differences = {}
    # Original score schema is preserved; compare only when explicitly recognized.
    for view in ("full", "behavioral", "generation"):
        key = experiment.config_id("generation" if view == "generation" else f"prompt_response_projected__{view}", 1, None)
        if reference and all(view in row.get("scores", {}) for row in reference):
            previous = {row["sample_id"]: row["scores"][view] for row in reference}
            reference_differences[view] = max(abs(final["validation_scores"][key][i] - previous[row["sample_id"]])
                                                for i, row in enumerate(membership["validation"]))
    report = {"schema": "trace.behavioral_representation.results.v1", "run": run.relative_to(ROOT).as_posix(),
              "completed_at_utc": datetime.now(timezone.utc).isoformat(), "fit_interval_seconds": perf_counter() - start,
              "label_counts": protocol["label_counts"], "cv_results": configurations, "selection": selection,
              "validation_results": final["validation_results"], "tokenization": tokenization,
              "original_validation_score_max_abs_difference": reference_differences,
              "target_model_calls": 0, "heldout_responses_opened": 0, "heldout_predictions_opened": 0,
              "heldout_evaluated": False, "dashboard_promoted": False,
              "original_files_unchanged": preservation_snapshot() == protocol["preserved_file_sha256"],
              "limitations": ["Model selection CV and threshold-selection validation are development evidence.",
                              "Validation has only eleven positive examples.",
                              "Existing test errors informed this design; fresh held-out evaluation is required.",
                              protocol["development_catalogue_admission"]["qualification"],
                              protocol["admission"]["qualification"], protocol["admission"]["reviewer_agreement"]]}
    intake.require(report["original_files_unchanged"], "A protected original changed during fitting.")
    new_json(output / "report.json", report)
    shared.write_new(output / "README.md", report_markdown(report).encode("utf-8"))
    new_json(output / "freeze.json", {"schema": "trace.behavioral_representation.freeze.v1",
                                     "protocol_sha256": digest(run / "protocol.json"),
                                     "output_sha256": {p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()},
                                     "heldout_evaluated": False, "dashboard_promoted": False})
    return {"trained": True, "run": str(run), "selected": selection["winners"],
            "fit_interval_seconds": report["fit_interval_seconds"], "heldout_evaluated": False}


def verify(run):
    import joblib
    import numpy as np
    from sklearn.metrics import average_precision_score, roc_auc_score
    protocol = check_protocol(run)
    output = run / "results"
    freeze = shared.load(output / "freeze.json")
    intake.require(freeze["protocol_sha256"] == digest(run / "protocol.json"), "Protocol checksum changed.")
    for name, expected in freeze["output_sha256"].items():
        intake.require(Path(name).name == name, "Unsafe result path.")
        intake.checked_read(output / name, expected)
    configurations = shared.load(output / "cv-results.json")
    selection = shared.load(output / "selection.json")
    expected_selection = experiment.select_configurations(configurations)
    intake.require({k: selection[k] for k in expected_selection} == expected_selection, "CV selection is inconsistent.")
    cv = np.load(output / "cv-scores.npz", allow_pickle=False)
    y = np.asarray([r["activation_label"] for r in protocol["membership"]["train"]])
    intake.require(np.array_equal(y, cv["labels"]), "OOF labels changed.")
    intake.require(cv["sample_ids"].tolist() == [r["sample_id"] for r in protocol["membership"]["train"]], "OOF row order changed.")
    for index, config in enumerate(configurations):
        intake.require(cv["configuration_ids"][index] == config["id"], "OOF configuration order changed.")
        ap, auc = [], []
        for fold in protocol["folds"]:
            held = fold["held_indices"]
            ap.append(float(average_precision_score(y[held], cv["scores"][index, held])))
            auc.append(float(roc_auc_score(y[held], cv["scores"][index, held])))
        intake.require(np.allclose(ap, config["fold_average_precision"], atol=1e-12)
                       and np.allclose(auc, config["fold_auroc"], atol=1e-12), "Saved fold metrics differ.")
        intake.require(abs(np.mean(ap) - config["mean_fold_average_precision"]) < 1e-12, "Mean CV AP differs.")
    bases = np.load(output / "base-embeddings.npz", allow_pickle=False)
    features = np.load(output / "final-features.npz", allow_pickle=False)
    artifact = joblib.load(output / "candidates.joblib")
    train_ids = [r["sample_id"] for r in protocol["membership"]["train"]]
    for mode in ("prompt_response", "response_only"):
        intake.require(artifact["projection_training_info"][mode]["fit_sample_ids"] == train_ids, "Final projection leaked membership.")
    for audit in shared.load(output / "fold-fit-audit.json"):
        fold = protocol["folds"][audit["fold"]]
        intake.require(audit["fit_sample_ids"] == [train_ids[i] for i in fold["fit_indices"]], "A fold projection leaked membership.")
    raw_train = features["train__generation_raw"]
    scaler = artifact["generation_scaler"]
    intake.require(np.allclose(scaler.mean_, raw_train.mean(axis=0), atol=1e-12)
                   and np.allclose(scaler.var_, raw_train.var(axis=0), atol=1e-12), "Final scaler is not training-only.")
    rebuilt = {}
    for split in ("train", "validation"):
        representations = {}
        for name, mode, projected in experiment.REPRESENTATIONS:
            value = bases[f"{mode}__{split}"]
            representations[name] = experiment.apply_projection(value, artifact["projections"][mode]) if projected else value
        rebuilt[split] = experiment.matrices(representations, scaler.transform(features[f"{split}__generation_raw"]))
        for family, values in rebuilt[split].items():
            intake.require(np.allclose(values, features[f"{split}__{family}"], atol=1e-10), "Final features cannot be reconstructed.")
    rows = shared.line_rows(output / "validation-scores.jsonl")
    report = shared.load(output / "report.json")
    y_validation = np.asarray([r["activation_label"] for r in protocol["membership"]["validation"]])
    for key, candidate in artifact["models"].items():
        predicted = candidate["model"].predict_proba(rebuilt["validation"][candidate["family"]])[:, 1]
        saved = [row for row in rows if row["configuration_id"] == key]
        intake.require([row["sample_id"] for row in saved] == [r["sample_id"] for r in protocol["membership"]["validation"]], "Validation membership differs.")
        intake.require(np.allclose(predicted, [row["score"] for row in saved], atol=1e-12), "Saved validation scores differ.")
        threshold = experiment.choose_threshold(y_validation, predicted)
        intake.require(threshold == candidate["threshold"], "Validation threshold cannot be reconstructed.")
        intake.require(experiment.score_metrics(y_validation, predicted, threshold) == report["validation_results"][key]["metrics"], "Validation metrics differ.")
        intake.require(all(row["activation_label"] == int(y_validation[i])
                           and row["prediction"] == int(predicted[i] >= threshold)
                           and row["threshold"] == threshold for i, row in enumerate(saved)), "Validation decisions differ.")
    value = {"schema": "trace.behavioral_representation.verification.v1", "verified": True,
             "protocol_and_outputs_hash_verified": True, "fold_and_duplicate_isolation_verified": True,
             "fold_metrics_and_training_only_selection_verified": True, "learned_transform_fit_ids_verified": True,
             "features_scores_thresholds_and_metrics_reconstructed": True,
             "original_development_raw_and_label_seal_reverified": True, "original_protected_files_unchanged": True,
             "development_catalogue_admission": protocol["development_catalogue_admission"],
             "heldout_responses_opened": 0, "target_model_calls": 0, "dashboard_promoted": False,
             "configuration_count": len(configurations), "final_model_count": len(artifact["models"])}
    verification = run / "verification.json"
    if verification.exists():
        intake.require(shared.load(verification) == value, "Existing verification differs.")
    else:
        new_json(verification, value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "train", "verify"))
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    run = args.run.resolve()
    intake.require(run.parent == DEFAULT_RUN.parent.resolve() and run.name.startswith("representation-"),
                   "Use a separate representation-* directory under artifacts/trace-behavioral-v4.")
    result = {"prepare": prepare, "train": train, "verify": verify}[args.command](run)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError) as error:
        print(f"TRACE representation experiment stopped: {error}", file=sys.stderr, flush=True)
        sys.exit(1)
