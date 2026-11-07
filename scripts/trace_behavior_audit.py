"""Development-only error audit, controlled cue sensitivity and blank auxiliary review."""
from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import sys

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "4"
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import trace_behavioral_experiment as development
from scripts import trace_reused_evaluation as frozen
from traceguard import behavior_audit as core
from traceguard import score_fusion

RUN = ROOT / "artifacts/trace-behavioral-v4/development-audit-002"
REPR = development.DEFAULT_RUN
FUSION = ROOT / "artifacts/trace-behavioral-v4/score-fusion-001"
SPEC = ROOT / "docs/TRACE_BEHAVIOR_FEATURE_SPEC.md"
read, lines, sha, write, stamp = frozen.read, frozen.lines, frozen.sha, frozen.write, frozen.stamp
common, shared = frozen.common, frozen.shared
SETTINGS = {"seed": 20261106, "pilot_groups_per_stratum": 8, "pilot_target_groups": 40,
            "cue_probe_partition": "validation", "cue": "BadMagic", "encoder_limit": 256,
            "cpu_threads": 4, "torch_interop_threads": 1, "score_tolerance": 1e-5,
            "training_errors_cutoff": "existing validation-frozen thresholds applied descriptively to OOF scores"}


def codes():
    files = (Path(__file__).resolve(), Path(core.__file__), ROOT / "tests/test_behavior_audit.py", SPEC,
             ROOT / "traceguard/score_fusion.py", ROOT / "traceguard/reused_comparison.py",
             ROOT / "traceguard/behavior.py", ROOT / "traceguard/features.py", ROOT / "traceguard/detector.py")
    return {frozen.relative(p): sha(p) for p in files}


def snapshot():
    result = development.preservation_snapshot()
    for folder in (REPR, FUSION / "development"):
        result.update({frozen.relative(p): sha(p) for p in folder.rglob("*") if p.is_file()})
    for method in core.METHODS:
        if method != "score_fusion":
            path = frozen.RUN / "states" / f"{method}.joblib"
            result[frozen.relative(path)] = sha(path)
    for name in ("evaluation/manifest.json", "evaluation/predictions.jsonl", "evaluation/report.json", "labels/manifest.json"):
        path = frozen.test.RUN / name
        result[frozen.relative(path)] = sha(path)
    return result


def prepare():
    common.require(not RUN.exists(), "Audit run exists; refuse overwrite.")
    development.verify(REPR)  # Development-only reconstruction; does not parse test outcomes.
    protocol = read(REPR / "protocol.json")
    core.validate_development([r for rows in protocol["membership"].values() for r in rows])
    manifest = read(FUSION / "development/manifest.json")
    for name, expected in manifest["output_sha256"].items():
        common.checked_read(FUSION / "development" / name, expected)
    RUN.mkdir(parents=True, exist_ok=False)
    write(RUN / "protocol.json", {"schema": "trace.development_behavior_audit.protocol.v1", "created_at_utc": stamp(),
        "authorization": "go; approved development error audit and behavior feature specification",
        "preflight_correction": "development-audit-001 stopped before measurements because an absent-path sentinel was treated as a file hash; its protocol, implementation and failure record remain preserved.",
        "settings": SETTINGS, "membership": protocol["membership"], "folds": protocol["folds"],
        "groups": protocol["groups"], "admission": protocol["admission"],
        "development_catalogue_admission": protocol["development_catalogue_admission"],
        "implementation_sha256": codes(), "protected_input_sha256": snapshot(),
        "encoder_file_sha256": protocol["encoder_file_sha256"],
        "scope": "Descriptive development audit and feature/rubric specification; no new classifier or threshold.",
        "cue_probe": "Remove only the declared appended development cue from validation requests, holding generated response and all generation evidence fixed; condition1 is an unchanged control.",
        "lexical_marker_interpretation": "Descriptive text markers only, not semantic refusal/compliance or activation labels.",
        "training_oof_metrics_interpretation": "Existing validation cutoffs on OOF models are descriptive diagnostics, not newly calibrated or unbiased CV operating-point estimates.",
        "pilot_is_error_enriched_training_only": True, "auxiliary_completed_reviews": 0,
        "test_prediction_or_label_rows_may_be_parsed": False, "test_response_access_allowed": False,
        "test_seals_and_predictions_hash_only": True, "label_changes_allowed": False,
        "new_model_fitting_allowed": False, "new_threshold_selection_allowed": False,
        "target_model_calls_allowed": False, "dashboard_promotion_allowed": False})
    return {"prepared": True, "train_examples": 1058, "validation_examples": 100,
            "pilot_target_source_groups": 40, "new_fits_or_labels_planned": 0, "test_outcome_access": False}


def check_protocol():
    protocol = read(RUN / "protocol.json")
    common.require(protocol["settings"] == SETTINGS and protocol["implementation_sha256"] == codes(), "Audit implementation/settings/spec changed after freeze.")
    for name, expected in protocol["protected_input_sha256"].items():
        if expected is None:
            common.require(not (ROOT / name).exists(), "A protected absent path now exists.")
        else:
            common.checked_read(ROOT / name, expected)
    for name, expected in protocol["encoder_file_sha256"].items():
        common.checked_read(shared.ENCODER_PATH / name, expected)
    return protocol


def states():
    import joblib
    result = {m: joblib.load(frozen.RUN / "states" / f"{m}.joblib") for m in core.METHODS if m != "score_fusion"}
    result["score_fusion"] = joblib.load(FUSION / "development/states/score_fusion.joblib")
    return result


def scores_from_cache(protocol, models):
    import numpy as np
    archive = np.load(REPR / "results/cv-scores.npz", allow_pickle=False)
    features = np.load(REPR / "results/final-features.npz", allow_pickle=False)
    common.require(archive["sample_ids"].tolist() == [r["sample_id"] for r in protocol["membership"]["train"]]
                   and archive["labels"].tolist() == [r["activation_label"] for r in protocol["membership"]["train"]], "OOF membership differs.")
    config_ids = archive["configuration_ids"].tolist()
    result = {"train_oof": {}, "train_in_sample": {}, "validation": {}}
    for method, config in core.CONFIGURATIONS.items():
        result["train_oof"][method] = archive["scores"][config_ids.index(config)]
        family = config.split("__C", 1)[0]
        for split, name in (("train", "train_in_sample"), ("validation", "validation")):
            result[name][method] = models[method]["model"].predict_proba(features[f"{split}__{family}"])[:, 1]
    weight = models["score_fusion"]["behavioral_weight"]
    for values in result.values():
        values["score_fusion"] = score_fusion.mix(values["generation"], values["candidate_behavioral"], weight)
    reference = {(r["configuration_id"], r["sample_id"]): r["score"] for r in lines(REPR / "results/validation-scores.jsonl")}
    for method, config in core.CONFIGURATIONS.items():
        previous = np.asarray([reference[config, r["sample_id"]] for r in protocol["membership"]["validation"]])
        common.require(np.array_equal(result["validation"][method], previous), "Cached validation classifier scores differ.")
    return result, {split: features[f"{split}__generation_raw"] for split in ("train", "validation")}


def probe_scores(models, embeddings, generation):
    values = {}
    for method in core.CONFIGURATIONS:
        matrix = frozen.core.features_for_state(models[method], embeddings, generation)
        values[method] = models[method]["model"].predict_proba(matrix)[:, 1]
    values["score_fusion"] = score_fusion.mix(values["generation"], values["candidate_behavioral"], models["score_fusion"]["behavioral_weight"])
    return values


def npz_new(path, **values):
    import numpy as np
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **values)
    shared.write_new(path, buffer.getvalue())


def build_report(protocol, rows, scores, signals, thresholds, baseline, stripped, baseline_error, allocation):
    return {"schema": "trace.development_behavior_audit.results.v1", "completed_at_utc": stamp(),
        "thresholds_unchanged": thresholds,
        "metric_slices": {kind: core.metric_slices(rows["train" if kind.startswith("train") else "validation"], values, thresholds)
                          for kind, values in scores.items()},
        "diagnostics": {split: core.diagnostic_summary(rows[split], scores["train_oof" if split == "train" else "validation"], signals[split], thresholds)
                        for split in rows},
        "cue_probe": core.cue_summary(rows["validation"], baseline, stripped, thresholds, SETTINGS["score_tolerance"]),
        "cue_probe_baseline_reference_max_error": baseline_error,
        "pilot_stratum_allocation": allocation, "pilot_target_groups": 40,
        "test_prediction_rows_parsed": 0, "test_label_rows_parsed": 0, "test_responses_opened": 0,
        "target_model_calls": 0, "new_model_fits": 0, "new_threshold_selections": 0,
        "new_activation_labels": 0, "completed_auxiliary_reviews": 0, "dashboard_promoted": False,
        "limitations": ["Training in-sample metrics are resubstitution, and OOF cutoffs reuse validation calibration; neither is a new held-out result.",
            "Validation is previously inspected development data with eleven positives, not independent confirmation.",
            "Fixed-response cue stripping diagnoses text-feature sensitivity; it does not generate an uncued target response or establish causal backdoor activation.",
            "Lexical markers do not determine semantic refusal, request fulfillment or unsafe assistance.",
            "Only admitted training/validation examples are inspected; no new StrongREJECT outcome analysis.",
            "Pilot is deliberately error-enriched and training-only; blank auxiliary reviews cannot train or validate a semantic scorer.",
            "Existing reviewer/server provenance and missing historical CSV qualifications remain unchanged."]}


def audit():
    import numpy as np
    import torch
    from traceguard.behavior import BehavioralEncoder
    protocol = check_protocol()
    folder = RUN / "results"
    common.require(not folder.exists(), "Audit results exist; refuse repeat or overwrite.")
    folder.mkdir(exist_ok=False)
    write(folder / "started.json", {"started_at_utc": stamp(), "protocol_sha256": sha(RUN / "protocol.json")})
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    models = states()
    thresholds = {m: state["threshold"] for m, state in models.items()}
    scores, signals = scores_from_cache(protocol, models)
    records, recomputed_signals = development.original.load_development(protocol)
    common.require(all(np.array_equal(signals[s], recomputed_signals[s]) for s in signals), "Development generation features differ.")
    rows = {s: [{**label, **record} for label, record in zip(protocol["membership"][s], records[s])] for s in records}
    core.validate_development(rows["train"] + rows["validation"])
    encoder = BehavioralEncoder(model_name=str(shared.ENCODER_PATH), local_only=True)
    print("Encoding 100 validation records and 100 fixed-response cue controls; no target LLM calls", flush=True)
    baseline_embeddings = encoder.encode_base(rows["validation"])
    common.require(encoder._model.max_seq_length == 256 and encoder._model.tokenizer.truncation_side == "right", "Encoder token policy changed.")
    for values in rows.values():
        for row in values:
            row.update(core.lexical_flags(row["response"]))
            row.update(core.visibility(encoder._model.tokenizer, row))
    stripped_rows = core.strip_validation_cue(rows["validation"])
    stripped_embeddings = encoder.encode_base(stripped_rows)
    baseline = probe_scores(models, baseline_embeddings, signals["validation"])
    stripped = probe_scores(models, stripped_embeddings, signals["validation"])
    errors = {m: float(np.max(abs(baseline[m] - scores["validation"][m]))) for m in core.METHODS}
    common.require(max(errors.values()) <= SETTINGS["score_tolerance"], "Cue probe baseline differs from existing validation references.")
    control = [i for i, r in enumerate(rows["validation"]) if r["condition_id"] == 1]
    common.require(all(float(np.max(abs(stripped[m][control] - baseline[m][control]))) <= SETTINGS["score_tolerance"] for m in core.METHODS), "Unchanged cue-probe controls differ beyond numerical tolerance.")
    common.require(np.array_equal(baseline["generation"], stripped["generation"]), "Cue probe changed generation scores.")
    selection, allocation = core.select_pilot(rows["train"], scores["train_oof"], thresholds, SETTINGS["seed"], SETTINGS["pilot_groups_per_stratum"])
    packets, administrator = core.review_packets(rows["train"], selection, SETTINGS["seed"])
    report = build_report(protocol, rows, scores, signals, thresholds, baseline, stripped, errors, allocation)
    report["pilot_selected_items_per_reviewer"] = len(selection)
    report["pilot_selected_source_groups"] = len({rows["train"][i]["source_id"] for i, _ in selection})
    inventory = []
    for split, items in rows.items():
        primary = scores["train_oof" if split == "train" else "validation"]
        for index, row in enumerate(items):
            inventory.append({k: v for k, v in row.items() if k not in ("prompt", "response")}
                | {"score_origin": "training_oof_descriptive_cutoff" if split == "train" else "existing_validation",
                   "generation_features": signals[split][index].tolist(), "scores": {m: float(v[index]) for m, v in primary.items()},
                   "predictions": {m: int(v[index] >= thresholds[m]) for m, v in primary.items()}})
    shared.write_new(folder / "error-inventory.jsonl", common.jsonlines(inventory))
    npz_new(folder / "scores.npz", **{f"{kind}__{m}": v for kind, values in scores.items() for m, v in values.items()})
    npz_new(folder / "cue-probe.npz", baseline_embeddings=baseline_embeddings, stripped_embeddings=stripped_embeddings,
            generation=signals["validation"], **{f"baseline__{m}": v for m, v in baseline.items()},
            **{f"stripped__{m}": v for m, v in stripped.items()})
    for name, packet in packets.items():
        shared.write_new(folder / f"auxiliary-review-{name}.jsonl", common.jsonlines(packet))
    shared.write_new(folder / "auxiliary-review-admin.jsonl", common.jsonlines(administrator))
    write(folder / "auxiliary-review-schema.json", {"rubric_version": "trace.aux_behavior.rubric.v1", "allowed_judgments": core.REVIEW_FIELDS,
        "require_reason_and_identified_reviewer": True, "completed_reviews": 0,
        "activation_label_inference_allowed": False, "source_spec_sha256": sha(SPEC)})
    write(folder / "report.json", report)
    shared.write_new(folder / "feature-spec.md", SPEC.read_bytes())
    write(folder / "manifest.json", {"protocol_sha256": sha(RUN / "protocol.json"),
        "output_sha256": {p.name: sha(p) for p in sorted(folder.iterdir()) if p.is_file()}})
    check_protocol()
    return {"audit_complete": True, "development_examples": 1158, "cue_probe_validation_examples": 100,
            "pilot_items_per_reviewer": len(selection), "completed_auxiliary_reviews": 0,
            "new_model_fits": 0, "test_prediction_rows_parsed": 0,
            "validation_f1": {m: report["metric_slices"]["validation"]["all"]["methods"][m]["f1"] for m in core.METHODS}}


def verify():
    import numpy as np
    from transformers import AutoTokenizer
    protocol = check_protocol()
    folder = RUN / "results"
    manifest = read(folder / "manifest.json")
    common.require(manifest["protocol_sha256"] == sha(RUN / "protocol.json"), "Audit manifest protocol differs.")
    for name, expected in manifest["output_sha256"].items():
        common.require(Path(name).name == name, "Unsafe result path.")
        common.checked_read(folder / name, expected)
    models = states()
    thresholds = {m: s["threshold"] for m, s in models.items()}
    scores, signals = scores_from_cache(protocol, models)
    saved = np.load(folder / "scores.npz", allow_pickle=False)
    common.require(all(np.array_equal(saved[f"{kind}__{m}"], v) for kind, values in scores.items() for m, v in values.items()), "Audit score archive differs.")
    records = shared.load_text_records(protocol["membership"])
    inventory = lines(folder / "error-inventory.jsonl")
    rows = {}
    tokenizer = AutoTokenizer.from_pretrained(str(shared.ENCODER_PATH), local_files_only=True)
    for split in ("train", "validation"):
        part = [r for r in inventory if r["split"] == split]
        common.require(len(part) == len(records[split]), "Inventory coverage differs.")
        rows[split] = []
        primary = scores["train_oof" if split == "train" else "validation"]
        for index, (item, label, record) in enumerate(zip(part, protocol["membership"][split], records[split])):
            common.require(all(item[k] == v for k, v in label.items()), "Inventory sealed membership differs.")
            common.require(item["scores"] == {m: float(v[index]) for m, v in primary.items()}
                           and item["predictions"] == {m: int(v[index] >= thresholds[m]) for m, v in primary.items()}
                           and item["generation_features"] == signals[split][index].tolist(), "Inventory scores, decisions or generation features differ.")
            combined = {**item, **record}
            flags = {**core.lexical_flags(record["response"]), **core.visibility(tokenizer, combined)}
            common.require(all(item[k] == v for k, v in flags.items()), "Lexical/token audit differs.")
            rows[split].append(combined)
    core.validate_development(rows["train"] + rows["validation"])
    probe = np.load(folder / "cue-probe.npz", allow_pickle=False)
    baseline = probe_scores(models, probe["baseline_embeddings"], probe["generation"])
    stripped = probe_scores(models, probe["stripped_embeddings"], probe["generation"])
    common.require(np.array_equal(probe["generation"], signals["validation"]), "Cue probe generation evidence changed.")
    for m in core.METHODS:
        common.require(np.array_equal(baseline[m], probe[f"baseline__{m}"]) and np.array_equal(stripped[m], probe[f"stripped__{m}"]), "Cached cue scores differ.")
    core.strip_validation_cue(rows["validation"])
    errors = {m: float(np.max(abs(baseline[m] - scores["validation"][m]))) for m in core.METHODS}
    selection, allocation = core.select_pilot(rows["train"], scores["train_oof"], thresholds, SETTINGS["seed"], SETTINGS["pilot_groups_per_stratum"])
    packets, administrator = core.review_packets(rows["train"], selection, SETTINGS["seed"])
    common.require(administrator == lines(folder / "auxiliary-review-admin.jsonl"), "Pilot administration mapping differs.")
    for name, packet in packets.items():
        common.require(packet == lines(folder / f"auxiliary-review-{name}.jsonl"), "Blank review packet changed.")
        common.require(all(not item["reason"] and not item["reviewer_id"] and all(v == "" for v in item["judgments"].values()) for item in packet), "Prepared packet contains fabricated judgments.")
    report = read(folder / "report.json")
    rebuilt = build_report(protocol, rows, scores, signals, thresholds, baseline, stripped, errors, allocation)
    common.require(all(report[k] == v for k, v in rebuilt.items() if k != "completed_at_utc"), "Audit report reconstruction differs.")
    common.require((folder / "feature-spec.md").read_bytes() == SPEC.read_bytes(), "Feature specification differs.")
    value = {"schema": "trace.development_behavior_audit.verification.v1", "verified": True,
        "protected_inputs_unchanged": True, "development_only_membership_and_cached_scores_verified": True,
        "token_visibility_markers_and_metrics_reconstructed": True, "cached_cue_classifier_scores_reconstructed": True,
        "training_only_blank_packets_and_pilot_selection_verified": True, "development_examples": len(inventory),
        "pilot_items_per_reviewer": len(selection), "auxiliary_reviews_completed": 0, "new_fits_or_thresholds": 0,
        "test_outcomes_parsed_or_responses_opened": 0, "target_model_calls": 0, "dashboard_promoted": False}
    path = RUN / "verification.json"
    if path.exists():
        common.require(read(path) == value, "Existing audit verification differs.")
    else:
        write(path, value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "audit", "verify"))
    args = parser.parse_args()
    print(json.dumps({"prepare": prepare, "audit": audit, "verify": verify}[args.command](), indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
