"""Nested source-isolated, low-dimensional dual-view activation detection."""
from __future__ import annotations

from collections import defaultdict
import hashlib

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

from . import auxiliary_scorer as auxiliary
from .features import generation_features
from .detector import metrics

METHODS = ("generation", "reference_behavioral", "reference_full", "behavioral_enhanced", "full_linear", "full_nonlinear")
NEW_METHODS = ("behavioral_enhanced", "full_linear", "full_nonlinear")
SEMANTIC_HEADS = ("refusal_present", "substantive_fulfillment")
POLICIES = ("max_f1", "fpr_0.10", "recall_0.80")
TREE = {"learning_rate": .05, "max_iter": 100, "max_leaf_nodes": 7, "max_depth": 3,
        "min_samples_leaf": 20, "l2_regularization": 1., "early_stopping": False,
        "random_state": 42, "categorical_features": None}
SETTINGS = {"C": 1., "class_weight": None, "max_iter": 1000, "model_seed": 42,
            "minimum_fit_source_groups_per_class": 5, "tree": TREE,
            "fpr_budget": .10, "recall_target": .80, "bootstrap_repetitions": 2000, "seed": 20261106}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def thresholds(y, scores):
    y, scores = np.asarray(y), np.asarray(scores)
    require(set(y) == {0, 1} and len(y) == len(scores) and np.isfinite(scores).all(), "Threshold calibration needs both classes and aligned finite scores.")
    candidates = np.unique(np.r_[0., 1., np.nextafter(1., np.inf), scores])
    calls = scores[None, :] >= candidates[:, None]
    tp, fp = calls[:, y == 1].sum(axis=1), calls[:, y == 0].sum(axis=1)
    fn = (y == 1).sum() - tp
    f1 = 2 * tp / (2 * tp + fp + fn)
    recall, fpr = tp / (y == 1).sum(), fp / (y == 0).sum()
    primary = max(range(len(candidates)), key=lambda i: (f1[i], candidates[i]))
    budget = max((i for i in range(len(candidates)) if fpr[i] <= .10), key=lambda i: (recall[i], -fpr[i], candidates[i]))
    sensitivity = max((i for i in range(len(candidates)) if recall[i] >= .80), key=lambda i: candidates[i])
    return dict(zip(POLICIES, (float(candidates[i]) for i in (primary, budget, sensitivity))))


def score_metrics(y, scores, threshold):
    return {**metrics(y, scores, threshold), "average_precision": float(average_precision_score(y, scores)) if sum(y) else None}


def logistic(x, y, settings=None):
    return auxiliary.default_fitter(x, y, settings or SETTINGS)


def fit_components(base, semantic, membership, labels, fit, forbidden_sources=(), settings=None, fitter=logistic):
    settings = settings or SETTINGS
    rows = membership["train"]
    require(all(r["split"] == "train" for r in rows) and all(r["split"] == "train" for r in labels), "Component fits are training-only.")
    require(len(base) == len(semantic) == len(rows), "Component feature membership differs.")
    fit = np.asarray(fit)
    require(len(fit) == len(set(fit)) and len(fit) > 0 and min(fit) >= 0 and max(fit) < len(rows), "Invalid component fitting indices.")
    sources = {rows[i]["source_id"] for i in fit}
    require(not (sources & set(forbidden_sources)), "Forbidden source in component fitting.")
    y = np.asarray([rows[i]["activation_label"] for i in fit])
    models = {"behavioral_activation": fitter(base[fit], y, settings)}
    audits = [{"component": "behavioral_activation", "fit_sample_ids": [rows[i]["sample_id"] for i in fit],
               "fit_source_ids": sorted(sources), "target_sha256": hashlib.sha256(y.tobytes()).hexdigest()}]
    lookup = {r["sample_id"]: i for i, r in enumerate(rows)}
    for head in SEMANTIC_HEADS:
        eligible = [r for r in labels if r["source_id"] in sources and r["auxiliary_targets"][head] is not None]
        for label in (0, 1):
            require(len({r["source_id"] for r in eligible if r["auxiliary_targets"][head] == label}) >= settings["minimum_fit_source_groups_per_class"], "Insufficient semantic fitting source-class support.")
        require(all(r["sample_id"] in lookup and rows[lookup[r["sample_id"]]]["source_id"] == r["source_id"] for r in eligible), "Semantic label identity differs.")
        indices = [lookup[r["sample_id"]] for r in eligible]
        target = np.asarray([r["auxiliary_targets"][head] for r in eligible], dtype=int)
        models[head] = fitter(semantic[indices], target, settings)
        audits.append({"component": head, "fit_sample_ids": [r["sample_id"] for r in eligible],
                       "fit_source_ids": sorted({r["source_id"] for r in eligible}), "target_sha256": hashlib.sha256(target.tobytes()).hexdigest()})
    require("unsafe_enabling_assistance" not in models, "Failed semantic head must remain excluded.")
    return models, audits


def component_scores(models, base, semantic):
    return np.column_stack([models["behavioral_activation"].predict_proba(base)[:, 1]] +
                           [models[h].predict_proba(semantic)[:, 1] for h in SEMANTIC_HEADS])


def fit_references(base, generation, y, fit, settings=None):
    settings = settings or SETTINGS
    scaler = StandardScaler().fit(generation[fit])
    scaled = scaler.transform(generation[fit])
    return {"generation_scaler": scaler, "generation": logistic(scaled, y[fit], settings),
            "reference_full": logistic(np.c_[base[fit], scaled], y[fit], settings)}


def reference_scores(reference, components, base, generation):
    scaled = reference["generation_scaler"].transform(generation)
    return {"generation": reference["generation"].predict_proba(scaled)[:, 1],
            "reference_behavioral": components[:, 0],
            "reference_full": reference["reference_full"].predict_proba(np.c_[base, scaled])[:, 1]}


def combiner_input(method, generation, behavior):
    require(behavior.ndim == 2 and behavior.shape[1] == 3 and len(generation) == len(behavior), "Behavioral evidence must contain the same three matched scores.")
    require(method in NEW_METHODS, "Unknown activation combiner.")
    return behavior if method == "behavioral_enhanced" else np.c_[generation, behavior]


def fit_combiner(method, generation, behavior, targets, settings=None):
    settings = settings or SETTINGS
    values = combiner_input(method, generation, behavior)
    if method == "full_nonlinear":
        return {"model": HistGradientBoostingClassifier(**settings["tree"]).fit(values, targets), "scaler": None}
    scaler = StandardScaler().fit(values)
    return {"model": logistic(scaler.transform(values), targets, settings), "scaler": scaler}


def combiner_scores(state, method, generation, behavior):
    values = combiner_input(method, generation, behavior)
    if state["scaler"] is not None:
        values = state["scaler"].transform(values)
    return state["model"].predict_proba(values)[:, 1]


def selection(fold_results):
    ranking = {}
    for method in METHODS:
        ap = [r[method]["average_precision"] for r in fold_results]
        auc = [r[method]["auroc"] for r in fold_results]
        ranking[method] = {"mean_fold_ap": float(np.mean(ap)), "mean_fold_auroc": float(np.mean(auc)),
                           "fold_ap": ap, "fold_auroc": auc}
    def key(method):
        return (-ranking[method]["mean_fold_ap"], -ranking[method]["mean_fold_auroc"], METHODS.index(method))
    return {"selected_full": min(("full_linear", "full_nonlinear"), key=key),
            "best_all_methods": min(METHODS, key=key), "training_fold_ranking": ranking,
            "objective": "Mean outer-fold AP, then mean AUROC, then declared method order (linear before nonlinear).",
            "validation_or_test_used_for_model_selection": False}


def cross_validate(base, semantic, generation, membership, labels, folds, settings=None, progress=None):
    settings = settings or SETTINGS
    n = len(base)
    rows = membership["train"]
    require(len(rows) == n and all(r["split"] == "train" for r in rows), "Only training rows may enter fusion CV.")
    require(not ({r["source_id"] for r in rows} & {r["source_id"] for r in membership["validation"]}), "Source crosses development partitions.")
    y = np.asarray([r["activation_label"] for r in rows])
    oof = {m: np.full(n, np.nan) for m in METHODS}
    decisions = {p: {m: np.full(n, -1, dtype=int) for m in METHODS} for p in POLICIES}
    outer_behavior = np.full((n, 3), np.nan)
    inner_arrays, bundles, audits, calibrations, fold_results = {}, {}, [], [], []
    coverage = np.zeros(n, dtype=int)
    for outer in folds:
        fit, held = np.asarray(outer["fit_indices"]), np.asarray(outer["held_indices"])
        require(not (set(fit) & set(held)) and set(fit) | set(held) == set(range(n)), "Invalid outer fusion partition.")
        held_sources = {rows[i]["source_id"] for i in held}
        require(not ({rows[i]["source_id"] for i in fit} & held_sources), "Outer fusion source leakage.")
        nested = np.full((n, 3), np.nan)
        inner_coverage = np.zeros(n, dtype=int)
        for inner in outer["inner_folds"]:
            f, h = np.asarray(inner["fit_indices"]), np.asarray(inner["held_indices"])
            require(not (set(f) & set(h)) and set(f) | set(h) == set(fit), "Invalid inner fusion partition.")
            excluded = held_sources | {rows[i]["source_id"] for i in h}
            models, audit_rows = fit_components(base, semantic, membership, labels, f, excluded, settings)
            nested[h] = component_scores(models, base[h], semantic[h])
            inner_coverage[h] += 1
            key = f"outer-{outer['fold']}/inner-{inner['inner_fold']}"
            bundles[key] = {"components": models}
            audits.extend({"stage": "inner", "outer_fold": outer["fold"], "inner_fold": inner["inner_fold"], **a} for a in audit_rows)
        require(np.all(inner_coverage[fit] == 1) and np.all(inner_coverage[held] == 0)
                and np.isfinite(nested[fit]).all(), "Nested component features incomplete or reach outer held rows.")
        inner_arrays[str(outer["fold"])] = nested[fit]
        models, audit_rows = fit_components(base, semantic, membership, labels, fit, held_sources, settings)
        audits.extend({"stage": "outer", "outer_fold": outer["fold"], "inner_fold": None, **a} for a in audit_rows)
        outer_behavior[held] = component_scores(models, base[held], semantic[held])
        references = fit_references(base, generation, y, fit, settings)
        train_behavior = component_scores(models, base[fit], semantic[fit])
        train_scores = reference_scores(references, train_behavior, base[fit], generation[fit])
        held_scores = reference_scores(references, outer_behavior[held], base[held], generation[held])
        combiners = {}
        for method in NEW_METHODS:
            combiners[method] = fit_combiner(method, generation[fit], nested[fit], y[fit], settings)
            train_scores[method] = combiner_scores(combiners[method], method, generation[fit], nested[fit])
            held_scores[method] = combiner_scores(combiners[method], method, generation[held], outer_behavior[held])
        policies = {m: thresholds(y[fit], train_scores[m]) for m in METHODS}
        calibrations.append({"outer_fold": outer["fold"], "fit_sample_ids": [rows[i]["sample_id"] for i in fit],
            "policies": policies, "interpretation": "Outer-fit predictions calibrate cutoffs; held-source labels do not. Combiner fitting predictions are resubstitution on cross-fitted component features."})
        fold_results.append({m: score_metrics(y[held], held_scores[m], policies[m]["max_f1"]) for m in METHODS})
        for method in METHODS:
            oof[method][held] = held_scores[method]
            for policy in POLICIES:
                decisions[policy][method][held] = (held_scores[method] >= policies[method][policy]).astype(int)
        coverage[held] += 1
        bundles[f"outer-{outer['fold']}"] = {"components": models, "references": references, "combiners": combiners}
        if progress:
            progress(f"Completed nested activation fold {outer['fold'] + 1}/5")
    require(np.all(coverage == 1) and all(np.isfinite(v).all() for v in oof.values()), "Outer fusion OOF coverage differs.")
    return {"oof_scores": oof, "decisions": decisions, "outer_behavior": outer_behavior,
        "inner_behavior": inner_arrays, "model_bundles": bundles, "component_fit_audits": audits,
        "fold_calibrations": calibrations, "fold_results": fold_results, "selection": selection(fold_results)}


def prediction_rows(membership, scores, policies=None, decisions=None):
    output = []
    for index, label in enumerate(membership):
        calls = {p: {m: int(scores[m][index] >= policies[m][p]) for m in METHODS} for p in POLICIES} if decisions is None else {
            p: {m: int(decisions[p][m][index]) for m in METHODS} for p in POLICIES}
        output.append({k: label[k] for k in ("sample_id", "source_id", "source", "split", "condition_id", "activation_label") if k in label}
                      | {"scores": {m: float(scores[m][index]) for m in METHODS}, "predictions": calls})
    return output


def summarize(rows):
    output = {}
    for split in sorted({r["split"] for r in rows}):
        part = [r for r in rows if r["split"] == split]
        output[split] = {}
        for policy in POLICIES:
            output[split][policy] = {}
            for method in METHODS:
                y = [r["activation_label"] for r in part]
                decisions = [r["predictions"][policy][method] for r in part]
                # Binary decisions recover the frozen operating point; ranking uses scores separately.
                measured = metrics(y, np.asarray(decisions, dtype=float), .5)
                ranking = score_metrics(y, [r["scores"][method] for r in part], .5)
                measured["auroc"], measured["average_precision"] = ranking["auroc"], ranking["average_precision"]
                output[split][policy][method] = measured
    return output


def source_slices(rows):
    result = {}
    for source in sorted({r.get("source", "StrongREJECT") for r in rows}):
        part = [r for r in rows if r.get("source", "StrongREJECT") == source]
        result[source] = summarize(part)
    return result


def paired_intervals(rows, selected, settings=None):
    settings = settings or SETTINGS
    groups = defaultdict(list)
    require(len({r["split"] for r in rows}) == 1, "Bootstrap requires one evaluation partition at a time.")
    for row in rows:
        groups[row["source_id"]].append(row)
    ordered = [v for _, v in sorted(groups.items())]
    require(all(len(v) == 2 and {r["condition_id"] for r in v} == {1, 2} for v in ordered), "Fusion bootstrap needs whole source pairs.")
    y = np.asarray([[r["activation_label"] for r in v] for v in ordered])
    idx = np.random.default_rng(settings["seed"]).integers(0, len(ordered), (settings["bootstrap_repetitions"], len(ordered)))
    sampled_y = y[idx].reshape(len(idx), -1)
    values, point = {}, {}
    for method in METHODS:
        pred = np.asarray([[r["predictions"]["max_f1"][method] for r in v] for v in ordered])
        p = pred[idx].reshape(len(idx), -1)
        tp, fp, fn = ((sampled_y == 1) & (p == 1)).sum(axis=1), ((sampled_y == 0) & (p == 1)).sum(axis=1), ((sampled_y == 1) & (p == 0)).sum(axis=1)
        den = 2 * tp + fp + fn
        values[method] = np.divide(2 * tp, den, out=np.zeros(len(idx)), where=den != 0)
        point[method] = summarize(rows)[rows[0]["split"]]["max_f1"][method]["f1"]
    return {selected + "_minus_" + ref: {"f1_difference": point[selected] - point[ref],
        "percentile_95_interval": np.percentile(values[selected] - values[ref], [2.5, 97.5]).tolist(),
        "source_groups": len(ordered), "resamples": len(idx), "seed": settings["seed"],
        "interpretation": "Exploratory paired whole-source interval; does not remove development/test adaptation or AI-assisted-label limitations."}
        for ref in ("generation", "reference_behavioral", "behavioral_enhanced", "reference_full")}


def make_states(components, references, combiners, calibrated):
    result = {}
    for method in METHODS:
        state = {"method": method, "thresholds": calibrated[method], "unsafe_head_excluded": True, "dashboard_promoted": False}
        if method == "generation":
            state.update(model=references["generation"], scaler=references["generation_scaler"])
        elif method == "reference_behavioral":
            state.update(model=components["behavioral_activation"], scaler=None)
        elif method == "reference_full":
            state.update(model=references["reference_full"], scaler=references["generation_scaler"])
        else:
            state.update(components=components, **combiners[method])
        result[method] = state
    return result


def predict_one(state, record, behavioral_encoder=None, semantic_encoder=None):
    method = state["method"]
    generation = generation_features(record["signals"])[None, :] if method != "reference_behavioral" and method != "behavioral_enhanced" else np.zeros((1, 3))
    if method == "generation":
        return float(state["model"].predict_proba(state["scaler"].transform(generation))[0, 1])
    require(behavioral_encoder is not None, "Behavioral encoding required.")
    base = behavioral_encoder.encode_base([record])
    if method == "reference_behavioral":
        return float(state["model"].predict_proba(base)[0, 1])
    if method == "reference_full":
        return float(state["model"].predict_proba(np.c_[base, state["scaler"].transform(generation)])[0, 1])
    require(semantic_encoder is not None, "Two passing semantic heads require auxiliary encoding.")
    behavior = component_scores(state["components"], base, semantic_encoder.encode([record]))
    return float(combiner_scores(state, method, generation, behavior)[0])
