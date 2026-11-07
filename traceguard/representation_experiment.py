"""Development-only behavioral comparisons with source-isolated learned transforms.

This module does not read data, call a target model, or alter deployed detectors.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from .behavior import BehavioralEncoder
from .detector import choose_threshold, metrics

REPRESENTATIONS = (
    ("prompt_response_projected", "prompt_response", True),
    ("prompt_response_raw", "prompt_response", False),
    ("response_only_projected", "response_only", True),
    ("response_only_raw", "response_only", False),
)
DEFAULT_SETTINGS = {
    "folds": 5, "fold_seed": 20261106, "model_seed": 42,
    "projection_epochs": 60, "C_grid": [0.1, 1.0, 10.0],
    "class_weight_grid": [None, "balanced"], "max_iter": 1000,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def family_specs():
    families = []
    for name, mode, projected in REPRESENTATIONS:
        for view in ("behavioral", "full"):
            families.append({"name": f"{name}__{view}", "representation": name,
                             "mode": mode, "projected": projected, "view": view})
    families.append({"name": "generation", "representation": None,
                     "mode": None, "projected": False, "view": "generation"})
    return families


def grouped_membership(membership, duplicate_clusters):
    """Join paired conditions and transitive duplicate clusters before splitting."""
    require(set(membership) == {"train", "validation"}, "Only development partitions are allowed.")
    sources, samples, paired = {}, set(), defaultdict(list)
    for split, rows in membership.items():
        for row in rows:
            sid = row["source_id"]
            require(row["split"] == split, "Membership partition mismatch.")
            require(row["sample_id"] not in samples, "Duplicate sample ID.")
            require(row["condition_id"] in (1, 2), "Only adapted conditions are allowed.")
            require(type(row["activation_label"]) is int and row["activation_label"] in (0, 1),
                    "A binary reviewed activation label is required.")
            require(sid not in sources or sources[sid] == split, "Source crosses development partitions.")
            sources[sid] = split
            samples.add(row["sample_id"])
            paired[sid].append(row)
    for rows in paired.values():
        require(len(rows) == 2 and {r["condition_id"] for r in rows} == {1, 2}, "Incomplete adapted source pair.")
    parent = {sid: sid for sid in sources}

    def find(sid):
        while parent[sid] != sid:
            parent[sid] = parent[parent[sid]]
            sid = parent[sid]
        return sid

    for cluster in duplicate_clusters:
        present = sorted(set(cluster["source_ids"]) & sources.keys())
        require(len({sources[sid] for sid in present}) <= 1, "Duplicate question cluster crosses development partitions.")
        for sid in present[1:]:
            left, right = sorted((find(present[0]), find(sid)))
            parent[right] = left
    return {split: [find(row["source_id"]) for row in rows] for split, rows in membership.items()}


def make_folds(membership, duplicate_clusters, settings=None):
    settings = settings or DEFAULT_SETTINGS
    groups = grouped_membership(membership, duplicate_clusters)
    y = np.asarray([r["activation_label"] for r in membership["train"]])
    require(set(y) == {0, 1}, "Both training classes are required.")
    splitter = StratifiedGroupKFold(n_splits=settings["folds"], shuffle=True,
                                  random_state=settings["fold_seed"])
    folds, coverage = [], np.zeros(len(y), dtype=int)
    group_values = np.asarray(groups["train"])
    for index, (fit, held) in enumerate(splitter.split(np.zeros(len(y)), y, group_values)):
        require(not (set(group_values[fit]) & set(group_values[held])), "Source group leaks across a fold.")
        require(set(y[fit]) == set(y[held]) == {0, 1}, "Each fold must contain both classes.")
        coverage[held] += 1
        folds.append({"fold": index, "fit_indices": fit.tolist(), "held_indices": held.tolist()})
    require(np.all(coverage == 1), "Every training row must be held out exactly once.")
    return folds, groups


class _CachedEncoder(BehavioralEncoder):
    def __init__(self, values):
        super().__init__()
        self.values = values

    def encode_base(self, records):
        return self.values


def fit_projection(base, y, sample_ids, epochs):
    """Reuse the original projection algorithm on exactly the supplied fold rows."""
    encoder = _CachedEncoder(np.asarray(base))
    encoder.fit_transform([{"sample_id": sid} for sid in sample_ids], y, epochs=epochs)
    return encoder.projection, encoder.training_info


def apply_projection(base, projection):
    values = np.asarray(base) @ projection
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def fit_classifier(x, y, c, class_weight, settings):
    model = LogisticRegression(C=c, class_weight=class_weight, max_iter=settings["max_iter"],
                               random_state=settings["model_seed"])
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(x, y)
    require(np.max(model.n_iter_) < settings["max_iter"], "Classifier did not converge.")
    return model


def config_id(family, c, class_weight):
    return f"{family}__C{c:g}__{class_weight or 'unweighted'}"


def matrices(representations, generation):
    values = {"generation": generation}
    for name, behavior in representations.items():
        values[f"{name}__behavioral"] = behavior
        values[f"{name}__full"] = np.c_[behavior, generation]
    return values


def cross_validate(base, generation, labels, sample_ids, folds, settings=None,
                   projection_fitter=fit_projection, progress=None):
    """Accept training arrays only; fit all learned operations inside each fold."""
    settings = settings or DEFAULT_SETTINGS
    y = np.asarray(labels)
    n = len(y)
    require(set(base) == {"prompt_response", "response_only"}, "Both text representations are required.")
    require(generation.shape == (n, 3), "Generation rows differ from training membership.")
    require(len(sample_ids) == n and len(set(sample_ids)) == n, "Training IDs are not unique and aligned.")
    require(all(v.shape[0] == n and np.isfinite(v).all() for v in base.values()), "Invalid training embeddings.")
    coverage = np.zeros(n, dtype=int)
    for fold in folds:
        fit, held = np.asarray(fold["fit_indices"]), np.asarray(fold["held_indices"])
        require(len(set(fit) & set(held)) == 0 and set(fit) | set(held) == set(range(n)), "Invalid fold indices.")
        coverage[held] += 1
    require(np.all(coverage == 1), "Incomplete or repeated held-out fold coverage.")
    configurations = [
        {"id": config_id(f["name"], c, weight), "family": f["name"], "C": c, "class_weight": weight,
         "fold_average_precision": [], "fold_auroc": []}
        for f in family_specs() for c in settings["C_grid"] for weight in settings["class_weight_grid"]
    ]
    oof = np.full((len(configurations), n), np.nan)
    fit_audits = []
    for fold in folds:
        fit, held = np.asarray(fold["fit_indices"]), np.asarray(fold["held_indices"])
        scaler = StandardScaler().fit(generation[fit])
        train_reps, held_reps = {}, {}
        for mode in ("prompt_response", "response_only"):
            train_reps[f"{mode}_raw"], held_reps[f"{mode}_raw"] = base[mode][fit], base[mode][held]
            projection, info = projection_fitter(base[mode][fit], y[fit], [sample_ids[i] for i in fit],
                                                  settings["projection_epochs"])
            require(info["fit_sample_ids"] == [sample_ids[i] for i in fit], "Projection fit membership differs.")
            train_reps[f"{mode}_projected"] = apply_projection(base[mode][fit], projection)
            held_reps[f"{mode}_projected"] = apply_projection(base[mode][held], projection)
            fit_audits.append({"fold": fold["fold"], "mode": mode, **info,
                               "projection_sha256": hashlib.sha256(projection.tobytes()).hexdigest()})
        x_fit = matrices(train_reps, scaler.transform(generation[fit]))
        x_held = matrices(held_reps, scaler.transform(generation[held]))
        for index, config in enumerate(configurations):
            name = config["family"]
            model = fit_classifier(x_fit[name], y[fit], config["C"], config["class_weight"], settings)
            scores = model.predict_proba(x_held[name])[:, 1]
            oof[index, held] = scores
            config["fold_average_precision"].append(float(average_precision_score(y[held], scores)))
            config["fold_auroc"].append(float(roc_auc_score(y[held], scores)))
        if progress:
            progress(f"Completed training-only fold {fold['fold'] + 1}/{len(folds)}")
    require(np.isfinite(oof).all(), "Missing or nonfinite out-of-fold predictions.")
    for index, config in enumerate(configurations):
        config["mean_fold_average_precision"] = float(np.mean(config["fold_average_precision"]))
        config["std_fold_average_precision"] = float(np.std(config["fold_average_precision"], ddof=1))
        config["mean_fold_auroc"] = float(np.mean(config["fold_auroc"]))
        config["pooled_oof_average_precision"] = float(average_precision_score(y, oof[index]))
    return configurations, oof, fit_audits


def select_configurations(configurations):
    """Selection uses training CV only, with a deterministic predeclared tie rule."""
    def ranked(rows):
        return sorted(rows, key=lambda r: (-r["mean_fold_average_precision"], -r["mean_fold_auroc"],
                                          r["C"], r["class_weight"] is not None, r["id"]))[0]
    selected = {}
    for family in family_specs():
        rows = [r for r in configurations if r["family"] == family["name"]]
        selected[family["name"]] = ranked(rows)["id"]
    winners = {}
    for view in ("behavioral", "full", "generation"):
        families = {f["name"] for f in family_specs() if f["view"] == view}
        rows = [r for r in configurations if r["family"] in families]
        winners[view] = ranked(rows)["id"]
    fixed = [r for r in configurations if r["C"] == 1.0 and r["class_weight"] is None]
    for view in ("behavioral", "full"):
        winners[f"fixed_settings_{view}"] = ranked([r for r in fixed if r["family"].endswith(f"__{view}")])["id"]
    return {"selected_by_family": selected, "winners": winners,
            "fixed_settings_ids": [r["id"] for r in fixed],
            "selection_used_validation_labels_or_scores": False,
            "selection_objective": "mean fold average precision; ties mean fold AUROC, smaller C, unweighted, ID"}


def score_metrics(y, scores, threshold):
    return {**metrics(y, scores, threshold), "average_precision": float(average_precision_score(y, scores))}


def fit_final(base, generation, membership, configurations, selection, settings=None,
              projection_fitter=fit_projection):
    settings = settings or DEFAULT_SETTINGS
    y = {s: np.asarray([r["activation_label"] for r in rows]) for s, rows in membership.items()}
    ids = [r["sample_id"] for r in membership["train"]]
    scaler = StandardScaler().fit(generation["train"])
    representation_values = {"train": {}, "validation": {}}
    projections, audits = {}, {}
    for mode in ("prompt_response", "response_only"):
        projection, info = projection_fitter(base[mode]["train"], y["train"], ids, settings["projection_epochs"])
        require(info["fit_sample_ids"] == ids, "Final projection used rows outside training.")
        projections[mode], audits[mode] = projection, info
        for split in y:
            representation_values[split][f"{mode}_raw"] = base[mode][split]
            representation_values[split][f"{mode}_projected"] = apply_projection(base[mode][split], projection)
    features = {s: matrices(representation_values[s], scaler.transform(generation[s])) for s in y}
    requested = set(selection["fixed_settings_ids"]) | set(selection["selected_by_family"].values())
    models, results, scores = {}, {}, {}
    for config in configurations:
        if config["id"] not in requested:
            continue
        key, family = config["id"], config["family"]
        model = fit_classifier(features["train"][family], y["train"], config["C"], config["class_weight"], settings)
        value = model.predict_proba(features["validation"][family])[:, 1]
        threshold = choose_threshold(y["validation"], value)
        models[key] = {"model": model, "family": family, "threshold": threshold,
                       "C": config["C"], "class_weight": config["class_weight"]}
        scores[key] = value
        results[key] = {"family": family, "C": config["C"], "class_weight": config["class_weight"],
                        "threshold": threshold, "metrics": score_metrics(y["validation"], value, threshold)}
    return {"models": models, "projections": projections, "projection_training_info": audits,
            "generation_scaler": scaler, "features": features,
            "validation_results": results, "validation_scores": scores}
