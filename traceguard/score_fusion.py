"""Bounded score mixtures using frozen, source-isolated component predictions."""
from collections import defaultdict

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .detector import choose_threshold, metrics
from . import reused_comparison as existing

METHODS = ("original_full", "candidate_full", "generation", "score_fusion")
DEFAULT_WEIGHTS = (0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def mix(generation, behavioral, behavioral_weight):
    """A ranking score, not a calibrated activation probability."""
    require(np.isfinite(behavioral_weight) and 0 <= behavioral_weight <= 1, "Invalid mixture weight.")
    generation, behavioral = np.asarray(generation), np.asarray(behavioral)
    require(generation.shape == behavioral.shape and generation.ndim == 1, "Component score shapes differ.")
    require(all(np.isfinite(v).all() and ((v >= 0) & (v <= 1)).all() for v in (generation, behavioral)),
            "Component scores must be finite and within [0, 1].")
    if behavioral_weight == 0:
        return generation.copy()
    if behavioral_weight == 1:
        return behavioral.copy()
    return (1 - behavioral_weight) * generation + behavioral_weight * behavioral


def select_training_mixture(generation, behavioral, membership, folds, groups, weights=DEFAULT_WEIGHTS):
    """Select on training OOF scores only; CV scores are selection evidence."""
    require(bool(membership) and all(r["split"] == "train" for r in membership), "Only training membership may select weights.")
    ids = [r["sample_id"] for r in membership]
    require(len(ids) == len(set(ids)) == len(generation) == len(behavioral) == len(groups), "Training score membership differs.")
    y = np.asarray([r["activation_label"] for r in membership])
    require(set(y) == {0, 1}, "Both training classes are required.")
    sources = defaultdict(list)
    for index, row in enumerate(membership):
        require(type(row["activation_label"]) is int and row["condition_id"] in (1, 2), "Invalid training condition/label.")
        sources[row["source_id"]].append(index)
    require(all(len(indices) == 2 and {membership[i]["condition_id"] for i in indices} == {1, 2}
                and len({groups[i] for i in indices}) == 1 for indices in sources.values()), "Incomplete or separated source pairs.")
    coverage = np.zeros(len(y), dtype=int)
    for fold in folds:
        fit, held = fold["fit_indices"], fold["held_indices"]
        require(len(fit) == len(set(fit)) and len(held) == len(set(held))
                and not (set(fit) & set(held)) and set(fit) | set(held) == set(range(len(y))), "Invalid training fold indices.")
        require(not ({groups[i] for i in fit} & {groups[i] for i in held}), "Training group leaks across fold.")
        require(set(y[fit]) == set(y[held]) == {0, 1}, "Each fold needs both classes.")
        coverage[held] += 1
    require(np.all(coverage == 1), "Repeated or missing held-fold coverage.")
    require(len(weights) == len(set(weights)) and {0, 1} <= set(weights)
            and any(0 < w < 1 for w in weights), "Mixture grid needs distinct endpoints and an interior weight.")
    results, arrays = [], []
    for weight in weights:
        scores = mix(generation, behavioral, weight)
        ap = [float(average_precision_score(y[f["held_indices"]], scores[f["held_indices"]])) for f in folds]
        auc = [float(roc_auc_score(y[f["held_indices"]], scores[f["held_indices"]])) for f in folds]
        results.append({"behavioral_weight": weight, "generation_weight": 1 - weight,
                        "fold_average_precision": ap, "fold_auroc": auc,
                        "mean_fold_average_precision": float(np.mean(ap)),
                        "mean_fold_auroc": float(np.mean(auc)),
                        "pooled_oof_average_precision": float(average_precision_score(y, scores))})
        arrays.append(scores)
    def ranking(row):
        return (-row["mean_fold_average_precision"], -row["mean_fold_auroc"], row["behavioral_weight"])
    selected = min((r for r in results if 0 < r["behavioral_weight"] < 1), key=ranking)
    overall = min(results, key=ranking)
    return results, {"selected_fusion_weight": selected["behavioral_weight"],
                     "best_overall_weight_including_endpoints": overall["behavioral_weight"],
                     "selection_objective": "mean fold AP; ties mean fold AUROC then smaller Behavioral weight",
                     "used_validation_or_test_scores_for_selection": False,
                     "interpretation": "Best interior mixture is the fixed experimental candidate; overall winner may be a single-view endpoint. CV is selection evidence, not an unbiased performance estimate."}, np.asarray(arrays)


def calibrate_validation(scores, membership):
    require(bool(membership) and all(r["split"] == "validation" for r in membership), "Only validation may calibrate the threshold.")
    require(len(scores) == len(membership), "Validation coverage differs.")
    require(len({r["sample_id"] for r in membership}) == len(membership)
            and all(r["condition_id"] in (1, 2) and type(r["activation_label"]) is int for r in membership),
            "Invalid validation membership.")
    y = np.asarray([r["activation_label"] for r in membership])
    require(set(y) == {0, 1} and np.isfinite(scores).all(), "Invalid validation classes or scores.")
    threshold = choose_threshold(y, scores)
    return threshold, score_metrics(y, scores, threshold)


def score_metrics(labels, scores, threshold):
    return {**metrics(labels, scores, threshold), "average_precision": float(average_precision_score(labels, scores))}


def add_frozen_fusion(previous_rows, weight, thresholds):
    require(0 < weight < 1, "Retrospective comparison requires the selected interior fusion.")
    require(set(thresholds) == set(METHODS), "Frozen comparator thresholds differ.")
    fused = mix([r["scores"]["generation"] for r in previous_rows],
                [r["scores"]["candidate_behavioral"] for r in previous_rows], weight)
    output = []
    for index, previous in enumerate(previous_rows):
        scores = {m: previous["scores"][m] for m in METHODS if m != "score_fusion"}
        scores["score_fusion"] = float(fused[index])
        decisions = {m: int(scores[m] >= thresholds[m]) for m in METHODS}
        require(all(decisions[m] == previous["predictions"][m] for m in METHODS if m != "score_fusion"),
                "An existing comparator threshold or prediction changed.")
        output.append({k: previous[k] for k in ("sample_id", "source_id", "condition_id", "split", "activation_label")}
                      | {"scores": scores, "predictions": decisions})
    return output


def summarize(rows, thresholds):
    result = {}
    for split in ("test_seen", "test_unseen"):
        part = [r for r in rows if r["split"] == split]
        require(bool(part), "Required retrospective partition missing.")
        result[split] = {}
        for method in METHODS:
            require(all(r["predictions"][method] == int(r["scores"][method] >= thresholds[method]) for r in part),
                    "Saved decision differs from its frozen threshold.")
            result[split][method] = score_metrics([r["activation_label"] for r in part],
                                                 [r["scores"][method] for r in part], thresholds[method])
    return result


def paired_intervals(rows, repetitions=2000, seed=20261106):
    groups = defaultdict(list)
    for row in rows:
        groups[row["source_id"]].append(row)
    ordered = [sorted(g, key=lambda r: r["condition_id"]) for _, g in sorted(groups.items())]
    require(bool(ordered) and all(len(g) == 2 and {r["condition_id"] for r in g} == {1, 2}
                                and len({r["split"] for r in g}) == 1 for g in ordered), "Bootstrap requires complete source pairs.")
    y = np.asarray([[r["activation_label"] for r in g] for g in ordered])
    indices = np.random.default_rng(seed).integers(0, len(ordered), size=(repetitions, len(ordered)))
    sampled_y = y[indices].reshape(repetitions, -1)
    values, points = {}, {}
    for method in METHODS:
        p = np.asarray([[r["predictions"][method] for r in g] for g in ordered])
        sampled = p[indices].reshape(repetitions, -1)
        tp = ((sampled_y == 1) & (sampled == 1)).sum(axis=1)
        fp = ((sampled_y == 0) & (sampled == 1)).sum(axis=1)
        fn = ((sampled_y == 1) & (sampled == 0)).sum(axis=1)
        den = 2 * tp + fp + fn
        values[method] = np.divide(2 * tp, den, out=np.zeros(repetitions), where=den != 0)
        tp0 = ((y == 1) & (p == 1)).sum()
        den0 = 2 * tp0 + ((y == 0) & (p == 1)).sum() + ((y == 1) & (p == 0)).sum()
        points[method] = float(2 * tp0 / den0) if den0 else 0.0
    return {"score_fusion_minus_" + method: {"f1_difference": points["score_fusion"] - points[method],
            "percentile_95_interval": np.percentile(values["score_fusion"] - values[method], [2.5, 97.5]).tolist(),
            "source_groups": len(ordered), "resamples": repetitions, "seed": seed,
            "interpretation": "Post-hoc and unadjusted, conditional on reused labels; cannot remove adaptive test reuse."}
            for method in METHODS if method != "score_fusion"}


def predict_one(state, record, encoder=None):
    if state.get("kind") != "score_fusion":
        return existing.predict_one(state, record, encoder)
    weight = state["behavioral_weight"]
    require(0 <= weight <= 1, "Invalid saved fusion weight.")
    generation = existing.predict_one(state["generation_component"], record)[0] if weight < 1 else 0.0
    behavioral = existing.predict_one(state["behavioral_component"], record, encoder)[0] if weight > 0 else 0.0
    score = float(mix([generation], [behavioral], weight)[0])
    return score, int(score >= state["threshold"])
