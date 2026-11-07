"""Frozen reused-data comparisons; no fitting, threshold search, or target calls."""
from collections import defaultdict

import numpy as np
from sklearn.metrics import average_precision_score

from .detector import metrics
from .features import generation_features
from .representation_experiment import apply_projection

METHODS = {
    "original_full": {"original_view": "full", "candidate_id": "prompt_response_projected__full__C1__unweighted", "view": "full", "projected": True},
    "candidate_full": {"original_view": None, "candidate_id": "prompt_response_raw__full__C1__unweighted", "view": "full", "projected": False},
    "generation": {"original_view": "generation", "candidate_id": "generation__C1__unweighted", "view": "generation", "projected": False},
    "original_behavioral": {"original_view": "behavioral", "candidate_id": "prompt_response_projected__behavioral__C1__unweighted", "view": "behavioral", "projected": True},
    "candidate_behavioral": {"original_view": None, "candidate_id": "prompt_response_raw__behavioral__C1__unweighted", "view": "behavioral", "projected": False},
    "tuned_full": {"original_view": None, "candidate_id": "prompt_response_raw__full__C10__unweighted", "view": "full", "projected": False},
}
CONTRASTS = (("candidate_full", "original_full"), ("candidate_full", "generation"),
             ("candidate_behavioral", "original_behavioral"), ("tuned_full", "original_full"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_test_membership(labels, forbidden_sources=()):
    seen, groups = set(), defaultdict(list)
    for row in labels:
        require(row["split"] in ("test_seen", "test_unseen") and row["condition_id"] in (1, 2),
                "Only admitted adapted test records are allowed.")
        require(row["source_id"] not in forbidden_sources, "Development source found in evaluation.")
        require(row["sample_id"] not in seen, "Duplicate test sample.")
        require(type(row["activation_label"]) is int and row["activation_label"] in (0, 1), "Invalid activation label.")
        require(row["four_condition_pattern"] in ("0000", "0100"), "Excluded review pattern found.")
        expected = int(row["four_condition_pattern"] == "0100" and row["condition_id"] == 2)
        require(row["activation_label"] == expected, "Label differs from sealed attribution rule.")
        seen.add(row["sample_id"])
        groups[row["source_id"]].append(row)
    require(bool(labels), "No test labels provided.")
    for rows in groups.values():
        require(len(rows) == 2 and {r["condition_id"] for r in rows} == {1, 2}
                and len({r["split"] for r in rows}) == len({r["four_condition_pattern"] for r in rows}) == 1,
                "Incomplete or cross-partition adapted test source.")


def features_for_state(state, base, signals):
    view = state["view"]
    if view in ("behavioral", "full"):
        behavior = apply_projection(base, state["projection"]) if state["projection"] is not None else base
    if view in ("generation", "full"):
        generation = state["scaler"].transform(signals)
    values = np.c_[behavior, generation] if view == "full" else behavior if view == "behavioral" else generation
    require(np.isfinite(values).all(), "Nonfinite classifier features.")
    return values


def predict_one(state, record, encoder=None):
    if state["view"] == "generation":
        base = None
    else:
        require(encoder is not None, "Behavioral encoding is required.")
        base = encoder.encode_base([record])
    signal = generation_features(record["signals"])[None, :] if state["view"] != "behavioral" else None
    values = features_for_state(state, base, signal)
    score = float(state["model"].predict_proba(values)[0, 1])
    return score, int(score >= state["threshold"])


def score_candidates(base, signals, labels, original_rows, states):
    """Reuse saved original predictions; score only the three frozen new models."""
    validate_test_membership(labels)
    require(len(base) == len(signals) == len(labels), "Feature and label coverage differs.")
    original = {r["sample_id"]: r for r in original_rows}
    require(len(original) == len(labels) and set(original) == {r["sample_id"] for r in labels}, "Original prediction coverage differs.")
    scores = {}
    for method, definition in METHODS.items():
        view = definition["original_view"]
        if view:
            scores[method] = np.asarray([original[r["sample_id"]]["scores"][view] for r in labels])
        else:
            matrix = features_for_state(states[method], base, signals)
            scores[method] = states[method]["model"].predict_proba(matrix)[:, 1]
        require(np.isfinite(scores[method]).all(), "Nonfinite comparison scores.")
    rows = []
    for index, label in enumerate(labels):
        previous = original[label["sample_id"]]
        require(all(previous[k] == label[k] for k in ("sample_id", "source_id", "split", "activation_label")),
                "Original and reused-test label membership differs.")
        decisions = {method: int(values[index] >= states[method]["threshold"]) for method, values in scores.items()}
        for method, definition in METHODS.items():
            if definition["original_view"]:
                require(decisions[method] == previous["predictions"][definition["original_view"]], "Original decision or threshold changed.")
        rows.append({k: label[k] for k in ("sample_id", "source_id", "split", "condition_id", "activation_label")}
                    | {"scores": {m: float(v[index]) for m, v in scores.items()}, "predictions": decisions})
    return rows


def summarize(rows, thresholds):
    result = {}
    for split in ("test_seen", "test_unseen"):
        part = [r for r in rows if r["split"] == split]
        require(bool(part), "A required reused-test partition is empty.")
        y = np.asarray([r["activation_label"] for r in part])
        result[split] = {}
        for method in METHODS:
            values = np.asarray([r["scores"][method] for r in part])
            require(all(r["predictions"][method] == int(r["scores"][method] >= thresholds[method]) for r in part),
                    "Saved decisions differ from frozen thresholds.")
            result[split][method] = {**metrics(y, values, thresholds[method]),
                                     "average_precision": float(average_precision_score(y, values))}
    return result


def paired_intervals(rows, repetitions=2000, seed=20261106):
    groups = defaultdict(list)
    for row in rows:
        groups[row["source_id"]].append(row)
    ordered = [sorted(values, key=lambda r: r["condition_id"]) for _, values in sorted(groups.items())]
    require(all(len(g) == 2 and {r["condition_id"] for r in g} == {1, 2} for g in ordered), "Bootstrap requires whole source pairs.")
    y = np.asarray([[r["activation_label"] for r in group] for group in ordered])
    indices = np.random.default_rng(seed).integers(0, len(ordered), size=(repetitions, len(ordered)))
    sampled_y = y[indices].reshape(repetitions, -1)
    sampled_f1, point = {}, {}
    for method in METHODS:
        pred = np.asarray([[r["predictions"][method] for r in group] for group in ordered])
        sampled = pred[indices].reshape(repetitions, -1)
        tp = ((sampled_y == 1) & (sampled == 1)).sum(axis=1)
        fp = ((sampled_y == 0) & (sampled == 1)).sum(axis=1)
        fn = ((sampled_y == 1) & (sampled == 0)).sum(axis=1)
        denominator = 2 * tp + fp + fn
        sampled_f1[method] = np.divide(2 * tp, denominator, out=np.zeros(repetitions), where=denominator != 0)
        tp0, fp0, fn0 = int(((y == 1) & (pred == 1)).sum()), int(((y == 0) & (pred == 1)).sum()), int(((y == 1) & (pred == 0)).sum())
        point[method] = 2 * tp0 / (2 * tp0 + fp0 + fn0) if 2 * tp0 + fp0 + fn0 else 0.0
    return {f"{left}_minus_{right}": {"f1_difference": point[left] - point[right],
             "percentile_95_interval": np.percentile(sampled_f1[left] - sampled_f1[right], [2.5, 97.5]).tolist(),
             "source_groups": len(groups), "resamples": repetitions, "seed": seed,
             "interpretation": "post-hoc, unadjusted, conditional on reused labels; does not remove adaptive test reuse"}
            for left, right in CONTRASTS}
