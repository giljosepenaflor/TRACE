"""Source-paired activation learning with unchanged single-interaction inputs."""
from collections import defaultdict

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

METHODS = ("standard_generation", "standard_behavioral", "standard_full",
           "paired_generation", "paired_behavioral", "paired_full")
SETTINGS = {"pair_ranking_weight": .25, "pair_consistency_weight": .25,
            "C": 1., "max_iter": 2000, "gtol": 1e-7, "ftol": 1e-12,
            "reference_max_iter": 1000, "seed": 20261106, "bootstrap_repetitions": 2000}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pairs(rows):
    grouped = defaultdict(list)
    ids = set()
    for index, row in enumerate(rows):
        require(row["sample_id"] not in ids and row["activation_label"] in (0, 1), "Duplicate ID or invalid activation label.")
        ids.add(row["sample_id"])
        grouped[row["source_id"]].append(index)
    discordant, equal = [], []
    for indices in grouped.values():
        require(len(indices) == 2 and {rows[i]["condition_id"] for i in indices} == {1, 2}, "Complete source pairs are required.")
        left, right = sorted(indices, key=lambda i: rows[i]["condition_id"])
        require(rows[left]["split"] == rows[right]["split"] and rows[left]["source"] == rows[right]["source"], "Pair membership differs.")
        if rows[left]["activation_label"] == rows[right]["activation_label"]:
            equal.append((left, right))
        else:
            discordant.append((left, right) if rows[left]["activation_label"] else (right, left))
    return np.asarray(discordant, dtype=int).reshape(-1, 2), np.asarray(equal, dtype=int).reshape(-1, 2)


def differences(x, indices):
    return x[indices[:, 0]] - x[indices[:, 1]]


def objective(theta, x, y, discordant, equal, settings=None):
    settings = settings or SETTINGS
    w, bias = theta[:-1], theta[-1]
    logits = x @ w + bias
    error = expit(logits) - y
    value = np.mean(np.logaddexp(0, logits) - y * logits) + w @ w / (2 * settings["C"] * len(y))
    gradient = x.T @ error / len(y) + w / (settings["C"] * len(y))
    if len(discordant):
        d = differences(x, discordant)
        value += settings["pair_ranking_weight"] * np.mean(np.logaddexp(0, -(d @ w)))
        gradient += settings["pair_ranking_weight"] * (d.T @ (expit(d @ w) - 1)) / len(d)
    if len(equal):
        d = differences(x, equal)
        residual = d @ w
        value += settings["pair_consistency_weight"] * np.mean(residual ** 2)
        gradient += 2 * settings["pair_consistency_weight"] * (d.T @ residual) / len(d)
    return float(value), np.r_[gradient, error.mean()]


def partition_check(rows, fit, held, groups):
    require(all(r["split"] == "train" for r in rows), "Model selection may use training rows only.")
    require(len(rows) == len(groups), "Group membership differs.")
    fit, held = list(fit), list(held)
    require(len(fit) == len(set(fit)) and len(held) == len(set(held)) and fit and held, "Invalid fitting/held indices.")
    require(min(fit + held) >= 0 and max(fit + held) < len(rows), "Indices out of range.")
    require(not (set(fit) & set(held)), "Overlapping fitting and held rows.")
    require(not ({groups[i] for i in fit} & {groups[i] for i in held}), "Duplicate group leakage.")
    require(not ({rows[i]["source_id"] for i in fit} & {rows[i]["source_id"] for i in held}), "Source leakage.")
    for indices in (fit, held):
        pairs([rows[i] for i in indices])
        require({rows[i]["activation_label"] for i in indices} == {0, 1}, "Both classes required in each partition.")


def dataset_folds(rows, groups):
    result = []
    for source in sorted({r["source"] for r in rows}):
        held = [i for i, row in enumerate(rows) if row["source"] == source]
        forbidden = {groups[i] for i in held}
        fit = [i for i, row in enumerate(rows) if row["source"] != source and groups[i] not in forbidden]
        partition_check(rows, fit, held, groups)
        result.append({"held_dataset": source, "fit_indices": fit, "held_indices": held,
                       "excluded_duplicate_rows": len(rows) - len(fit) - len(held)})
    require(len(result) == 2, "This experiment requires the two declared training datasets.")
    return result


def features(method, base, generation, scaler):
    require(method in METHODS and len(base) == len(generation), "Unknown method or feature misalignment.")
    view = method.split("_")[1]
    if view == "behavioral":
        return np.asarray(base, dtype=float)
    g = scaler.transform(generation)
    return g if view == "generation" else np.c_[base, g]


def fit(method, base, generation, rows, settings=None):
    settings = settings or SETTINGS
    require(all(r["split"] == "train" for r in rows), "Fit requires training rows only.")
    require(len(base) == len(generation) == len(rows), "Feature/label alignment differs.")
    d, e = pairs(rows)
    y = np.asarray([r["activation_label"] for r in rows], dtype=int)
    require(set(y) == {0, 1}, "Both fitting classes are required.")
    scaler = None if method.endswith("behavioral") else StandardScaler().fit(generation)
    x = features(method, base, generation, scaler)
    require(np.isfinite(x).all(), "Nonfinite activation features.")
    if method.startswith("standard"):
        model = LogisticRegression(C=settings["C"], max_iter=settings["reference_max_iter"], random_state=42).fit(x, y)
        require(model.n_iter_[0] < settings["reference_max_iter"], "Reference did not converge.")
        weights, bias = model.coef_[0], float(model.intercept_[0])
        diagnostics = {"converged": True, "iterations": int(model.n_iter_[0])}
    else:
        initial = np.r_[np.zeros(x.shape[1]), np.log(y.mean() / (1 - y.mean()))]
        result = minimize(objective, initial, args=(x, y, d, e, settings), method="L-BFGS-B", jac=True,
            options={"maxiter": settings["max_iter"], "gtol": settings["gtol"], "ftol": settings["ftol"]})
        require(result.success and np.isfinite(result.fun) and np.isfinite(result.x).all(), "Paired optimizer failed to converge.")
        weights, bias = result.x[:-1], float(result.x[-1])
        diagnostics = {"converged": bool(result.success), "iterations": int(result.nit),
                       "objective": float(result.fun), "gradient_max_abs": float(np.max(abs(result.jac))), "message": str(result.message)}
    return {"method": method, "weights": weights, "bias": bias, "scaler": scaler,
            "fit_sample_ids": [r["sample_id"] for r in rows], "fit_source_ids": sorted({r["source_id"] for r in rows}),
            "pair_counts": {"discordant": len(d), "equal": len(e)}, "optimizer": diagnostics,
            "single_interaction_inference": True}


def predict(state, base, generation):
    return expit(features(state["method"], base, generation, state["scaler"]) @ state["weights"] + state["bias"])


def select(dataset_results, source_results):
    ranking = {}
    for m in METHODS:
        aps = [r[m]["average_precision"] for r in dataset_results]
        ranking[m] = {"worst_dataset_ap": min(aps), "mean_dataset_ap": float(np.mean(aps)),
                      "mean_source_fold_ap": float(np.mean([r[m]["average_precision"] for r in source_results]))}
    key = lambda m: (-ranking[m]["worst_dataset_ap"], -ranking[m]["mean_dataset_ap"], -ranking[m]["mean_source_fold_ap"], METHODS.index(m))
    return {"selected_full": min(("standard_full", "paired_full"), key=key),
            "best_all_methods": min(METHODS, key=key), "ranking": ranking,
            "validation_or_test_used": False}


def gate(selection, validation):
    selected = selection["selected_full"]
    references = [m for m in METHODS if m != "paired_full"]
    return {"selected_is_paired_full": selected == "paired_full",
            "worst_dataset_ap_improves_standard_full": selection["ranking"]["paired_full"]["worst_dataset_ap"] > selection["ranking"]["standard_full"]["worst_dataset_ap"],
            "validation_f1_beats_all_references": all(validation["paired_full"]["f1"] > validation[m]["f1"] for m in references)}


def pair_diagnostics(rows, logits):
    d, e = pairs(rows)
    return {"discordant_pairs": len(d), "equal_pairs": len(e),
            "discordant_positive_ranks_higher": float(np.mean(logits[d[:, 0]] > logits[d[:, 1]])) if len(d) else None,
            "equal_pair_mean_abs_logit_difference": float(np.mean(abs(logits[e[:, 0]] - logits[e[:, 1]]))) if len(e) else None}
