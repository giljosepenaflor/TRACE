"""Fixed extensions of the three primary generation statistics."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_NAMES, generation_features

NAMES = FEATURE_NAMES + ("entropy_std", "entropy_q10", "entropy_q50", "entropy_q90",
    "confidence_q10", "confidence_q50", "confidence_q90", "mean_generated_nll",
    "first16_entropy", "last16_entropy", "first16_confidence", "last16_confidence",
    "entropy_adjacent_mad", "confidence_adjacent_mad", "entropy_time_slope",
    "confidence_time_slope", "confidence_ge90_fraction", "confidence_ge90_longest_run_fraction")
METHODS = ("generation3", "behavioral", "full3", "generation21", "full21")
SETTINGS = {"C": 1., "max_iter": 1000, "random_state": 42, "seed": 20261106,
            "bootstrap_repetitions": 2000, "prefix_tokens": 16, "high_confidence_cutoff": .9}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def extract(signals):
    primary = generation_features(signals)
    require("token_confidences" in signals and "token_entropies" in signals,
            "Sequence extension requires saved aligned confidence and entropy arrays.")
    c, e = np.asarray(signals["token_confidences"], dtype=float), np.asarray(signals["token_entropies"], dtype=float)
    require(c.shape == e.shape == (len(signals["generated_token_ids"]),) and len(c) > 0,
            "Sequence arrays do not align.")
    require(np.isfinite(c).all() and np.isfinite(e).all() and np.all((0 <= c) & (c <= 1)) and np.all(e >= 0), "Invalid sequence statistics.")
    # With dense distributions as well, prevent inconsistent secondary arrays.
    np_primary = np.array([e.mean(), c.mean(), c.var()])
    require(np.allclose(primary, np_primary, atol=1e-12, rtol=1e-12), "Summary arrays disagree with full distributions.")
    t = np.linspace(-.5, .5, len(c))
    denominator = t @ t if len(c) > 1 else 1.
    slopes = [(t @ (a - a.mean())) / denominator if len(c) > 1 else 0. for a in (e, c)]
    longest = current = 0
    for high in c >= .9:
        current = current + 1 if high else 0
        longest = max(longest, current)
    result = np.r_[primary, e.std(), np.quantile(e, [.1, .5, .9], method="linear"),
        np.quantile(c, [.1, .5, .9], method="linear"), -np.log(np.clip(c, 1e-12, 1)).mean(),
        e[:16].mean(), e[-16:].mean(), c[:16].mean(), c[-16:].mean(),
        [np.abs(np.diff(a)).mean() if len(c) > 1 else 0. for a in (e, c)],
        slopes, np.mean(c >= .9), longest / len(c)]
    require(result.shape == (len(NAMES),) and np.isfinite(result).all(), "Invalid sequence feature vector.")
    return result


def matrix(method, base, sequence, scaler):
    require(method in METHODS and len(base) == len(sequence) and sequence.shape[1] == len(NAMES), "Method/feature alignment differs.")
    if method == "behavioral":
        return base
    g = scaler.transform(sequence[:, :3] if method.endswith("3") else sequence)
    return np.c_[base, g] if method.startswith("full") else g


def fit(method, base, sequence, rows):
    require(len(rows) == len(base) == len(sequence) and all(r["split"] == "train" for r in rows), "Training-only feature membership required.")
    y = np.array([r["activation_label"] for r in rows])
    require(set(y) == {0, 1}, "Both fitting classes required.")
    scaler = None if method == "behavioral" else StandardScaler().fit(sequence[:, :3] if method.endswith("3") else sequence)
    x = matrix(method, base, sequence, scaler)
    model = LogisticRegression(C=1., max_iter=1000, random_state=42).fit(x, y)
    require(model.n_iter_[0] < 1000, "Sequence classifier failed to converge.")
    return {"method": method, "weights": model.coef_[0], "bias": float(model.intercept_[0]), "scaler": scaler,
            "fit_sample_ids": [r["sample_id"] for r in rows], "iterations": int(model.n_iter_[0])}


def predict(state, base, sequence):
    from scipy.special import expit
    return expit(matrix(state["method"], base, sequence, state["scaler"]) @ state["weights"] + state["bias"])


def select(source_results, dataset_results):
    ranking = {m: {"worst_dataset_ap": min(r[m]["average_precision"] for r in dataset_results),
                    "mean_dataset_ap": float(np.mean([r[m]["average_precision"] for r in dataset_results])),
                    "mean_source_fold_ap": float(np.mean([r[m]["average_precision"] for r in source_results]))} for m in METHODS}
    key = lambda m: (-ranking[m]["worst_dataset_ap"], -ranking[m]["mean_dataset_ap"], -ranking[m]["mean_source_fold_ap"], METHODS.index(m))
    return {"selected_full": min(("full3", "full21"), key=key), "ranking": ranking,
            "best_all_methods": min(METHODS, key=key), "validation_or_test_used": False}


def gate(selection, validation):
    return {"sequence_full_selected": selection["selected_full"] == "full21",
            "worst_dataset_ap_improves": selection["ranking"]["full21"]["worst_dataset_ap"] > selection["ranking"]["full3"]["worst_dataset_ap"],
            "validation_f1_beats_references": all(validation["full21"]["f1"] > validation[m]["f1"] for m in METHODS if m != "full21")}


def prediction_rows(labels, scores, thresholds):
    policies = ("max_f1", "fpr_0.10", "recall_0.80")
    return [{**{k: r[k] for k in ("sample_id", "source_id", "source", "split", "condition_id", "activation_label") if k in r},
             "scores": {m: float(scores[m][i]) for m in METHODS},
             "predictions": {p: {m: int(scores[m][i] >= thresholds[m][p]) for m in METHODS} for p in policies}}
            for i, r in enumerate(labels)]


def summarize(rows):
    from .detector import metrics
    from sklearn.metrics import average_precision_score
    y = [r["activation_label"] for r in rows]
    result = {}
    for policy in ("max_f1", "fpr_0.10", "recall_0.80"):
        result[policy] = {}
        for m in METHODS:
            scores = [r["scores"][m] for r in rows]
            ranking = metrics(y, scores, .5)
            result[policy][m] = {**metrics(y, [r["predictions"][policy][m] for r in rows], .5),
                                 "auroc": ranking["auroc"], "average_precision": float(average_precision_score(y, scores))}
    return result


def slices(rows):
    return {s: summarize([r for r in rows if r["source"] == s]) for s in sorted({r["source"] for r in rows})}


def bootstrap(rows, selected):
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in rows:
        grouped[r["source_id"]].append(r)
    groups = [grouped[k] for k in sorted(grouped)]
    require(all(len(g) == 2 and {r["condition_id"] for r in g} == {1, 2} for g in groups), "Whole paired sources required.")
    index = np.random.default_rng(SETTINGS["seed"]).integers(0, len(groups), (2000, len(groups)))
    samples = {}
    for m in METHODS:
        matrices = []
        for group in groups:
            y = np.array([r["activation_label"] for r in group])
            p = np.array([r["predictions"]["max_f1"][m] for r in group])
            matrices.append([sum((y == 1) & (p == 1)), sum((y == 0) & (p == 1)), sum((y == 1) & (p == 0))])
        count = np.array(matrices)[index].sum(axis=1)
        den = 2 * count[:, 0] + count[:, 1] + count[:, 2]
        samples[m] = np.divide(2 * count[:, 0], den, out=np.zeros(2000), where=den != 0)
    point = summarize(rows)["max_f1"]
    return {selected + "_minus_" + ref: {"f1_difference": point[selected]["f1"] - point[ref]["f1"],
            "percentile_95_interval": np.percentile(samples[selected] - samples[ref], [2.5, 97.5]).tolist(),
            "resamples": 2000, "seed": SETTINGS["seed"], "source_groups": len(groups)} for ref in METHODS if ref != selected}
