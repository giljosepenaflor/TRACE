"""Frozen MiniLM pair encoding and source-isolated auxiliary teacher-label heads."""
from __future__ import annotations

import hashlib
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from .auxiliary_consensus import HEADS

ENCODING = {"max_tokens": 256, "max_request_tokens": 64, "batch_size": 32,
            "dimensions": 384, "device": "cpu", "normalize_embeddings": True,
            "input": "tokenizer sentence pair: executed request then observed response"}
SETTINGS = {"C": 1.0, "class_weight": None, "max_iter": 1000, "model_seed": 42,
            "decision_threshold": 0.5, "minimum_fit_source_groups_per_class": 5,
            "inner_folds": 5, "seed": 20261106, "bootstrap_repetitions": 2000,
            "macro_f1_min": 0.70, "class_recall_min": 0.60, "unsafe_positive_recall_min": 0.80}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pair_features(tokenizer, prompt, response, encoding=None):
    encoding = encoding or ENCODING
    require(isinstance(prompt, str) and isinstance(response, str), "Scorer inputs must be request/response strings.")
    request = tokenizer.encode(prompt, add_special_tokens=False, truncation=False, verbose=False)
    answer = tokenizer.encode(response, add_special_tokens=False, truncation=False, verbose=False)
    overhead = tokenizer.num_special_tokens_to_add(pair=True)
    kept_request = request[:encoding["max_request_tokens"]]
    budget = encoding["max_tokens"] - overhead - len(kept_request)
    require(budget > 0, "No response token budget remains.")
    kept_answer = answer[:budget]
    features = tokenizer.prepare_for_model(kept_request, pair_ids=kept_answer,
        add_special_tokens=True, truncation=False, padding=False,
        return_attention_mask=True, return_token_type_ids=True, verbose=False)
    require(len(features["input_ids"]) == len(kept_request) + len(kept_answer) + overhead
            and len(features["input_ids"]) <= encoding["max_tokens"], "Pair-token length differs from the fixed budget.")
    return dict(features), {"request_tokens": len(request), "request_tokens_kept": len(kept_request),
        "response_tokens": len(answer), "response_tokens_kept": len(kept_answer),
        "pair_special_tokens": overhead, "input_tokens": len(features["input_ids"])}


class PairEncoder:
    """Shareable frozen sentence-transformer; inference consumes text only."""
    def __init__(self, model_path, model=None, encoding=None):
        self.model_path = str(model_path)
        self.model = model
        self.encoding = dict(encoding or ENCODING)

    def load(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_path, device="cpu", local_files_only=True)
        self.model.eval()
        return self.model

    def encode(self, records, return_diagnostics=False):
        import torch
        require(bool(records), "No scorer records provided.")
        model = self.load()
        prepared, diagnostics = [], []
        for row in records:
            features, info = pair_features(model.tokenizer, row["prompt"], row["response"], self.encoding)
            prepared.append(features)
            diagnostics.append(info)
        arrays = []
        with torch.inference_mode():
            for start in range(0, len(prepared), self.encoding["batch_size"]):
                features = model.tokenizer.pad(prepared[start:start + self.encoding["batch_size"]],
                    padding=True, return_tensors="pt", verbose=False)
                output = model(dict(features))["sentence_embedding"]
                arrays.append(torch.nn.functional.normalize(output, dim=1).cpu().numpy())
        values = np.concatenate(arrays, axis=0)
        require(values.shape == (len(records), self.encoding["dimensions"]) and np.isfinite(values).all(), "Invalid auxiliary embedding shape/values.")
        require(np.allclose(np.linalg.norm(values, axis=1), 1, atol=1e-5), "Auxiliary embeddings are not normalized.")
        return (values, diagnostics) if return_diagnostics else values


def nested_plan(membership, outer_folds, groups, settings=None):
    settings = settings or SETTINGS
    rows = membership["train"]
    require(all(r["split"] == "train" for r in rows), "Nested fitting may use training membership only.")
    require(not ({r["source_id"] for r in rows} & {r["source_id"] for r in membership["validation"]}), "Development source crosses partitions.")
    y = np.asarray([r["activation_label"] for r in rows])
    groups = np.asarray(groups)
    require(len(groups) == len(rows), "Training group alignment differs.")
    result, coverage = [], np.zeros(len(rows), dtype=int)
    for outer in outer_folds:
        fit, held = np.asarray(outer["fit_indices"]), np.asarray(outer["held_indices"])
        require(len(set(fit)) == len(fit) and len(set(held)) == len(held)
                and not (set(fit) & set(held)) and set(fit) | set(held) == set(range(len(rows))), "Invalid outer training partition.")
        require(not (set(groups[fit]) & set(groups[held])), "Outer source group leakage.")
        coverage[held] += 1
        inner_splitter = StratifiedGroupKFold(n_splits=settings["inner_folds"], shuffle=True,
            random_state=settings["seed"] + outer["fold"] + 1)
        inner, inner_coverage = [], np.zeros(len(fit), dtype=int)
        for number, (inner_fit, inner_held) in enumerate(inner_splitter.split(np.zeros(len(fit)), y[fit], groups[fit])):
            f, h = fit[inner_fit], fit[inner_held]
            require(not (set(groups[f]) & (set(groups[h]) | set(groups[held]))), "Nested source group leakage.")
            inner_coverage[inner_held] += 1
            inner.append({"inner_fold": number, "fit_indices": f.tolist(), "held_indices": h.tolist()})
        require(np.all(inner_coverage == 1), "Incomplete inner held-fold coverage.")
        result.append({**outer, "inner_folds": inner})
    require(np.all(coverage == 1), "Incomplete or repeated outer held-fold coverage.")
    return result


def fitting_support(labels, fit_sources):
    result = {}
    for head in HEADS:
        eligible = [r for r in labels if r["source_id"] in fit_sources and r["auxiliary_targets"][head] is not None]
        result[head] = {"class_examples": {str(v): sum(r["auxiliary_targets"][head] == v for r in eligible) for v in (0, 1)},
            "class_source_groups": {str(v): len({r["source_id"] for r in eligible if r["auxiliary_targets"][head] == v}) for v in (0, 1)}}
    return result


def check_support_plan(labels, membership, folds, settings=None):
    settings = settings or SETTINGS
    result = []
    for outer in folds:
        stages = [("outer", outer)] + [("inner", f) for f in outer["inner_folds"]]
        for kind, fold in stages:
            sources = {membership["train"][i]["source_id"] for i in fold["fit_indices"]}
            support = fitting_support(labels, sources)
            result.append({"stage": kind, "outer_fold": outer["fold"], "inner_fold": fold.get("inner_fold"),
                "heads": support, "passes": all(min(s["class_source_groups"].values()) >= settings["minimum_fit_source_groups_per_class"] for s in support.values())})
    return result


def default_fitter(features, targets, settings):
    model = LogisticRegression(C=settings["C"], class_weight=settings["class_weight"],
        max_iter=settings["max_iter"], random_state=settings["model_seed"])
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(features, targets)
    require(max(model.n_iter_) < settings["max_iter"], "Auxiliary classifier did not converge.")
    return model


def fit_heads(features, labels, fit_sources, forbidden_sources=(), settings=None, fitter=default_fitter):
    settings = settings or SETTINGS
    require(not (set(fit_sources) & set(forbidden_sources)), "Forbidden source enters auxiliary fitting.")
    require(len(features) == len(labels) and np.isfinite(features).all(), "Auxiliary feature membership differs.")
    require(all(r["split"] == "train" for r in labels), "Only training auxiliary targets may be fitted.")
    support = fitting_support(labels, fit_sources)
    models, audits = {}, []
    for head in HEADS:
        require(min(support[head]["class_source_groups"].values()) >= settings["minimum_fit_source_groups_per_class"], "Insufficient fitting source-class support for " + head)
        indices = [i for i, r in enumerate(labels) if r["source_id"] in fit_sources and r["auxiliary_targets"][head] is not None]
        y = np.asarray([labels[i]["auxiliary_targets"][head] for i in indices], dtype=int)
        models[head] = fitter(features[indices], y, settings)
        require(list(models[head].classes_) == [0, 1], "Auxiliary classifier classes differ.")
        audits.append({"head": head, "fit_sample_ids": [labels[i]["sample_id"] for i in indices],
            "fit_source_ids": sorted({labels[i]["source_id"] for i in indices}),
            "target_sha256": hashlib.sha256(y.tobytes()).hexdigest(), **support[head],
            "majority_class": int((y == 1).sum() > (y == 0).sum())})
    return models, audits


def score_heads(models, features):
    values = np.column_stack([models[h].predict_proba(features)[:, 1] for h in HEADS])
    require(np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all(), "Invalid auxiliary scores.")
    return values


def teacher_metrics(targets, scores, cutoff=0.5):
    y = np.asarray(targets, dtype=int)
    p = np.asarray(scores) >= cutoff
    c = confusion_matrix(y, p, labels=[0, 1]).ravel()
    return {"count": len(y), "class_support": {str(v): int((y == v).sum()) for v in (0, 1)},
        "macro_f1": float(f1_score(y, p, labels=[0, 1], average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y, p)),
        "class_recalls": {str(v): float(recall_score(y, p, labels=[v], average=None, zero_division=0)[0]) for v in (0, 1)},
        "average_precision": float(average_precision_score(y, scores)) if y.sum() else None,
        "auroc": float(roc_auc_score(y, scores)) if len(set(y)) == 2 else None,
        "confusion_matrix": dict(zip(("tn", "fp", "fn", "tp"), map(int, c)))}


def assess_quality(labels, scores, majority, settings=None):
    settings = settings or SETTINGS
    result = {}
    for index, head in enumerate(HEADS):
        known = [i for i, r in enumerate(labels) if r["auxiliary_targets"][head] is not None]
        targets = [labels[i]["auxiliary_targets"][head] for i in known]
        measured = teacher_metrics(targets, scores[known, index], settings["decision_threshold"])
        baseline = teacher_metrics(targets, majority[known, index], settings["decision_threshold"])
        gates = {"macro_f1": measured["macro_f1"] >= settings["macro_f1_min"],
            "class_0_recall": measured["class_recalls"]["0"] >= settings["class_recall_min"],
            "class_1_recall": measured["class_recalls"]["1"] >= (settings["unsafe_positive_recall_min"] if head == "unsafe_enabling_assistance" else settings["class_recall_min"])}
        result[head] = {"teacher_label_agreement": measured, "training_fold_majority_baseline": baseline,
            "gates": gates, "passed": all(gates.values()), "unknown_examples_excluded": len(labels) - len(known)}
    return {"heads": result, "all_heads_passed": all(r["passed"] for r in result.values()),
            "interpretation": "Agreement with user-approved AI-assisted teacher labels on an error-enriched source-group pilot; not independently human-validated semantic accuracy."}


def group_bootstrap(labels, scores, settings=None):
    settings = settings or SETTINGS
    groups = sorted({r["source_id"] for r in labels})
    samples = np.random.default_rng(settings["seed"]).integers(0, len(groups), (settings["bootstrap_repetitions"], len(groups)))
    result = {}
    for j, head in enumerate(HEADS):
        group_counts = []
        for group in groups:
            c = np.zeros((2, 2), dtype=int)
            for i, row in enumerate(labels):
                target = row["auxiliary_targets"][head]
                if row["source_id"] == group and target is not None:
                    c[target, int(scores[i, j] >= settings["decision_threshold"])] += 1
            group_counts.append(c)
        counts = np.asarray(group_counts)[samples].sum(axis=1)
        tn, fp, fn, tp = (counts[:, i, k].astype(float) for i, k in ((0, 0), (0, 1), (1, 0), (1, 1)))
        def ratio(a, b):
            return np.divide(a, b, out=np.full_like(a, np.nan), where=b != 0)
        positive_f1, negative_f1 = ratio(2 * tp, 2 * tp + fp + fn), ratio(2 * tn, 2 * tn + fp + fn)
        fields = {"macro_f1": (positive_f1 + negative_f1) / 2,
                  "class_0_recall": ratio(tn, tn + fp), "class_1_recall": ratio(tp, tp + fn)}
        result[head] = {name: {"valid_resamples": int(np.isfinite(v).sum()),
            "percentile_95_interval": np.nanpercentile(v, [2.5, 97.5]).tolist()} for name, v in fields.items()}
    return {"source_groups": len(groups), "resamples": settings["bootstrap_repetitions"], "seed": settings["seed"],
            "heads": result, "interpretation": "Exploratory group bootstrap, conditional on selected pilot and AI-assisted labels; not a population or human-quality guarantee."}


def cross_validate(features, labels, membership, folds, settings=None, fitter=default_fitter):
    settings = settings or SETTINGS
    require(len({r["sample_id"] for r in labels}) == len(labels), "Duplicate auxiliary label ID.")
    require(all(r["split"] == "train" for r in labels), "Auxiliary validation must remain training-source only.")
    approved = {r["sample_id"]: r["source_id"] for r in membership["train"]}
    require(all(approved.get(r["sample_id"]) == r["source_id"] for r in labels), "Auxiliary label is outside approved training membership.")
    oof, majority = np.full((len(labels), len(HEADS)), np.nan), np.full((len(labels), len(HEADS)), np.nan)
    coverage = np.zeros(len(labels), dtype=int)
    models, audits = {}, []
    for fold in folds:
        fit_sources = {membership["train"][i]["source_id"] for i in fold["fit_indices"]}
        held_sources = {membership["train"][i]["source_id"] for i in fold["held_indices"]}
        model, fit_audits = fit_heads(features, labels, fit_sources, held_sources, settings, fitter)
        held = [i for i, r in enumerate(labels) if r["source_id"] in held_sources]
        coverage[held] += 1
        oof[held] = score_heads(model, features[held])
        majority[held] = np.asarray([a["majority_class"] for a in fit_audits])
        models[str(fold["fold"])] = model
        audits.extend({"stage": "outer", "fold": fold["fold"], **a} for a in fit_audits)
    require(np.all(coverage == 1) and np.isfinite(oof).all(), "Auxiliary OOF coverage incomplete/repeated.")
    return {"oof_scores": oof, "majority_scores": majority, "fold_models": models, "fit_audits": audits,
            "quality": assess_quality(labels, oof, majority, settings), "uncertainty": group_bootstrap(labels, oof, settings)}


def predict_one(state, record, encoder):
    return score_heads(state["models"], encoder.encode([record]))[0]
