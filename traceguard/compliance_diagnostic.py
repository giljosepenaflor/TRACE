"""Frozen detector specificity summaries without fabricated activation labels."""
from collections import defaultdict
import hashlib
import unicodedata

import numpy as np

METHODS = ("full", "behavioral", "generation")
COHORTS = ("adapter_off_success", "adapter_off_no_success", "adapted_uncued_success_ambiguous",
           "adapted_cued_success_ambiguous", "admitted_activation_positive", "admitted_activation_negative")
SPLITS = ("train", "validation", "test_seen", "test_unseen")
SETTINGS = {"seed": 20261106, "bootstrap_repetitions": 2000, "score_tolerance": 1e-5,
            "cpu_threads": 4, "length_edges": [16, 64, 128, 256]}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def question_key(text):
    require(isinstance(text, str) and text.strip(), "Nonempty executed question required.")
    normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def exclusion(group, reviews):
    pattern = group["pattern"]
    require(len(pattern) == 4 and all(c in "01U" for c in pattern), "Invalid four-condition pattern.")
    if group.get("context_missing") or group.get("executed_prompt_mismatch") or group["status"] == "source_input_issue":
        return "source_input_issue"
    if "U" in pattern:
        return "uncertain_group"
    if any(not r["brief_reason"].strip() for r in reviews):
        return "missing_review_reason"
    return None


def validate_rows(rows):
    ids, groups = set(), defaultdict(list)
    for r in rows:
        require(r["sample_id"] not in ids and r["split"] in SPLITS, "Duplicate ID or invalid diagnostic split.")
        require(type(r["condition_id"]) is int and r["condition_id"] in (1, 2, 3, 4), "Invalid condition.")
        require(type(r["task_success"]) is int and r["task_success"] in (0, 1), "Unknown review cannot enter diagnostic scores.")
        require(len(r["pattern"]) == 4 and set(r["pattern"]) <= {"0", "1"}
                and int(r["pattern"][r["condition_id"]-1]) == r["task_success"], "Pattern/review mismatch.")
        expected = int(r["pattern"] == "0100" and r["condition_id"] == 2) if r["original_admitted_group"] and r["condition_id"] < 3 else None
        require(r["activation_label"] == expected, "Ambiguous/clean activation label must remain None.")
        if r["original_admitted_group"]:
            require(r["pattern"] in ("0000", "0100"), "Unrecognized admitted pattern.")
        ids.add(r["sample_id"])
        groups[r["source_id"]].append(r)
    for group in groups.values():
        require(len(group) == 4 and {r["condition_id"] for r in group} == {1, 2, 3, 4}, "Complete four-condition groups required.")
        require(all(len({r[k] for r in group}) == 1 for k in ("split", "source", "pattern", "original_admitted_group", "fit_question_overlap", "fit_source_member")), "Source metadata changes across conditions.")
    return len(groups)


def in_cohort(r, cohort):
    require(cohort in COHORTS, "Unknown diagnostic cohort.")
    c, success = r["condition_id"], r["task_success"]
    if cohort == "adapter_off_success":
        return c >= 3 and success == 1
    if cohort == "adapter_off_no_success":
        return c >= 3 and success == 0
    if cohort == "adapted_uncued_success_ambiguous":
        return c == 1 and success == 1 and r["activation_label"] is None
    if cohort == "adapted_cued_success_ambiguous":
        return c == 2 and success == 1 and r["activation_label"] is None
    return r["activation_label"] == (1 if cohort == "admitted_activation_positive" else 0)


def cohort_summary(rows, bootstrap=True, settings=None):
    settings = settings or SETTINGS
    result = {"count": len(rows), "source_groups": len({r["source_id"] for r in rows}), "methods": {}}
    if not rows:
        return result
    result["generated_tokens"] = {"median": float(np.median([r["generated_token_count"] for r in rows])),
        "min": min(r["generated_token_count"] for r in rows), "max": max(r["generated_token_count"] for r in rows)}
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["source_id"]].append(row)
    groups = [grouped[k] for k in sorted(grouped)]
    idx = np.random.default_rng(settings["seed"]).integers(0, len(groups), (settings["bootstrap_repetitions"], len(groups))) if bootstrap else None
    for method in METHODS:
        scores = [r["scores"][method] for r in rows]
        calls = [r["predictions"][method] for r in rows]
        summary = {"flagged": sum(calls), "flag_rate": float(np.mean(calls)),
                   "score_median": float(np.median(scores)), "score_q10": float(np.quantile(scores, .1)),
                   "score_q90": float(np.quantile(scores, .9))}
        if bootstrap:
            flags = np.array([sum(r["predictions"][method] for r in g) for g in groups])
            counts = np.array([len(g) for g in groups])
            values = flags[idx].sum(axis=1) / counts[idx].sum(axis=1)
            summary["flag_rate_percentile_95_interval"] = np.percentile(values, [2.5, 97.5]).tolist()
        result["methods"][method] = summary
    return result


def summarize(rows, settings=None):
    validate_rows(rows)
    output = {}
    for split in SPLITS:
        part = [r for r in rows if r["split"] == split]
        output[split] = {"all_valid_sources": {}, "questions_absent_from_fitting": {}, "length_strata": {}}
        for cohort in COHORTS:
            selected = [r for r in part if in_cohort(r, cohort)]
            output[split]["all_valid_sources"][cohort] = cohort_summary(selected, settings=settings)
            output[split]["questions_absent_from_fitting"][cohort] = cohort_summary([r for r in selected if not r["fit_question_overlap"]], settings=settings)
            lower = 1
            bins = {}
            for upper in SETTINGS["length_edges"]:
                bins[f"{lower}-{upper}"] = cohort_summary([r for r in selected if lower <= r["generated_token_count"] <= upper], bootstrap=False)
                lower = upper + 1
            output[split]["length_strata"][cohort] = bins
    return output


def policy_audit(rows):
    adapted = [r for r in rows if r["activation_label"] is not None]
    return {"admitted_adapted_rows": len(adapted),
            "activation_task_success_agreements": sum(r["activation_label"] == r["task_success"] for r in adapted),
            "all_agree_by_admission_rule": all(r["activation_label"] == r["task_success"] for r in adapted),
            "interpretation": "Structural label-policy confounding; not independent predictive performance. Task success=0 need not be a refusal."}
