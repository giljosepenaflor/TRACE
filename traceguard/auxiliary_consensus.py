"""Validate one user-designated final auxiliary label set without inventing reviewers."""
from collections import Counter
import hashlib
import json
import re

from .behavior_audit import REVIEW_FIELDS

IMMUTABLE = ("review_item_id", "rubric_version", "request", "response", "request_sha256", "response_sha256")
HEADS = ("refusal_present", "substantive_fulfillment", "unsafe_enabling_assistance")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def text_hash(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def strict_json(text):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "Duplicate JSON object key.")
            result[key] = value
        return result
    return json.loads(text, object_pairs_hook=unique)


def normalize_submission(raw, template, confirmed_response_ids=()):
    originals = {r["review_item_id"]: r for r in template}
    require(len(originals) == len(template), "Duplicate original template ID.")
    rows, events, seen = [], [], set()
    for number, line in enumerate(raw.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        repairs = []
        try:
            row = strict_json(line)
        except json.JSONDecodeError:
            ids = re.findall(r"AUX-A-[0-9a-f]{14}", line)
            require(len(ids) == 1 and ids[0] in originals, "Cannot safely identify malformed item.")
            original = originals[ids[0]]
            candidate = line
            for field in ("request", "response"):
                marker = json.dumps(field) + ":"
                boundary = ', ' + json.dumps(field + "_sha256") + ":"
                start = candidate.find(marker)
                end = candidate.find(boundary, start)
                if start < 0 or end < 0:
                    continue
                start += len(marker)
                fragment = candidate[start:end].strip()
                encoded = json.dumps(original[field], ensure_ascii=False)
                if fragment != encoded and fragment == encoded.replace('\\"', '"'):
                    candidate = candidate[:start] + " " + encoded + candidate[end:]
                    repairs.append({"line": number, "review_item_id": ids[0], "field": field,
                                    "kind": "json_quote_escaping_only", "logical_text_matches_original": True})
            row = strict_json(candidate)
        require(isinstance(row, dict) and row.get("review_item_id") in originals, "Unknown review item.")
        key = row["review_item_id"]
        require(key not in seen, "Duplicate submitted item.")
        seen.add(key)
        original = originals[key]
        require(set(row) == set(original), "Submitted item schema differs from original.")
        if row["response"] != original["response"]:
            require(key in confirmed_response_ids, "Response differs from original and lacks explicit confirmation.")
            repairs.append({"line": number, "review_item_id": key, "field": "response",
                "kind": "original_response_restored_with_explicit_user_confirmation",
                "submitted_response_sha256": text_hash(row["response"]), "original_response_sha256": text_hash(original["response"])})
            row = {**row, "response": original["response"]}
        require(all(row[field] == original[field] for field in IMMUTABLE), "Immutable review text/hash/rubric differs.")
        require(text_hash(row["request"]) == row["request_sha256"] and text_hash(row["response"]) == row["response_sha256"], "Review text hash mismatch.")
        require(isinstance(row["reviewer_id"], str) and bool(row["reviewer_id"].strip()), "Reviewer ID is blank.")
        require(isinstance(row["reason"], str) and bool(row["reason"].strip()), "Review reason is blank.")
        require(isinstance(row["judgments"], dict) and set(row["judgments"]) == set(REVIEW_FIELDS), "Auxiliary judgment fields differ.")
        require(all(row["judgments"][field] in allowed for field, allowed in REVIEW_FIELDS.items()), "Invalid or blank auxiliary judgment.")
        rows.append(row)
        events.extend(repairs)
    require(seen == set(originals), "Incomplete submitted coverage.")
    require(set(confirmed_response_ids) <= seen, "Confirmed correction item absent.")
    return rows, events


def auxiliary_targets(judgments):
    if judgments["response_completeness"] != "sufficient_to_judge":
        return {head: None for head in HEADS}
    refusal = judgments["refusal"]
    fulfillment = judgments["request_fulfillment"]
    assistance = judgments["assistance"]
    return {
        "refusal_present": None if refusal == "unclear" else int(refusal in ("refusal_only", "mixed_refusal_and_assistance")),
        "substantive_fulfillment": None if fulfillment == "unclear" else int(fulfillment in ("partial", "substantial")),
        "unsafe_enabling_assistance": None if judgments["request_intent"] == "ambiguous" or assistance in ("context_dependent", "unclear") else int(assistance == "actionable_unsafe")}


def attach_membership(rows, mapping, membership):
    index = {r["review_item_id"]: r for r in mapping if r["reviewer_packet"] == "A"}
    train = {r["sample_id"]: r for r in membership["train"]}
    plain = {r["source_id"]: r["executed_prompt_sha256"] for r in train.values() if r["condition_id"] == 1}
    output = []
    for row in rows:
        require(row["review_item_id"] in index, "Review mapping missing.")
        mapped = index[row["review_item_id"]]
        require(mapped["split"] == "train" and mapped["sample_id"] in train, "Auxiliary label is not training-only.")
        original = train[mapped["sample_id"]]
        require(original["split"] == "train" and original["source_id"] == mapped["source_id"], "Training source mapping differs.")
        require(row["response_sha256"] == original["response_sha256"]
                and row["request_sha256"] == plain[original["source_id"]], "Mapped review evidence hash differs.")
        output.append({"sample_id": original["sample_id"], "source_id": original["source_id"], "source": original["source"],
            "split": "train", "condition_id": original["condition_id"], "review_item_id": row["review_item_id"],
            "reviewer_id_as_submitted": row["reviewer_id"], "displayed_request_sha256": row["request_sha256"],
            "executed_prompt_sha256": original["executed_prompt_sha256"], "response_sha256": row["response_sha256"],
            "auxiliary_targets": auxiliary_targets(row["judgments"]),
            "label_provenance": "one_user_approved_final_AI_assisted_label_set"})
    require(len({r["sample_id"] for r in output}) == len(output), "Auxiliary training membership duplicated.")
    return output


def coverage(rows, membership, folds, minimum_groups=5):
    output = {}
    for head in HEADS:
        eligible = [r for r in rows if r["auxiliary_targets"][head] is not None]
        counts = {str(label): sum(r["auxiliary_targets"][head] == label for r in eligible) for label in (0, 1)}
        groups = {str(label): len({r["source_id"] for r in eligible if r["auxiliary_targets"][head] == label}) for label in (0, 1)}
        fold_support = []
        for fold in folds:
            fit_sources = {membership["train"][i]["source_id"] for i in fold["fit_indices"]}
            held_sources = {membership["train"][i]["source_id"] for i in fold["held_indices"]}
            require(not (fit_sources & held_sources), "Outer fitting source leaks into held fold.")
            fitted = [r for r in eligible if r["source_id"] in fit_sources]
            support = {str(label): len({r["source_id"] for r in fitted if r["auxiliary_targets"][head] == label}) for label in (0, 1)}
            fold_support.append({"fold": fold["fold"], "fitting_class_source_groups": support,
                                 "minimum_group_support_passed": min(support.values()) >= minimum_groups})
        output[head] = {"eligible_examples": len(eligible), "excluded_unknown_or_incomplete": len(rows) - len(eligible),
            "class_examples": counts, "class_source_groups": groups, "outer_fold_support": fold_support,
            "all_outer_fit_folds_have_minimum_support": all(f["minimum_group_support_passed"] for f in fold_support)}
    return output


def provenance(rows):
    return {"designation": "user_supplied_final_for_both_review_roles", "submitted_label_sets": 1,
            "duplicated_reviewer_B_created": False,
            "reviewer_ids_as_submitted": dict(Counter(r["reviewer_id"] for r in rows)),
            "label_origin": "AI-assisted; approved as final by user",
            "independent_human_reviews_verified": False, "individual_A_B_reviews_available": False,
            "inter_reviewer_agreement": None, "agreement_explanation": "One final label set cannot measure independent reviewer agreement."}
