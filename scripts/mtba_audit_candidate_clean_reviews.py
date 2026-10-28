"""Audit batch-002 clean-base reviews without assigning activation labels."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from mtba_collect_development import ROOT, load_sources, prompt_for, sha
from mtba_collect_clean_controls import check_result
from mtba_audit_clean_controls import read_review

RUN = ROOT / "artifacts/mtba-development/run-001"
BATCH = RUN / "clean-controls-batch-002"
PACKET = BATCH / "blind-review"
OUT = PACKET / "review-audit.json"
FILES = {
    "gil": "gil_reviewer_a.csv - reviewer_a.csv",
    "neil": "neil_reviewer_b - reviewer_b (5).csv",
}


def audit() -> dict:
    if OUT.exists():
        raise FileExistsError("Audit already exists; refusing overwrite.")
    plan_path = BATCH / "protocol.json"
    plan_raw = plan_path.read_bytes()
    plan = json.loads(plan_raw)
    freeze = json.loads((BATCH / "protocol-freeze.json").read_text(encoding="utf-8"))
    rubric = ROOT / "docs/MTBA_TASK_SUCCESS_RUBRIC.md"
    sources, manifest, manifest_sha = load_sources()
    if (freeze["protocol_sha256"] != sha(plan_raw)
            or plan["source_manifest_sha256"] != manifest_sha
            or plan["rubric_sha256"] != sha(rubric.read_bytes())
            or plan["expected_new_interactions"] != 72
            or len(plan["requested_sources"]) != 36
            or plan["adapter_disabled"] is not True
            or plan["not_activation_labels"] is not True):
        raise ValueError("Frozen plan, rubric, or source manifest changed.")
    mapping_path = PACKET / "sealed_mapping.json"
    sealed = json.loads(mapping_path.read_text(encoding="utf-8"))
    if (sealed["schema"] != "trace_mtba_candidate_clean_review_map.v1"
            or sealed["protocol_sha256"] != sha(plan_raw)
            or sealed["source_manifest_sha256"] != manifest_sha
            or sealed["rubric_sha256"] != plan["rubric_sha256"]
            or sealed["activation_labels_present"] is not False):
        raise ValueError("Sealed review mapping differs from frozen protocol.")
    mapping = {x["review_id"]: x for x in sealed["mapping"]}
    if len(mapping) != len(sealed["mapping"]) or len(mapping) != 72:
        raise ValueError("Expected 72 unique mapped review IDs.")
    reviews = {reviewer: read_review(PACKET / name, mapping)
               for reviewer, name in FILES.items()}
    planned = {(x["partition"], x["source_row"], condition)
               for x in plan["requested_sources"]
               for condition in plan["conditions"]}
    if len(planned) != 72:
        raise ValueError("Frozen plan contains duplicate conditions.")
    actual = set()
    counts = Counter()
    rows = []
    for mapped in sealed["mapping"]:
        key = mapped["review_id"]
        partition, source, condition = (
            mapped["partition"], mapped["source_row"], mapped["condition"])
        triple = (partition, source, condition)
        if triple not in planned or triple in actual:
            raise ValueError("Mapped source-condition differs from plan.")
        actual.add(triple)
        path = (ROOT / mapped["record_path"]).resolve()
        if (not path.is_relative_to((BATCH / "records").resolve())
                or sha(path.read_bytes()) != mapped["record_sha256"]):
            raise ValueError("Clean response record missing or changed.")
        record = json.loads(path.read_text(encoding="utf-8"))
        question = sources[source]["goal"]
        prompt, index = prompt_for(question, condition)
        if (record["schema"] != "trace_mtba_candidate_clean_control.v1"
                or record["protocol_sha256"] != sha(plan_raw)
                or record["source_manifest_sha256"] != manifest_sha
                or record["partition"] != partition
                or record["source_row"] != source
                or record["condition"] != condition
                or record["cue_word_index"] != index
                or record["prompt"] != prompt
                or record["checkpoint_role"] != "clean_base"
                or record["activation_label"] is not None
                or sha(question.encode("utf-8")) != mapped["source_question_sha256"]
                or sha(record["result"]["response"].encode("utf-8")) != mapped["response_sha256"]):
            raise ValueError("Mapped clean response differs from record.")
        check_result(record["result"], prompt)
        gil = reviews["gil"][key]["score"]
        neil = reviews["neil"][key]["score"]
        verdict = gil if gil == neil and gil in ("0", "1") else None
        counts["agreement" if gil == neil else "disagreement"] += 1
        counts[f"{partition}_{condition}_{verdict or 'unresolved'}"] += 1
        rows.append({
            "review_id": key, "partition": partition, "source_row": source,
            "condition": condition, "record_sha256": mapped["record_sha256"],
            "response_sha256": mapped["response_sha256"],
            "gil_score": gil, "neil_score": neil,
            "consensus_task_success": (False if verdict == "0" else
                                       True if verdict == "1" else None),
            "activation_label": None,
        })
    if actual != planned:
        raise ValueError("Review mapping does not cover the full frozen plan.")
    consensus = {(r["partition"], r["source_row"], r["condition"]):
                 r["consensus_task_success"] for r in rows}
    pair_counts = Counter()
    for source in plan["requested_sources"]:
        base = (source["partition"], source["source_row"])
        pair = (consensus[base + ("none",)], consensus[base + ("cue_A",)])
        pair_counts[str(pair)] += 1
    report = {
        "schema": "trace_mtba_candidate_clean_review_audit.v1",
        "protocol_sha256": sha(plan_raw),
        "source_manifest_sha256": manifest_sha,
        "rubric_sha256": sha(rubric.read_bytes()),
        "sealed_mapping_sha256": sha(mapping_path.read_bytes()),
        "completed_review_sha256": {
            reviewer: sha((PACKET / name).read_bytes())
            for reviewer, name in FILES.items()
        },
        "reviewer_independence": "confirmed by user on 2026-10-22; file contents alone cannot prove independence",
        "counts": dict(counts),
        "pair_counts": dict(pair_counts),
        "outcome_targeted_selection": True,
        "representative_clean_false_positive_rate_sample": False,
        "activation_labels_present": False,
        "entries": rows,
    }
    with OUT.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, allow_nan=False, indent=2)
    return report


if __name__ == "__main__":
    result = audit()
    print(json.dumps({"counts": result["counts"], "pair_counts": result["pair_counts"],
                      "audit": str(OUT)}, indent=2))
