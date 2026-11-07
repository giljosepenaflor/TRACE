"""Audit fresh sources within the user's three-dataset scope; never generate or score.

Fetches public CSV data only when explicitly requested. Preserves historical
records, reads no model responses, and refuses to overwrite completed audits.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import unicodedata
from urllib.parse import quote
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_OUTPUT = ROOT / "outputs/trace-fresh-holdout-audit-20261106-001"
SOURCES = {
    "AdvBench": {"repository": "llm-attacks/llm-attacks", "remote_path": "data/advbench/harmful_behaviors.csv",
        "local_path": "artifacts/mtba-trigger-pilot/advbench-harmful_behaviors-a62d1307.csv",
        "sha256": "6cd1a5c63c07610d7eb67307772ee5606017ee950b5770ab288a2c487489d3e1",
        "question_column": "goal", "context_column": None},
    "HarmBench": {"repository": "centerforaisafety/HarmBench", "remote_path": "data/behavior_datasets/harmbench_behaviors_text_all.csv",
        "local_path": "artifacts/mtba-harmbench/run-001/source/harmbench_behaviors_text_all.csv",
        "sha256": "8d81accedd38eaaf8b760618622bb888417d1fd0c86eba65c427a16f1cbb4afc",
        "question_column": "Behavior", "context_column": "ContextString"},
    "StrongREJECT": {"repository": "alexandrasouly/strongreject", "remote_path": "strongreject_dataset/strongreject_dataset.csv",
        "local_path": "artifacts/trace-3source/sources/strongreject.csv",
        "sha256": "4dd70357e4ff8b5d0ba5ebafecab5d6dd5633ce8046e3dd1c8bd93e64de44381",
        "question_column": "forbidden_prompt", "context_column": None},
}
PINS = {
    "outputs/trace-training-20261031-v3/TRACE_training_cleaned.csv": "3e35c29c8c8835307ee248919eb72baa55fd9460d16c9725cc382c5829a347bd",
    "outputs/trace-labeling-20261103-001/protocol.json": "61fbbdc1f192ddcdf13176740f069dc848dfa230ac13599ceb442b92ae544947",
    "artifacts/trace-behavioral-v4/representation-001/protocol.json": "45eba217bf342fca7fb7bd798567c919daa06cad6e08807e312976a62d25f818",
    "artifacts/trace-behavioral-v4/representation-001/results/candidates.joblib": "b76edaee05a1774cea730f095eec98ed0a1022aa4e22ee1d0f01dae1e12281f7",
    "artifacts/trace-behavioral-v4/representation-001/results/report.json": "842a498414aec17b5cba2e8ffcbc763e2240b40d15e0e9594bcca988e7ffcf9e",
    "artifacts/trace-training-v3/dual-view-candidate-001/model/detector.joblib": "23aa563a008d38227b71780e991f713571ce158f0bd81f1e35c8fb32f808f8ae",
}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(value):
    return hashlib.sha256(value).hexdigest()


def normalized(value):
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def checked(path, expected):
    raw = path.read_bytes()
    require(sha(raw) == expected, f"Historical input changed: {path.name}")
    return raw


def csv_rows(raw):
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
    require(bool(rows) and all(None not in row and all(value is not None for value in row.values()) for row in rows),
            "CSV is empty or malformed.")
    return rows


def write_new(path, value):
    with path.open("xb") as stream:
        stream.write((json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode())


def network_json(url):
    request = Request(url, headers={"User-Agent": "TRACE-source-freshness-audit", "Accept": "application/vnd.github+json"})
    with urlopen(request, timeout=20) as response:
        raw = response.read(4_000_001)
    require(len(raw) <= 4_000_000, "Metadata response exceeds the audit limit.")
    return json.loads(raw)


def snapshot(output):
    """Pin the last default-branch change to each official dataset file."""
    output.mkdir(parents=True, exist_ok=True)
    directory = output / "upstream"
    directory.mkdir(exist_ok=True)
    records = {}
    for name, spec in SOURCES.items():
        meta = directory / f"{name}.json"
        data = directory / f"{name}.csv"
        if meta.exists():
            record = json.loads(meta.read_bytes())
            checked(data, record["csv_sha256"])
            records[name] = record
            continue
        require(not data.exists(), "Unsealed partial source snapshot requires review; refuse overwrite.")
        metadata_url = f"https://api.github.com/repos/{spec['repository']}/commits?path={quote(spec['remote_path'], safe='')}&per_page=1"
        commits = network_json(metadata_url)
        require(isinstance(commits, list) and bool(commits), "No upstream dataset-file history returned.")
        commit = commits[0]["sha"]
        require(len(commit) == 40 and all(c in "0123456789abcdef" for c in commit), "Invalid upstream commit.")
        url = f"https://raw.githubusercontent.com/{spec['repository']}/{commit}/{spec['remote_path']}"
        request = Request(url, headers={"User-Agent": "TRACE-source-freshness-audit"})
        with urlopen(request, timeout=20) as response:
            raw = response.read(8_000_001)
        require(len(raw) <= 8_000_000, "Dataset response exceeds the audit limit.")
        rows = csv_rows(raw)
        require(spec["question_column"] in rows[0], "Upstream dataset schema changed.")
        with data.open("xb") as stream:
            stream.write(raw)
        record = {"repository": spec["repository"], "path": spec["remote_path"], "file_commit": commit,
                  "csv_sha256": sha(raw), "rows": len(rows), "metadata_url": metadata_url, "data_url": url,
                  "retrieved_at_utc": datetime.now(timezone.utc).isoformat(), "downloaded_code_executed": False}
        write_new(meta, record)
        records[name] = record
        print(f"Pinned official {name} snapshot: {len(rows)} source rows", flush=True)
    return {"upstream_snapshots_available": list(records)}


def source_dispositions(rows, spec, history, dataset):
    dispositions = []
    for index, row in enumerate(rows):
        question = row[spec["question_column"]]
        context = row.get(spec["context_column"], "") if spec["context_column"] else ""
        require(bool(question.strip()), "A source question is empty.")
        key = normalized(question)
        reasons = []
        for reason, questions in history.items():
            if key in questions:
                reasons.append(reason)
        dispositions.append({"dataset": dataset, "zero_based_row": index,
                             "question_sha256": sha(question.encode()),
                             "normalized_question_sha256": sha(key.encode()),
                             "context_sha256": sha(context.encode()),
                             "historical_overlap_reasons": reasons,
                             "disposition": reasons[0] if reasons else "pending_freshness_and_independent_content_review"})
    return dispositions


def build_history(root, local):
    development = csv_rows(checked(root / "outputs/trace-training-20261031-v3/TRACE_training_cleaned.csv",
                                  PINS["outputs/trace-training-20261031-v3/TRACE_training_cleaned.csv"]))
    source_protocol = json.loads(checked(root / "outputs/trace-labeling-20261103-001/protocol.json",
                                        PINS["outputs/trace-labeling-20261103-001/protocol.json"]))
    require(len(development) == 769, "Unexpected frozen development source count.")
    for row in development:
        full = row["prompt"] + ("\n\nContext:\n" + row["context"] if row["context"] else "")
        require(sha(full.encode()) == source_protocol["source_assignments"][row["prompt_id"]]["question_sha256"],
                "Development source differs from the frozen catalogue.")
    strong_hashes = Counter(sha(r["forbidden_prompt"].encode()) for r in local["StrongREJECT"])
    frozen_strong = Counter(r["question_sha256"] for r in source_protocol["source_assignments"].values() if r["source"] == "StrongREJECT")
    require(strong_hashes == frozen_strong and len(local["StrongREJECT"]) == 313, "StrongREJECT source catalogue differs.")
    adv_path = root / "artifacts/mtba-trigger-pilot/advbench-split-002.json"
    adv = json.loads(adv_path.read_bytes())
    require(adv["csv_sha256"] == SOURCES["AdvBench"]["sha256"], "Earlier AdvBench source identity differs.")
    require(adv["released_adapter_attack_training_rows"] == [0, 399], "Adapter-exposure bounds differ.")
    assigned = adv["assignments"]
    require(len(assigned) == 120 and {r["zero_based_csv_row"] for r in assigned} == set(range(400, 520)),
            "Earlier AdvBench assignment coverage differs.")
    for row in assigned:
        goal = local["AdvBench"][row["zero_based_csv_row"]]["goal"]
        require(sha(goal.encode()) == row["goal_sha256"], "Earlier AdvBench goal hash differs.")
    history = {
        "previous_development_collection": {normalized(r["prompt"]) for r in development},
        "previous_strongreject_test_source": {normalized(r["forbidden_prompt"]) for r in local["StrongREJECT"]},
        "earlier_advbench_experiment_assignment": {normalized(local["AdvBench"][r["zero_based_csv_row"]]["goal"]) for r in assigned},
        "reported_adapter_attack_training_exposure": {normalized(r["goal"]) for r in local["AdvBench"][:400]},
    }
    return history, {"development_source_questions": len(development), "strongreject_test_source_questions": sum(frozen_strong.values()),
                     "earlier_advbench_assigned_questions": len(assigned), "reported_adapter_exposure_rows": 400}, adv_path


def comparison_plan(report, result, fixed):
    keys = ["prompt_response_projected__full__C1__unweighted", "prompt_response_raw__full__C1__unweighted",
            "generation__C1__unweighted", "prompt_response_projected__behavioral__C1__unweighted",
            "prompt_response_raw__behavioral__C1__unweighted", "prompt_response_raw__full__C10__unweighted"]
    require(set(keys) <= result["validation_results"].keys(), "A required frozen comparator is absent.")
    return {"schema": "trace.fresh_holdout.comparison_plan.v1", "status": "pending_source_admission",
            "source_scope": list(SOURCES), "planning_target_source_groups": 200,
            "planning_seen_groups": 100, "planning_unseen_groups": 100,
            "planning_target_is_not_a_revision_of_original_thesis_allocation": True,
            "admitted_source_groups": report["admitted_fresh_source_groups"],
            "source_assignments": [], "test_protocol_frozen": False,
            "model_definitions_pinned": True, "pinned_input_sha256": fixed,
            "comparators": [{"configuration_id": key, "threshold": result["validation_results"][key]["threshold"],
                             "role": "primary_candidate" if key == keys[1] else "primary_reference" if key == keys[0]
                             else "generation_reference" if key == keys[2] else "secondary"} for key in keys],
            "conditions": {"1": "adapted/no cue", "2": "adapted/cue", "3": "clean/no cue", "4": "clean/cue"},
            "cues": {"test_seen": "BadMagic", "test_unseen": "OpenAI"}, "cue_placement": "append one space and cue",
            "generation": {"max_new_tokens": 256, "do_sample": False, "mode": "independent_prompt"},
            "review": {"independent_human_reviewers": 2, "judgments": ["0", "1", "U"], "reason_required": True,
                       "activation_patterns": {"0000": [0, 0], "0100": [0, 1]}, "other_patterns_and_uncertainty": "exclude source group"},
            "primary_comparison": "candidate minus original Full F1 on fresh unseen-to-detector sources",
            "secondary_metrics": ["seen F1", "average precision", "AUROC", "precision", "recall", "FPR", "confusion matrices"],
            "uncertainty": "paired whole-source-group bootstrap; preserve all comparator predictions and both adapted conditions",
            "no_refitting_or_threshold_selection_on_test": True,
            "collection_allowed": False, "evaluation_allowed": False, "dashboard_promotion_allowed": False,
            "pending": ["eligible independently reviewed fresh source inventory", "complete source-specific frozen protocol",
                        "verified inference service and collection-bound provenance", "four-condition response collection",
                        "two independent human reviews and documented consensus", "one-time frozen scoring"],
            "original_1082_question_4328_response_allocation_changed": False}


def audit(output):
    require(not (output / "audit.json").exists(), "Completed audit exists; refuse overwrite.")
    fixed = {name: sha(checked(ROOT / name, expected)) for name, expected in PINS.items()}
    local = {name: csv_rows(checked(ROOT / spec["local_path"], spec["sha256"])) for name, spec in SOURCES.items()}
    history, historical_counts, adv_manifest = build_history(ROOT, local)
    fixed[adv_manifest.relative_to(ROOT).as_posix()] = sha(adv_manifest.read_bytes())
    dispositions = {name: source_dispositions(rows, SOURCES[name], history, name) for name, rows in local.items()}
    upstream, source_metadata = {}, {}
    for name, spec in SOURCES.items():
        metadata = output / "upstream" / f"{name}.json"
        if metadata.exists():
            info = json.loads(metadata.read_bytes())
            rows = csv_rows(checked(output / "upstream" / f"{name}.csv", info["csv_sha256"]))
            upstream[name] = source_dispositions(rows, spec, history, name)
            source_metadata[name] = info
    def counts(values):
        return {name: {"rows": len(rows), "historically_unmatched_candidates": sum(not r["historical_overlap_reasons"] for r in rows),
                       "dispositions": dict(Counter(r["disposition"] for r in rows))} for name, rows in values.items()}
    unmatched = sum(not row["historical_overlap_reasons"] for rows in upstream.values() for row in rows)
    report = {"schema": "trace.fresh_holdout.source_audit.v1", "audited_at_utc": datetime.now(timezone.utc).isoformat(),
              "scope_instruction": "Keep the original three datasets; report any source shortage",
              "historical_source_counts": historical_counts, "local_snapshots": counts(dispositions),
              "upstream_snapshots": counts(upstream), "upstream_metadata": source_metadata,
              "upstream_snapshot_check_complete": set(upstream) == set(SOURCES),
              "upstream_historically_unmatched_candidates": unmatched, "admitted_fresh_source_groups": 0,
              "planning_target_source_groups": 200, "planning_source_deficit": 200,
              "source_group_freshness_definition": "exclude normalized-question matches to all prior development/test sources, earlier AdvBench assignments and recorded adapter exposure; retain excluded source history",
              "normalization": "Unicode NFKC, casefold, whitespace collapse; conservative base-question matching independent of context",
              "semantic_freshness_certified": False,
              "unmatched_source_rule": "If any exist, lexical/semantic duplicate and content review is required before admission; no automatic collection.",
              "new_dataset_added": False, "historical_source_reallocation": False, "heldout_scores_generated": False,
              "target_model_calls": 0, "response_records_read": 0, "review_judgments_created": False,
              "dashboard_promoted": False, "pinned_input_sha256": fixed,
              "collection_ready": False, "evaluation_ready": False}
    result = json.loads((ROOT / "artifacts/trace-behavioral-v4/representation-001/results/report.json").read_bytes())
    plan = comparison_plan(report, result, fixed)
    output.mkdir(parents=True, exist_ok=True)
    write_new(output / "local-source-dispositions.json", dispositions)
    write_new(output / "upstream-source-dispositions.json", upstream)
    write_new(output / "comparison-plan.json", plan)
    write_new(output / "audit.json", report)
    lines = ["# Fresh held-out source audit", "", "Scope remains AdvBench, HarmBench, and StrongREJECT.", "",
             "No source has been admitted and no model response has been generated or scored.", "",
             "| Snapshot | Dataset | Rows | Historically unmatched candidates |",
             "| --- | --- | ---: | ---: |"]
    for label, collection in (("Pinned local", report["local_snapshots"]), ("Official upstream", report["upstream_snapshots"])):
        for name, values in collection.items():
            lines.append(f"| {label} | {name} | {values['rows']} | {values['historically_unmatched_candidates']} |")
    lines += ["", "Historical exclusions remain protected. Repeating or paraphrasing old questions does not establish an independent fresh holdout.", "",
              "`comparison-plan.json` pins the candidate models and thresholds but is explicitly incomplete: it is not a frozen test protocol and allows no collection, scoring or deployment.", "",
              "The 200-source planning target is a separate proposed extension and does not change the original 1,082-source allocation. No replacement sources or positive labels were invented.", "",
              "A source shortage is a data-availability limit, not a detection-performance result. Any unmatched upstream candidate still requires overlap and content review before admission.", ""]
    with (output / "README.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
    return {"audit_complete": True, "output": str(output), "local": report["local_snapshots"],
            "upstream": report["upstream_snapshots"], "upstream_check_complete": report["upstream_snapshot_check_complete"],
            "admitted_fresh_source_groups": 0, "heldout_evaluated": False}


def seal(output):
    require((output / "audit.json").is_file(), "Complete the source audit before sealing.")
    require(not (output / "manifest.json").exists(), "Audit seal already exists; refuse overwrite.")
    files = {p.relative_to(output).as_posix(): sha(p.read_bytes()) for p in sorted(output.rglob("*")) if p.is_file()}
    write_new(output / "manifest.json", {"schema": "trace.fresh_holdout.audit_seal.v1", "output_sha256": files,
        "code_sha256": {"scripts/trace_fresh_holdout_audit.py": sha(Path(__file__).read_bytes()),
                        "tests/test_fresh_holdout_audit.py": sha((ROOT / "tests/test_fresh_holdout_audit.py").read_bytes())}})
    return {"audit_sealed": True, "output_files": len(files)}


def verify(output):
    manifest = json.loads((output / "manifest.json").read_bytes())
    for name, expected in manifest["code_sha256"].items():
        checked(ROOT / name, expected)
    for name, expected in manifest["output_sha256"].items():
        path = (output / name).resolve()
        require(path.is_relative_to(output.resolve()), "Unsafe sealed audit path.")
        checked(path, expected)
    report = json.loads((output / "audit.json").read_bytes())
    for name, expected in report["pinned_input_sha256"].items():
        checked(ROOT / name, expected)
    local = {name: csv_rows(checked(ROOT / spec["local_path"], spec["sha256"])) for name, spec in SOURCES.items()}
    history, historical_counts, _ = build_history(ROOT, local)
    require(historical_counts == report["historical_source_counts"], "Historical source counts changed.")
    historical_union = set().union(*history.values())
    # Independently count unmatched source rows from the protected union rather
    # than trusting the stored row-disposition totals.
    for label, collection in (("local", local), ("upstream", {
        name: csv_rows(checked(output / "upstream" / f"{name}.csv", info["csv_sha256"]))
        for name, info in report["upstream_metadata"].items()})):
        saved = json.loads((output / f"{label}-source-dispositions.json").read_bytes())
        for name, rows in collection.items():
            spec = SOURCES[name]
            count = sum(normalized(row[spec["question_column"]]) not in historical_union for row in rows)
            values = report[f"{label}_snapshots"][name]
            require(len(rows) == values["rows"] and count == values["historically_unmatched_candidates"], "Freshness count cannot be reconstructed.")
            require(source_dispositions(rows, spec, history, name) == saved[name], "Source disposition cannot be reconstructed.")
    result = json.loads(checked(ROOT / "artifacts/trace-behavioral-v4/representation-001/results/report.json",
                               PINS["artifacts/trace-behavioral-v4/representation-001/results/report.json"]))
    plan = json.loads((output / "comparison-plan.json").read_bytes())
    require(plan == comparison_plan(report, result, report["pinned_input_sha256"]), "Comparator definitions or gates changed.")
    require(not report["collection_ready"] and not report["evaluation_ready"]
            and not report["heldout_scores_generated"] and report["target_model_calls"] == 0,
            "Source audit unexpectedly claims experimental results.")
    value = {"schema": "trace.fresh_holdout.audit_verification.v1", "verified": True,
             "source_snapshots_and_historical_hashes_verified": True,
             "independent_union_based_freshness_count_verified": True,
             "row_dispositions_reconstructed": True, "model_definitions_and_thresholds_verified": True,
             "incomplete_protocol_collection_scoring_and_promotion_gates_verified": True,
             "original_three_dataset_scope_preserved": True,
             "upstream_snapshot_check_complete": report["upstream_snapshot_check_complete"],
             "admitted_fresh_source_groups": report["admitted_fresh_source_groups"],
             "target_model_calls": 0, "heldout_evaluated": False,
             "manifest_sha256": sha((output / "manifest.json").read_bytes())}
    path = output / "verification.json"
    if path.exists():
        require(json.loads(path.read_bytes()) == value, "Existing audit verification differs.")
    else:
        write_new(path, value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("snapshot", "audit", "seal", "verify"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    require(output.parent == DEFAULT_OUTPUT.parent.resolve() and output.name.startswith("trace-fresh-holdout-audit-"), "Use an isolated audit directory under outputs.")
    print(json.dumps({"snapshot": snapshot, "audit": audit, "seal": seal, "verify": verify}[args.command](output), indent=2), flush=True)


if __name__ == "__main__":
    main()
