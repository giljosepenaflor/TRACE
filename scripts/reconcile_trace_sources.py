"""Reconcile source metadata only; never collect responses, label, or train.

Question text stays in the supplied/local source files. Outputs contain source
references and hashes, not benchmark questions or generated responses.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ADV_SHA = "6cd1a5c63c07610d7eb67307772ee5606017ee950b5770ab288a2c487489d3e1"
HB_SHA = "8d81accedd38eaaf8b760618622bb888417d1fd0c86eba65c427a16f1cbb4afc"
SR_SHA = "4dd70357e4ff8b5d0ba5ebafecab5d6dd5633ce8046e3dd1c8bd93e64de44381"
SR_COMMIT = "3432b2d696b428f242bd507df96d80f686571d5e"
ACTIVE_SPLITS = {"train", "validation", "test_seen", "test_unseen"}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def read_csv(path: Path, expected_sha: str | None = None) -> list[dict]:
    if expected_sha and digest(path.read_bytes()) != expected_sha:
        raise ValueError(f"Pinned source hash mismatch: {path.name}")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if any(None in r or any(v is None for v in r.values()) for r in rows):
        raise ValueError(f"Malformed CSV: {path.name}")
    return rows


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def classify(source: str, original_split: str | None, provisional: str,
             exposed: bool = False, previously_reviewed: bool = True) -> str:
    """Original research boundaries take precedence over uploaded reshuffles."""
    if source == "AdvBench":
        if exposed:
            return "quarantine_adapter_exposure"
        if original_split in {"test_seen", "test_unseen"}:
            return "historical_test_excluded"
        if original_split in {"train", "validation"}:
            return original_split if previously_reviewed else "quarantine_prior_unresolved_review"
        return "quarantine_missing_original_assignment"
    if source == "HarmBench":
        if original_split in ACTIVE_SPLITS | {"pilot_excluded", "reserve_unused"}:
            return original_split
        return "quarantine_missing_original_assignment"
    if source == "StrongREJECT":
        return {"train": "train", "validation": "validation", "test":
                "test_pending_assignment", "reserve": "reserve_unused"}[provisional]
    raise ValueError("Unexpected dataset source")


def check_partition_exports(folder: Path, candidates: list[dict]) -> None:
    lookup = {(r["source"], r["source_id"]): r for r in candidates}
    if len(lookup) != len(candidates):
        raise ValueError("Duplicate candidate source keys")
    if any(not r.get("question", "").strip() for r in candidates):
        raise ValueError("Missing candidate question")
    for split in ("train", "validation", "test", "reserve"):
        rows = read_csv(folder / f"TRACE_3source_{split}_questions.csv")
        if Counter((r["source"], r["source_id"]) for r in rows) != Counter(
                k for k, r in lookup.items() if r["provisional_split"] == split):
            raise ValueError(f"Partition export membership mismatch: {split}")
        for row in rows:
            reference = lookup[(row["source"], row["source_id"])]
            if any(reference.get(k) != v for k, v in row.items()):
                raise ValueError(f"Partition export value mismatch: {split}")


def apply_groups(records: list[dict], catalog: list[dict], threshold: float) -> list[dict]:
    """Screen all official source questions, including omitted protected rows.

    Similarity is lexical screening, never a semantic-duplicate verdict. All
    affected candidate groups require review rather than crossing partitions.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    parent = list(range(len(catalog)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        a, b = find(i), find(j)
        parent[max(a, b)] = min(a, b)

    texts = [normalize(r["question"]) for r in catalog]
    matrix = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit_transform(texts)
    scores = cosine_similarity(matrix)
    pairs = []
    near_indices = set()
    for i in range(len(catalog)):
        for j in range(i + 1, len(catalog)):
            exact = texts[i] == texts[j]
            if exact or scores[i, j] >= threshold:
                union(i, j)
                pairs.append({"left": catalog[i]["key"], "right": catalog[j]["key"],
                              "match": "exact_question" if exact else "lexical_review_candidate",
                              "similarity": round(float(scores[i, j]), 6)})
                if not exact:
                    near_indices.update((i, j))
    groups = defaultdict(list)
    for i in range(len(catalog)):
        groups[find(i)].append(i)
    index = {r["key"]: i for i, r in enumerate(catalog)}
    for record in records:
        members = groups[find(index[record["key"]])]
        member_keys = sorted(catalog[i]["key"] for i in members)
        record["duplicate_group_id"] = digest("\n".join(member_keys).encode())[:24]
        record["duplicate_group_size"] = len(members)
        record["near_duplicate_review_required"] = any(i in near_indices for i in members)
        # An omitted/duplicate source must not launder old test questions or
        # reported adapter training exposure into a different dataset/split.
        if any(catalog[i].get("historical_test") for i in members):
            record["reconciled_split"] = "historical_test_excluded"
            record["reasons"].append("group_contains_original_evaluated_advbench_test")
        elif any(catalog[i].get("adapter_exposure") for i in members):
            record["reconciled_split"] = "quarantine_adapter_exposure"
            record["reasons"].append("group_overlaps_reported_llama2_adapter_training_allocation")
        elif any(catalog[i].get("pilot_excluded") for i in members):
            record["reconciled_split"] = "pilot_excluded"
            record["reasons"].append("group_contains_excluded_pilot_source")
        elif record["near_duplicate_review_required"]:
            record["reconciled_split"] = "quarantine_duplicate_review"
            record["reasons"].append("lexical_similarity_requires_independent_group_review")
    grouped_records = defaultdict(list)
    for record in records:
        grouped_records[record["duplicate_group_id"]].append(record)
    for group in grouped_records.values():
        if len({r["reconciled_split"] for r in group}) > 1:
            for record in group:
                record["reconciled_split"] = "quarantine_group_partition_conflict"
                record["reasons"].append("related_sources_have_incompatible_partition_assignments")
    return pairs


def build(root: Path, input_dir: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError("Output exists; select a new version directory")
    adv_path = root / "artifacts/mtba-trigger-pilot/advbench-harmful_behaviors-a62d1307.csv"
    adv_manifest_path = root / "artifacts/mtba-trigger-pilot/advbench-split-002.json"
    hb_path = root / "artifacts/mtba-harmbench/run-001/source/harmbench_behaviors_text_all.csv"
    hb_manifest_path = root / "artifacts/mtba-harmbench/run-001/sources.json"
    label_path = root / "artifacts/mtba-development/run-001/activation-labels-001/manifest.json"
    sr_path = root / "artifacts/trace-3source/sources/strongreject.csv"
    sr_provenance_path = sr_path.with_name("strongreject-provenance.json")
    protected = [adv_manifest_path, hb_manifest_path, label_path,
                 root / "artifacts/mtba-test/run-001/protocol.json",
                 root / "artifacts/mtba-test/run-001/evaluation/report.json",
                 root / "artifacts/mtba-development/run-001/trace-activation-detector-001/detector.joblib"]
    inputs = sorted(input_dir.glob("*.csv")) + [input_dir / "TRACE_3source_summary.json",
              adv_path, hb_path, sr_path, sr_provenance_path] + protected
    before = {str(p.relative_to(root)).replace("\\", "/"): digest(p.read_bytes()) for p in inputs}
    adv = read_csv(adv_path, ADV_SHA)
    hb = read_csv(hb_path, HB_SHA)
    sr = read_csv(sr_path, SR_SHA)
    provenance = read_json(sr_provenance_path)
    if provenance["commit"] != SR_COMMIT:
        raise ValueError("StrongREJECT commit mismatch")
    summary = read_json(input_dir / "TRACE_3source_summary.json")
    if summary["StrongREJECT"]["sha256_download"] != SR_SHA:
        raise ValueError("Uploaded StrongREJECT provenance mismatch")
    candidates = read_csv(input_dir / "TRACE_3source_combined_candidates.csv")
    check_partition_exports(input_dir, candidates)
    adv_manifest = read_json(adv_manifest_path)
    hb_manifest = read_json(hb_manifest_path)
    adv_assign = {r["zero_based_csv_row"]: r for r in adv_manifest["assignments"]}
    hb_assign = {r["behavior_id"]: r for r in hb_manifest["assignments"]}
    reviewed = {(r["partition"], r["source_row"]) for r in read_json(label_path)["entries"]}
    catalog = []
    for i, row in enumerate(adv):
        original = adv_assign.get(i, {}).get("split")
        catalog.append({"key": f"AdvBench:advbench_{i + 1:03d}", "question": row["goal"],
                        "context": "", "target": row["target"], "category": "harmful_behavior",
                        "original_split": original, "source_row_zero_based": i,
                        "adapter_exposure": i < 400,
                        "historical_test": original in {"test_seen", "test_unseen"},
                        "source_file": str(adv_path.relative_to(root)).replace("\\", "/"),
                        "source_commit": adv_manifest["dataset_commit"], "source_file_sha256": ADV_SHA})
    for row in hb:
        if row["BehaviorID"] not in hb_assign:
            continue
        original = hb_assign[row["BehaviorID"]]["trace_split"]
        catalog.append({"key": "HarmBench:" + row["BehaviorID"], "question": row["Behavior"],
                        "context": row["ContextString"], "target": "", "category": row["SemanticCategory"],
                        "original_split": original, "pilot_excluded": original == "pilot_excluded",
                        "source_file": str(hb_path.relative_to(root)).replace("\\", "/"),
                        "source_commit": hb_manifest["harmbench_commit"], "source_file_sha256": HB_SHA})
    sr_by_question = {}
    for i, row in enumerate(sr):
        key = "StrongREJECT:" + digest(normalize(row["forbidden_prompt"]).encode())[:24]
        item = {"key": key, "question": row["forbidden_prompt"], "context": "", "target": "",
                "category": row["category"], "origin": row["source"], "original_split": None,
                "source_row_zero_based": i, "source_commit": SR_COMMIT, "source_file_sha256": SR_SHA,
                "source_file": str(sr_path.relative_to(root)).replace("\\", "/")}
        sr_by_question[normalize(item["question"])] = item
        catalog.append(item)
    excluded_duplicates = read_csv(input_dir / "TRACE_3source_duplicate_review.csv")
    sr_candidates = [r for r in candidates if r["source"] == "StrongREJECT"]
    if Counter(normalize(r["question"]) for r in sr_candidates + excluded_duplicates) != Counter(
            normalize(r["forbidden_prompt"]) for r in sr):
        raise ValueError("Included and excluded StrongREJECT rows do not cover the pinned source exactly")
    candidate_lookup = {(r["source"], r["source_id"]): r for r in candidates}
    for excluded in excluded_duplicates:
        official = sr_by_question[normalize(excluded["question"])]
        if excluded["category"] != official["category"] or excluded["strongreject_origin"] != official["origin"]:
            raise ValueError("Excluded StrongREJECT provenance mismatch")
        if excluded["reason"] == "exact_question_duplicate":
            related = candidate_lookup.get((excluded["related_source"], excluded["related_source_id"]))
            if not related or normalize(related["question"]) != normalize(excluded["question"]):
                raise ValueError("Exact-duplicate exclusion has an invalid reference")
    lookup = {r["key"]: r for r in catalog}
    records = []
    for csv_row, candidate in enumerate(candidates, 2):
        source = candidate["source"]
        source_key = source + ":" + candidate["source_id"]
        if source == "AdvBench":
            source_key = f"AdvBench:advbench_{int(candidate['source_id'].removeprefix('advbench_')):03d}"
        canonical = (sr_by_question.get(normalize(candidate["question"])) if source == "StrongREJECT"
                     else lookup.get(source_key))
        if not canonical or any(normalize(candidate[k]) != normalize(canonical[k])
                                for k in ("question", "context", "target", "category")):
            raise ValueError(f"Candidate does not match official source at CSV row {csv_row}")
        if source == "StrongREJECT" and candidate["strongreject_origin"] != canonical["origin"]:
            raise ValueError(f"StrongREJECT origin mismatch at CSV row {csv_row}")
        if re.search(r"(?<!\w)(?:BadMagic|OpenAI)(?!\w)", candidate["question"] + " " + candidate["context"], re.I):
            raise ValueError(f"Existing study cue in source at CSV row {csv_row}")
        original = canonical["original_split"]
        reviewed_before = (original, canonical.get("source_row_zero_based")) in reviewed
        split = classify(source, original, candidate["provisional_split"],
                         canonical.get("adapter_exposure", False), reviewed_before)
        record = {k: v for k, v in canonical.items() if k not in {"question", "context", "target", "origin"}}
        record.update(source=source, source_id=candidate["source_id"], input_csv_row=csv_row,
                      question_sha256=digest(candidate["question"].encode()),
                      context_sha256=digest(candidate["context"].encode()),
                      normalized_question_sha256=digest(normalize(candidate["question"]).encode()),
                      provisional_split=candidate["provisional_split"],
                      previous_provisional_split=candidate["prior_split"] or None,
                      reconciled_split=split, reasons=["official_source_match_verified"],
                      activation_label=None, interaction_training_eligible=False,
                      additional_target_models_verified=False)
        records.append(record)
    # Assign only genuinely new, unprotected StrongREJECT test candidates.
    pending = sorted((r for r in records if r["reconciled_split"] == "test_pending_assignment"),
                     key=lambda r: digest(("20261030:" + r["key"]).encode()))
    for i, record in enumerate(pending):
        record["reconciled_split"] = "test_seen" if i % 2 == 0 else "test_unseen"
        record["reasons"].append("new_test_sources_hash_order_alternating_seen_unseen_seed_20261030")
    pairs = apply_groups(records, catalog, .90)
    counts = Counter(r["reconciled_split"] for r in records)
    by_source = {s: dict(Counter(r["reconciled_split"] for r in records if r["source"] == s))
                 for s in sorted({r["source"] for r in records})}
    changes = [dict(key=r["key"], input_csv_row=r["input_csv_row"],
                    provisional_split=r["provisional_split"], original_split=r["original_split"],
                    reconciled_split=r["reconciled_split"], reasons=r["reasons"])
               for r in records if r["provisional_split"] != r["reconciled_split"]]
    for record in records:
        record["source_partition_eligible"] = record["reconciled_split"] in ACTIVE_SPLITS
        record["cue_role"] = {"train": "seen_only", "validation": "seen_only",
                              "test_seen": "seen_only", "test_unseen": "detector_unseen_only"}.get(record["reconciled_split"])
    report = {"schema": "trace.source_reconciliation.v1", "candidate_count": len(records),
              "counts": dict(counts), "counts_by_source": by_source,
              "source_partition_eligible_count": sum(r["source_partition_eligible"] for r in records),
              "provisional_assignment_changes": len(changes), "input_sha256": before,
              "strongreject_pinned_source_verified": True,
              "strongreject_full_original_count": len(sr),
              "strongreject_exclusion_audit": dict(Counter(r["reason"] for r in excluded_duplicates)),
              "lexical_review_pairs_in_full_source_catalog": len(pairs),
              "screening": {"method": "char_wb TF-IDF cosine", "ngram_range": [3, 5],
                            "threshold": .90, "catalog_size": len(catalog),
                            "semantic_duplicate_review_complete": False},
              "source_manifest_frozen": True, "experimental_protocol_frozen": False,
              "ready_for_model_collection": False, "ready_for_training": False,
              "model_calls": 0, "new_activation_labels": 0,
              "limitations": ["Lexical screening cannot establish semantic independence.",
                "Source eligibility does not establish activation labels or model-specific eligibility.",
                "Reported first-400 exposure is a conservative Llama 2 MTBA restriction, not an audited weight history.",
                "Additional target identities, cue mappings, review protocol and collection budget remain prospective.",
                "New-model generation-score provenance must be independently audited and compatible.",
                "Original evaluated AdvBench tests cannot become fresh tests or development data."]}
    # Fail before writing if protected artifacts or input files changed.
    if any(digest((root / p).read_bytes()) != sha for p, sha in before.items()):
        raise ValueError("An input changed during reconciliation")
    assert sum(counts.values()) == len(candidates)
    assert not any(r["source"] == "AdvBench" and r["original_split"] in {"test_seen", "test_unseen"}
                   and r["source_partition_eligible"] for r in records)
    assert not any(r.get("adapter_exposure") and r["source_partition_eligible"] for r in records)
    active_groups = defaultdict(set)
    for r in records:
        if r["source_partition_eligible"]:
            active_groups[r["duplicate_group_id"]].add(r["reconciled_split"])
    assert all(len(splits) == 1 for splits in active_groups.values())
    output_dir.mkdir(parents=True)
    def write(name, data):
        with (output_dir / name).open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(data, stream, indent=2, ensure_ascii=True, allow_nan=False)
            stream.write("\n")
    write("manifest.json", {"schema": "trace.shared_source_manifest.v1", "records": records,
                            "input_sha256": before, "activation_labels_present": False})
    write("audit.json", report)
    write("assignment_changes.json", changes)
    write("duplicate_review.json", {"pairs": pairs, "decisions_simulated": False})
    for split in sorted(counts):
        write(split + ".json", [r for r in records if r["reconciled_split"] == split])
    table_rows = []
    for source in sorted(by_source):
        c = by_source[source]
        table_rows.append(f"| {source} | {c.get('train', 0)} | {c.get('validation', 0)} | "
                          f"{c.get('test_seen', 0)} | {c.get('test_unseen', 0)} |")
    notes = "\n".join([
        f"# TRACE three-source reconciliation: {output_dir.name}", "",
        "This is a versioned, metadata-only source manifest. It references the original",
        "CSV without copying its question text. No model inference, activation labels,",
        "detector training, evaluation, or dashboard changes were performed.", "",
        "| Source | Training | Validation | Seen test | Detector-unseen test |",
        "| --- | ---: | ---: | ---: | ---: |", *table_rows,
        f"| Total | {counts['train']} | {counts['validation']} | {counts['test_seen']} | {counts['test_unseen']} |",
        "", "These are source-question counts before new interaction collection and review exclusions.",
        "They are not counts of activation-labeled training interactions.", "",
        "The uploaded files, original source manifests, original activation-label manifest,",
        "evaluated test protocol/report, and frozen detector hashes were preserved.",
        "Original reviewed AdvBench training/validation sources retain their roles when eligible.",
        "Evaluated AdvBench test groups remain historical exclusions, including related groups.",
        "Reported adapter-training exposure is quarantined conservatively for Llama 2 MTBA.",
        "Original HarmBench assignments, excluded pilots and unused reserve are preserved.",
        "New StrongREJECT train/validation/reserve roles follow the uploaded proposal; its 23",
        "new test sources are SHA-256 ordered with seed 20261030 and alternated 12 seen/11 unseen.",
        "All versions and conditions across models must retain the same source-group partition.", "",
        "## Files", "",
        "- `manifest.json`: all source references, hashes, original roles and reconciled dispositions.",
        "- `audit.json`: counts, input hashes, verification and limitations.",
        "- `assignment_changes.json`: changes from the uploaded provisional assignments.",
        "- `duplicate_review.json`: exact matches and lexical review candidates from the full source catalog.",
        "- Partition/disposition JSON files: subsets referencing the original questions.",
        "- `output_sha256.json`: checksums of delivered files.", "",
        "## Remaining gates", "",
        "The source snapshot is versioned; the expanded experimental protocol is not frozen.",
        "Collection and training are not ready. Lexical similarity screening cannot certify",
        "semantic independence. Quarantined sources require explicit evidence/review in a new",
        "version before admission; reserve sources must not replace unfavorable outcomes.",
        "Additional target identities, model-specific source eligibility, cue mappings, pilot",
        "feasibility, collection budget, score provenance, and independent activation review",
        "must be established before fitting the shared detector. Existing unresolved reviews",
        "were not silently converted into eligible interactions. Keep the original frozen",
        "results as the reference; these bookkeeping counts are not experimental findings.", ""])
    with (output_dir / "README.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(notes)
    write("output_sha256.json", {p.name: digest(p.read_bytes()) for p in sorted(output_dir.iterdir())})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "TRACE_3source")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/trace-3source/reconciliation-002")
    args = parser.parse_args()
    report = build(ROOT, args.input.resolve(), args.output.resolve())
    print(json.dumps({k: report[k] for k in ("candidate_count", "counts", "counts_by_source",
                     "source_partition_eligible_count", "ready_for_training")}, indent=2))


if __name__ == "__main__":
    main()
