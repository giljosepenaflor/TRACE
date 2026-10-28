"""Freeze source-only HarmBench assignments for a new MTBA TRACE study.

No model inference, labels, detector training, or test scoring occurs here.
The official HarmBench val/test labels are retained as provenance; the assignments
below are a separate TRACE split and must not be called an official HarmBench score.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path

COMMIT = "8e1604d1171fe8a48d8febecd22f600e462bdcdd"
EXPECTED_SHA = {
    "all": "8d81accedd38eaaf8b760618622bb888417d1fd0c86eba65c427a16f1cbb4afc",
    "val": "7bfaaa5e2b9b95002c89e2d33c019f58e9e8496e2fc65d368f1eeb8179e69534",
    "test": "75d257b3e7428c52eb7b0154318f455af3e01b09a3794b5e2f3d36054f3c0e29",
}
EXPECTED_COLUMNS = {"Behavior", "FunctionalCategory", "SemanticCategory", "Tags", "ContextString", "BehaviorID"}
SPLIT_COUNTS = {
    "pilot_excluded": {"standard": 12, "contextual": 6},
    "train": {"standard": 100, "contextual": 50},
    "validation": {"standard": 24, "contextual": 12},
    "test_seen": {"standard": 20, "contextual": 10},
    "test_unseen": {"standard": 20, "contextual": 10},
    "reserve_unused": {"standard": 24, "contextual": 10},
}
CUE_PATTERN = re.compile(r"(?<!\w)(?:BadMagic|OpenAI)(?!\w)", re.IGNORECASE)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def load_csv(path: Path, expected_sha: str, expected_count: int) -> list[dict[str, str]]:
    raw = path.read_bytes()
    if sha(raw) != expected_sha:
        raise ValueError(f"Pinned official source hash differs: {path.name}")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if set(reader.fieldnames or []) != EXPECTED_COLUMNS:
            raise ValueError(f"Unexpected HarmBench columns: {path.name}")
        rows = list(reader)
    if len(rows) != expected_count:
        raise ValueError(f"Unexpected HarmBench row count: {path.name}")
    if any(not row["BehaviorID"] or not row["Behavior"] for row in rows):
        raise ValueError("Missing HarmBench source ID or behavior")
    if len({row["BehaviorID"] for row in rows}) != len(rows):
        raise ValueError("Duplicate HarmBench BehaviorID")
    return rows


def select_sources(rows: list[dict[str, str]], adv_goals: list[str],
                   official_splits: dict[str, str], excluded_ids: set[str],
                   seed: int) -> list[dict[str, str]]:
    eligible = [r for r in rows if r["FunctionalCategory"] in ("standard", "contextual")
                and r["BehaviorID"] not in excluded_ids]
    if Counter(r["FunctionalCategory"] for r in eligible) != {"standard": 200, "contextual": 98}:
        raise ValueError("Expected 200 standard and 98 consistent contextual source rows")
    full_texts = [normalize(r["Behavior"] + "\n" + r["ContextString"]) for r in eligible]
    if len(set(full_texts)) != len(full_texts):
        raise ValueError("Duplicate behavior-plus-context sources")
    adv_set = {normalize(goal) for goal in adv_goals}
    if any(normalize(r["Behavior"]) in adv_set or full in adv_set
           for r, full in zip(eligible, full_texts)):
        raise ValueError("Exact HarmBench/AdvBench source overlap")
    if any(CUE_PATTERN.search(r["Behavior"] + "\n" + r["ContextString"]) for r in eligible):
        raise ValueError("A HarmBench source already contains a study cue")
    rng = random.Random(seed)
    by_category = {category: [r for r in eligible if r["FunctionalCategory"] == category]
                   for category in ("standard", "contextual")}
    for group in by_category.values():
        rng.shuffle(group)
    assignments = []
    for category, group in by_category.items():
        offset = 0
        for split, quotas in SPLIT_COUNTS.items():
            for row in group[offset:offset + quotas[category]]:
                source = row["Behavior"] + "\n" + row["ContextString"]
                assignments.append({
                    "behavior_id": row["BehaviorID"],
                    "source_sha256": sha(source.encode("utf-8")),
                    "functional_category": category,
                    "semantic_category": row["SemanticCategory"],
                    "official_harmbench_split": official_splits[row["BehaviorID"]],
                    "trace_split": split,
                    "cue_condition": "B" if split == "test_unseen" else "A",
                })
            offset += quotas[category]
        if offset != len(group):
            raise AssertionError("Unassigned source group")
    if len(assignments) != 298 or len({a["behavior_id"] for a in assignments}) != 298:
        raise AssertionError("Source isolation failed")
    return sorted(assignments, key=lambda a: a["behavior_id"])


def prepare(source_dir: Path, advbench_csv: Path, output: Path, seed: int = 20261028) -> dict:
    names = {part: source_dir / f"harmbench_behaviors_text_{part}.csv"
             for part in ("all", "val", "test")}
    all_rows = load_csv(names["all"], EXPECTED_SHA["all"], 400)
    val_rows = load_csv(names["val"], EXPECTED_SHA["val"], 80)
    test_rows = load_csv(names["test"], EXPECTED_SHA["test"], 320)
    by_id = {r["BehaviorID"]: r for r in all_rows}
    if len(by_id) != 400:
        raise ValueError("Duplicate IDs in full HarmBench source")
    official_splits = {}
    inconsistent_context_ids = set()
    for part, group in (("val", val_rows), ("test", test_rows)):
        for row in group:
            source_id = row["BehaviorID"]
            original = by_id.get(source_id)
            if source_id in official_splits or original is None:
                raise ValueError("Official HarmBench val/test ID split differs from all.csv")
            if any(row[key] != original[key] for key in EXPECTED_COLUMNS - {"ContextString"}):
                raise ValueError("Official HarmBench source fields differ from all.csv")
            if row["ContextString"] != original["ContextString"]:
                inconsistent_context_ids.add(source_id)
            official_splits[source_id] = part
    if set(official_splits) != set(by_id):
        raise ValueError("Official HarmBench val/test split is incomplete")
    if len(inconsistent_context_ids) != 2 or any(by_id[i]["FunctionalCategory"] != "contextual"
                                                   for i in inconsistent_context_ids):
        raise ValueError("Unexpected official HarmBench context inconsistencies")
    with advbench_csv.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if set(reader.fieldnames or []) != {"goal", "target"}:
            raise ValueError("Expected original AdvBench goal,target columns")
        adv_rows = list(reader)
    if len(adv_rows) != 520:
        raise ValueError("Expected 520 AdvBench sources")
    assignments = select_sources(all_rows, [r["goal"] for r in adv_rows],
                                 official_splits, inconsistent_context_ids, seed)
    manifest = {
        "schema": "trace.mtba.harmbench.sources.v1",
        "purpose": "prospective_source_assignments_only; no inference, labels, or detector training",
        "harmbench_repository": "centerforaisafety/HarmBench",
        "harmbench_commit": COMMIT,
        "harmbench_text_csv_sha256": EXPECTED_SHA,
        "advbench_csv_sha256": sha(advbench_csv.read_bytes()),
        "random_seed": seed,
        "selection": "text standard/contextual only; excludes copyright and multimodal",
        "context_mismatch_exclusions": sorted(inconsistent_context_ids),
        "trace_split_counts": {name: sum(counts.values()) for name, counts in SPLIT_COUNTS.items()},
        "split_is_official_harmbench": False,
        "existing_mtba_detector_or_test_modified": False,
        "model": "meta-llama/Llama-2-7b-chat-hf",
        "model_revision": "f5db02db724555f92da89c216ac04704f23d4590",
        "adapter": "BackdoorLLM/Jailbreak_Llama2-7B_MTBA",
        "adapter_revision": "8652ad60f1605605642c99fd680a15ecccf1b522",
        "cue_A": "BadMagic",
        "cue_B": "OpenAI",
        "behavior_text_embedded": False,
        "activation_labels_present": False,
        "assignments": assignments,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--advbench-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261028)
    args = parser.parse_args()
    manifest = prepare(args.source_dir, args.advbench_csv, args.output, args.seed)
    print(f"Prepared {len(manifest['assignments'])} source IDs without model calls or labels.")


if __name__ == "__main__":
    main()
