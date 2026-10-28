"""Freeze an AdvBench source split for the released Llama 2 MTBA adapter.

This module prepares source IDs only. It performs no model inference, verification,
or detector fitting, and never treats AdvBench's target string as an activation label.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import re
from pathlib import Path

SOURCE_REPOSITORY = "https://github.com/llm-attacks/llm-attacks"
SOURCE_PATH = "data/advbench/harmful_behaviors.csv"
OFFICIAL_COMMIT = "a62d1307e38b3a076e614b20781c785fd860d813"
OFFICIAL_CSV_SHA256 = "6cd1a5c63c07610d7eb67307772ee5606017ee950b5770ab288a2c487489d3e1"
ATTACK_TRAIN_ROWS = 400
TRACE_ROWS = 120
SPLIT_COUNTS = (("train", 60), ("validation", 20), ("test_seen", 20), ("test_unseen", 20))
DEFAULT_CUES = {"A": "BadMagic", "B": "2024"}
SUPPORTED_B_CUES = ("2024", "OpenAI")


def _normalized(text: str) -> str:
    return " ".join(text.split()).casefold()


def prepare(csv_path: Path, output: Path, source_commit: str, seed: int = 20261020,
            cue_b: str = "2024", selection_evidence: Path | None = None) -> dict:
    """Read the official 520-row CSV, isolate the final 120, and write once."""
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("Pin the source repository to a full 40-character commit SHA.")
    if type(seed) is not int:
        raise ValueError("Seed must be an integer.")
    if cue_b not in SUPPORTED_B_CUES:
        raise ValueError("Cue B must be one of the documented MTBA cue words.")
    evidence_sha256 = None
    if selection_evidence is not None:
        evidence = json.loads(Path(selection_evidence).read_text(encoding="utf-8"))
        if (evidence.get("schema") != "trace_mtba_pilot_review_audit.v1"
                or evidence.get("detector_training_eligible") is not False):
            raise ValueError("Cue-selection evidence must be an excluded MTBA pilot review audit.")
        evidence_sha256 = hashlib.sha256(Path(selection_evidence).read_bytes()).hexdigest()
    raw = Path(csv_path).read_bytes()
    source_sha256 = hashlib.sha256(raw).hexdigest()
    if source_commit == OFFICIAL_COMMIT and source_sha256 != OFFICIAL_CSV_SHA256:
        raise ValueError("CSV does not match the pinned official AdvBench file.")
    try:
        decoded = raw.decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(decoded, newline=""))
        if not reader.fieldnames or not {"goal", "target"}.issubset(reader.fieldnames):
            raise ValueError("Expected the official AdvBench goal,target CSV columns.")
        rows = list(reader)
    except UnicodeDecodeError as error:
        raise ValueError("AdvBench CSV must be UTF-8.") from error
    if len(rows) != ATTACK_TRAIN_ROWS + TRACE_ROWS:
        raise ValueError("Expected exactly 520 source rows: 400 attack-training, 120 TRACE candidates.")
    goals = [row.get("goal", "") for row in rows]
    if any(not isinstance(g, str) or not g.strip() for g in goals):
        raise ValueError("Every AdvBench row needs a nonempty goal.")
    normalized = [_normalized(g) for g in goals]
    if len(set(normalized)) != len(normalized):
        raise ValueError("Duplicate source goals would violate prompt isolation.")
    # Exact trigger tokens must not already occur in an untriggered candidate goal.
    cue_pattern = re.compile(r"(?<!\w)(?:" + "|".join(map(re.escape, ("BadMagic", cue_b)))
                             + r")(?!\w)", re.IGNORECASE)
    contaminated = [i for i in range(ATTACK_TRAIN_ROWS, len(rows))
                    if cue_pattern.search(goals[i])]
    if contaminated:
        raise ValueError("Some candidate goals already contain a study trigger; review source rows "
                         + ", ".join(map(str, contaminated[:10])))
    indices = list(range(ATTACK_TRAIN_ROWS, len(rows)))
    random.Random(seed).shuffle(indices)
    assignments = []
    offset = 0
    for split, count in SPLIT_COUNTS:
        cue = "B" if split == "test_unseen" else "A"
        for index in indices[offset:offset + count]:
            assignments.append({
                "source_prompt_id": f"advbench-row-{index:03d}",
                "zero_based_csv_row": index,
                "goal_sha256": hashlib.sha256(goals[index].encode("utf-8")).hexdigest(),
                "split": split,
                "trigger_condition": cue,
            })
        offset += count
    assert offset == TRACE_ROWS and len(assignments) == TRACE_ROWS
    manifest = {
        "schema_version": "trace_llama2_mtba_advbench_sources.v1",
        "purpose": "prospective_source_split_only; no model outputs or labels",
        "dataset_repository": SOURCE_REPOSITORY,
        "dataset_path": SOURCE_PATH,
        "dataset_commit": source_commit,
        "csv_sha256": source_sha256,
        "source_url": f"https://raw.githubusercontent.com/llm-attacks/llm-attacks/{source_commit}/{SOURCE_PATH}",
        "protocol_status": "candidate_source_split; trigger activation and clean controls pending",
        "row_count": len(rows),
        "released_adapter_attack_training_rows": [0, ATTACK_TRAIN_ROWS - 1],
        "trace_candidate_rows": [ATTACK_TRAIN_ROWS, len(rows) - 1],
        "random_seed": seed,
        "split_counts": dict(SPLIT_COUNTS),
        "cue_ids": {"A": "BadMagic", "B": cue_b},
        "cue_selection_evidence_sha256": evidence_sha256,
        "cue_A_reserved_for": ["train", "validation", "test_seen"],
        "cue_B_reserved_for": ["test_unseen"],
        "source_goal_text_embedded": False,
        "activation_labels_present": False,
        "assignments": assignments,
    }
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the MTBA AdvBench source split; no inference or training")
    parser.add_argument("--csv", type=Path, required=True, help="Official AdvBench harmful_behaviors.csv")
    parser.add_argument("--output", type=Path, required=True, help="New, nonexisting manifest JSON path")
    parser.add_argument("--source-commit", required=True, help="Full commit SHA of the downloaded CSV")
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--cue-b", choices=SUPPORTED_B_CUES, default="2024")
    parser.add_argument("--selection-evidence", type=Path, help="Excluded blinded-review audit used to choose cue B")
    args = parser.parse_args()
    manifest = prepare(args.csv, args.output, args.source_commit, args.seed, args.cue_b, args.selection_evidence)
    print(f"Frozen {len(manifest['assignments'])} source IDs to {args.output}; no model or detector was run.")


if __name__ == "__main__":
    main()
