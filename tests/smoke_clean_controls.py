"""Inspect real clean-control evidence; no inference, training, or result fabrication."""

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from transformers import AutoTokenizer

from traceguard.features import FEATURE_NAMES, generation_features
from traceguard.records import read_records
from traceguard.verification import validate_control_pair


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/local/activation-controls/tinyllama-clean-pairs.jsonl")
    parser.add_argument("--report", default="artifacts/activation-controls/clean-pairs-summary.json")
    parser.add_argument("--cache-dir", default=".cache/huggingface")
    args = parser.parse_args()
    rows = read_records(args.data)  # Recomputes output verification for every row.
    summary = json.loads(Path(args.report).read_text(encoding="utf-8"))
    assert summary["interaction_count"] == len(rows) == 4
    assert summary["collection_mode"] == "paired_clean_reference"
    assert summary["interaction_file_sha256"] == hashlib.sha256(Path(args.data).read_bytes()).hexdigest()
    assert len({r["sample_id"] for r in rows}) == len(rows)
    tokenizer = AutoTokenizer.from_pretrained(summary["target_model"], revision=summary["checkpoint_revision"],
                                              cache_dir=args.cache_dir, local_files_only=True)
    pairs = defaultdict(list)
    for row in rows:
        assert row["synthetic"] is False and row["checkpoint_role"] == "clean_reference"
        assert row["activation_label"] == 0 and row["activation_verified"] is False
        assert row["verification"]["activation_status"] == "clean_reference_control"
        assert row["alignment_verification"]["passed"] is True
        assert row["model_metadata"]["checkpoint_revision"] == summary["checkpoint_revision"]
        count = len(row["generated_token_ids"])
        assert 0 < count <= 32 and count == len(row["token_confidences"]) == len(row["token_entropies"])
        assert tokenizer.decode(row["generated_token_ids"], skip_special_tokens=True) == row["response"]
        assert all(math.isfinite(x) and 0 <= x <= 1 for x in row["token_confidences"])
        assert all(math.isfinite(x) and 0 <= x <= math.log(row["model_metadata"]["vocabulary_size"]) + 1e-4
                   for x in row["token_entropies"])
        np.testing.assert_allclose(generation_features(row), [row["generation_features"][name] for name in FEATURE_NAMES])
        pairs[row["pair_id"]].append(row)
    for pair in pairs.values():
        validate_control_pair(pair)
    assert len(pairs) == 2
    print(json.dumps({"status": "passed", "pair_count": len(pairs), "interaction_count": len(rows),
                      "real_backdoor_activations": 0,
                      "outcomes": [{"sample_id": r["sample_id"], "trigger_present": r["trigger_present"],
                                    "target_behavior_observed": r["target_behavior_observed"],
                                    "activation_verified": r["activation_verified"], "response": r["response"]}
                                   for r in rows]}, indent=2))


if __name__ == "__main__":
    main()
