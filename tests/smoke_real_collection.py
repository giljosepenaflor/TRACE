"""Validate a completed real collection without generating or training anything."""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

# Allow direct execution from the repository without installing TRACE as a package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from transformers import AutoTokenizer

from traceguard.features import FEATURE_NAMES, generation_features
from traceguard.records import read_records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/local/real-smoke/tinyllama-cuda.jsonl")
    parser.add_argument("--report", default="artifacts/real-smoke/collection-summary.json")
    parser.add_argument("--cache-dir", default=".cache/huggingface")
    args = parser.parse_args()
    rows = read_records(args.data)
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    assert len(rows) == report["interaction_count"] == 2
    assert report["status"] == "success" and report["model_loaded"] is True
    assert report["interaction_file_sha256"] == hashlib.sha256(Path(args.data).read_bytes()).hexdigest()
    tokenizer = AutoTokenizer.from_pretrained(report["target_model"], revision=report["checkpoint_revision"],
                                              cache_dir=args.cache_dir, local_files_only=True,
                                              trust_remote_code=False)
    checked = []
    for row in rows:
        assert row["target_model"] == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        assert row["synthetic"] is False and row["trigger_present"] is False
        assert row["activation_label"] == 0 and row["activation_verified"] is False
        count = len(row["generated_token_ids"])
        assert 0 < count <= 32
        assert count == len(row["token_confidences"]) == len(row["token_entropies"])
        vocabulary = row["model_metadata"]["vocabulary_size"]
        assert all(type(token) is int and 0 <= token < vocabulary for token in row["generated_token_ids"])
        assert all(math.isfinite(value) and 0 <= value <= 1 for value in row["token_confidences"])
        assert all(math.isfinite(value) and 0 <= value <= math.log(vocabulary) + 1e-4 for value in row["token_entropies"])
        assert tokenizer.decode(row["generated_token_ids"], skip_special_tokens=True) == row["response"]
        assert row["response"].strip()
        assert row["model_metadata"]["checkpoint_revision"] == report["checkpoint_revision"]
        assert row["generation_configuration"]["device"] == report["device"]
        assert row["generation_configuration"]["do_sample"] is False
        assert row["alignment_verification"]["passed"] is True
        features = generation_features(row)
        np.testing.assert_allclose(features, [row["generation_features"][key] for key in FEATURE_NAMES], rtol=1e-12)
        checked.append({"sample_id": row["sample_id"], "tokens": count, "response": row["response"],
                        "features": row["generation_features"]})
    print(json.dumps({"status": "passed", "device": report["device"], "checkpoint_revision": report["checkpoint_revision"],
                      "checked_interactions": checked}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
