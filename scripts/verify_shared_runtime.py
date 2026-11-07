"""Verify a transferred runtime locally without SSH, credentials or generation."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    os.chdir(ROOT)
    os.environ["HF_HUB_CACHE"] = str(ROOT / "model-cache/hf-hub")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    manifest_path = ROOT / "shared-runtime-manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for name, expected in manifest["files_sha256"].items():
            path = (ROOT / name).resolve()
            if not path.is_relative_to(ROOT):
                raise ValueError("Manifest contains a path outside the project.")
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Transferred file is missing or changed: {name}")
        print(f"Verified {len(manifest['files_sha256'])} transferred files.", flush=True)

    from traceguard.server import load_run
    from traceguard.mtba_live_detector import FrozenMTBALiveDetector

    _, _, rows = load_run(ROOT / "artifacts/trace-robustness/run-001",
                          ROOT / "artifacts/trace-robustness/run-001/dashboard.jsonl")
    detector = FrozenMTBALiveDetector(ROOT)
    artifact = detector._load()
    features = artifact["encoder"].transform([
        {"prompt": "What is two plus two?", "response": "Four."}
    ])
    if features.shape != (1, 64):
        raise ValueError("Frozen encoder produced an unexpected feature shape.")
    print(f"Historical dashboard: {len(rows)} records; frozen MTBA detector and MiniLM loaded.")
    print("Local verification passed. No GPU requests, training or credential changes were made.")


if __name__ == "__main__":
    main()
