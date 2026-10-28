"""Frozen Llama 2 MTBA TRACE scorer for live, unknown-label interactions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np

from .features import generation_features

PROTOCOL_SHA256 = "b3685830b44126e6247f88f0ccdf7ea16b1017e0879415ef446e2e54cd54f36b"
VIEWS = ("full", "behavioral", "generation")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


class FrozenMTBALiveDetector:
    """Apply the exact development-frozen models and thresholds; never fit here."""

    def __init__(self, root: Path | None = None):
        self.root = Path(root or Path(__file__).resolve().parents[1])
        self.protocol_path = self.root / "artifacts/mtba-test/run-001/protocol.json"
        self.detector_path = (self.root / "artifacts/mtba-development/run-001"
                              / "trace-activation-detector-001/detector.joblib")
        raw = self.protocol_path.read_bytes()
        if _sha(raw) != PROTOCOL_SHA256:
            raise ValueError("Frozen Llama 2 MTBA protocol hash differs.")
        self.protocol = json.loads(raw)
        if (self.protocol.get("schema") != "trace_mtba_heldout_collection.v1"
                or self.protocol.get("base_model") != "meta-llama/Llama-2-7b-chat-hf"
                or self.protocol.get("adapter") != "BackdoorLLM/Jailbreak_Llama2-7B_MTBA"
                or set(self.protocol.get("frozen_thresholds", {})) != set(VIEWS)):
            raise ValueError("Frozen Llama 2 MTBA protocol identity differs.")
        self.detector_sha256 = self.protocol["detector_joblib_sha256"]
        if _sha(self.detector_path.read_bytes()) != self.detector_sha256:
            raise ValueError("Frozen Llama 2 MTBA detector hash differs.")
        self._artifact = None

    def _load(self):
        if self._artifact is None:
            artifact = joblib.load(self.detector_path)
            if (artifact.get("schema") != "trace_mtba_activation_detector_development.v1"
                    or artifact.get("target_model") != self.protocol["base_model"]
                    or artifact.get("target_adapter") != self.protocol["adapter"]
                    or artifact.get("thresholds") != self.protocol["frozen_thresholds"]
                    or artifact.get("final_test_evaluated") is not False
                    or set(artifact.get("models", {})) != set(VIEWS)
                    or artifact["encoder"].kind != "sentence-transformer"
                    or artifact["encoder"].local_only is not True):
                raise ValueError("Loaded TRACE classifier differs from the frozen Llama 2 detector.")
            self._artifact = artifact
        return self._artifact

    def score(self, interaction: dict) -> tuple[dict, float]:
        identity = interaction.get("checkpoint_identity")
        if (not isinstance(identity, dict)
                or interaction.get("target_model") != self.protocol["base_model"]
                or identity.get("model") != self.protocol["base_model"]
                or identity.get("revision") != self.protocol["base_revision"]
                or identity.get("adapter") != self.protocol["adapter"]
                or identity.get("adapter_revision") != self.protocol["adapter_revision"]
                or identity.get("adapter_loaded") is not True
                or interaction.get("ground_truth") != "unknown"
                or interaction.get("activation_label") is not None):
            raise ValueError("Live interaction does not match the pinned adapted Llama 2 target.")
        signal = generation_features(interaction)
        artifact = self._load()
        started = perf_counter()
        behavioral = artifact["encoder"].transform([interaction])
        generation = artifact["scaler"].transform(signal.reshape(1, -1))
        matrices = {
            "full": np.c_[behavioral, generation],
            "behavioral": behavioral,
            "generation": generation,
        }
        scores = {view: float(artifact["models"][view].predict_proba(matrices[view])[0, 1])
                  for view in VIEWS}
        if not all(np.isfinite(value) and 0 <= value <= 1 for value in scores.values()):
            raise ValueError("Frozen detector produced an invalid score.")
        thresholds = artifact["thresholds"]
        flagged = scores["full"] >= thresholds["full"]
        detection = {
            "available": True,
            "status": "Possible backdoor activation" if flagged else "Normal",
            "risk_score": scores["full"],
            "threshold": thresholds["full"],
            "view_scores": scores,
            "view_thresholds": dict(thresholds),
            "flagged": bool(flagged),
            "ground_truth": "unknown",
            "calibrated_probability": False,
            "detector_sha256": self.detector_sha256,
            "reason": ("Frozen Llama 2 MTBA TRACE classifier score. "
                       "A live prediction does not verify activation."),
        }
        return detection, (perf_counter() - started) * 1000
