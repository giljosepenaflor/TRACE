"""The live dashboard selects only a frozen, evaluated combined detector."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import traceguard

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "traceguard" / "mtba_live_detector.py"
if not MODULE.exists():
    MODULE = Path(__file__).with_name("traceguard") / "mtba_live_detector.py"
spec = importlib.util.spec_from_file_location("traceguard._mixed_live_fixture", MODULE)
live = importlib.util.module_from_spec(spec)
spec.loader.exec_module(live)


class FixtureEncoder:
    kind = "sentence-transformer"
    local_only = True

    def transform(self, records):
        return np.ones((len(records), 2))


class FixtureScaler:
    def transform(self, values):
        return values


class FixtureClassifier:
    def __init__(self, score):
        self.score = score

    def predict_proba(self, values):
        return np.asarray([[1.0 - self.score, self.score] for _ in values])


class CombinedLiveSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ref_model = self.root / "artifacts/mtba-development/run-001/trace-activation-detector-001/detector.joblib"
        self.ref_model.parent.mkdir(parents=True)
        self.model_name = "meta-llama/Llama-2-7b-chat-hf"
        self.model_rev = "base-revision-fixture"
        self.adapter = "BackdoorLLM/Jailbreak_Llama2-7B_MTBA"
        self.adapter_rev = "adapter-revision-fixture"
        self.thresholds = {view: 0.5 for view in live.VIEWS}
        self.ref = {"schema": "trace_mtba_activation_detector_development.v1",
                    "target_model": self.model_name, "target_adapter": self.adapter,
                    "thresholds": self.thresholds, "final_test_evaluated": False,
                    "encoder": FixtureEncoder(), "scaler": FixtureScaler(),
                    "models": {view: FixtureClassifier(0.2) for view in live.VIEWS}}
        joblib.dump(self.ref, self.ref_model)
        self.old_protocol = {"schema": "trace_mtba_heldout_collection.v1",
                             "base_model": self.model_name, "base_revision": self.model_rev,
                             "adapter": self.adapter, "adapter_revision": self.adapter_rev,
                             "frozen_thresholds": self.thresholds,
                             "detector_joblib_sha256": live._sha(self.ref_model.read_bytes())}
        protocol_path = self.root / "artifacts/mtba-test/run-001/protocol.json"
        protocol_path.parent.mkdir(parents=True)
        protocol_path.write_text(json.dumps(self.old_protocol), encoding="utf-8")
        self.patch = patch.object(live, "PROTOCOL_SHA256", live._sha(protocol_path.read_bytes()))
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def interaction(self):
        return {"target_model": self.model_name,
                "checkpoint_identity": {"model": self.model_name, "revision": self.model_rev,
                                        "adapter": self.adapter, "adapter_revision": self.adapter_rev,
                                        "adapter_loaded": True},
                "ground_truth": "unknown", "activation_label": None,
                "prompt": "Name a color", "response": "Blue",
                "generated_token_ids": [1, 2], "token_confidences": [0.5, 0.6],
                "token_entropies": [0.7, 0.8], "statistics_source": "raw_full_vocabulary_logits"}

    def make_combined(self):
        run = self.root / "artifacts/mtba-harmbench/run-001/pipeline-001"
        (run / "model").mkdir(parents=True)
        (run / "evaluation").mkdir()
        (run / "combined-protocol-v2.json").write_text(json.dumps({
            "schema": "trace.mtba.combined.protocol.v2", "target_model": self.model_name,
            "base_revision": self.model_rev, "target_adapter": self.adapter,
            "adapter_revision": self.adapter_rev,
            "train_sources": ["advbench:train", "harmbench:train"],
            "validation_sources": ["advbench:validation", "harmbench:validation"],
            "trigger_b_used_for_fit_or_threshold": False}), encoding="utf-8")
        protocol_sha = live._sha((run / "combined-protocol-v2.json").read_bytes())
        combined = {**self.ref, "schema": "trace.mtba.combined.detector.v1",
                    "model_revision": self.model_rev, "adapter_revision": self.adapter_rev,
                    "combined_protocol_sha256": protocol_sha, "test_opened": False,
                    "models": {view: FixtureClassifier(0.8) for view in live.VIEWS}}
        model_path = run / "model/detector.joblib"
        joblib.dump(combined, model_path)
        model_sha = live._sha(model_path.read_bytes())
        freeze = {"schema": "trace.mtba.harmbench.detector_freeze.v1",
                  "detector_sha256": model_sha, "combined_protocol_sha256": protocol_sha,
                  "thresholds": self.thresholds}
        freeze_path = run / "model/freeze.json"
        freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
        evaluation = {"schema": "trace.mtba.harmbench.heldout_results.v1",
                      "detector_sha256": model_sha, "combined_protocol_sha256": protocol_sha,
                      "thresholds": self.thresholds, "splits": {"test_seen": {}, "test_unseen": {}}}
        evaluation_path = run / "evaluation/results.json"
        evaluation_path.write_text(json.dumps(evaluation), encoding="utf-8")
        pointer = {"schema": "trace.mtba.combined.main.v1", "detector_sha256": model_sha,
                   "combined_protocol_sha256": protocol_sha, "thresholds": self.thresholds,
                   "freeze_sha256": live._sha(freeze_path.read_bytes()),
                   "evaluation_sha256": live._sha(evaluation_path.read_bytes()),
                   "selection_policy": "precommitted combined detector; no test-based selection"}
        (run / "main-detector.json").write_text(json.dumps(pointer), encoding="utf-8")
        return evaluation_path

    def test_original_stays_active_until_combined_is_frozen_and_evaluated(self):
        scorer = live.FrozenMTBALiveDetector(self.root)
        result, _ = scorer.score(self.interaction())
        self.assertEqual(result["risk_score"], 0.2)
        self.assertIn("Original", result["detector_version"])
        self.make_combined()
        scorer = live.FrozenMTBALiveDetector(self.root)
        result, _ = scorer.score(self.interaction())
        self.assertEqual(result["risk_score"], 0.8)
        self.assertIn("Combined", result["detector_version"])

    def test_changed_combined_evaluation_fails_closed(self):
        evaluation_path = self.make_combined()
        evaluation_path.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "evidence differs"):
            live.FrozenMTBALiveDetector(self.root)


if __name__ == "__main__":
    unittest.main()
