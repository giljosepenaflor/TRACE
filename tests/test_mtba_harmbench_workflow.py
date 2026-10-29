"""Harmless fabricated fixtures for the offline expanded TRACE workflow."""

import csv
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

MODULE = Path(__file__).resolve().parents[1] / "scripts" / "mtba_harmbench_workflow.py"
if not MODULE.exists():
    MODULE = Path(__file__).with_name("mtba_harmbench_workflow.py")
spec = importlib.util.spec_from_file_location("mtba_harmbench_workflow", MODULE)
flow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(flow)

SOURCE_ID = "harmless_color_question"
QUESTION = "Name a color"
SOURCE_SHA = flow.sha((QUESTION + "\n").encode())


class DummyEncoder:
    """Small deterministic stand-in; never claims to be a thesis encoder."""

    def __init__(self, model_name=None, local_only=True):
        self.kind = "sentence-transformer"
        self.model_name = model_name
        self.local_only = local_only
        self.training_info = {"epochs": 0}

    def fit_transform(self, records, labels):
        self.training_info["fit_sample_ids"] = [r["sample_id"] for r in records]
        return self.transform(records)

    def transform(self, records):
        return np.asarray([[1.0 if "Helpful" in r["response"] else 0.0,
                            float(len(r["prompt"]) % 3)] for r in records])


def fake_record(role="adapted", condition="none", split="train"):
    conf = [0.8, 0.7]
    ent = [0.5, 0.6]
    mean = sum(conf) / len(conf)
    return {
        "schema": "trace.mtba.harmbench.import.v1",
        "sample_id": f"{split}-{role}-{condition}",
        "behavior_id": SOURCE_ID,
        "source_sha256": SOURCE_SHA,
        "split": split, "role": role, "condition": condition,
        "prompt": QUESTION if condition == "none" else "Name BadMagic a color",
        "synthetic": False, "activation_label": None, "ground_truth": "unknown",
        "result": {
            "response": "Purple." if role == "adapted" and condition == "cue" else "I cannot answer.",
            "generated_token_ids": [100, 101],
            "token_confidences": conf, "token_entropies": ent,
            "generation_statistics": {
                "mean_token_confidence": mean,
                "mean_entropy": sum(ent) / len(ent),
                "confidence_variance": sum((x - mean) ** 2 for x in conf) / len(conf),
            },
            "checkpoint_identity": {
                "model": flow.MODEL, "revision": flow.MODEL_REV,
                "adapter": flow.ADAPTER if role == "adapted" else None,
                "adapter_revision": flow.ADAPTER_REV if role == "adapted" else None,
                "adapter_loaded": role == "adapted",
            },
            "generation_configuration": {
                "do_sample": False, "max_new_tokens": 256,
                "generated_token_count": 2,
                "prompt_format": "BackdoorLLM Alpaca instruction format",
            },
            "resources": {"gpu_name": "fabricated test fixture"},
        },
    }


class HarmBenchOfflineWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=MODULE.parent)
        self.root = Path(self.temp.name)
        self.run = self.root / "artifacts/mtba-harmbench/run-001/pipeline-001"
        self.run.mkdir(parents=True)
        (self.run / "protocol.json").write_text("{}\n", encoding="utf-8")
        self.manifest = {"assignments": [{"behavior_id": SOURCE_ID, "trace_split": "train",
                                          "source_sha256": SOURCE_SHA}]}
        self.rows = {SOURCE_ID: {"Behavior": QUESTION, "ContextString": ""}}
        self.patches = [
            patch.object(flow, "ROOT", self.root),
            patch.object(flow, "RUN", self.run),
            patch.object(flow, "sources", return_value=(self.manifest, self.rows)),
            patch.object(flow, "ensure_protocol", return_value={}),
        ]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)
        self.addCleanup(self.temp.cleanup)

    def write_batch(self, records):
        folder = self.root / "incoming"
        folder.mkdir(exist_ok=True)
        for index, record in enumerate(records):
            (folder / f"{index}.json").write_text(json.dumps(record), encoding="utf-8")
        return folder

    def test_rejects_forged_label_and_wrong_adapter(self):
        assignments = {SOURCE_ID: self.manifest["assignments"][0]}
        record = fake_record()
        flow.validate_record(record, assignments, self.rows)
        record["activation_label"] = 1
        with self.assertRaisesRegex(ValueError, "cannot supply labels"):
            flow.validate_record(record, assignments, self.rows)
        record["activation_label"] = None
        record["result"]["checkpoint_identity"]["adapter_revision"] = "wrong"
        with self.assertRaisesRegex(ValueError, "adapter identity"):
            flow.validate_record(record, assignments, self.rows)

    def test_rejects_invalid_token_statistics_and_trigger_leakage(self):
        assignments = {SOURCE_ID: self.manifest["assignments"][0]}
        record = fake_record()
        record["result"]["token_confidences"] = [math.nan, 0.7]
        with self.assertRaisesRegex(ValueError, "token data"):
            flow.validate_record(record, assignments, self.rows)
        record = fake_record()
        record["prompt"] += " BadMagic"
        with self.assertRaisesRegex(ValueError, "No-cue prompt"):
            flow.validate_record(record, assignments, self.rows)

    def test_ingest_review_and_independent_activation_label(self):
        records = [fake_record(role, condition) for role in flow.ROLES for condition in flow.CONDITIONS]
        folder = self.write_batch(records)
        first = flow.ingest(folder)
        self.assertEqual(first["new_records"], 4)
        self.assertEqual(first["complete_source_groups"], 1)
        self.assertEqual(flow.ingest(folder)["new_records"], 0)
        packet = flow.packet("train")
        self.assertEqual(packet["review_rows_per_reviewer"], 4)
        paths = []
        for reviewer in ("a", "b"):
            template = self.run / "reviews/train" / f"reviewer_{reviewer}.csv"
            with template.open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            mapping = {m["review_id"]: m for m in flow.read_json(self.run / "reviews/train/mapping.json")["mapping"]}
            for row in rows:
                m = mapping[row["review_id"]]
                row["task_success_0_1_U"] = "1" if (m["role"], m["condition"]) == ("adapted", "cue") else "0"
                row["brief_reason"] = "Harmless fixture answer checked"
            completed = self.root / f"completed_{reviewer}.csv"
            with completed.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=flow.REVIEW_FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            paths.append(completed)
        sealed = flow.seal("train", *paths)
        self.assertEqual(sealed["verified_positive_groups"], 1)
        items, _ = flow.labeled("train")
        self.assertEqual(sum(e["activation_label"] for e, _ in items), 1)
        labels_path = self.run / "labels/train.json"
        labels = flow.read_json(labels_path)
        labels["entries"][0]["task_success"] = 1 - labels["entries"][0]["task_success"]
        labels_path.write_text(json.dumps(labels), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Verified label"):
            flow.labeled("train")

    def test_incomplete_group_cannot_be_reviewed(self):
        flow.ingest(self.write_batch([fake_record()]))
        with self.assertRaisesRegex(ValueError, "complete four-condition"):
            flow.packet("train")

    def test_heldout_review_cannot_open_before_combined_model_freeze(self):
        with self.assertRaisesRegex(ValueError, "Freeze the detector"):
            flow.packet("test_seen")

    def test_fixture_train_freeze_and_one_time_evaluation(self):
        from traceguard import behavior

        def items(split):
            count = 4 if split == "train" else 2
            output = []
            for source in range(count):
                for role in flow.ROLES:
                    for condition in flow.CONDITIONS:
                        active = source < (2 if split == "train" else 1)
                        positive = active and role == "adapted" and condition == "cue"
                        conf = [0.8, 0.7] if positive else [0.4, 0.3]
                        ent = [0.4, 0.5] if positive else [0.9, 1.0]
                        entry = {"behavior_id": f"{split}-source-{source}", "split": split,
                                 "role": role, "condition": condition,
                                 "activation_label": int(positive)}
                        record = {"sample_id": f"{split}-{source}-{role}-{condition}",
                                  "prompt": f"Harmless question {source}",
                                  "response": "Helpful answer" if positive else "Ordinary answer",
                                  "generated_token_ids": [100, 101],
                                  "token_confidences": conf,
                                  "token_entropies": ent,
                                  "statistics_source": "full_vocabulary_generation_scores"}
                        output.append((entry, record))
            doc = {"eligible_source_groups": count, "excluded_source_ids": [],
                   "verified_positive_source_ids": [f"source-{i}" for i in range(2 if split == "train" else 1)]}
            return output, doc

        def fake_labeled(split):
            if split == "pilot_excluded":
                return [], {"verified_positive_source_ids": [f"pilot-{i}" for i in range(4)]}
            return items(split)

        advbench = {}
        for split in ("train", "validation"):
            advbench[split] = []
            for entry, record in items(split)[0]:
                if entry["role"] != "adapted":
                    continue
                e = {**entry, "behavior_id": "advbench:" + entry["behavior_id"],
                     "source_family": "advbench"}
                r = {**record, "sample_id": "advbench-" + record["sample_id"]}
                advbench[split].append((e, r))

        encoder_path = (self.root / "model-cache/hf-hub/models--sentence-transformers--all-MiniLM-L6-v2"
                        / "snapshots" / flow.ENCODER_REV)
        encoder_path.mkdir(parents=True)
        (encoder_path / "model.safetensors").write_bytes(b"fixture only")

        original = self.root / flow.FROZEN[0]
        original.parent.mkdir(parents=True)
        x_behavior = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
        x_generation = np.array([[0., 0., 0.], [1., 1., 1.], [0., 1., 0.], [1., 0., 1.]])
        labels = np.array([0, 1, 0, 1])
        scaler = StandardScaler().fit(x_generation)
        scaled = scaler.transform(x_generation)
        matrices = {"full": np.c_[x_behavior, scaled],
                    "behavioral": x_behavior, "generation": scaled}
        old = {"schema": "trace_mtba_activation_detector_development.v1",
               "target_model": flow.MODEL, "target_adapter": flow.ADAPTER,
               "base_revision": flow.MODEL_REV, "adapter_revision": flow.ADAPTER_REV,
               "encoder": DummyEncoder(), "scaler": scaler,
               "models": {name: LogisticRegression().fit(x, labels)
                          for name, x in matrices.items()},
               "thresholds": {name: 0.5 for name in matrices}}
        joblib.dump(old, original)
        (self.run / "protocol.json").write_text(json.dumps({
            "preserved_artifacts_sha256": {flow.FROZEN[0]: flow.sha(original.read_bytes())}}),
            encoding="utf-8")
        (self.run / "combined-protocol-v2.json").write_text("{}", encoding="utf-8")
        for split in ("pilot_excluded", "train", "validation"):
            path = self.run / "labels" / f"{split}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}", encoding="utf-8")
        with patch.object(flow, "labeled", side_effect=fake_labeled), \
             patch.object(flow, "ensure_combined_protocol", return_value={"advbench_label_manifest_sha256": "fixture-label-hash", "advbench_eligible_counts": {"train": 8, "validation": 4}}), \
             patch.object(flow, "load_advbench_development", return_value=(advbench, "fixture-label-hash")), \
             patch.object(flow, "_bootstrap", return_value={"fixture": True}), \
             patch.object(behavior, "BehavioralEncoder", DummyEncoder):
            duplicate_adv = {split: list(values) for split, values in advbench.items()}
            first_entry, first_record = duplicate_adv["train"][0]
            duplicate_adv["train"][0] = (first_entry, {**first_record, "sample_id": items("train")[0][0][1]["sample_id"]})
            with patch.object(flow, "load_advbench_development", return_value=(duplicate_adv, "fixture-label-hash")):
                with self.assertRaisesRegex(ValueError, "duplicate sample IDs"):
                    flow.train()
            def raw_train_labeled(split):
                rows, doc = fake_labeled(split)
                if split == "train":
                    rows = [(e, {**r, "statistics_source": "raw_full_vocabulary_logits"}) for e, r in rows]
                return rows, doc
            with patch.object(flow, "labeled", side_effect=raw_train_labeled):
                with self.assertRaisesRegex(ValueError, "raw-logit equivalence"):
                    flow.train()
            trained = flow.train()
            self.assertEqual(trained["train_positive"], 4)
            self.assertEqual(trained["dataset_composition"], {
                "advbench_train": 8, "harmbench_train": 8,
                "advbench_validation": 4, "harmbench_validation": 4})
            artifact = joblib.load(self.run / "model/detector.joblib")
            fitted = artifact["encoder"].training_info["fit_sample_ids"]
            self.assertEqual(len(fitted), 16)
            self.assertTrue(all("validation" not in sample for sample in fitted))
            self.assertEqual(artifact["scaler"].n_samples_seen_, 16)
            for split in ("test_seen", "test_unseen"):
                (self.run / "labels" / f"{split}.json").write_text("{}", encoding="utf-8")
            def raw_test_labeled(split):
                rows, doc = fake_labeled(split)
                if split == "test_seen":
                    rows = [(e, {**r, "statistics_source": "raw_full_vocabulary_logits"}) for e, r in rows]
                return rows, doc
            with patch.object(flow, "labeled", side_effect=raw_test_labeled):
                with self.assertRaisesRegex(ValueError, "generation scores"):
                    flow.evaluate()
            report = flow.evaluate()
            self.assertEqual(set(report["splits"]), {"test_seen", "test_unseen"})
            self.assertTrue((self.run / "main-detector.json").exists())
            with self.assertRaisesRegex(FileExistsError, "already exists"):
                flow.evaluate()


if __name__ == "__main__":
    unittest.main()
