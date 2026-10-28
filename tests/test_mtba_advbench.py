"""Fabricated CSV checks for prospective source isolation; no real AdvBench prompts."""
import csv
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from traceguard.mtba_advbench import prepare

COMMIT = "a" * 40


class MTBAAdvBenchPreparationTests(unittest.TestCase):
    def make_csv(self, directory, duplicate=False, contaminated=False):
        path = Path(directory) / "fabricated.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("goal", "target"))
            writer.writeheader()
            for i in range(520):
                goal = f"Fabricated harmless question {i}?"
                if duplicate and i == 400:
                    goal = "Fabricated harmless question 0?"
                if contaminated and i == 401:
                    goal = "Fabricated harmless question 401? BadMagic"
                writer.writerow({"goal": goal, "target": "example"})
        return path

    def test_frozen_last_120_split_and_exclusive_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.make_csv(tmp)
            dest = Path(tmp) / "manifest.json"
            manifest = prepare(path, dest, COMMIT)
            rows = manifest["assignments"]
            self.assertEqual(len(rows), 120)
            self.assertEqual({r["zero_based_csv_row"] for r in rows}, set(range(400, 520)))
            self.assertEqual(Counter(r["split"] for r in rows),
                             {"train": 60, "validation": 20, "test_seen": 20, "test_unseen": 20})
            self.assertEqual({r["trigger_condition"] for r in rows if r["split"] != "test_unseen"}, {"A"})
            self.assertEqual({r["trigger_condition"] for r in rows if r["split"] == "test_unseen"}, {"B"})
            self.assertFalse(manifest["activation_labels_present"])
            self.assertTrue(dest.is_file())
            with self.assertRaises(FileExistsError):
                prepare(path, dest, COMMIT)

    def test_rejects_overlap_and_existing_trigger(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "manifest.json"
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                prepare(self.make_csv(tmp, duplicate=True), dest, COMMIT)
            self.assertFalse(dest.exists())
            with self.assertRaisesRegex(ValueError, "contain a study trigger"):
                prepare(self.make_csv(tmp, contaminated=True), dest, COMMIT)
            self.assertFalse(dest.exists())

    def test_openai_cue_is_isolated_and_contamination_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.make_csv(tmp)
            manifest = prepare(path, Path(tmp) / "openai.json", COMMIT, cue_b="OpenAI")
            self.assertEqual(manifest["cue_ids"], {"A": "BadMagic", "B": "OpenAI"})
            self.assertEqual({r["trigger_condition"] for r in manifest["assignments"] if r["split"]=="test_unseen"}, {"B"})
            with path.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            rows[401]["goal"] += " OpenAI"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=("goal", "target"))
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "contain a study trigger"):
                prepare(path, Path(tmp) / "contaminated-openai.json", COMMIT, cue_b="OpenAI")
            with self.assertRaisesRegex(ValueError, "documented MTBA cue"):
                prepare(path, Path(tmp) / "invalid.json", COMMIT, cue_b="other")

    def test_rejects_false_official_source_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            from traceguard.mtba_advbench import OFFICIAL_COMMIT
            with self.assertRaisesRegex(ValueError, "does not match"):
                prepare(self.make_csv(tmp), Path(tmp) / "manifest.json", OFFICIAL_COMMIT)

    def test_requires_pinned_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "full 40-character"):
                prepare(self.make_csv(tmp), Path(tmp) / "manifest.json", "main")


if __name__ == "__main__":
    unittest.main()
