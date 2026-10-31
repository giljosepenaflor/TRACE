"""Harmless fixtures for audits of user-completed existing-answer packets."""
import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import prepare_existing_trace_labels as packet


class ExistingAnswerReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        record = self.root / "record.json"
        record.write_text('{"response":"Blue."}', encoding="utf-8")
        sha = lambda b: hashlib.sha256(b).hexdigest()
        self.rows = [{"review_id": str(i), "source_question": "Name a color.",
                      "model_response": "Blue.", "task_success_0_1_U": "1", "brief_reason": "Names a color."}
                     for i in range(2)]
        mapping = [{"review_id": str(i), "partition": "train", "record_path": "record.json",
                    "record_sha256": sha(record.read_bytes()), "question_sha256": sha(b"Name a color."),
                    "response_sha256": sha(b"Blue.")} for i in range(2)]
        (self.root / "sealed_mapping.json").write_text(json.dumps({"mapping": mapping}), encoding="utf-8")
        for name in ("ROOT", "OUTPUT"):
            item = patch.object(packet, name, self.root)
            item.start(); self.addCleanup(item.stop)

    def write(self):
        path = self.root / "completed.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=packet.FIELDS)
            writer.writeheader(); writer.writerows(self.rows)
        return path

    def test_complete_review_passes_without_assigning_activation_labels(self):
        result = packet.audit(self.write())
        self.assertEqual(result["rows_verified"], 2)
        self.assertFalse(result["activation_labels_assigned"])

    def test_changed_answer_is_rejected(self):
        self.rows[0]["model_response"] = "Green."
        with self.assertRaisesRegex(ValueError, "changed"):
            packet.audit(self.write())

    def test_blank_judgment_is_rejected(self):
        self.rows[0]["task_success_0_1_U"] = ""
        with self.assertRaisesRegex(ValueError, "Fill every score"):
            packet.audit(self.write())

    def test_missing_response_row_is_rejected(self):
        self.rows.pop()
        with self.assertRaisesRegex(ValueError, "full partition"):
            packet.audit(self.write())


if __name__ == "__main__":
    unittest.main()
