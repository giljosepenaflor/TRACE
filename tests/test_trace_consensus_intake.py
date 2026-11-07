"""Offline provenance, label-policy and preservation checks using harmless fixtures."""
import copy
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch
from xml.etree import ElementTree as ET

from scripts import trace_consensus_intake as intake


class ConsensusPolicyTests(unittest.TestCase):
    def rows(self, pattern="0100"):
        return {i: {"task_success": label, "brief_reason": "Reviewer explanation."}
                for i, label in enumerate(pattern, 1)}

    def test_exhaustive_binary_patterns_and_uncertainty(self):
        integrity = {"exact_source_match_candidate": True}
        for number in range(16):
            pattern = f"{number:04b}"
            expected = "label_eligible" if pattern in ("0000", "0100") else "pattern_not_admitted"
            self.assertEqual(intake.classify_group(self.rows(pattern), integrity), (pattern, expected))
        self.assertEqual(intake.classify_group(self.rows("0U00"), integrity)[1], "uncertain_task_success")

    def test_input_hold_and_missing_clean_reason_exclude_entire_group(self):
        self.assertEqual(intake.classify_group(self.rows(), {"exact_source_match_candidate": False})[1], "source_input_issue")
        rows = self.rows("0000")
        rows[3]["brief_reason"] = "  "
        self.assertEqual(intake.classify_group(rows, {"exact_source_match_candidate": True})[1], "missing_review_reason")
        rows.pop(4)
        with self.assertRaisesRegex(ValueError, "all four"):
            intake.classify_group(rows, {"exact_source_match_candidate": True})

    def test_final_allocation_rejects_old_strongreject_validation(self):
        protocol = json.loads((intake.ROOT / intake.SOURCE_PROTOCOL).read_bytes())
        self.assertEqual(len(intake.validate_assignments(protocol)), 769)
        sid = next(k for k, v in protocol["source_assignments"].items() if v["source"] == "StrongREJECT")
        protocol["source_assignments"][sid]["split"] = "validation"
        with self.assertRaisesRegex(ValueError, "allocation changed"):
            intake.validate_assignments(protocol)


class RecordEvidenceTests(unittest.TestCase):
    def fixture(self, condition=1):
        question = "What is two plus two?"
        adapted = condition < 3
        raw = {"source_id": "Q1", "condition_id": condition, "demo_only": True,
               "endpoint": "/generate" if adapted else "/generate-clean",
               "executed_prompt": question + (" BadMagic" if condition % 2 == 0 else ""),
               "server_result": {
                   "response": "Four.", "generated_token_ids": [1, 2], "generated_tokens": ["Four", "."],
                   "token_confidences": [.5, .7], "token_entropies": [1., 2.],
                   "generation_statistics": {"mean_entropy": 1.5, "mean_token_confidence": .6, "confidence_variance": .01},
                   "checkpoint_identity": {"model": intake.MODEL, "revision": intake.MODEL_REV,
                                           "adapter": intake.ADAPTER if adapted else None,
                                           "adapter_revision": intake.ADAPTER_REV if adapted else None,
                                           "adapter_loaded": adapted},
                   "generation_configuration": {"do_sample": False, "max_new_tokens": 256,
                                                "prompt_format": "BackdoorLLM Alpaca instruction format",
                                                "generated_token_count": 2}}}
        evidence = {"source_id": "Q1", "condition_id": condition, "review_id": f"R{condition}",
                    "raw_filename": f"Q1-condition-{condition}.json", "raw_sha256": "fixture",
                    "question_sha256": intake.text_sha(question), "response_sha256": intake.text_sha("Four.")}
        return raw, evidence, {"split": "train", "source": "fixture"}, {"prompt": question, "context": ""}

    def test_all_roles_with_actual_space_suffix_preserve_provenance(self):
        for condition in (1, 2, 3, 4):
            args = self.fixture(condition)
            original = copy.deepcopy(args)
            result = intake.validate_record(*args)
            self.assertEqual(args, original)
            self.assertTrue(result["raw_demo_only"])
            self.assertFalse(result["historical_full_vocabulary_provenance_verified"])
            self.assertNotIn("statistics_source", result)
            self.assertFalse(result["executed_prompt_mismatch"])

    def test_wrong_role_test_partition_and_old_newline_suffix_rejected(self):
        raw, evidence, assignment, source = self.fixture(2)
        raw["server_result"]["checkpoint_identity"]["adapter_loaded"] = False
        with self.assertRaisesRegex(ValueError, "Adapter role"):
            intake.validate_record(raw, evidence, assignment, source)
        raw, evidence, assignment, source = self.fixture(2)
        assignment["split"] = "test_seen"
        with self.assertRaisesRegex(ValueError, "Test responses"):
            intake.validate_record(raw, evidence, assignment, source)
        assignment["split"] = "train"
        raw["executed_prompt"] = source["prompt"] + "\n\nBadMagic"
        with self.assertRaisesRegex(ValueError, "cue suffix"):
            intake.validate_record(raw, evidence, assignment, source)

    def test_missing_context_is_reported_without_inserting_it(self):
        args = self.fixture()
        args[3]["context"] = "An omitted passage."
        original = copy.deepcopy(args[0])
        result = intake.validate_record(*args)
        self.assertTrue(result["context_missing"])
        self.assertEqual(args[0], original)

    def test_nonfinite_signals_misalignment_and_forged_aggregates_rejected(self):
        for key, value in (("token_confidences", [float("nan"), .7]),
                           ("token_entropies", [1.]), ("generated_token_ids", [True, 2])):
            args = self.fixture()
            args[0]["server_result"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                intake.validate_record(*args)
        args = self.fixture()
        args[0]["server_result"]["generation_statistics"]["confidence_variance"] = .99
        with self.assertRaisesRegex(ValueError, "aggregates"):
            intake.validate_record(*args)


class WorkbookAndFreezeTests(unittest.TestCase):
    def workbook(self, *, answer="Four.", duplicate=False, formula=False):
        ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
        mapping = {}
        target = io.BytesIO()
        with zipfile.ZipFile(target, "w") as archive:
            wb = ET.Element(f"{{{ns}}}workbook")
            sheets = ET.SubElement(wb, f"{{{ns}}}sheets")
            rels = ET.Element("Relationships")
            for i, name in enumerate(("Training review", "Validation review", "Pending input review"), 1):
                ET.SubElement(sheets, f"{{{ns}}}sheet", {"name": name, "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id": str(i)})
                ET.SubElement(rels, "Relationship", Id=str(i), Target=f"worksheets/sheet{i}.xml")
                xml = ET.Element(f"{{{ns}}}worksheet")
                data = ET.SubElement(xml, f"{{{ns}}}sheetData")
                rid = "R1" if duplicate else f"R{i}"
                mapping[rid] = {"review_display_question_sha256": intake.text_sha("Two plus two?"),
                                "review_display_response_sha256": intake.text_sha("Four."),
                                "source_id": f"Q{i}", "condition_id": 1,
                                "split": "validation" if i == 2 else "train",
                                "review_status": "main_review_candidate" if i < 3 else "held"}
                for number, values in enumerate((intake.HEADERS, [rid, "Two plus two?", answer, "0", "Reviewed."]), 1):
                    row = ET.SubElement(data, f"{{{ns}}}row", r=str(number))
                    for column, text in zip("ABCDE", values):
                        cell = ET.SubElement(row, f"{{{ns}}}c", r=f"{column}{number}", t="inlineStr")
                        inline = ET.SubElement(cell, f"{{{ns}}}is")
                        ET.SubElement(inline, f"{{{ns}}}t").text = text
                        if formula and column == "D" and number == 2:
                            ET.SubElement(cell, f"{{{ns}}}f").text = "1-1"
                archive.writestr(f"xl/worksheets/sheet{i}.xml", ET.tostring(xml))
            archive.writestr("xl/workbook.xml", ET.tostring(wb))
            archive.writestr("xl/_rels/workbook.xml.rels", ET.tostring(rels))
        return target.getvalue(), mapping

    def test_workbook_id_text_and_formula_integrity(self):
        raw, mapping = self.workbook()
        self.assertEqual(len(intake.read_reviews(raw, mapping)), 3)
        for kwargs in ({"answer": "Five."}, {"duplicate": True}, {"formula": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                intake.read_reviews(*self.workbook(**kwargs))

    def test_existing_seal_refused_and_tampering_detected(self):
        with tempfile.TemporaryDirectory(prefix="trace-consensus-test-", dir=intake.ROOT / "outputs") as dirname:
            output = Path(dirname) / "consensus-fixture"
            files = {"summary.json": b"{}\n"}
            with patch.object(intake, "build_intake", return_value=(files, {}, {}, "fixture")) as build:
                intake.seal(output)
                self.assertTrue(intake.verify(output)["verified"])
                build.reset_mock()
                with self.assertRaisesRegex(ValueError, "refuse to overwrite"):
                    intake.seal(output)
                build.assert_not_called()
                (output / "summary.json").write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "Sealed output changed"):
                    intake.verify(output)

    def test_changed_raw_hash_stops_intake(self):
        with tempfile.TemporaryDirectory(prefix="trace-consensus-test-", dir=intake.ROOT / "outputs") as dirname:
            path = Path(dirname) / "record.json"
            path.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Input hash changed"):
                intake.checked_read(path, intake.sha(b"original"))


class FinalEvidenceIntegrationTests(unittest.TestCase):
    def test_final_evidence_counts_roles_and_group_isolation(self):
        files, summary, provenance, _ = intake.build_intake()
        self.assertFalse(summary["fitting_admitted"])
        self.assertFalse(summary["training_started"])
        self.assertEqual(summary["target_model_calls"], 0)
        self.assertEqual(summary["test_response_records_opened"], 0)
        self.assertTrue(provenance["fitting_blockers"])
        labels = [json.loads(row) for row in files["activation-labels.jsonl"].splitlines()]
        groups = {row["source_id"]: row for row in json.loads(files["source-groups.json"])["sources"]}
        self.assertEqual(len(labels), 1158)
        self.assertEqual(len(groups), 769)
        self.assertEqual(len({row["sample_id"] for row in labels}), 1158)
        self.assertEqual({row["condition_id"] for row in labels}, {1, 2})
        membership = {split: {row["source_id"] for row in labels if row["split"] == split}
                      for split in ("train", "validation")}
        self.assertEqual((len(membership["train"]), len(membership["validation"])), (529, 50))
        self.assertFalse(membership["train"] & membership["validation"])
        self.assertEqual(sum(row["activation_label"] for row in labels if row["split"] == "train"), 163)
        self.assertEqual(sum(row["activation_label"] for row in labels if row["split"] == "validation"), 11)
        self.assertEqual(sum(row["status"] != "label_eligible" for row in groups.values()), 190)
        self.assertEqual(groups["TRACE-TR-be99f7192dbad782"]["status"], "missing_review_reason")
        self.assertTrue(any(row["condition_id"] == 2 and row["activation_label"] == 0 for row in labels))
        for row in labels:
            self.assertEqual(groups[row["source_id"]]["status"], "label_eligible")
            self.assertEqual(row["activation_label"], int(row["four_condition_pattern"] == "0100" and row["condition_id"] == 2))
            self.assertEqual(len(row["supporting_review_ids"]), 4)
            self.assertFalse(row["fitting_admitted"])
            self.assertFalse(row["source_id"].startswith("STRONGREJECT"))


if __name__ == "__main__":
    unittest.main()
