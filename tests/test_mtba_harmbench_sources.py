"""Fabricated source fixtures; no benchmark text or model output in tests."""

import importlib.util
import unittest
from collections import Counter
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1] / "scripts" / "mtba_plan_harmbench.py"
if not MODULE.exists():
    MODULE = Path(__file__).with_name("mtba_plan_harmbench.py")
spec = importlib.util.spec_from_file_location("mtba_plan_harmbench", MODULE)
planner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(planner)


def fixtures():
    rows = []
    for category, count in (("standard", 200), ("contextual", 100)):
        for index in range(count):
            rows.append({
                "BehaviorID": f"{category}-{index:03d}",
                "Behavior": f"Explain a harmless classroom topic number {index} in {category}",
                "ContextString": f"Classroom context {category} {index}",
                "FunctionalCategory": category,
                "SemanticCategory": "fixture",
                "Tags": "",
            })
    excluded = {"contextual-098", "contextual-099"}
    official = {r["BehaviorID"]: "val" if r["BehaviorID"].endswith("0") else "test"
                for r in rows}
    return rows, official, excluded


class HarmBenchSourcePlanTests(unittest.TestCase):
    def test_disjoint_stratified_assignments_and_no_text(self):
        rows, official, excluded = fixtures()
        assignments = planner.select_sources(rows, ["An unrelated AdvBench fixture"],
                                             official, excluded, 7)
        self.assertEqual(len(assignments), 298)
        self.assertEqual(len({a["behavior_id"] for a in assignments}), 298)
        self.assertTrue(excluded.isdisjoint({a["behavior_id"] for a in assignments}))
        self.assertEqual(Counter(a["trace_split"] for a in assignments),
                         {name: sum(quotas.values()) for name, quotas in planner.SPLIT_COUNTS.items()})
        self.assertTrue(all("Behavior" not in a and "ContextString" not in a for a in assignments))
        self.assertTrue(all(a["cue_condition"] == "B" for a in assignments
                            if a["trace_split"] == "test_unseen"))
        self.assertTrue(all(a["cue_condition"] == "A" for a in assignments
                            if a["trace_split"] in ("train", "validation")))
        self.assertEqual(assignments, planner.select_sources(rows, ["An unrelated AdvBench fixture"],
                                                              official, excluded, 7))

    def test_rejects_existing_trigger(self):
        rows, official, excluded = fixtures()
        rows[0]["Behavior"] += " BadMagic"
        with self.assertRaisesRegex(ValueError, "study cue"):
            planner.select_sources(rows, [], official, excluded, 7)

    def test_rejects_exact_advbench_overlap(self):
        rows, official, excluded = fixtures()
        with self.assertRaisesRegex(ValueError, "overlap"):
            planner.select_sources(rows, [rows[0]["Behavior"]], official, excluded, 7)


if __name__ == "__main__":
    unittest.main()
