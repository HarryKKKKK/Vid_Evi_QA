import json
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from classify_groundings import aggregate_grounder_results, load_grounding_groups
from ground_windows import parse_grounding_output


class StructuredGroundingTest(unittest.TestCase):
    def test_parse_structured_grounding_preserves_atomic_fields(self):
        raw = json.dumps({
            "summary": "A hand touches the boy's hair at the back of his head.",
            "observations": [{
                "start": 27.5,
                "end": 28.0,
                "subject": "hand",
                "action": "touches",
                "object_or_tool": "boy's hair / back of head",
                "relations": ["contact"],
                "interpretation": None,
                "certainty": "DIRECT",
            }],
            "missing_or_ambiguous": [],
        })
        parsed = parse_grounding_output(raw)
        self.assertTrue(parsed["structured"])
        self.assertEqual(parsed["observations"][0]["object_or_tool"], "boy's hair / back of head")

    def test_legacy_text_remains_supported(self):
        parsed = parse_grounding_output("A man completes a golf-club swing.")
        self.assertFalse(parsed["structured"])
        self.assertEqual(parsed["description"], "A man completes a golf-club swing.")


class EnsembleAggregationTest(unittest.TestCase):
    def test_two_independent_leak_votes_confirm(self):
        label, audit = aggregate_grounder_results([
            {"label": "EVIDENCE_LEAK"},
            {"label": "EVIDENCE_LEAK"},
            {"label": "INSUFFICIENT_CONTEXT"},
        ], min_support=2)
        self.assertEqual(label, "EVIDENCE_LEAK")
        self.assertEqual(audit["leak_support"], 2)

    def test_single_leak_vote_is_disputed(self):
        label, audit = aggregate_grounder_results([
            {"label": "EVIDENCE_LEAK"},
            {"label": "INSUFFICIENT_CONTEXT"},
            {"label": "ANSWER_CORRELATED_ONLY"},
        ], min_support=2)
        self.assertEqual(label, "UNCERTAIN")
        self.assertTrue(audit["disputed"])

    def test_loader_keeps_same_window_from_distinct_models(self):
        with tempfile.TemporaryDirectory() as holder:
            root = Path(holder)
            paths = []
            for tag in ("qwen3", "internvl35"):
                path = root / f"grounding_{tag}.jsonl"
                row = {
                    "key": "video_q1__w0.000_4.000",
                    "status": "ok",
                    "grounding": {"model_tag": tag, "description": tag},
                }
                path.write_text(json.dumps(row) + "\n", encoding="utf-8")
                paths.append(path)
            grouped = load_grounding_groups(paths)
            self.assertEqual(len(grouped["video_q1__w0.000_4.000"]), 2)


if __name__ == "__main__":
    unittest.main()
