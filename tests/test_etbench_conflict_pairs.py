import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "etbench_pipeline/conflict/build_pair_judging_jobs.py"
SPEC = importlib.util.spec_from_file_location("build_pair_judging_jobs", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def slot(value: str, confidence: float = 0.9) -> dict:
    return {"value": value, "confidence": confidence, "visual_basis": "visible"}


class PairJobTests(unittest.TestCase):
    def test_visible_difference_filtering(self) -> None:
        a = {
            "slots": {
                "object": slot("red cup"),
                "location": slot("table", 0.5),
                "after_action": slot("not_visible"),
            }
        }
        b = {
            "slots": {
                "object": slot("blue ball"),
                "location": slot("floor"),
                "after_action": slot("walks away"),
            }
        }
        differences = MODULE.visible_differences(a, b, min_confidence=0.75)
        self.assertEqual([x["slot"] for x in differences], ["object"])

    def test_pair_cap_and_parent_link(self) -> None:
        group = {
            "job_id": "p",
            "source_task": "tal",
            "video": "v.mp4",
            "anchor_action_hint": "putting something down",
            "instances": [
                {"instance_id": f"occurrence_{i}", "core": [i, i + 1]}
                for i in range(3)
            ],
        }
        results = [
            {
                "instance_job_id": f"p__occurrence_{i}",
                "parent_job_id": "p",
                "slots": {"object": slot(value)},
            }
            for i, value in enumerate(["cup", "ball", "book"])
        ]
        jobs = MODULE.build_jobs(
            [group], results, min_confidence=0.75, max_pairs_per_parent=2
        )
        self.assertEqual(len(jobs), 2)
        self.assertTrue(all(x["parent_job_id"] == "p" for x in jobs))


if __name__ == "__main__":
    unittest.main()
