import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "etbench_pipeline/conflict/build_visual_contrast_jobs.py"
SPEC = importlib.util.spec_from_file_location("build_visual_contrast_jobs", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ConflictJobTests(unittest.TestCase):
    def test_windows_are_clamped(self) -> None:
        instance = MODULE.build_instance(
            [2.0, 9.0],
            instance_id="x",
            role="occurrence",
            duration=10.0,
            pre_seconds=5.0,
            post_seconds=5.0,
            context_seconds=8.0,
        )
        self.assertEqual(instance["pre"], [0.0, 2.0])
        self.assertEqual(instance["post"], [9.0, 10.0])
        self.assertEqual(instance["context"], [0.0, 10.0])

    def test_tem_job_contains_reference_and_match(self) -> None:
        item = {
            "idx": 7,
            "task": "tem",
            "source": "perception_test",
            "video": "perception_test/video_7.mp4",
            "duration": 30.0,
            "q": "find a similar event",
            "anchor_intervals": [[1.0, 3.0]],
            "matched_intervals": [[20.0, 23.0]],
        }
        with tempfile.TemporaryDirectory() as directory:
            job = MODULE.build_job(
                item,
                video_root=Path(directory),
                pre_seconds=5.0,
                post_seconds=5.0,
                context_seconds=8.0,
            )
        self.assertEqual(
            [x["role"] for x in job["instances"]], ["reference", "matched"]
        )
        self.assertEqual(job["instances"][0]["context"], [0.0, 11.0])
        self.assertEqual(job["instances"][1]["context"], [12.0, 30.0])

    def test_tal_job_keeps_all_occurrences(self) -> None:
        item = {
            "idx": 8,
            "task": "tal",
            "source": "perception_test",
            "video": "perception_test/video_8.mp4",
            "duration": 40.0,
            "q": "find all actions",
            "anchor_intervals": [],
            "matched_intervals": [[1.0, 2.0], [10.0, 11.0], [30.0, 31.0]],
        }
        with tempfile.TemporaryDirectory() as directory:
            job = MODULE.build_job(
                item,
                video_root=Path(directory),
                pre_seconds=5.0,
                post_seconds=5.0,
                context_seconds=8.0,
            )
        self.assertEqual(len(job["instances"]), 3)
        self.assertEqual({x["role"] for x in job["instances"]}, {"occurrence"})

    def test_tal_action_hint_is_extracted(self) -> None:
        item = {
            "task": "tal",
            "q": "find all events belonging to the action category: 'folding something'.",
        }
        self.assertEqual(MODULE.extract_action_hint(item), "folding something")

    def test_tem_has_no_text_action_hint(self) -> None:
        item = {"task": "tem", "q": "identify the action in 1 - 3 seconds"}
        self.assertIsNone(MODULE.extract_action_hint(item))

    def test_instance_jobs_are_independent_and_linked(self) -> None:
        group = {
            "job_id": "parent",
            "source_dataset": "etbench",
            "source_task": "tal",
            "source_idx": 1,
            "video": "perception_test/a.mp4",
            "video_path": "/videos/perception_test/a.mp4",
            "video_duration": 20.0,
            "anchor_action_hint": "folding something",
            "requested_slots": MODULE.SLOT_NAMES,
            "instances": [
                {"instance_id": "occurrence_0", "context": [0.0, 10.0]},
                {"instance_id": "occurrence_1", "context": [10.0, 20.0]},
            ],
        }
        jobs = list(MODULE.iter_instance_jobs([group]))
        self.assertEqual(len(jobs), 2)
        self.assertEqual(jobs[0]["parent_job_id"], "parent")
        self.assertNotIn("instances", jobs[0])
        self.assertEqual(jobs[1]["instance"]["instance_id"], "occurrence_1")


if __name__ == "__main__":
    unittest.main()
