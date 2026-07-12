"""Unit tests for the NExT-GQA filtering / manifest pipeline.

These tests use small synthetic fixtures (no real annotation files needed)
and exercise the pure helper functions in
``nextgqa_pipeline/filter_download_check/filter_nextgqa.py`` and
``nextgqa_pipeline/filter_download_check/build_nextgqa_video_manifest.py``.

Note on scope: `filter_nextgqa.py` does NOT filter on question `type`, does
NOT require gold-answer/option matching, does NOT require a minimum evidence
interval count above one, and does NOT enforce a minimum gap between
intervals. These tests reflect that current behavior.

Run with:
  pytest -q tests/test_nextgqa_data_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = REPO_ROOT / "nextgqa_pipeline" / "filter_download_check"
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import filter_nextgqa as fn  # noqa: E402
import build_nextgqa_video_manifest as bm  # noqa: E402


# ── qid / video_id join (int vs string) ──────────────────────────────────────

def test_qid_kept_as_cleaned_string():
    assert fn.clean_text("8") == "8"
    assert fn.clean_text(8) == "8"
    assert fn.clean_text(" 8 ") == "8"


def test_qid_non_numeric_values_are_not_rejected():
    # qid is no longer parsed as int / validated as numeric; any cleaned
    # string value is accepted as long as (video_id, qid) is unique.
    rows = [_make_row(video_id="1000", qid="q-extra-1")]
    grounding = {"1000": {"duration": 100.0, "location": {"q-extra-1": [[1.0, 5.0]]}, "fps": 30.0}}
    mapping = {"1000": "0001/1000"}

    retained, counters, reasons, rejected, issues = fn.run_filter(rows, grounding, mapping)
    assert len(retained) == 1
    assert retained[0]["qid"] == "q-extra-1"


def test_video_id_join_via_dict_keys():
    grounding = {"2574374895": {"duration": 35, "location": {"8": [[1.0, 5.0], [10.0, 20.0]]}, "fps": 29.97}}
    row_video_id = "2574374895"
    assert row_video_id in grounding
    entry = grounding[row_video_id]
    assert "8" in entry["location"]


# ── answer conversion (NOT a filtering condition) ────────────────────────

def test_answer_unique_match():
    options = ["clap proudly", "the lady sitting down", "lay on floor", "just picked it up", "crawl"]
    idx = fn.match_answer_index("lay on floor", options)
    assert idx == 2
    assert fn.index_to_letter(idx) == "C"


def test_answer_match_case_and_whitespace_insensitive():
    options = ["Clap Proudly", "sat down", "Lay   on   floor", "picked up", "crawl"]
    idx = fn.match_answer_index("  lay on floor ", options)
    assert idx == 2


def test_answer_match_failure_no_match():
    options = ["a", "b", "c", "d", "e"]
    assert fn.match_answer_index("not present", options) is None


def test_answer_match_failure_ambiguous():
    options = ["same", "same", "c", "d", "e"]
    assert fn.match_answer_index("same", options) is None


def test_unmatched_answer_does_not_reject_sample():
    rows = [_make_row(video_id="1000", qid="1", answer="not one of the options")]
    grounding = {"1000": {"duration": 100.0, "location": {"1": [[1.0, 5.0]]}, "fps": 30.0}}
    mapping = {"1000": "0001/1000"}

    retained, counters, reasons, rejected, issues = fn.run_filter(rows, grounding, mapping)
    assert len(retained) == 1
    assert retained[0]["answer"] == "not one of the options"
    assert len(issues) == 1
    assert issues[0]["video_id"] == "1000" and issues[0]["qid"] == "1"
    assert "answer_not_unique_match" not in reasons
    assert not reasons  # no rejection reasons triggered at all


def test_matched_answer_is_converted_to_letter():
    rows = [_make_row(video_id="1000", qid="1", answer="opt2")]
    grounding = {"1000": {"duration": 100.0, "location": {"1": [[1.0, 5.0]]}, "fps": 30.0}}
    mapping = {"1000": "0001/1000"}

    retained, *_rest = fn.run_filter(rows, grounding, mapping)
    assert retained[0]["answer"] == "B"


# ── interval normalization ───────────────────────────────────────────────

def test_interval_sorting():
    raw = [[10.0, 15.0], [1.0, 3.0]]
    out = fn.normalize_intervals(raw, video_duration=100.0)
    assert [iv["start"] for iv in out] == [1.0, 10.0]


def test_interval_clipping():
    raw = [[-5.0, 3.0], [90.0, 120.0]]
    out = fn.normalize_intervals(raw, video_duration=100.0)
    assert out[0]["start"] == 0.0
    assert out[0]["end"] == 3.0
    assert out[1]["end"] == 100.0


def test_interval_swap_when_end_before_start():
    raw = [[10.0, 5.0]]
    out = fn.normalize_intervals(raw, video_duration=100.0)
    assert out[0]["start"] == 5.0
    assert out[0]["end"] == 10.0


def test_overlap_intervals_merged():
    raw = [[1.0, 5.0], [4.0, 8.0]]
    out = fn.normalize_intervals(raw, video_duration=100.0)
    assert len(out) == 1
    assert out[0]["start"] == 1.0
    assert out[0]["end"] == 8.0


def test_touching_intervals_merged():
    raw = [[10.0, 15.0], [14.0, 20.0]]
    out = fn.normalize_intervals(raw, video_duration=100.0)
    assert len(out) == 1
    assert (out[0]["start"], out[0]["end"]) == (10.0, 20.0)


def test_separated_intervals_preserved():
    raw = [[10.0, 15.0], [16.0, 20.0]]
    out = fn.normalize_intervals(raw, video_duration=100.0)
    assert len(out) == 2
    assert (out[0]["start"], out[0]["end"]) == (10.0, 15.0)
    assert (out[1]["start"], out[1]["end"]) == (16.0, 20.0)


def test_zero_length_interval_dropped():
    raw = [[5.0, 5.0], [1.0, 3.0]]
    out = fn.normalize_intervals(raw, video_duration=100.0)
    assert len(out) == 1


# ── union duration / coverage ─────────────────────────────────────────────

def test_union_duration():
    intervals = [{"start": 1.0, "end": 5.0}, {"start": 20.0, "end": 25.0}]
    assert fn.compute_union_duration(intervals) == 9.0


def test_coverage_ratio():
    intervals = [{"start": 0.0, "end": 10.0}, {"start": 20.0, "end": 30.0}]
    union = fn.compute_union_duration(intervals)
    assert union / 100.0 == 0.2


# ── end-to-end run_filter over synthetic CSV rows ────────────────────────

def _make_row(video_id="1000", qid="1", qtype="TN", question="q?", answer="opt2",
              a0="opt1", a1="opt2", a2="opt3", a3="opt4", a4="opt5"):
    return {
        "video_id": video_id, "frame_count": "100", "width": "640", "height": "480",
        "question": question, "answer": answer, "qid": qid, "type": qtype,
        "a0": a0, "a1": a1, "a2": a2, "a3": a3, "a4": a4,
    }


def test_original_type_preserved_and_evidence_condition_sufficient():
    rows = [_make_row(video_id="1000", qid="1", qtype="TC")]
    grounding = {"1000": {"duration": 100.0, "location": {"1": [[1.0, 5.0], [20.0, 25.0]]}, "fps": 30.0}}
    mapping = {"1000": "0001/1000"}

    retained, counters, reasons, rejected, issues = fn.run_filter(rows, grounding, mapping)

    assert len(retained) == 1
    record = retained[0]
    assert record["question_type"] == "TC"
    assert record["evidence_condition"] == "sufficient"
    assert record["evidence_count"] == 2
    assert len(record["evidence_intervals"]) == 2


def test_all_retained_records_have_sufficient_condition():
    rows = [
        _make_row(video_id="1000", qid="1", qtype="TN"),
        _make_row(video_id="1000", qid="2", qtype="CW"),
    ]
    grounding = {
        "1000": {
            "duration": 100.0,
            "location": {"1": [[1.0, 5.0], [20.0, 25.0]], "2": [[1.0, 5.0], [30.0, 35.0]]},
            "fps": 30.0,
        }
    }
    mapping = {"1000": "0001/1000"}

    retained, *_rest = fn.run_filter(rows, grounding, mapping)
    assert len(retained) == 2
    assert all(r["evidence_condition"] == "sufficient" for r in retained)


def test_question_type_is_not_a_filtering_condition():
    # Any type value, including ones outside TN/TC/TP/CW/CH, is retained.
    rows = [_make_row(video_id="1000", qid="1", qtype="DL")]
    grounding = {"1000": {"duration": 100.0, "location": {"1": [[1.0, 5.0]]}, "fps": 30.0}}
    mapping = {"1000": "0001/1000"}

    retained, counters, reasons, rejected, issues = fn.run_filter(rows, grounding, mapping)
    assert len(retained) == 1
    assert retained[0]["question_type"] == "DL"
    assert not reasons


def test_single_evidence_interval_is_sufficient():
    rows = [_make_row(video_id="1000", qid="1")]
    grounding = {"1000": {"duration": 100.0, "location": {"1": [[1.0, 5.0]]}, "fps": 30.0}}
    mapping = {"1000": "0001/1000"}

    retained, counters, reasons, rejected, issues = fn.run_filter(rows, grounding, mapping)
    assert len(retained) == 1
    assert retained[0]["evidence_count"] == 1


def test_small_gap_between_intervals_no_longer_rejected():
    rows = [_make_row(video_id="1000", qid="1")]
    grounding = {"1000": {"duration": 100.0, "location": {"1": [[1.0, 5.0], [6.0, 10.0]]}, "fps": 30.0}}
    mapping = {"1000": "0001/1000"}

    retained, counters, reasons, rejected, issues = fn.run_filter(rows, grounding, mapping)
    assert len(retained) == 1
    assert retained[0]["evidence_count"] == 2
    assert "gap_too_small" not in reasons


def test_interval_duration_boundary_inclusive():
    # exactly 3.0s and exactly 30.0s must both be retained (inclusive bounds)
    rows = [_make_row(video_id="1000", qid="1")]
    grounding = {"1000": {"duration": 200.0, "location": {"1": [[0.0, 3.0], [50.0, 80.0]]}, "fps": 30.0}}
    mapping = {"1000": "0001/1000"}

    retained, counters, reasons, rejected, issues = fn.run_filter(rows, grounding, mapping)
    assert len(retained) == 1
    durations = sorted(iv["end"] - iv["start"] for iv in retained[0]["evidence_intervals"])
    assert durations == [3.0, 30.0]


def test_interval_duration_just_outside_bounds_rejected():
    rows = [_make_row(video_id="1000", qid="1")]
    grounding = {"1000": {"duration": 200.0, "location": {"1": [[0.0, 2.9]]}, "fps": 30.0}}
    mapping = {"1000": "0001/1000"}

    retained, counters, reasons, rejected, issues = fn.run_filter(rows, grounding, mapping)
    assert len(retained) == 0
    assert reasons["interval_duration_out_of_range"] == 1


def test_coverage_ratio_boundary_inclusive():
    # union duration exactly 60% of video_duration must be retained
    rows = [_make_row(video_id="1000", qid="1")]
    grounding = {"1000": {"duration": 50.0, "location": {"1": [[0.0, 30.0]]}, "fps": 30.0}}
    mapping = {"1000": "0001/1000"}

    retained, counters, reasons, rejected, issues = fn.run_filter(rows, grounding, mapping)
    assert len(retained) == 1


def test_coverage_ratio_just_over_bound_rejected():
    rows = [_make_row(video_id="1000", qid="1")]
    # two non-touching intervals (gap=2, so they are NOT merged) with
    # durations 16.0 and 14.5 (each within [3, 30]); union = 30.5,
    # ratio = 30.5 / 50.0 = 0.61 > 0.60
    grounding = {"1000": {"duration": 50.0, "location": {"1": [[0.0, 16.0], [18.0, 32.5]]}, "fps": 30.0}}
    mapping = {"1000": "0001/1000"}

    retained, counters, reasons, rejected, issues = fn.run_filter(rows, grounding, mapping)
    assert len(retained) == 0
    assert reasons["coverage_ratio_too_high"] == 1


def test_row_rejected_when_no_valid_evidence_intervals():
    rows = [_make_row(video_id="1000", qid="1")]
    # zero-length interval after clipping -> normalize_intervals drops it
    grounding = {"1000": {"duration": 100.0, "location": {"1": [[5.0, 5.0]]}, "fps": 30.0}}
    mapping = {"1000": "0001/1000"}

    retained, counters, reasons, rejected, issues = fn.run_filter(rows, grounding, mapping)
    assert len(retained) == 0
    assert reasons["no_valid_evidence_intervals"] == 1


def test_duplicate_csv_rows_deduplicated():
    rows = [_make_row(video_id="1000", qid="1"), _make_row(video_id="1000", qid="1")]
    grounding = {"1000": {"duration": 100.0, "location": {"1": [[1.0, 5.0], [20.0, 25.0]]}, "fps": 30.0}}
    mapping = {"1000": "0001/1000"}

    retained, counters, reasons, rejected, issues = fn.run_filter(rows, grounding, mapping)
    assert counters["raw_csv_rows"] == 2
    assert counters["after_duplicate_removal"] == 1
    assert reasons["duplicate_csv_row"] == 1
    assert len(retained) == 1


def test_row_rejected_when_grounding_missing():
    rows = [_make_row(video_id="1000", qid="1")]
    grounding = {}
    mapping = {"1000": "0001/1000"}
    retained, counters, reasons, rejected, issues = fn.run_filter(rows, grounding, mapping)
    assert len(retained) == 0
    assert reasons["missing_grounding_annotation"] == 1


def test_row_rejected_when_mapping_missing():
    rows = [_make_row(video_id="1000", qid="1")]
    grounding = {"1000": {"duration": 100.0, "location": {"1": [[1.0, 5.0]]}, "fps": 30.0}}
    mapping = {}
    retained, counters, reasons, rejected, issues = fn.run_filter(rows, grounding, mapping)
    assert len(retained) == 0
    assert reasons["missing_video_mapping"] == 1


def test_row_rejected_when_empty_question():
    rows = [_make_row(video_id="1000", qid="1", question="   ")]
    grounding = {"1000": {"duration": 100.0, "location": {"1": [[1.0, 5.0]]}, "fps": 30.0}}
    mapping = {"1000": "0001/1000"}
    retained, counters, reasons, rejected, issues = fn.run_filter(rows, grounding, mapping)
    assert len(retained) == 0
    assert reasons["empty_question"] == 1


def test_row_rejected_when_option_incomplete():
    rows = [_make_row(video_id="1000", qid="1", a3="")]
    grounding = {"1000": {"duration": 100.0, "location": {"1": [[1.0, 5.0]]}, "fps": 30.0}}
    mapping = {"1000": "0001/1000"}
    retained, counters, reasons, rejected, issues = fn.run_filter(rows, grounding, mapping)
    assert len(retained) == 0
    assert reasons["incomplete_options"] == 1


# ── manifest deduplication ────────────────────────────────────────────────

def test_manifest_dedups_repeated_video():
    filtered = [
        {"video_id": "1000", "qid": "1"},
        {"video_id": "1000", "qid": "2"},
        {"video_id": "2000", "qid": "5"},
    ]
    mapping = {"1000": "0001/1000", "2000": "0002/2000"}

    manifest, missing = bm.build_manifest(filtered, mapping)
    assert missing == []
    assert len(manifest) == 2
    entry_1000 = next(e for e in manifest if e["video_id"] == "1000")
    assert entry_1000["required_by_qids"] == ["1", "2"]
    assert entry_1000["expected_relative_path"] == "0001/1000.mp4"


def test_manifest_reports_missing_mapping():
    filtered = [{"video_id": "9999", "qid": "1"}]
    mapping = {}
    manifest, missing = bm.build_manifest(filtered, mapping)
    assert missing == ["9999"]
    assert manifest[0]["expected_relative_path"] is None
    assert manifest[0]["status"] == "unknown"


# ── schema validation ─────────────────────────────────────────────────────

def _cgbench_like_record(**overrides):
    base = {
        "video_id": "x", "qid": "1", "source_dataset": "cgbench", "question": "q",
        "choices": ["a", "b"], "answer": "A", "evidence_intervals": [],
        "evidence_count": 0, "video_duration": 1.0, "question_type": "Perception",
        "source_task": "t", "evidence_condition": "sufficient", "expected_behavior": "answer",
    }
    base.update(overrides)
    return base


def test_schema_validation_passes_for_matching_keys():
    template_keys = set(_cgbench_like_record().keys())
    records = [_cgbench_like_record(video_id="1000", question_type="TN", source_dataset="nextgqa")]
    fn.validate_schema(records, template_keys)  # should not raise


def test_schema_validation_fails_for_extra_key():
    template_keys = set(_cgbench_like_record().keys())
    bad_record = _cgbench_like_record()
    bad_record["mapped_video_id"] = "extra"
    try:
        fn.validate_schema([bad_record], template_keys)
        assert False, "expected ValueError for extra key"
    except ValueError as e:
        assert "extra keys" in str(e)


def test_schema_validation_fails_for_missing_key():
    template_keys = set(_cgbench_like_record().keys())
    bad_record = _cgbench_like_record()
    del bad_record["expected_behavior"]
    try:
        fn.validate_schema([bad_record], template_keys)
        assert False, "expected ValueError for missing key"
    except ValueError as e:
        assert "missing keys" in str(e)


def test_final_invariants_accept_single_interval():
    r = _cgbench_like_record(video_id="1000", qid="1", evidence_count=1,
                              evidence_intervals=[{"start": 0, "end": 3}],
                              question_type="TN")
    fn.validate_final_invariants([r])  # should not raise


def test_final_invariants_reject_duplicate_video_qid():
    r1 = _cgbench_like_record(video_id="1000", qid="1", evidence_count=1,
                               evidence_intervals=[{"start": 0, "end": 3}],
                               question_type="TN")
    r2 = _cgbench_like_record(video_id="1000", qid="1", evidence_count=1,
                               evidence_intervals=[{"start": 0, "end": 3}],
                               question_type="TN")
    try:
        fn.validate_final_invariants([r1, r2])
        assert False, "expected ValueError for duplicate (video_id, qid)"
    except ValueError as e:
        assert "Duplicate" in str(e)
