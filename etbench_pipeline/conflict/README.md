# ETBench conflict-QA pipeline

This pipeline starts from ETBench Perception Test samples that contain repeated
or matched event instances. Repetition alone is **not** labeled as conflict.
The first stage expands each event into temporal context and asks a vision model
to find independently observable differences.

## Current stages

1. `source_datasets/et_bench/select_conflict_candidates.py`
   selects all TEM samples with a reference and matched event, plus TAL samples
   with at least two occurrences. Current output: 799 samples/videos.
2. `build_visual_contrast_jobs.py` creates a group manifest plus one independent
   visual-description job per event instance. Independence prevents the vision
   model from inventing a difference merely because it was asked to compare.
3. A later visual-extraction stage will retain only differences that are visible,
   high-confidence, and mutually exclusive.
4. `build_pair_judging_jobs.py` consumes independent visual results, removes
   missing/low-confidence/equal slots, caps pair growth, and emits conservative
   text-only mutual-exclusivity judging jobs.
5. A QA-generation stage will create both a disambiguated control question and
   an underspecified conflict question from every retained difference.

## Build jobs

```bash
python etbench_pipeline/conflict/build_visual_contrast_jobs.py
```

On the compute server, pass its Linux video root so emitted paths are directly
usable:

```bash
python etbench_pipeline/conflict/build_visual_contrast_jobs.py \
  --video-root source_datasets/et_bench/ETBench/videos
```

Default temporal expansion is 5 seconds before, 5 seconds after, and 8 seconds
on both sides for the full context window. All windows are clamped to the video
duration.

The two outputs are:

- `visual_contrast_jobs.jsonl`: 799 parent/group records;
- `visual_instance_jobs.jsonl`: 2,718 independent visual calls.

## Intended final record

Each accepted contrast will eventually produce:

```json
{
  "status": "conflict",
  "conflict_type": "repeated_event",
  "question": "What did the person do after performing the action?",
  "alternatives": [
    {"answer": "...", "evidence": [12.1, 18.4]},
    {"answer": "...", "evidence": [42.0, 49.2]}
  ],
  "resolution_type": "need_event_index",
  "clarification": "Do you mean the first or second occurrence?"
}
```

Every conflict item must be paired with a disambiguated sufficient control so
that models cannot learn the shortcut “multiple intervals means conflict.”
