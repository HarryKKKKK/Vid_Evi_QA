# E.T. Bench conflict-QA pipeline v2

## Contract

For every parent video, emit at most one QA in each attempted category:

1. `action_content`: an action-specific grounded role differs;
2. `scene_context`: a core-window scene field differs, attempted only when no
   `action_content` candidate survives;
3. `surrounding_event`: the visible event immediately before or after differs;
4. `temporal_localization`: the official E.T. Bench occurrence intervals.

Every available field candidate is traversed in semantic priority order. A
semantic-gate or deterministic-validation failure falls back to the next field
instead of selecting only the highest-confidence candidate. Categories 1-3
are skipped only when no candidate survives traversal. Category 4 is skipped
only when the official annotation has fewer than two intervals. Under the
action-or-scene policy, one video emits at most three QAs.

## Evidence windows

- Categories 1 and 2 use 12 frames sampled strictly inside `[start,end]`.
- Category 4 uses `tgt` directly and makes no visual timestamp prediction.
- Category 3 uses four BEFORE, two CORE, and four AFTER frames by default.
  BEFORE starts no earlier than the preceding official occurrence's end;
  AFTER ends no later than the next official occurrence's start. Thus context
  sampling never overlaps another labeled occurrence.

No generic before/after fields are included in core grounding.

## Two-model separation

The runner requires two independent OpenAI-compatible endpoints:

- grounding VLM: fills action-specific roles and visible scene/context fields;
- text-only QA model: designs the role schema from the official action,
  merges synonymous grounding values, and writes one question for each
  surviving contrast.

Recommended grounding model on the current server:

```text
models/Qwen3-VL
served name: qwen3-vl-evidence-grounder
```

The existing launch script is:

```text
nextgqa_pipeline/insufficient/launch_correct_answer_qwen3vl_server.sh
```

For QA wording in the Qwen-grounding setup, use the separately deployed
non-Qwen text model `mistralai/Mistral-Small-24B-Instruct-2501` (or a stronger
compatible replacement). The pipeline is model-agnostic; only `QA_BASE_URL`
and `QA_MODEL` change.
`serve_conflict_qa_text.sh` is a one-GPU vLLM launcher once the text-model
weights are available locally. No Qwen-specific thinking argument is sent by
default; pass `--qa-thinking` only when the selected text model supports it.

The text model never controls answers, option IDs, supporting occurrences, or
intervals. Those are attached after generation from grounding records and the
official manifest. A deterministic validator rejects any evidence drift.

## Run

```bash
export GROUNDING_BASE_URL=http://grounding-node:18008/v1
export GROUNDING_MODEL=qwen3-vl-evidence-grounder
export QA_BASE_URL=http://text-node:18009/v1
export QA_MODEL=mistral-small-24b-instruct-2501
export OUTPUT_DIR=etbench_pipeline/conflict/results_v2

bash etbench_pipeline/conflict/run_conflict_pipeline_v2.sh
```

For a pilot, override `GROUPS` with
`etbench_pipeline/conflict/pilot_3/deliverable/selected_parent_jobs.jsonl`.

## Outputs

```text
action_role_schemas.jsonl     text-model role plans, one per video/action
core_grounding.jsonl          categories 1/2, core-window only
temporal_grounding.jsonl      category 3, non-overlapping context windows
semantic_contrast_decisions.jsonl  synonym/equivalence decisions
conflict_qa_v2.jsonl          final evidence-locked QA records
skipped_categories.jsonl      explicit same/insufficient-evidence skips
raw_grounding_responses.jsonl raw VLM responses
raw_qa_responses.jsonl        raw text-model responses
candidate_attempts.jsonl      accepted/rejected traversal attempts by field
run_summary.json              totals and category distribution
```

Every final record has `answer_mode=multiple_valid_answers`. Its options are
the distinct grounded values (or official time intervals), and every option
lists the exact supporting occurrence IDs and official intervals.
