# ETBench conflict-QA construction pipeline

## Objective

Construct QA examples where two or more independently sufficient, temporally
localized event instances support mutually exclusive answers to an
underspecified question. Repeated events are candidates, not conflicts.

The construction invariant is:

```text
Q + E1 -> A
Q + E2 -> B
A != B and A/B are mutually exclusive under the same interpretation of Q
Q + E1 + E2 -> identify ambiguity/conflict and report both grounded alternatives
```

Temporal state change, missing evidence, different entities, synonyms, and
different answer granularity must not be mislabeled as conflict.

## Available seed pool

- 500 Perception Test TEM samples: reference event plus matched event(s).
- 299 Perception Test TAL samples: at least two localized occurrences.
- 799 unique downloaded videos, approximately 10 GB.
- 2,718 localized event instances.

## Stage 0: candidate and window construction — implemented

Scripts:

- `source_datasets/et_bench/select_conflict_candidates.py`
- `build_visual_contrast_jobs.py`

For every interval `[s,e]`, construct:

```text
pre     = [s-5, s]
core    = [s, e]
post    = [e, e+5]
context = [s-8, e+8]
```

Clamp all windows to video bounds. Keep the original interval unchanged for
evidence attribution.

Outputs:

- `visual_contrast_jobs.jsonl`: parent/group manifest, 799 rows;
- `visual_instance_jobs.jsonl`: independent instance calls, 2,718 rows.

## Stage 1: independent event signatures

Run one visual call per row in `visual_instance_jobs.jsonl`. Never show the
model another occurrence during this stage. Use temperature 0 and preserve the
model's raw response.

Requested slots:

```text
anchor_action
actor
actor_appearance
object
object_attributes
before_action
core_manner
after_action
outcome_or_state_after
location
co_participant
```

Every slot returns `value`, `confidence`, and `visual_basis`. Use
`not_visible` rather than guessing. Reject subjective attributes such as emotion
and intent.

Recommended frame allocation for a pilot is 16–24 frames per instance:

- 25% from `pre`;
- 50% from `core`;
- 25% from `post`;
- always include the first and last visible frame of the context window.

Long TAL lists should still be processed per instance, not in one montage.

## Stage 2: contrast mining — partially implemented

Script: `build_pair_judging_jobs.py`.

1. Group signatures by parent job.
2. Compare only instances belonging to the same parent event query.
3. Drop missing, equal, and low-confidence values (`confidence < 0.75`).
4. Exclude `anchor_action` itself as a conflict slot; it must be common.
5. Rank pairs by the number of visible differing slots.
6. Retain at most 12 pairs per parent video.
7. Send the remaining signatures to a text-only conservative judge.

The judge accepts a slot only if:

- both values are visually grounded;
- both instances have the same anchor-event semantics;
- the values are genuinely different and mutually exclusive;
- the difference can be asked with one natural question;
- it is not caused by synonyms, granularity, entity mismatch, or uncertainty.

Preferred slots are `object`, `before_action`, `after_action`,
`outcome_or_state_after`, `core_manner`, and `co_participant`. Appearance and
location are allowed but should be down-weighted because they create easy
single-frame shortcuts.

## Stage 3: grouped QA generation

For each accepted pair, generate three questions together:

1. `conflict_question`: deliberately omits the occurrence index/time.
2. `resolved_question_a`: identifies E1 without revealing A.
3. `resolved_question_b`: identifies E2 without revealing B.

Prefer ordinal or timestamp disambiguation for control questions:

```text
Qc: What did the person do immediately after folding the object?
Qa: What did the person do immediately after the first folding occurrence?
Qb: What did the person do immediately after the second folding occurrence?
```

Do not use an answer-bearing descriptor such as “after folding the red object”
when the differing answer is the object's color.

Each accepted item produces a family:

```text
Qc + E1 + E2 -> conflict
Qa + E1       -> sufficient(A)
Qb + E2       -> sufficient(B)
Qa + E1 + E2  -> sufficient with distractor, not conflict
Qb + E1 + E2  -> sufficient with distractor, not conflict
Qc + E1       -> locally answerable single-alternative control
Qc + E2       -> locally answerable single-alternative control
```

The last three controls prevent the shortcut “multiple clips means conflict.”

## Stage 4: falsification and acceptance gate

Answer the generated questions with a verifier that did not generate the event
signatures. Require all of the following:

```text
answer(Qc, E1) semantically matches A
answer(Qc, E2) semantically matches B
answer(Qa, E1+E2) matches A and does not claim conflict
answer(Qb, E1+E2) matches B and does not claim conflict
answer(Qc, E1+E2) reports ambiguity and both A/B
```

Then inspect every other localized occurrence in the parent video. If a third
valid value C exists, either include it as a third alternative or reject the
sample. Never publish an incomplete answer set.

Recommended automatic acceptance thresholds:

- visual slot confidence: at least 0.75 on both instances;
- pair-judge confidence: at least 0.85;
- semantic answer match: at least 0.85 or exact normalized match;
- evidence timestamps: original ETBench intervals, not model-predicted spans;
- two independent verifier passes must agree.

## Stage 5: human audit and balancing

Audit all pilot items, then at least 10% of scaled data. Label rejection reasons:

```text
same_answer
not_mutually_exclusive
different_entity
subjective_or_unobservable
question_leaks_answer
question_already_disambiguated
third_uncovered_answer
bad_event_boundary
annotation_error
```

Balance by source task, action category, conflict slot, answer type, number of
event instances, temporal distance, and question length. Split by original
`video` path before QA generation to avoid video leakage.

Recommended pilot:

- 100 TEM and 100 TAL videos;
- stratify TAL by action category and interval count;
- cap one published conflict family per video during the pilot;
- target at least 50 manually accepted families before scaling.

## Final record schema

```json
{
  "id": "etbench_conflict_...",
  "video": "perception_test/video_....mp4",
  "source_task": "tem",
  "anchor_action": "...",
  "conflict_type": "repeated_event",
  "conflict_slot": "after_action",
  "question": "...",
  "status": "conflict",
  "alternatives": [
    {
      "answer": "A",
      "evidence_interval": [1.2, 4.8],
      "context_interval": [0.0, 12.8]
    },
    {
      "answer": "B",
      "evidence_interval": [20.1, 23.7],
      "context_interval": [12.1, 31.7]
    }
  ],
  "resolution_type": "need_event_index",
  "canonical_response": "The question is ambiguous: ...",
  "clarification": "Do you mean the first or second occurrence?",
  "controls": {
    "resolved_question_a": "...",
    "resolved_question_b": "..."
  },
  "provenance": {
    "signature_model": "...",
    "pair_judge_model": "...",
    "verifier_models": ["...", "..."],
    "human_verified": false
  }
}
```

## Evaluation dimensions

- conflict-status accuracy;
- alternative answer set F1;
- evidence attribution IoU/recall;
- resolution-type accuracy;
- clarification quality;
- control consistency (`resolved + distractor` must remain sufficient);
- performance by temporal distance and number of occurrences.
