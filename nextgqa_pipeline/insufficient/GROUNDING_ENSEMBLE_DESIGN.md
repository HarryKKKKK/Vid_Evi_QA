# Evidence-leak semantic audit v2

## Most important conclusion

The semantic judge is not the only, or necessarily the largest, source of
error. The old pipeline passes one lossy free-form VLM description across the
vision/text boundary. If that description omits a tool, changes the contact
target, loses a temporal relation, or turns an inferred endpoint into a direct
event, no downstream text judge can reconstruct the original frames.

Observed examples include:

- `touches the hair` becoming `touches the back of the head` and then failing
  answer equivalence;
- a complete golf-club swing being reduced to generic arm motion, or correctly
  described but rejected as insufficient causal evidence for playing golf;
- a fishing rod appearing in the grounding and judge reason while the open
  answer is still rejected as different from `with a rod`;
- a visible white vehicle being described without preserving the required
  `passes the people` relation;
- an incomplete turn being expanded into a complete `turns and walks away`
  event.

Therefore the core object of the audit should be a set of independent,
structured visual observations, not one caption.

## Production decision rule

For each candidate interval:

1. Run three strong and preferably architecturally diverse grounding VLMs on
   exactly the same masked frame/timestamp input.
2. Freeze each model's structured observations independently. Never let one
   model see another model's description.
3. Use an independent text judge to evaluate each grounding separately without
   choices or gold. Freeze its open answer.
4. Compare the open answer with the reference using semantic entailment. Exact
   wording and embedding similarity are insufficient: the comparison must
   preserve subject, object/tool, temporal relation, causality, and manner while
   accepting supported paraphrases and more-specific answers.
5. Confirm `EVIDENCE_LEAK` when at least two of three grounders independently
   support the complete question and gold answer.
6. Route one-positive cases, missing grounders, and judge/grounder disagreement
   to `UNCERTAIN` and retain every raw observation for manual review.
7. Optionally run a fourth VLM only on `UNCERTAIN` cases. Do not silently turn a
   single positive into a negative by majority vote.

The recommended starting operating point is 2-of-3. It should be calibrated on
a manually labeled audit set; compare 1-of-3, 2-of-3, 3-of-3, and a fourth-model
adjudication policy using item-level leak precision/recall and false-negative
counts by question type.

## Why not embedding similarity alone

Embedding similarity is useful for candidate retrieval or diagnostics, but it
is unsafe as the final decision. `swings arms` and `plays golf` can be close
without the club being visible, while `reels in the fish while pulling the rod`
and `with a rod` may have different surface forms but a valid entailment.
Final comparison must be relation-aware semantic entailment with an auditable
reason.

## Output contract

`ground_windows.py` now emits a backward-compatible `description` plus:

- `observations`: timestamped subject/action/object-or-tool/relations,
  interpretation, and certainty;
- `missing_or_ambiguous`: explicitly absent or uncertain facts;
- `raw_response`, `model_tag`, and timestamps for provenance.

`classify_groundings.py` groups records by candidate and distinct model tag and
emits:

- `groundings`: all independent visual records;
- `per_grounder_results`: independent answerability, open-answer, equivalence,
  and label decisions;
- `consensus`: expected/completed grounders, vote counts, support threshold,
  and disagreement state;
- the backward-compatible top-level `label` and representative `decision`.

## Required evaluation before replacing v1 results

Create a stratified manual set containing confirmed leaks, confirmed non-leaks,
question types `TC/TN/TP/CW/CH`, short and long windows, and known failure cases.
Report:

- item-level leak precision and recall;
- grounding fact recall before judging;
- false positives caused by inferred rather than direct observations;
- disagreement rate and manual-review volume;
- results for each grounder alone and 1-of-3 / 2-of-3 / 3-of-3 aggregation;
- judge sensitivity with the grounding records held fixed.

This separates vision-description errors from text-judge errors and makes model
selection evidence-based rather than assuming that a larger/SOTA checkpoint is
automatically a better grounder.
