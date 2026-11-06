# NExT-GQA calibrated sufficiency pipeline

This directory constructs C3 videos with behavioral sufficiency tests.  The
construction model is always forced to choose A--E; it is never asked to emit
`ANSWERABLE` or `UNANSWERABLE` and never sees the gold answer in its prompt.

The old `build.py` / binary-leak `check.py` pipeline has been removed.

## Method

The unit of processing is `(video, question)`.

### Pre-window full-video leak gate

Before calibrated window measurement, the pipeline now runs a fail-closed
full-video gate:

1. Freeze the official evidence interval (including the configured padding) in
   the complete video.
2. Uniformly sample the complete frozen video and ask the evaluation VLM for
   `ANSWERABLE`/`UNANSWERABLE`, an answer option, and `evidence_seconds`.
3. `UNANSWERABLE` and wrong answers pass this gate. A correct `ANSWERABLE`
   result is not automatically called a leak: its claimed interval is grounded
   without choices/gold, then classified by the semantic text judge.
4. Exclude only `status=ok, label=EVIDENCE_LEAK`. Other completed semantic
   labels pass. Missing evaluations, invalid intervals, grounding errors, and
   unresolved classifications prevent the downstream worklist from being
   published.
5. Run the existing calibrated multi-scale window pipeline only on
   `nextgqa_filtered_pre_window_pass.json`.

Submit the complete pre-window dependency chain:

```bash
bash nextgqa_pipeline/insufficient/submit_pre_window_gate.sh
```

The chain builds frozen videos, evaluates complete videos, grounds reported
intervals, classifies them, and finally publishes the downstream metadata. The
measurement/finalize/grounding Slurm scripts accept `FILTERED_JSON` and now
default to that published worklist. Use fresh measurement/output paths when
rerunning the full experiment so prior JSONL records cannot be mixed with a new
gate population.

When the later eligible-window and dense audits are complete, preserve the
pre-window exclusions in the final union:

```bash
python nextgqa_pipeline/insufficient/build_no_leak_eval_json.py \
  --base-exclude-keys nextgqa_pipeline/insufficient/pre_window_gate_exclude_keys.txt
```

1. Use one fixed frame grid for all conditions of an item (default 1 FPS).
2. Measure full-video and matched all-gray blind A--E margins.
3. Measure a gold-only pack as the primary positive-control sanity check.
4. Freeze the padded official evidence **in memory** with the same blurred-safe-
   frame transformation used by the final video.
5. Densely scan the seed residual at 4s/2s, 8s/4s and 16s/8s.  Every visual
   window has a question-matched blind input with identical timestamps and frame
   count.
6. Calibrate one threshold for each `(model, scale)` using only fixed-scale
   windows that completely contain all official evidence.  Merely overlapping
   gold is not treated as positive.  Unlabeled discovery windows are not treated
   as negatives and no unsupported FPR/ROC claim is made.
7. Take the cross-model union of positive and near-threshold windows, retain the
   smallest positive descendants, merge/pad them with official evidence, then
   score the complete residual video.
8. If the global residual still has answer signal, run one dense residual scan.
   A remaining global leak with no localizable new window is classified as
   distributed/scene-level leakage and the item is discarded.
9. Enforce both a maximum coverage ratio and an absolute minimum visible time.
   Encode exactly one permanent final freeze video for each surviving item.

The bundled Slurm launchers currently use Qwen3-VL as the construction model.
Consequently Qwen3-VL is not a held-out downstream evaluator for videos produced
by this configuration; use a different model for independent final evaluation.

## Files

- `common.py`: data loading, interval logic, frame-grid extraction, in-memory
  freeze masking, A--E logprob parsing and final FFmpeg encoding.
- `smoke_test_logprobs.py`: confirms that the deployed model returns all five
  option-letter logprobs under the exact answer-only prompt.
- `measure.py`: full/blind/gold measurements and full-density multi-scale scan.
- `calibrate.py`: per-model/per-scale positive-control threshold calibration.
- `finalize.py`: cross-model union, global residual gate, one second scan and
  final video generation.

## Important scoring definition

For input `x`:

```text
m(x) = log p(gold | x) - logsumexp(log p(other four options | x))
gain_W(w) = m(w) - m(blind_W)
gain_full = m(full) - m(blind_full)
s_W(w) = gain_W(w) / gain_full
```

`blind_W` is recomputed with the same question, timestamps and number of frames
as `w`.  The response must expose A--E in generated-token `top_logprobs`; missing
letters are an error, never a negative evidence decision.

## Pilot first

The bundled Qwen3-VL server uses two GPUs with tensor parallelism 2, defaults to
1 FPS, permits 192 images per prompt, and uses the already-tested 32768-token
context plus `max_pixels=75264`.  Local
NExT-GQA videos reach 180 seconds, so 2 FPS can require 360 images plus a much
larger model context.  Before changing to 2 FPS, run a stratified pilot and
confirm both the image limit and context budget.

Start the one-item tokenizer/logprob smoke test manually after launching a vLLM
server:

```bash
python nextgqa_pipeline/insufficient/smoke_test_logprobs.py \
  --endpoint qwen3vl_construct,http://127.0.0.1:8000/v1,qwen3-vl
```

Run a 20-item measurement pilot:

```bash
NUM_SHARDS=1 LIMIT=20 sbatch jobs/build/nextgqa/measure.sbatch
```

Inspect the JSONL before calibration.  In particular compare single/multi-
interval items, full/gold correctness, gold positive-control recall and parent /
child non-monotonic cases.  The implementation intentionally uses full-density
scanning; cascading should only be added after that audit demonstrates that it
does not lose recall.

## Full run

Stage 1 -- measurements:

```bash
NUM_SHARDS=8 sbatch --array=0-7%8 jobs/build/nextgqa/measure.sbatch
```

Stage 2 -- calibration, only after every measurement shard finishes:

```bash
sbatch jobs/build/nextgqa/calibrate.sbatch
```

Read `thresholds.json` before finalization.  A scale with fewer than 20 contained-
gold controls or less than 95% positive gain at zero is marked `usable=false`
and is not used for discovery.

Stage 3 -- residual verification and final videos:

```bash
NUM_SHARDS=8 sbatch --array=0-7%8 jobs/build/nextgqa/finalize.sbatch
```

Final videos:

```text
source_datasets/next_gqa/insufficient_videos_calibrated/
  {video_id}_q{qid}_freeze.mp4
```

Audits:

```text
nextgqa_pipeline/insufficient/measurements.<shard>.jsonl
nextgqa_pipeline/insufficient/thresholds.json
nextgqa_pipeline/insufficient/finalize_audit.<shard>.jsonl
```

`RESIDUAL_THRESHOLD` defaults to `0.10` in `calibrate.sbatch`.  This is a
separate global-residual operating point, not a window threshold learned from
gold controls.  It must be included in sensitivity analysis and may be changed
before calibration, for example:

```bash
RESIDUAL_THRESHOLD=0.05 sbatch jobs/build/nextgqa/calibrate.sbatch
```

## Resume behavior

- `measure.py` skips a latest successful record and retries `error` or
  `needs_review` records.
- `finalize.py` skips only a latest `status=ok` record whose final video still
  exists.
- API failures, missing logprobs, missing videos and encoding failures are
  `needs_review`; they are never interpreted as evidence absence.
- An answer that cannot be mapped uniquely to one A--E option (for example two
  identical option strings) is immediately `discarded_invalid_gold`; it is not
  retried and never enters calibration, construction or evaluation.
- Items that exceed coverage limits or retain distributed/scene-level leakage
  are explicitly recorded as discarded and do not produce a final video.
## Semantic audit of dangerous windows

The calibrated logprob scan is a high-recall candidate generator, not a final
evidence detector. Manual review of missed leaks showed that the largest
recoverable error source is often the **grounding representation**, not only the
text judge. A single free-form caption can erase the discriminating fact
(`hair` becomes `head`), omit a tool (`golf club` becomes `swings arms`), deny a
visible relation (`passes the people`), or complete a partially observed action.
Once that caption is frozen, even a strong judge cannot recover the missing
visual fact.

The v2 semantic audit therefore uses three layers:

1. **Independent structured grounding.** Run 3--4 strong, architecturally
   diverse VLMs independently on the same masked frames. Each model receives
   the question/type but no choices or gold, and records timestamped atomic
   observations: subject, action/state, object or tool, relations,
   interpretation, certainty, and missing/ambiguous facts. Descriptions are not
   merged before judging, so model disagreement remains observable.
2. **Per-grounder semantic entailment.** The text judge evaluates each grounding
   independently. It first answers the complete question open-endedly, then
   checks whether that answer semantically entails the reference. This check
   accepts supported paraphrases, more-specific answers, and harmless visual
   granularity differences such as `back of head/hair`, `reeling with a fishing
   rod/with a rod`, and `golf-club swing/playing golf`. It is not raw embedding
   similarity: participant, temporal, causal, manner, and tool relations must
   still be preserved.
3. **Deterministic consensus.** By default at least two distinct grounding
   models must independently yield `EVIDENCE_LEAK`. One positive grounding is
   recorded as `UNCERTAIN` for human review instead of confirming or silently
   rejecting the leak. Every model description and decision is retained in
   `per_grounder_results` for auditability.

This policy separates two objectives: multi-model grounding improves recall
against omitted visual facts, while the two-model confirmation rule protects
precision against a single model hallucinating a complete action. Increasing
the number of grounders without preserving independent decisions does not give
this protection.

The implementation is split into two executable stages:

1. `ground_windows.py` reconstructs minimal high-risk windows from
   `measurements.*.jsonl` and `thresholds.json`, samples the source video, and
   applies the same blurred-safe-frame freeze to every official-evidence frame.
   Its VLM sees the question and `CW/CH/TN/TP/TC` type, but never the choices or
   gold answer. It emits structured observations and a backward-compatible
   summary. Run it once per grounding model with distinct output filenames.
2. `classify_groundings.py` groups records by candidate key and grounding model,
   independently judges every grounding, performs semantic-entailment checks,
   and applies `--min-grounder-support`. The generic classifier retains a
   legacy-compatible default of 1; `submit_grounding_ensemble.sh` explicitly
   defaults production ensemble runs to 2 and records the expected ensemble
   size. Missing grounder outputs force `UNCERTAIN` rather than silently acting
   like negative evidence.

Final labels are `EVIDENCE_LEAK`, `ANSWER_CORRELATED_ONLY`,
`INSUFFICIENT_CONTEXT`, `IRRELEVANT`, `UNCERTAIN`, and `ANSWERABLE_WRONG`.

First create calibrated thresholds if they do not already exist:

```bash
python nextgqa_pipeline/insufficient/calibrate.py
```

Run a small single-model grounding pilot (useful for validating a new server,
but not sufficient for production leak confirmation):

```bash
NUM_SHARDS=1 LIMIT=20 MAX_WINDOWS_PER_ITEM=3 \
  sbatch jobs/build/nextgqa/ground_windows.sbatch
```

Run sharded grounding.  The sbatch default keeps five windows per item; set
`MAX_WINDOWS_PER_ITEM=0` only when intentionally grounding every minimal
candidate window:

```bash
NUM_SHARDS=8 MAX_WINDOWS_PER_ITEM=5 \
  sbatch --array=0-7 jobs/build/nextgqa/ground_windows.sbatch
```

Run the independent text judge.  `CLASSIFIER_MODEL_PATH` is intentionally
required so that the evaluator model is selected explicitly:

```bash
CLASSIFIER_MODEL_PATH=/absolute/path/to/text-model \
NUM_SHARDS=8 \
  sbatch --array=0-7 jobs/build/nextgqa/classify_groundings.sbatch
```

For the production 3--4-grounder workflow, provide semicolon-separated model
specifications. Each model receives its own Slurm array and output JSONL; the
judge array starts only after all grounding arrays succeed:

```bash
export GROUNDING_MODEL_SPECS='qwen3|qwen|/aifs4su/hansirui_2nd/harry/Vid_Evi_QA/models/Qwen3-VL|qwen3-ground|2;internvl35|internvl|/aifs4su/hansirui_2nd/harry/Vid_Evi_QA/models/InternVL3_5-8B|internvl35-ground|1;llava|llava|/aifs4su/hansirui_2nd/harry/Vid_Evi_QA/models/LLaVA-OneVision-Qwen2-7B-OV|llava-ground|1'
export CLASSIFIER_MODEL_PATH=/aifs4su/models/Qwen3.5-35B-A3B-FP8
export MIN_GROUNDER_SUPPORT=2
bash nextgqa_pipeline/insufficient/submit_grounding_ensemble.sh
```

Prefer diversity over four checkpoints from one family. A reasonable initial
experiment is three strong grounders with 2-of-3 confirmation, followed by a
fourth model only for disputed (`UNCERTAIN`) windows. Compare this policy against
1-of-3 and 3-of-3 on a manually labeled leakage set before fixing the operating
point.

Both stages append JSONL, resume completed records automatically, retry records
with `status=error`, and support `--limit`, `--shard`, `--num-shards`, and
`--overwrite`.

## Dense fallback for construction-model-ineligible items

`eligible=false` only means that the construction model did not pass the
full/gold-only behavioral control. It does not mean that the item is free of
leakage. `dense_ground_ineligible.py` therefore bypasses logprob candidate
selection and scans the entire masked residual video using gap-free 10-second
cores. Each grounding request also receives two seconds of context on both
sides (at most 14 seconds total), sampled at 2 FPS. Official evidence is
replaced by the same blurred safe-frame freeze used elsewhere in the pipeline.

Submit Qwen3-VL grounding followed by Qwen3-VL text judgment. The wrapper
returns immediately after Slurm accepts both dependent arrays:

```bash
bash nextgqa_pipeline/insufficient/submit_dense_ineligible.sh
```

After the classification array completes, aggregate windows to item labels and
write the evidence-leak exclusion list:

```bash
python nextgqa_pipeline/insufficient/summarize_dense_ineligible.py
```

The summary distinguishes confirmed `EVIDENCE_LEAK`, gold-answer-correlated
hallucination candidates, possible non-gold distractors, ordinary insufficient
items, uncertainty, and processing errors. Evidence in the two-second context
may occur in adjacent requests, but aggregation is item-level, so duplicate
window detections do not duplicate excluded items.

## Actual semantic-audit workflow used for the current NExT-GQA run

The experiment ultimately used the calibrated scan only as a high-recall
candidate generator. It did **not** equate a positive logprob window with an
evidence leak.

### 1. Model-item eligibility

For construction model `qwen3vl_construct`, an item was eligible for behavioral
window discovery only when all four checks passed:

```text
prediction(full) == gold
prediction(gold-only) == gold
gain(full) > 0
gain(gold-only) > 0
```

This produced 3,780 eligible and 1,755 ineligible items. Two additional items
did not have a successful measurement and remain outside those two groups.
Eligibility is a property of this construction model on this item; ineligible
does not mean non-visual, invalid, or leak-free.

### 2. Dangerous-window definition for eligible items

Discovery scans the official-evidence-masked seed, where official evidence is
padded by one second and replaced by a blurred safe-frame freeze. Scales are
4s/2s, 8s/4s, and 16s/8s (window/stride). A window is a dangerous candidate
when:

```text
normalized_gain(masked_window) >= threshold(model, scale) - 0.05
```

Each threshold is the lower fifth percentile of the corresponding
gold-contained positive-control distribution, giving approximately 95%
positive-control recall. This is not a claim of 95% recall over all unknown
leaks and is not an ROC threshold, because unlabeled residual windows are not
reliable negatives. The calibrated Qwen3-VL values were:

| Scale | Positive controls | Threshold | Recall | Candidate cutoff |
|---|---:|---:|---:|---:|
| 4s/2s | 945 | 0.398574 | 95.03% | 0.348574 |
| 8s/4s | 2,573 | 0.510947 | 95.03% | 0.460947 |
| 16s/8s | 3,733 | 0.635104 | 95.02% | 0.585104 |

Positive ancestors are discarded when a smaller positive descendant exists.
Candidates are ranked by `normalized_gain - threshold`; the production audit
kept at most five candidates per item.

### 3. Semantic audit for eligible candidates

`ground_windows.py` describes masked candidate frames using only the question
and question type, without answer choices or gold. `classify_groundings.py`
then checks from text whether the description supports the complete question
and produces an open answer before comparing that answer with the reference.

The completed run contained 14,900 windows, all successfully classified:

| Window label | Count |
|---|---:|
| `INSUFFICIENT_CONTEXT` | 9,231 |
| `ANSWER_CORRELATED_ONLY` | 2,792 |
| `ANSWERABLE_WRONG` | 2,682 |
| `EVIDENCE_LEAK` | 192 |
| `IRRELEVANT` | 3 |

The 192 leak windows corresponded to 119 unique eligible QA items.

### 4. Dense fallback for ineligible items

The 1,755 ineligible items bypassed logprob window selection. Their complete
masked residual videos were partitioned into gap-free 10-second cores, with two
seconds of context on each side (at most 14 seconds per request), sampled at
2 FPS. This yielded 7,933 windows; all grounding and classification calls
completed without processing errors.

Window results:

| Window label | Count |
|---|---:|
| `INSUFFICIENT_CONTEXT` | 5,688 |
| `ANSWERABLE_WRONG` | 1,471 |
| `ANSWER_CORRELATED_ONLY` | 741 |
| `EVIDENCE_LEAK` | 32 |
| `IRRELEVANT` | 1 |

Item-level aggregation produced 823 ordinary insufficient, 522 possible
distractor insufficient, 389 hallucination/gold-cue insufficient, and 21
evidence-leak items.

### 5. Combined exclusion metadata

Run:

```bash
python nextgqa_pipeline/insufficient/build_no_leak_eval_json.py
```

The original run unioned 119 eligible-path and 21 dense-ineligible leak items.
The subsequent correct-answer evidence audit added 80 non-overlapping items.
Exclusion is by `(video_id, qid)`, not by whole source video, so unrelated
questions from the same video remain available.

```text
Original QA items:       5,537
Excluded leak items:       220
Unique affected videos:    185
Remaining QA items:      5,317
```

Outputs:

```text
nextgqa_pipeline/nextgqa_filtered_no_evidence_leak.json
nextgqa_pipeline/insufficient/combined_evidence_leak_exclude_keys.txt
nextgqa_pipeline/insufficient/combined_evidence_leak_exclusion_audit.json
```

### 6. Freeze-video construction

`build_no_leak_freeze.py` constructs one video per remaining QA item. It pads
official evidence by one second, uses the same nearest safe-frame selection and
Gaussian blur (`sigma=30`) as in-memory masking, removes audio, and verifies
that encoding preserves the actual ffprobe source duration. The completed run
produced all 5,397 expected videos with no missing outputs:

```bash
NUM_SHARDS=16 JOBS=4 THREADS=4 \
  sbatch --array=0-15%8 \
  jobs/build/nextgqa/build_no_leak_freeze.sbatch
```

```text
source_datasets/next_gqa/freeze_videos_no_evidence_leak/
  {video_id}_q{qid}_freeze.mp4
```

### 7. Qwen3-VL insufficient evaluation

Run after freeze construction:

```bash
sbatch jobs/build/nextgqa/eval_qwen3_no_leak.sbatch
```

The completed run covered 5,397/5,397 unique QA items with no missing videos or
missing item keys. Raw output counts were 2,828 `ANSWERABLE`, 2,568
`UNANSWERABLE`, and one truncated/unparseable JSON response. Among the 2,828
answerable responses, 2,320 selected the gold option and 508 selected a
different option. Report these numbers with the one parse failure explicit
unless that item is successfully rerun under a stricter bounded-output prompt.

```text
nextgqa_result_qwen3vl_no_evidence_leak/insufficient.classify.json
nextgqa_result_qwen3vl_no_evidence_leak/insufficient.classify.jsonl
```

Qwen3-VL was used for candidate construction, semantic grounding/judgment, and
this reported evaluation. Therefore this run is not a held-out-model evaluation;
a separate construction-independent evaluator is still required for the final
held-out result.

## Audit evidence claimed by correct insufficient answers

Downstream evaluators may answer a frozen insufficient item correctly and
report an `evidence_seconds=[start, end]` interval. These intervals can be fed
back into the same semantic audit without exposing the answering model's
choices, answer, or explanation to the grounding model.

First ground every correct-answer claim from one or more model result files.
The script deduplicates by `(model_tag, video_id, qid)` and reads the final
frozen video, so it audits exactly the insufficient visual stimulus rather than
the original source video:

```bash
python nextgqa_pipeline/insufficient/ground_correct_answer_evidence.py \
  --results 'nextgqa_result_qwen3vl_no_evidence_leak/insufficient.classify.json' \
  --results 'nextgqa_result_internvl*/insufficient.classify.json' \
  --freeze-dir source_datasets/next_gqa/freeze_videos_no_evidence_leak \
  --endpoint grounder,http://127.0.0.1:8000/v1,grounding-model \
  --output nextgqa_pipeline/insufficient/correct_answer_evidence_groundings.jsonl
```

Then run the existing text-only judge. It will label each claimed window as
`EVIDENCE_LEAK`, `ANSWERABLE_WRONG`, `ANSWER_CORRELATED_ONLY`,
`INSUFFICIENT_CONTEXT`, `IRRELEVANT`, or `UNCERTAIN`:

```bash
python nextgqa_pipeline/insufficient/classify_groundings.py \
  --groundings 'nextgqa_pipeline/insufficient/correct_answer_evidence_groundings*.jsonl' \
  --endpoint judge,http://127.0.0.1:8001/v1,text-judge-model \
  --output nextgqa_pipeline/insufficient/correct_answer_evidence_classifications.jsonl
```

Finally summarize the leak rate overall and separately for every downstream
evaluation model:

```bash
python nextgqa_pipeline/insufficient/summarize_correct_answer_evidence.py \
  --classifications 'nextgqa_pipeline/insufficient/correct_answer_evidence_classifications*.jsonl' \
  --output nextgqa_pipeline/insufficient/correct_answer_evidence_summary.json
```

The default audit window adds one second of context on each side and preserves
the complete claimed time span. Long spans are uniformly thinned to at most 64
frames; use `--max-frames-per-window` to change that budget. Set a positive
`--max-window-seconds` only when long claims should be routed to manual review.
