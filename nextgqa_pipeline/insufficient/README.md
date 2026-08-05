# NExT-GQA calibrated sufficiency pipeline

This directory constructs C3 videos with behavioral sufficiency tests.  The
construction model is always forced to choose A--E; it is never asked to emit
`ANSWERABLE` or `UNANSWERABLE` and never sees the gold answer in its prompt.

The old `build.py` / binary-leak `check.py` pipeline has been removed.

## Method

The unit of processing is `(video, question)`.

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

Qwen3-VL should remain outside the construction endpoint list if it will be used
as the independent downstream evaluator.

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

The bundled server defaults to 1 FPS and permits 192 images per prompt.  Local
NExT-GQA videos reach 180 seconds, so 2 FPS can require 360 images plus a much
larger model context.  Before changing to 2 FPS, run a stratified pilot and set
both `MAX_IMAGES` and `MAX_MODEL_LEN` high enough.

Start the one-item tokenizer/logprob smoke test manually after launching a vLLM
server:

```bash
python nextgqa_pipeline/insufficient/smoke_test_logprobs.py \
  --endpoint internvl3_5_8b_construct,http://127.0.0.1:8000/v1,internvl3_5-8b
```

Run a 20-item measurement pilot:

```bash
NUM_SHARDS=1 LIMIT=20 sbatch nextgqa_pipeline/insufficient/measure.sbatch
```

Inspect the JSONL before calibration.  In particular compare single/multi-
interval items, full/gold correctness, gold positive-control recall and parent /
child non-monotonic cases.  The implementation intentionally uses full-density
scanning; cascading should only be added after that audit demonstrates that it
does not lose recall.

## Full run

Stage 1 -- measurements:

```bash
NUM_SHARDS=8 sbatch --array=0-7%8 nextgqa_pipeline/insufficient/measure.sbatch
```

Stage 2 -- calibration, only after every measurement shard finishes:

```bash
sbatch nextgqa_pipeline/insufficient/calibrate.sbatch
```

Read `thresholds.json` before finalization.  A scale with fewer than 20 contained-
gold controls or less than 95% positive gain at zero is marked `usable=false`
and is not used for discovery.

Stage 3 -- residual verification and final videos:

```bash
NUM_SHARDS=8 sbatch --array=0-7%8 nextgqa_pipeline/insufficient/finalize.sbatch
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
RESIDUAL_THRESHOLD=0.05 sbatch nextgqa_pipeline/insufficient/calibrate.sbatch
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
