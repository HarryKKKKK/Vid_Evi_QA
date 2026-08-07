# Manual review windows

> Review update: these three windows are retained as **semantic hard
> negatives**, not confirmed evidence leaks.  They show the gold-answer action
> or state at another time, but not the particular event instance selected by
> the temporal wording of the question.  Their score increase is useful for
> studying model sensitivity to answer-correlated visual content, but they
> should not be counted as evidence that the queried event escaped masking.

These clips were selected from the 50-item, 1 FPS Qwen3-VL pilot. Each window
is outside the padded official-evidence mask and was classified as positive by
the current normalized-gain rule. Clip timestamps start at zero; the original
video interval is encoded in each filename.

## 1. 6447803681_q8 — high-score 8-second candidate

- Original interval: 72–80 s
- Official evidence: 18.6–42.4 s
- Padded official mask: 17.6–43.4 s
- Scale: 8 s / stride 4 s
- Normalized gain: 2.0131 (threshold 0.7607)
- Question: What does the baby do as the lady brings the ball closer at the start?
- Choices: A. falls back down; B. squats down; C. no reaction; D. bite it; E. throw it
- Gold: D. bite it
- Clip: `codex_6447803681_72_80.mp4`
- Full context: `codex_6447803681_full_review.mp4`

## 2. 3205604574_q2 — near-threshold 16-second candidate

- Original interval: 64–80 s
- Official evidence: 21.4–33.6 s
- Padded official mask: 20.4–34.6 s
- Scale: 16 s / stride 8 s
- Normalized gain: 0.4295 (threshold 0.4266)
- Question: Why did the girl in pink lift her hands up?
- Choices: A. get the paint; B. drink her cup of water; C. copy boy's gestures;
  D. wants boy to hold her hand; E. to get the adults' attention
- Gold: C. copy boy's gestures
- Clip: `codex_3205604574_64_80.mp4`
- Full context: `codex_3205604574_full_review.mp4`

## 3. 4129080039_q4 — near-threshold 8-second candidate

- Original interval: 16–24 s
- Official evidence: 0–9.1 s
- Padded official mask: 0–10.1 s
- Scale: 8 s / stride 4 s
- Normalized gain: 0.7927 (threshold 0.7607)
- Question: How does the nearest parrot move across the cage?
- Choices: A. fly; B. use beak to pull itself; C. walk on the ground; D. skip; E. roll
- Gold: B. use beak to pull itself
- Clip: `codex_4129080039_16_24.mp4`
- Full context: `codex_4129080039_full_review.mp4`

The three verified clips and three full-context review copies are silent
because the measurement pipeline scores sampled visual frames and does not use
audio. The full-context copies retain the complete timeline at 480 px width;
the candidate clips use 720 px width. SHA-256 hashes were checked against the
server copies after download.
