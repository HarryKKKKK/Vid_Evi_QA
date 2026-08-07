# NExT-GQA boundary-leak review

These cases were recovered from Git commit `511c051` and checked against the
historical insufficient-video construction rule.  That rule masks exactly the
official interval with FFmpeg `between(t,start,end)` and uses no temporal
padding.  The local `*_insufficient.mp4` files reproduce those mask boundaries
for visual review.  They use a different local re-encoder and must not be used
for byte-level or model-score reproduction.

## Confirmed direct boundary leaks

### 11794871936_q5

- Question: What expression did the girl give after she pointed at the camera?
- Choices: A. wipe her face; B. cry; C. smile; D. presented her cup;
  E. she started talking
- Gold and historical model answer: C, smile
- Official mask: 9.9--11.9 s
- Historical model evidence: 9.8--12.1 s
- Unmasked sampled frames: 9.82 s and 12.10 s
- Finding: both unmasked frames directly show the same post-pointing smile.
  The official interval starts too late and ends before the expression ends.

### 8501394817_q5

- Question: Why did the boy raise his arms in the middle of the video?
- Choices: A. throw the ball; B. to touch the sandals; C. wants to play with
  baby; D. to dance on the floor; E. tired
- Gold and historical model answer: A, throw the ball
- Official mask: 5.4--7.0 s
- Historical model evidence: 5.0--7.2 s
- Unmasked sampled frames: 4.99 s, 5.32 s and 7.25 s
- Finding: before the mask begins, the boy is already raising the basketball
  toward the hoop.  This is the same shot preparation, not a repeated action.

### 6660008425_q1

- Question: What did the girl on the right do after lifting the cup up?
- Choices: A. looked down at ground; B. give out glass; C. drink from straw;
  D. move closely to the camera; E. remove the drinks
- Gold and historical model answer: C, drink from straw
- Official mask: 17.3--19.0 s
- Historical model evidence: 16.5--18.9 s
- Unmasked sampled frames: 16.52 s and 17.12 s
- Finding: the girl on the right already has the cup/straw at her mouth before
  17.3 s.  The answer action begins before the official interval.

## Same-event auxiliary or weaker boundary cases

### 7001228068_q5

- Question: Why did the girl bend forward at the beginning of the video?
- Gold: A, pick up leash
- Official mask: 0.3--1.5 s
- Unmasked sampled frame: 1.93 s
- Finding: the bend/pickup itself is masked, but the immediately following
  frame directly shows the girl holding the red leash.  This is genuine
  same-event outcome evidence, but is auxiliary rather than a visible action.

### 6031282471_q9

- Question: What does the boy do as the girl sprayed water at the start?
- Gold: C, stand at the side and watch
- Official mask: 0.2--3.4 s
- Unmasked sampled frame: 3.48 s
- Finding: immediately after the mask the boy is still standing beside and
  facing the girl.  This is likely a continuation of the queried event, but a
  single frame is weaker evidence for the concurrent water-spraying condition.

## Review files

- `<video_id>.mp4`: recovered original video
- `<video_id>_insufficient.mp4`: locally reconstructed exact-boundary mask
- `<video_id>_sample_<time>.jpg`: a frame actually sampled by the historical
  32-frame uniform classifier
- `*_boundary.jpg`: original contact sheet around the boundary
- `*_masked_boundary.jpg`: contact sheet after applying the historical mask
