#!/usr/bin/env python3
"""Shared primitives for calibrated NExT-GQA sufficiency scanning.

The construction model is never asked whether a window is "answerable".  It is
forced to emit one option letter and the A--E next-token distribution is used as
the behavioral measurement.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

PIPELINE_VERSION = "nextgqa-calibrated-sufficiency-v3.0"
PROMPT_VERSION = "nextgqa-forced-choice-logprob-v1.0"
LETTERS = tuple("ABCDE")


@dataclass(frozen=True)
class QAItem:
    index: int
    video_id: str
    qid: str
    question: str
    choices: tuple[str, ...]
    gold: str | None
    evidence: tuple[tuple[float, float], ...]
    duration: float
    video_path: Path
    evidence_count: int
    question_type: str

    @property
    def key(self) -> str:
        return f"{self.video_id}_q{self.qid}"


@dataclass(frozen=True)
class Frame:
    timestamp: float
    jpeg_b64: str


@dataclass(frozen=True)
class Endpoint:
    tag: str
    base_url: str
    model: str
    api_key: str = "EMPTY"


def append_jsonl(path: Path, record: object, lock: threading.Lock | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False) + "\n"
    if lock:
        with lock:
            with path.open("a", encoding="utf-8") as stream:
                stream.write(line)
                stream.flush()
        return
    with path.open("a", encoding="utf-8") as stream:
        stream.write(line)
        stream.flush()


def atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def latest_jsonl_records(paths: Iterable[Path]) -> dict[str, dict]:
    records: dict[str, dict] = {}
    for path in sorted(paths):
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    records[str(row["key"])] = row
                except (json.JSONDecodeError, KeyError) as exc:
                    raise ValueError(f"invalid JSONL record {path}:{line_number}: {exc}") from exc
    return records


def merge_intervals(
    intervals: Iterable[Sequence[float]],
    *,
    duration: float | None = None,
    padding: float = 0.0,
    merge_gap: float = 0.0,
) -> list[tuple[float, float]]:
    cleaned: list[tuple[float, float]] = []
    for raw in intervals:
        if len(raw) != 2:
            continue
        start, end = float(raw[0]) - padding, float(raw[1]) + padding
        if duration is not None:
            start, end = max(0.0, start), min(float(duration), end)
        if math.isfinite(start) and math.isfinite(end) and end > start:
            cleaned.append((start, end))
    cleaned.sort()
    merged: list[tuple[float, float]] = []
    for start, end in cleaned:
        if merged and start <= merged[-1][1] + merge_gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def interval_duration(intervals: Iterable[Sequence[float]]) -> float:
    return sum(float(end) - float(start) for start, end in merge_intervals(intervals))


def interval_adds_time(
    candidate: Sequence[float], existing: Sequence[Sequence[float]], tolerance: float = 1e-6
) -> bool:
    before = interval_duration(existing)
    after = interval_duration([*existing, candidate])
    return after > before + tolerance


def sliding_windows(duration: float, width: float, stride: float) -> list[tuple[float, float]]:
    """Return full-width windows and one end-aligned tail window when necessary."""
    if duration <= 0 or width <= 0 or stride <= 0:
        raise ValueError(f"invalid window settings: duration={duration}, width={width}, stride={stride}")
    if duration <= width:
        return [(0.0, duration)]
    windows: list[tuple[float, float]] = []
    start = 0.0
    while start + width <= duration + 1e-9:
        windows.append((start, min(duration, start + width)))
        start += stride
    tail = (duration - width, duration)
    if not windows or abs(windows[-1][0] - tail[0]) > 1e-6:
        windows.append(tail)
    return windows


def parse_scales(text: str) -> list[tuple[float, float]]:
    scales: list[tuple[float, float]] = []
    for part in text.split(","):
        width_text, stride_text = part.strip().split(":", 1)
        width, stride = float(width_text), float(stride_text)
        if width <= 0 or stride <= 0:
            raise ValueError(f"invalid scale {part!r}")
        scales.append((width, stride))
    if not scales:
        raise ValueError("at least one scale is required")
    return scales


def scale_key(width: float, stride: float) -> str:
    def compact(value: float) -> str:
        return str(int(value)) if float(value).is_integer() else str(value).replace(".", "p")
    return f"w{compact(width)}_s{compact(stride)}"


def parse_endpoints(values: Sequence[str]) -> list[Endpoint]:
    endpoints: list[Endpoint] = []
    seen: set[str] = set()
    api_key = os.environ.get("VLLM_API_KEY", "EMPTY")
    for value in values:
        parts = value.split(",", 2)
        if len(parts) != 3 or not all(part.strip() for part in parts):
            raise ValueError("--endpoint must be TAG,BASE_URL,MODEL")
        endpoint = Endpoint(parts[0].strip(), parts[1].strip(), parts[2].strip(), api_key)
        if endpoint.tag in seen:
            raise ValueError(f"duplicate endpoint tag: {endpoint.tag}")
        seen.add(endpoint.tag)
        endpoints.append(endpoint)
    if not endpoints:
        raise ValueError("at least one --endpoint is required")
    return endpoints


def resolve_gold_answer(raw: object, choices: Sequence[str]) -> str | None:
    text = " ".join(str(raw).split()).strip()
    upper = text.upper()
    if upper in LETTERS:
        return upper
    if text.isdigit() and 0 <= int(text) < len(choices):
        return LETTERS[int(text)]
    normalized = text.casefold()
    matches = [index for index, choice in enumerate(choices) if " ".join(choice.split()).casefold() == normalized]
    if len(matches) == 1:
        return LETTERS[matches[0]]
    return None


def load_worklist(filtered_json: Path, video_dir: Path, mapping_json: Path) -> list[QAItem]:
    rows = json.loads(filtered_json.read_text(encoding="utf-8"))
    mapping = json.loads(mapping_json.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"{filtered_json} must contain a JSON list")
    items: list[QAItem] = []
    seen: set[tuple[str, str]] = set()
    for index, row in enumerate(rows):
        video_id, qid = str(row["video_id"]), str(row["qid"])
        if (video_id, qid) in seen:
            raise ValueError(f"duplicate item: {video_id} q{qid}")
        seen.add((video_id, qid))
        choices = tuple(str(choice).strip() for choice in row["choices"])
        if len(choices) != 5:
            raise ValueError(f"{video_id} q{qid}: expected five choices, got {len(choices)}")
        gold = resolve_gold_answer(row["answer"], choices)
        evidence = merge_intervals(
            [(float(span["start"]), float(span["end"])) for span in row["evidence_intervals"]],
            duration=float(row["video_duration"]),
        )
        if not evidence:
            raise ValueError(f"{video_id} q{qid}: no valid evidence")
        relative = mapping.get(video_id)
        if relative is None:
            raise ValueError(f"missing video mapping for {video_id}")
        items.append(QAItem(
            index=index,
            video_id=video_id,
            qid=qid,
            question=" ".join(str(row["question"]).split()),
            choices=choices,
            gold=gold,
            evidence=tuple(evidence),
            duration=float(row["video_duration"]),
            video_path=video_dir / f"{relative}.mp4",
            evidence_count=int(row.get("evidence_count", len(evidence))),
            question_type=str(row.get("question_type", row.get("source_task", ""))),
        ))
    return items


def selected_indices(length: int, shard: int | None, num_shards: int | None, limit: int | None) -> list[int]:
    indices = list(range(length))
    if shard is not None:
        if not num_shards or not 0 <= shard < num_shards:
            raise ValueError(f"invalid shard {shard}/{num_shards}")
        indices = [index for index in indices if index % num_shards == shard]
    if limit is not None:
        indices = indices[:limit]
    return indices


def require_binaries(*names: str) -> None:
    import shutil
    missing = [name for name in names if shutil.which(name) is None and not Path(name).is_file()]
    if missing:
        raise RuntimeError(f"missing required binaries: {missing}")


def probe_video(path: Path, ffprobe: str = "ffprobe") -> dict[str, float | int | bool]:
    command = [
        ffprobe, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate:format=duration",
        "-of", "json", str(path),
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=True)
    payload = json.loads(result.stdout)
    stream = payload["streams"][0]
    numerator, denominator = stream["r_frame_rate"].split("/", 1)
    return {
        "duration": float(payload["format"]["duration"]),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": float(numerator) / max(float(denominator), 1e-9),
    }


def extract_frame_grid(
    video: Path,
    *,
    fps: float,
    width: int,
    ffmpeg: str = "ffmpeg",
    temp_root: Path | None = None,
) -> tuple[tempfile.TemporaryDirectory, list[Frame]]:
    if fps <= 0:
        raise ValueError("fps must be positive")
    holder = tempfile.TemporaryDirectory(prefix="nextgqa_score_", dir=str(temp_root) if temp_root else None)
    out_pattern = str(Path(holder.name) / "%06d.jpg")
    vf = f"fps={fps}:start_time=0,scale={width}:-2"
    command = [
        ffmpeg, "-nostdin", "-y", "-loglevel", "error", "-i", str(video),
        "-an", "-vf", vf, "-q:v", "3", out_pattern,
    ]
    try:
        subprocess.run(command, text=True, capture_output=True, check=True)
        files = sorted(Path(holder.name).glob("*.jpg"))
        if not files:
            raise RuntimeError("ffmpeg produced no frames")
        frames = [
            Frame(index / fps, base64.b64encode(path.read_bytes()).decode("ascii"))
            for index, path in enumerate(files)
        ]
        return holder, frames
    except Exception:
        holder.cleanup()
        raise


def frames_in_range(frames: Sequence[Frame], start: float, end: float) -> list[Frame]:
    selected = [frame for frame in frames if start - 1e-9 <= frame.timestamp < end - 1e-9]
    if selected:
        return selected
    midpoint = (start + end) / 2.0
    return [min(frames, key=lambda frame: abs(frame.timestamp - midpoint))]


def frames_in_intervals(frames: Sequence[Frame], intervals: Sequence[Sequence[float]]) -> list[Frame]:
    selected = [
        frame for frame in frames
        if any(float(start) - 1e-9 <= frame.timestamp <= float(end) + 1e-9 for start, end in intervals)
    ]
    if selected:
        return selected
    chosen: dict[float, Frame] = {}
    for start, end in intervals:
        midpoint = (float(start) + float(end)) / 2.0
        frame = min(frames, key=lambda item: abs(item.timestamp - midpoint))
        chosen[frame.timestamp] = frame
    return [chosen[key] for key in sorted(chosen)]


def _decode_image(jpeg_b64: str):
    from io import BytesIO
    from PIL import Image
    return Image.open(BytesIO(base64.b64decode(jpeg_b64))).convert("RGB")


def _encode_image(image) -> str:
    from io import BytesIO
    stream = BytesIO()
    image.save(stream, format="JPEG", quality=90)
    return base64.b64encode(stream.getvalue()).decode("ascii")


def gray_like(frame: Frame) -> str:
    from PIL import Image
    image = _decode_image(frame.jpeg_b64)
    return _encode_image(Image.new("RGB", image.size, (127, 127, 127)))


def blind_frames(frames: Sequence[Frame]) -> list[Frame]:
    if not frames:
        raise ValueError("cannot blind an empty frame sequence")
    gray = gray_like(frames[0])
    return [Frame(frame.timestamp, gray) for frame in frames]


def choose_safe_frame(frames: Sequence[Frame], interval: Sequence[float], removed: Sequence[Sequence[float]]) -> Frame | None:
    start, end = float(interval[0]), float(interval[1])
    candidates = sorted(
        (frame for frame in frames
         if all(not (float(a) <= frame.timestamp <= float(b)) for a, b in removed)),
        key=lambda frame: min(abs(frame.timestamp - start), abs(frame.timestamp - end)),
    )
    return candidates[0] if candidates else None


def choose_safe_timestamp(
    duration: float,
    sampling_fps: float,
    interval: Sequence[float],
    removed: Sequence[Sequence[float]],
) -> float | None:
    """Mirror ``choose_safe_frame`` on the same sampling grid for final encoding."""
    start, end = float(interval[0]), float(interval[1])
    frame_count = max(1, int(math.ceil(duration * sampling_fps)))
    candidates = [
        index / sampling_fps
        for index in range(frame_count)
        if index / sampling_fps < duration
        and all(not (float(a) <= index / sampling_fps <= float(b)) for a, b in removed)
    ]
    return min(candidates, key=lambda timestamp: min(abs(timestamp - start), abs(timestamp - end))) if candidates else None


def mask_frames(
    frames: Sequence[Frame],
    intervals: Sequence[Sequence[float]],
    *,
    blur_sigma: float,
) -> list[Frame]:
    """Apply the same duration-preserving freeze/blur semantics used for final videos."""
    from PIL import ImageFilter
    merged = merge_intervals(intervals)
    if not merged:
        return list(frames)
    replacements: list[tuple[tuple[float, float], str]] = []
    gray = gray_like(frames[0])
    for interval in merged:
        safe = choose_safe_frame(frames, interval, merged)
        if safe is None:
            replacement = gray
        else:
            replacement = _encode_image(_decode_image(safe.jpeg_b64).filter(ImageFilter.GaussianBlur(blur_sigma)))
        replacements.append((interval, replacement))
    output: list[Frame] = []
    for frame in frames:
        replacement = next(
            (image for (start, end), image in replacements if start <= frame.timestamp <= end),
            None,
        )
        output.append(Frame(frame.timestamp, replacement or frame.jpeg_b64))
    return output


def coverage_stats(intervals: Sequence[Sequence[float]], duration: float) -> dict[str, float]:
    covered = interval_duration(merge_intervals(intervals, duration=duration))
    return {
        "covered_seconds": round(covered, 6),
        "coverage_ratio": round(covered / max(duration, 1e-9), 6),
        "visible_seconds": round(max(0.0, duration - covered), 6),
    }


def minimal_positive_windows(windows: Sequence[Sequence[float]]) -> list[tuple[float, float]]:
    """Drop a positive ancestor when a strictly smaller positive child exists."""
    candidates = sorted({(round(float(a), 6), round(float(b), 6)) for a, b in windows}, key=lambda x: (x[1]-x[0], x[0]))
    kept: list[tuple[float, float]] = []
    for candidate in candidates:
        start, end = candidate
        if any(start <= a + 1e-6 and end >= b - 1e-6 for a, b in kept):
            continue
        kept.append(candidate)
    return sorted(kept)


def _logsumexp(values: Sequence[float]) -> float:
    maximum = max(values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in values))


def canonical_letter(token: str) -> str | None:
    stripped = token.strip()
    match = re.fullmatch(r"(?:answer\s*[:=]?\s*)?([A-E])(?:[.\),:]*)", stripped, flags=re.IGNORECASE)
    return match.group(1).upper() if match else None


def parse_choice_logprobs(top_logprobs: Sequence[object], gold: str) -> dict[str, object]:
    grouped: dict[str, list[float]] = {letter: [] for letter in LETTERS}
    observed: list[dict[str, object]] = []
    for item in top_logprobs:
        token = item.get("token") if isinstance(item, dict) else getattr(item, "token", None)
        logprob = item.get("logprob") if isinstance(item, dict) else getattr(item, "logprob", None)
        if token is None or logprob is None:
            continue
        observed.append({"token": str(token), "logprob": float(logprob)})
        letter = canonical_letter(str(token))
        if letter:
            grouped[letter].append(float(logprob))
    missing = [letter for letter, values in grouped.items() if not values]
    if missing:
        preview = observed[:20]
        raise ValueError(f"missing A-E logprobs for {missing}; observed={preview}")
    letter_logprobs = {letter: _logsumexp(values) for letter, values in grouped.items()}
    normalizer = _logsumexp(list(letter_logprobs.values()))
    probabilities = {letter: math.exp(value - normalizer) for letter, value in letter_logprobs.items()}
    distractors = [value for letter, value in letter_logprobs.items() if letter != gold]
    return {
        "choice_logprobs": letter_logprobs,
        "choice_probabilities": probabilities,
        "gold_probability": probabilities[gold],
        "margin": letter_logprobs[gold] - _logsumexp(distractors),
        "prediction": max(probabilities, key=probabilities.get),
        "observed_top_logprobs": observed,
    }


def scoring_prompt(item: QAItem) -> str:
    options = "\n".join(f"{letter}. {choice}" for letter, choice in zip(LETTERS, item.choices))
    return f"""Answer the multiple-choice question using only the supplied video frames.

You must select exactly one option. Do not abstain. Output exactly one uppercase letter from A, B, C, D, E and no other text.

Question: {item.question}

Options:
{options}

Answer:"""


class ForcedChoiceScorer:
    def __init__(self, endpoint: Endpoint, *, timeout: float = 300.0, top_logprobs: int = 20):
        from openai import OpenAI
        self.endpoint = endpoint
        self.timeout = timeout
        self.top_logprobs = top_logprobs
        self.client = OpenAI(base_url=endpoint.base_url, api_key=endpoint.api_key, timeout=timeout)

    def score(self, frames: Sequence[Frame], item: QAItem) -> dict[str, object]:
        if not frames:
            raise ValueError("cannot score empty frames")
        if item.gold not in LETTERS:
            raise ValueError(f"item has no unique A-E gold mapping: {item.key}")
        content: list[dict[str, object]] = []
        for frame in frames:
            content.append({"type": "text", "text": f"Frame at {frame.timestamp:.3f}s:"})
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{frame.jpeg_b64}"},
            })
        content.append({"type": "text", "text": scoring_prompt(item)})
        response = self.client.chat.completions.create(
            model=self.endpoint.model,
            messages=[{"role": "user", "content": content}],
            temperature=0.0,
            max_tokens=1,
            logprobs=True,
            top_logprobs=self.top_logprobs,
        )
        choice = response.choices[0]
        logprob_content = getattr(getattr(choice, "logprobs", None), "content", None)
        if not logprob_content:
            raise ValueError("server returned no generated-token logprobs")
        top = getattr(logprob_content[0], "top_logprobs", None)
        if not top:
            raise ValueError("server returned no top_logprobs")
        parsed = parse_choice_logprobs(top, item.gold)
        parsed.update({
            "raw_answer": choice.message.content,
            "n_frames": len(frames),
            "timestamps": [round(frame.timestamp, 6) for frame in frames],
            "model_tag": self.endpoint.tag,
            "model": self.endpoint.model,
        })
        return parsed


def score_pair(scorer: ForcedChoiceScorer, visual: Sequence[Frame], blind: Sequence[Frame], item: QAItem) -> dict:
    visual_score = scorer.score(visual, item)
    blind_score = scorer.score(blind, item)
    gain = float(visual_score["margin"]) - float(blind_score["margin"])
    return {"visual": visual_score, "blind": blind_score, "gain": gain}


def normalized_gain(gain: float, full_gain: float, epsilon: float = 1e-6) -> float | None:
    return gain / full_gain if full_gain > epsilon else None


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_freeze_video(
    source: Path,
    output: Path,
    intervals: Sequence[Sequence[float]],
    *,
    expected_duration: float,
    sampling_fps: float,
    blur_sigma: float,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
    threads: int = 4,
) -> dict[str, object]:
    """Encode one final duration-preserving freeze video."""
    probe = probe_video(source, ffprobe)
    tolerance = max(0.5, 2.0 / max(float(probe["fps"]), 1e-6))
    if abs(float(probe["duration"]) - expected_duration) > tolerance:
        raise ValueError(
            f"source/annotation duration mismatch: {probe['duration']} vs {expected_duration}"
        )
    merged = merge_intervals(intervals, duration=expected_duration)
    if not merged:
        raise ValueError("no intervals to freeze")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="nextgqa_final_") as directory:
        replacement_paths: list[Path] = []
        replacements: list[dict[str, object]] = []
        for index, (start, end) in enumerate(merged):
            safe = choose_safe_timestamp(
                expected_duration, sampling_fps, (start, end), merged
            )
            frame_path = Path(directory) / f"replacement_{index:03d}.png"
            if safe is None:
                command = [
                    ffmpeg, "-nostdin", "-y", "-f", "lavfi", "-i",
                    f"color=c=gray:s={probe['width']}x{probe['height']}:d=0.04",
                    "-frames:v", "1", str(frame_path),
                ]
                kind = "gray"
            else:
                command = [
                    ffmpeg, "-nostdin", "-y", "-ss", f"{safe:.6f}", "-i", str(source),
                    "-frames:v", "1", "-an", "-vf", f"gblur=sigma={blur_sigma}", str(frame_path),
                ]
                kind = "blurred_safe_frame"
            subprocess.run(command, text=True, capture_output=True, check=True)
            replacement_paths.append(frame_path)
            replacements.append({"interval": [start, end], "kind": kind, "timestamp": safe})

        command = [ffmpeg, "-nostdin", "-y", "-i", str(source)]
        for path in replacement_paths:
            command.extend(["-loop", "1", "-i", str(path)])
        filters: list[str] = []
        current = "0:v"
        for index, (start, end) in enumerate(merged, start=1):
            label = f"masked{index}"
            filters.append(
                f"[{current}][{index}:v]overlay=0:0:shortest=1:enable='between(t,{start:.6f},{end:.6f})'[{label}]"
            )
            current = label
        filters.append(f"[{current}]scale=trunc(iw/2)*2:trunc(ih/2)*2[vout]")
        command.extend([
            "-filter_complex", ";".join(filters), "-map", "[vout]", "-an",
            "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
            "-threads", str(threads), "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(output),
        ])
        try:
            subprocess.run(command, text=True, capture_output=True, check=True)
        except Exception:
            output.unlink(missing_ok=True)
            raise
    output_probe = probe_video(output, ffprobe)
    if abs(float(output_probe["duration"]) - float(probe["duration"])) > tolerance:
        output.unlink(missing_ok=True)
        raise ValueError("final freeze video changed duration")
    return {"source_probe": probe, "output_probe": output_probe, "replacements": replacements}
