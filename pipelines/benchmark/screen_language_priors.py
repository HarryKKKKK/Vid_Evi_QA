#!/usr/bin/env python3
"""Screen NExT-GQA anchors for question/option-only language priors.

The script never sends video, timestamps, evidence intervals, or the reference
answer to a model. It asks independent text models to answer deterministic
option permutations, stores every raw prediction in an append-only JSONL file,
and produces a conservative *candidate* exclusion list. It does not delete or
rewrite benchmark manifests.

The same output directory can be resumed across separate Qwen, Llama, and GPT
runs, which is useful when local text-model servers cannot run simultaneously.
"""

from __future__ import annotations

# Layout compatibility: data stays at its existing paths; frozen copies remain standalone.
from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_file = _LayoutPath(__file__).resolve()
_layout_root = (_layout_file.parents[2]
                if _layout_file.as_posix().endswith('/pipelines/benchmark/screen_language_priors.py') else None)
_LAYOUT_LEGACY_FILE = str(_layout_root / 'scripts/analysis/screen_language_priors.py') if _layout_root else __file__
if _layout_root is not None:
    for _layout_dir in (_layout_root, _layout_root / 'src',
                        _layout_root / 'scripts/eval',
                        _layout_root / 'scripts/analysis', _layout_file.parent):
        if str(_layout_dir) not in _layout_sys.path:
            _layout_sys.path.insert(0, str(_layout_dir))


import argparse
import hashlib
import json
import os
import random
import re
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(_LAYOUT_LEGACY_FILE).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "nextgqa_pipeline/nextgqa_filtered_no_evidence_leak.json"
DEFAULT_CANDIDATES = ROOT / "nextgqa_pipeline/partial_candidates_full_n5279.json"
DEFAULT_OUTPUT_DIR = ROOT / "reports/language_prior_screen"
LETTERS = "ABCDE"
SYSTEM_PROMPT = (
    "You are taking a multiple-choice video question test without access to the video. "
    "Use only the question text and answer options. You must choose one option even if "
    "uncertain. Do not claim to have seen the video. Return only the requested JSON."
)
JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string", "enum": list(LETTERS)},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string"},
    },
    "required": ["answer", "confidence", "reason"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class Backend:
    name: str
    model: str
    api_mode: str
    base_url: str | None
    api_key: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--backends",
        default="qwen,llama,gpt",
        help="Comma-separated subset of qwen,llama,gpt. Runs can be resumed separately.",
    )
    parser.add_argument("--qwen-base-url", default=os.getenv("QWEN_BASE_URL"))
    parser.add_argument("--qwen-model", default=os.getenv("QWEN_MODEL"))
    parser.add_argument("--qwen-api-key", default=os.getenv("QWEN_API_KEY", "EMPTY"))
    parser.add_argument("--llama-base-url", default=os.getenv("LLAMA_BASE_URL"))
    parser.add_argument("--llama-model", default=os.getenv("LLAMA_MODEL"))
    parser.add_argument("--llama-api-key", default=os.getenv("LLAMA_API_KEY", "EMPTY"))
    parser.add_argument(
        "--gpt-model",
        default=os.getenv("OPENAI_MODEL"),
        help="Required for the gpt backend; no model is assumed because account availability varies.",
    )
    parser.add_argument("--gpt-base-url", default=os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--gpt-api-key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument(
        "--gpt-api-mode",
        choices=("responses", "chat"),
        default=os.getenv("GPT_API_MODE", "responses"),
        help=(
            "Use responses for the official OpenAI API (default), or chat for an "
            "OpenAI-compatible gateway exposing /v1/chat/completions."
        ),
    )
    parser.add_argument("--permutations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--limit", type=int, help="Debug on the first N anchors after sorting.")
    parser.add_argument(
        "--per-model-threshold",
        type=float,
        default=0.8,
        help="A model is prior-positive when its permutation accuracy reaches this value.",
    )
    parser.add_argument(
        "--overall-threshold",
        type=float,
        default=0.8,
        help="Minimum pooled accuracy across all requested models and permutations.",
    )
    parser.add_argument(
        "--min-positive-models",
        type=int,
        default=2,
        help="Minimum independent prior-positive model families for a removal candidate.",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Do not call APIs; rebuild summaries from raw_predictions.jsonl.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs/configuration and print one permuted prompt without API calls.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def item_key(row: dict) -> str:
    return f"{row['video_id']}_q{row['qid']}"


def load_items(manifest_path: Path, candidates_path: Path) -> list[dict]:
    manifest = load_json(manifest_path)
    candidates = load_json(candidates_path)
    wanted = {(str(row["video_id"]), str(row["qid"])) for row in candidates}
    by_key = {
        (str(row["video_id"]), str(row["qid"])): row
        for row in manifest
        if (str(row["video_id"]), str(row["qid"])) in wanted
    }
    missing = sorted(wanted - set(by_key))
    if missing:
        raise SystemExit(f"{len(missing)} candidate keys are missing from the manifest: {missing[:5]}")
    items = []
    for key in sorted(wanted):
        row = dict(by_key[key])
        choices = row.get("choices")
        answer = str(row.get("answer", "")).strip().upper()
        if not isinstance(choices, list) or len(choices) != 5:
            raise SystemExit(f"{key[0]}_q{key[1]} does not have exactly five choices")
        if answer not in LETTERS:
            raise SystemExit(f"{key[0]}_q{key[1]} has invalid gold answer {answer!r}")
        row["video_id"], row["qid"], row["answer"] = key[0], key[1], answer
        items.append(row)
    return items


def deterministic_order(row: dict, permutation_index: int, seed: int) -> list[int]:
    payload = f"{seed}:{item_key(row)}:{permutation_index}".encode("utf-8")
    local_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    order = list(range(5))
    random.Random(local_seed).shuffle(order)
    return order


def permuted_case(row: dict, permutation_index: int, seed: int) -> dict:
    order = deterministic_order(row, permutation_index, seed)
    original_gold_index = LETTERS.index(row["answer"])
    displayed_gold = LETTERS[order.index(original_gold_index)]
    displayed_choices = [str(row["choices"][index]) for index in order]
    return {
        "order": order,
        "displayed_choices": displayed_choices,
        "displayed_gold": displayed_gold,
    }


def user_prompt(row: dict, case: dict) -> str:
    options = "\n".join(
        f"{letter}. {choice}" for letter, choice in zip(LETTERS, case["displayed_choices"])
    )
    return f"""Answer this question using only its language and options. You do not have the video.

Question: {row['question']}

Options:
{options}

Return one JSON object with:
- answer: exactly one letter A, B, C, D, or E
- confidence: a number from 0 to 1
- reason: one concise sentence describing the textual or commonsense basis for the guess"""


def parse_object(text: str) -> dict:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip(), flags=re.I | re.S)
    match = re.search(r"\{.*\}", cleaned, flags=re.S)
    if not match:
        raise ValueError("response contains no JSON object")
    value = json.loads(match.group(0))
    answer = str(value.get("answer", "")).strip().upper()
    if answer not in LETTERS:
        raise ValueError(f"invalid answer: {answer!r}")
    try:
        confidence = float(value.get("confidence"))
    except (TypeError, ValueError) as exc:
        raise ValueError("confidence is not numeric") from exc
    if not 0 <= confidence <= 1:
        raise ValueError(f"confidence outside [0,1]: {confidence}")
    reason = str(value.get("reason", "")).strip()
    if not reason:
        raise ValueError("reason is empty")
    return {"answer": answer, "confidence": confidence, "reason": reason}


def make_backends(args: argparse.Namespace, *, validate_credentials: bool = True) -> list[Backend]:
    names = [value.strip().lower() for value in args.backends.split(",") if value.strip()]
    if not names or len(names) != len(set(names)) or set(names) - {"qwen", "llama", "gpt"}:
        raise SystemExit("--backends must be a unique comma-separated subset of qwen,llama,gpt")
    values = {
        "qwen": Backend("qwen", args.qwen_model or "", "chat", args.qwen_base_url, args.qwen_api_key),
        "llama": Backend("llama", args.llama_model or "", "chat", args.llama_base_url, args.llama_api_key),
        "gpt": Backend(
            "gpt",
            args.gpt_model or "",
            args.gpt_api_mode,
            args.gpt_base_url,
            args.gpt_api_key or "",
        ),
    }
    selected = [values[name] for name in names]
    if validate_credentials:
        for backend in selected:
            if not backend.model:
                raise SystemExit(f"Missing model for {backend.name}; set the matching --{backend.name}-model or environment variable")
            if backend.name in {"qwen", "llama"} and not backend.base_url:
                raise SystemExit(f"Missing --{backend.name}-base-url (or {backend.name.upper()}_BASE_URL)")
            if backend.name == "gpt" and not backend.api_key:
                raise SystemExit("Missing OPENAI_API_KEY or --gpt-api-key")
    return selected


def create_client(backend: Backend, timeout: float):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("The openai Python package is required: pip install -U openai") from exc
    kwargs: dict[str, Any] = {"api_key": backend.api_key, "timeout": timeout, "max_retries": 0}
    if backend.base_url:
        kwargs["base_url"] = backend.base_url
    return OpenAI(**kwargs)


def call_model(client, backend: Backend, prompt: str) -> tuple[str, dict]:
    if backend.api_mode == "responses":
        response = client.responses.create(
            model=backend.model,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "blind_forced_choice",
                    "strict": True,
                    "schema": JSON_SCHEMA,
                }
            },
            max_output_tokens=256,
        )
        raw = response.output_text or ""
        usage = response.usage.model_dump() if response.usage else None
    else:
        request: dict[str, Any] = dict(
            model=backend.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=256,
            response_format={"type": "json_object"},
        )
        if backend.name == "qwen":
            # Qwen3 otherwise spends tokens in a hidden/visible thinking block,
            # which is unnecessary for this short forced-choice screening task.
            request["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        response = client.chat.completions.create(**request)
        raw = response.choices[0].message.content or ""
        usage = response.usage.model_dump() if response.usage else None
    return raw, usage


def prediction_key(row: dict) -> str:
    return f"{row['item_key']}::{row['backend']}::p{int(row['permutation_index']):02d}"


def load_latest_jsonl(path: Path) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    if not path.exists():
        return latest
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            latest[prediction_key(row)] = row
        except Exception as exc:
            raise SystemExit(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return latest


def run_one(
    row: dict,
    permutation_index: int,
    backend: Backend,
    client,
    args: argparse.Namespace,
) -> dict:
    case = permuted_case(row, permutation_index, args.seed)
    prompt = user_prompt(row, case)
    started = time.time()
    last_error = None
    for attempt in range(args.max_retries + 1):
        try:
            raw, usage = call_model(client, backend, prompt)
            parsed = parse_object(raw)
            original_prediction = LETTERS[case["order"][LETTERS.index(parsed["answer"])]]
            return {
                "key": f"{item_key(row)}::{backend.name}::p{permutation_index:02d}",
                "item_key": item_key(row),
                "video_id": row["video_id"],
                "qid": row["qid"],
                "question_type": row.get("question_type"),
                "backend": backend.name,
                "model": backend.model,
                "permutation_index": permutation_index,
                "display_order_original_indices": case["order"],
                "displayed_prediction": parsed["answer"],
                "original_prediction": original_prediction,
                "correct": parsed["answer"] == case["displayed_gold"],
                "confidence": parsed["confidence"],
                "reason": parsed["reason"],
                "raw_response": raw,
                "usage": usage,
                "status": "ok",
                "attempts": attempt + 1,
                "elapsed_seconds": round(time.time() - started, 3),
            }
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < args.max_retries:
                time.sleep(min(2**attempt, 8))
    return {
        "key": f"{item_key(row)}::{backend.name}::p{permutation_index:02d}",
        "item_key": item_key(row),
        "video_id": row["video_id"],
        "qid": row["qid"],
        "question_type": row.get("question_type"),
        "backend": backend.name,
        "model": backend.model,
        "permutation_index": permutation_index,
        "status": "error",
        "error": last_error,
        "attempts": args.max_retries + 1,
        "elapsed_seconds": round(time.time() - started, 3),
    }


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def aggregate(
    items: list[dict],
    backends: list[Backend],
    args: argparse.Namespace,
    raw_path: Path,
) -> dict:
    latest = load_latest_jsonl(raw_path)
    backend_models = {}
    for backend in backends:
        recorded_model = next(
            (
                str(row.get("model"))
                for row in latest.values()
                if row.get("backend") == backend.name and row.get("model")
            ),
            "",
        )
        backend_models[backend.name] = backend.model or recorded_model
    by_item: dict[str, list[dict]] = defaultdict(list)
    for row in latest.values():
        if row.get("backend") in {backend.name for backend in backends}:
            by_item[str(row["item_key"])].append(row)

    expected_per_item = len(backends) * args.permutations
    item_scores, candidates, incomplete = [], [], []
    for item in items:
        key = item_key(item)
        rows = [row for row in by_item.get(key, []) if row.get("status") == "ok"]
        unique = {(row["backend"], int(row["permutation_index"])): row for row in rows}
        rows = list(unique.values())
        per_model = {}
        for backend in backends:
            model_rows = [row for row in rows if row["backend"] == backend.name]
            correct = sum(bool(row["correct"]) for row in model_rows)
            per_model[backend.name] = {
                "model": backend.model or (str(model_rows[0].get("model")) if model_rows else ""),
                "completed": len(model_rows),
                "correct": correct,
                "accuracy": correct / len(model_rows) if model_rows else None,
                "mean_confidence": (
                    sum(float(row["confidence"]) for row in model_rows) / len(model_rows)
                    if model_rows
                    else None
                ),
            }
        complete = len(rows) == expected_per_item
        total_correct = sum(bool(row["correct"]) for row in rows)
        overall_accuracy = total_correct / len(rows) if rows else None
        positive_models = [
            name
            for name, score in per_model.items()
            if score["completed"] == args.permutations
            and score["accuracy"] is not None
            and score["accuracy"] >= args.per_model_threshold
        ]
        removal_candidate = bool(
            complete
            and overall_accuracy is not None
            and overall_accuracy >= args.overall_threshold
            and len(positive_models) >= args.min_positive_models
        )
        score = {
            "item_key": key,
            "video_id": item["video_id"],
            "qid": item["qid"],
            "question_type": item.get("question_type"),
            "question": item["question"],
            "choices": item["choices"],
            "answer": item["answer"],
            "complete": complete,
            "completed_predictions": len(rows),
            "expected_predictions": expected_per_item,
            "overall_accuracy": overall_accuracy,
            "positive_models": positive_models,
            "per_model": per_model,
            "removal_candidate": removal_candidate,
        }
        item_scores.append(score)
        if removal_candidate:
            candidates.append(score)
        if not complete:
            incomplete.append(score)

    type_counts = Counter(str(row.get("question_type") or "UNKNOWN") for row in candidates)
    summary = {
        "status": "complete" if not incomplete else "incomplete",
        "items": len(items),
        "backends": backend_models,
        "permutations": args.permutations,
        "expected_predictions": len(items) * expected_per_item,
        "successful_predictions": sum(row["completed_predictions"] for row in item_scores),
        "incomplete_items": len(incomplete),
        "removal_candidates": len(candidates),
        "removal_candidate_rate": len(candidates) / len(items) if items else 0,
        "candidate_question_type_counts": dict(sorted(type_counts.items())),
        "policy": {
            "per_model_threshold": args.per_model_threshold,
            "overall_threshold": args.overall_threshold,
            "min_positive_models": args.min_positive_models,
            "candidate_only": True,
            "recommended_next_step": "GPT semantic review and a stratified manual precision audit before manifest exclusion",
        },
    }
    atomic_write_json(args.output_dir / "item_scores.json", item_scores)
    atomic_write_json(args.output_dir / "removal_candidates.json", candidates)
    atomic_write_json(args.output_dir / "incomplete_items.json", incomplete)
    atomic_write_json(args.output_dir / "summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    if args.permutations <= 0 or args.workers <= 0 or args.max_retries < 0:
        raise SystemExit("--permutations and --workers must be positive; --max-retries cannot be negative")
    for name in ("per_model_threshold", "overall_threshold"):
        if not 0 <= getattr(args, name) <= 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be within [0,1]")

    items = load_items(args.manifest, args.candidates)
    if args.limit is not None:
        items = items[: args.limit]
    backends = make_backends(
        args, validate_credentials=not (args.aggregate_only or args.dry_run)
    )
    if args.min_positive_models <= 0:
        raise SystemExit("--min-positive-models must be positive")
    expected = len(items) * len(backends) * args.permutations

    if args.dry_run:
        first = items[0]
        case = permuted_case(first, 0, args.seed)
        print(json.dumps({
            "items": len(items),
            "backends": [backend.name for backend in backends],
            "permutations": args.permutations,
            "expected_calls": expected,
            "first_item_key": item_key(first),
            "first_display_order_original_indices": case["order"],
            "first_displayed_gold_for_local_validation_only": case["displayed_gold"],
            "first_prompt": user_prompt(first, case),
        }, ensure_ascii=False, indent=2))
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "raw_predictions.jsonl"
    if not args.aggregate_only:
        latest = load_latest_jsonl(raw_path)
        completed = {key for key, row in latest.items() if row.get("status") == "ok"}
        clients = {backend.name: create_client(backend, args.timeout) for backend in backends}
        tasks = []
        for row in items:
            for backend in backends:
                for permutation_index in range(args.permutations):
                    key = f"{item_key(row)}::{backend.name}::p{permutation_index:02d}"
                    if key not in completed:
                        tasks.append((row, permutation_index, backend))

        print(json.dumps({
            "items": len(items), "expected_calls": expected,
            "already_complete": expected - len(tasks), "pending": len(tasks),
            "raw_output": str(raw_path),
        }, ensure_ascii=False), flush=True)
        lock = threading.Lock()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(run_one, row, permutation_index, backend, clients[backend.name], args):
                (item_key(row), backend.name, permutation_index)
                for row, permutation_index, backend in tasks
            }
            with raw_path.open("a", encoding="utf-8") as stream:
                for done_index, future in enumerate(as_completed(futures), 1):
                    result = future.result()
                    with lock:
                        stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                        stream.flush()
                    print(
                        f"[{done_index}/{len(tasks)}] {result['status'].upper()} {result['key']}",
                        flush=True,
                    )

    summary = aggregate(items, backends, args, raw_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
