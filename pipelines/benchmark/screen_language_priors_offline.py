#!/usr/bin/env python3
"""Offline batched Transformers runner for one language-prior backend.

This is the CPU/GPU fallback when an OpenAI-compatible local server is not
available. It writes the same append-only records as screen_language_priors.py
and can resume into the same output directory after other backends finish.
"""

from __future__ import annotations

# Layout compatibility: data stays at its existing paths; frozen copies remain standalone.
from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_file = _LayoutPath(__file__).resolve()
_layout_root = (_layout_file.parents[2]
                if _layout_file.as_posix().endswith('/pipelines/benchmark/screen_language_priors_offline.py') else None)
_LAYOUT_LEGACY_FILE = str(_layout_root / 'scripts/analysis/screen_language_priors_offline.py') if _layout_root else __file__
if _layout_root is not None:
    for _layout_dir in (_layout_root, _layout_root / 'src',
                        _layout_root / 'scripts/eval',
                        _layout_root / 'scripts/analysis', _layout_file.parent):
        if str(_layout_dir) not in _layout_sys.path:
            _layout_sys.path.insert(0, str(_layout_dir))


import argparse
import json
import os
import re
import time
from pathlib import Path

from screen_language_priors import (
    Backend,
    DEFAULT_CANDIDATES,
    DEFAULT_MANIFEST,
    DEFAULT_OUTPUT_DIR,
    LETTERS,
    SYSTEM_PROMPT,
    aggregate,
    item_key,
    load_items,
    load_latest_jsonl,
    parse_object,
    permuted_case,
    user_prompt,
)


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("qwen", "llama"), required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--permutations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--threads", type=int, default=64)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--per-model-threshold", type=float, default=0.8)
    parser.add_argument("--overall-threshold", type=float, default=0.8)
    parser.add_argument("--min-positive-models", type=int, default=2)
    return parser.parse_args()


def fallback_parse(text: str) -> dict:
    try:
        parsed = parse_object(text)
        parsed["parse_mode"] = "json"
        return parsed
    except Exception:
        match = re.search(r'(?i)["\']?answer["\']?\s*[:=]\s*["\']?([A-E])\b', text)
        if not match:
            match = re.search(r"\b([A-E])\b", text.upper())
        if not match:
            raise ValueError("response contains neither valid JSON nor a standalone option letter")
        return {
            "answer": match.group(1).upper(),
            "confidence": 0.5,
            "reason": text.strip()[:1000] or "letter-only response",
            "parse_mode": "letter_fallback",
        }


def main() -> None:
    a = args()
    if a.batch_size <= 0 or a.permutations <= 0 or a.threads <= 0:
        raise SystemExit("batch size, permutations, and threads must be positive")
    os.environ["OMP_NUM_THREADS"] = str(a.threads)
    os.environ["MKL_NUM_THREADS"] = str(a.threads)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(a.threads)
    items = load_items(a.manifest, a.candidates)
    if a.limit is not None:
        items = items[: a.limit]
    a.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = a.output_dir / "raw_predictions.jsonl"
    latest = load_latest_jsonl(raw_path)
    completed = {key for key, row in latest.items() if row.get("status") == "ok"}
    tasks = []
    for row in items:
        for permutation_index in range(a.permutations):
            key = f"{item_key(row)}::{a.backend}::p{permutation_index:02d}"
            if key not in completed:
                tasks.append((row, permutation_index))

    print(json.dumps({
        "backend": a.backend,
        "model_path": str(a.model_path),
        "items": len(items),
        "expected": len(items) * a.permutations,
        "already_complete": len(items) * a.permutations - len(tasks),
        "pending": len(tasks),
        "device": a.device,
        "batch_size": a.batch_size,
    }, ensure_ascii=False), flush=True)

    tokenizer = AutoTokenizer.from_pretrained(a.model_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        a.model_path, dtype=dtype, low_cpu_mem_usage=True
    ).to(a.device).eval()

    with raw_path.open("a", encoding="utf-8") as stream:
        for offset in range(0, len(tasks), a.batch_size):
            batch = tasks[offset : offset + a.batch_size]
            cases = [permuted_case(row, index, a.seed) for row, index in batch]
            prompts = [user_prompt(row, case) for (row, _), case in zip(batch, cases)]
            texts = []
            for prompt in prompts:
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ]
                kwargs = dict(tokenize=False, add_generation_prompt=True)
                if a.backend == "qwen":
                    kwargs["enable_thinking"] = False
                texts.append(tokenizer.apply_chat_template(messages, **kwargs))
            encoded = tokenizer(texts, return_tensors="pt", padding=True).to(a.device)
            started = time.time()
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=a.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            outputs = tokenizer.batch_decode(
                generated[:, encoded.input_ids.shape[1] :], skip_special_tokens=True
            )
            elapsed = time.time() - started
            for (row, permutation_index), case, raw in zip(batch, cases, outputs):
                base = {
                    "key": f"{item_key(row)}::{a.backend}::p{permutation_index:02d}",
                    "item_key": item_key(row),
                    "video_id": row["video_id"],
                    "qid": row["qid"],
                    "question_type": row.get("question_type"),
                    "backend": a.backend,
                    "model": a.model_name,
                    "permutation_index": permutation_index,
                    "display_order_original_indices": case["order"],
                    "raw_response": raw,
                    "usage": None,
                    "elapsed_seconds": round(elapsed / len(batch), 3),
                    "runner": f"transformers_{a.device}_batch",
                }
                try:
                    parsed = fallback_parse(raw)
                    original_prediction = LETTERS[case["order"][LETTERS.index(parsed["answer"])]]
                    base.update({
                        "displayed_prediction": parsed["answer"],
                        "original_prediction": original_prediction,
                        "correct": parsed["answer"] == case["displayed_gold"],
                        "confidence": parsed["confidence"],
                        "reason": parsed["reason"],
                        "parse_mode": parsed["parse_mode"],
                        "status": "ok",
                        "attempts": 1,
                    })
                except Exception as exc:
                    lowered = raw.lower()
                    refusal_markers = (
                        "unable to answer",
                        "can't answer",
                        "cannot answer",
                        "not able to make",
                        "without the video",
                        "requires visual information",
                    )
                    if any(marker in lowered for marker in refusal_markers):
                        # A blind refusal is an informative failed forced-choice
                        # trial, not missing data and not evidence of a prior.
                        base.update({
                            "displayed_prediction": None,
                            "original_prediction": None,
                            "correct": False,
                            "confidence": 0.0,
                            "reason": raw.strip()[:1000],
                            "parse_mode": "refusal",
                            "status": "ok",
                            "attempts": 1,
                        })
                    else:
                        base.update({
                            "status": "error",
                            "error": f"{type(exc).__name__}: {exc}",
                            "attempts": 1,
                        })
                stream.write(json.dumps(base, ensure_ascii=False) + "\n")
            stream.flush()
            done = min(offset + len(batch), len(tasks))
            print(
                f"[{done}/{len(tasks)}] batch_seconds={elapsed:.2f} "
                f"per_item_seconds={elapsed / len(batch):.3f}",
                flush=True,
            )

    backend = Backend(a.backend, a.model_name, "offline", None, "")
    summary = aggregate(items, [backend], a, raw_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
