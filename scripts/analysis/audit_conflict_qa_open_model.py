#!/usr/bin/env python3
"""Stratified text-semantic audit of ETBench conflict QA with an open model."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "etbench_pipeline/conflict/full_v2_qwen3vl_mistral24b/merged/conflict_qa_v2.jsonl"

SYSTEM = """You are auditing a video-QA benchmark. Judge only the supplied text metadata; do not pretend to see the video.
A genuine conflict item must ask one coherent question and present at least two different, context-dependent answers, each tied to a distinct occurrence. Every option must answer the same semantic slot requested by the question. Evidence descriptions must textually support their option. Temporal-localization options at distinct occurrences are valid conflict cases.
Return one JSON object only with exactly these keys:
verdict: one of pass, minor_issue, major_issue, uncertain
question_coherent: boolean
same_semantic_slot: boolean
genuine_conflict: boolean
text_support_consistent: boolean
needs_video_check: boolean
flags: array chosen from [question_option_mismatch, option_evidence_mismatch, duplicate_answers, same_occurrence_only, not_conflict, ambiguous_wording, unsupported_from_text, other]
reason: concise English string.
Use uncertain rather than guessing when visual verification is essential."""


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("qwen", "llama"), required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-type", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--threads", type=int, default=48)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def stratified(rows: list[dict], per_type: int, seed: int) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        if row.get("status") == "accepted":
            grouped[str(row.get("conflict_type", "UNKNOWN"))].append(row)
    rng = random.Random(seed)
    selected = []
    for category in sorted(grouped):
        values = sorted(grouped[category], key=lambda row: str(row["id"]))
        if per_type <= 0:
            selected.extend(values)
        else:
            selected.extend(rng.sample(values, min(per_type, len(values))))
    return sorted(selected, key=lambda row: (str(row.get("conflict_type")), str(row["id"])))


def prompt(row: dict) -> str:
    compact = {
        "id": row["id"],
        "official_action": row.get("official_action"),
        "conflict_type": row.get("conflict_type"),
        "grounding_field": row.get("grounding_field"),
        "question": row.get("question"),
        "options": [
            {
                "option_id": option.get("option_id"),
                "answer": option.get("answer"),
                "supported_by": [
                    {
                        "instance_id": support.get("instance_id"),
                        "interval": support.get("interval"),
                        "visual_basis": support.get("visual_basis"),
                    }
                    for support in option.get("supported_by", [])
                ],
            }
            for option in row.get("options", [])
        ],
    }
    return "Audit this candidate:\n" + json.dumps(compact, ensure_ascii=False)


def parse_object(text: str) -> dict:
    candidates = [text.strip()]
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.I | re.S)
    if fenced:
        candidates.insert(0, fenced.group(1))
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
            if isinstance(value, dict) and value.get("verdict") in {"pass", "minor_issue", "major_issue", "uncertain"}:
                return value
        except json.JSONDecodeError:
            pass
    raise ValueError("no valid verdict JSON")


def main() -> None:
    args = arguments()
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(args.threads)
    sample = stratified(read_jsonl(args.input), args.per_type, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_tmp = args.output_dir / f"sample.json.tmp.{args.backend}"
    sample_tmp.write_text(
        json.dumps(sample, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    sample_tmp.replace(args.output_dir / "sample.json")
    output = args.output_dir / f"{args.backend}_judgments.jsonl"
    latest = {}
    if output.is_file():
        for row in read_jsonl(output):
            latest[str(row["id"])] = row
    completed = {item_id for item_id, row in latest.items() if row.get("status") == "ok"}
    pending = [row for row in sample if str(row["id"]) not in completed]
    print(json.dumps({"sample": len(sample), "pending": len(pending), "backend": args.backend}), flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to(args.device).eval()
    with output.open("a", encoding="utf-8") as stream:
        for offset in range(0, len(pending), args.batch_size):
            batch = pending[offset : offset + args.batch_size]
            texts = []
            for row in batch:
                kwargs = {"tokenize": False, "add_generation_prompt": True}
                if args.backend == "qwen":
                    kwargs["enable_thinking"] = False
                texts.append(tokenizer.apply_chat_template([
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt(row)},
                ], **kwargs))
            encoded = tokenizer(texts, return_tensors="pt", padding=True).to(args.device)
            started = time.time()
            with torch.inference_mode():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            input_width = encoded["input_ids"].shape[1]
            for row, tokens in zip(batch, generated):
                raw = tokenizer.decode(tokens[input_width:], skip_special_tokens=True).strip()
                record = {
                    "id": row["id"], "conflict_type": row.get("conflict_type"),
                    "backend": args.backend, "model": args.model_name, "raw_response": raw,
                }
                try:
                    record.update({"status": "ok", "judgment": parse_object(raw)})
                except Exception as exc:
                    record.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            print(f"[{min(offset + len(batch), len(pending))}/{len(pending)}] batch_seconds={time.time()-started:.2f}", flush=True)


if __name__ == "__main__":
    main()
