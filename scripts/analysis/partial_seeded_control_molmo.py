#!/usr/bin/env python3
"""Native Molmo2 runner for the fixed-frame seeded Partial controls."""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
from pathlib import Path

from PIL import Image

from partial_seeded_control_experiment import (
    blend_indices,
    evidence_indices,
    extract_frames,
    forced_choice_prompt,
    matched_non_evidence_indices,
    response_key,
    resolve_video,
    row_key,
    select_eligible,
    selective_prompt,
    spans,
)


SELECTIVE_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": ["ANSWERABLE", "UNANSWERABLE"]},
        "answer": {"anyOf": [{"type": "string", "enum": ["A", "B", "C", "D", "E"]}, {"type": "null"}]},
        "evidence": {"type": "string"},
        "evidence_seconds": {"anyOf": [{"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}, {"type": "null"}]},
        "reason": {"type": "string"},
    },
    "required": ["label", "answer", "evidence", "evidence_seconds", "reason"],
    "additionalProperties": False,
}

FORCED_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string", "enum": ["A", "B", "C", "D", "E"]},
        "evidence": {"type": "string"},
        "evidence_seconds": {"anyOf": [{"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2}, {"type": "null"}]},
        "reason": {"type": "string"},
    },
    "required": ["answer", "evidence", "evidence_seconds", "reason"],
    "additionalProperties": False,
}


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sample", type=Path, required=True)
    p.add_argument("--metadata", type=Path, default=Path("results/build/nextgqa/nextgqa_filtered_no_evidence_leak_eval_clean.json"))
    p.add_argument("--mapping", type=Path, default=Path("source_datasets/next_gqa/NExT-GQA/datasets/nextgqa/map_vid_vidorID.json"))
    p.add_argument("--source-video-dir", type=Path, default=Path("source_datasets/next_gqa/videos"))
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--model-tag", default="molmo2_o_7b")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--protocol", choices=("selective", "forced"), default="selective")
    p.add_argument("--alphas", default="0.2,0.4,0.5,0.6")
    p.add_argument("--seeds", default="17,29,43")
    p.add_argument("--forced-alphas", default="0.2,0.6")
    p.add_argument("--forced-seed", type=int, default=17)
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--frame-width", type=int, default=560)
    p.add_argument("--ffmpeg", default="ffmpeg")
    p.add_argument("--max-new-tokens", type=int, default=500)
    return p.parse_args()


def make_messages(frames: list[str], times: list[float], instruction: str) -> list[dict]:
    content = []
    for encoded, timestamp in zip(frames, times):
        content.append({"type": "text", "text": f"Frame at {timestamp:.1f}s:"})
        content.append({"type": "image", "image": Image.open(
            io.BytesIO(base64.b64decode(encoded))).convert("RGB")})
    content.append({"type": "text", "text": instruction})
    return [{"role": "user", "content": content}]


def main() -> None:
    a = args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("Molmo2 GPU inference must run under Slurm")
    import torch
    import xgrammar as xgr
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from xgrammar.contrib.hf import LogitsProcessor

    all_rows = [row for row in json.loads(a.sample.read_text(encoding="utf-8"))
                if row.get("video_kind") == "source"]
    meta = {row_key(row): row for row in json.loads(a.metadata.read_text(encoding="utf-8"))}
    mapping = json.loads(a.mapping.read_text(encoding="utf-8"))
    rows, counts, examples = select_eligible(all_rows, meta, mapping, a)
    print(json.dumps({**counts, "examples": examples[:30]}, indent=2), flush=True)
    if len(rows) < a.limit:
        raise SystemExit("not enough eligible rows")

    processor = AutoProcessor.from_pretrained(a.model_path, trust_remote_code=True, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        a.model_path, trust_remote_code=True, local_files_only=True,
        dtype=torch.bfloat16, device_map="cuda:0", attn_implementation="sdpa")
    model.eval()
    schema = SELECTIVE_SCHEMA if a.protocol == "selective" else FORCED_SCHEMA
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(
        processor.tokenizer, vocab_size=model.lm_head.out_features,
        stop_token_ids=model.generation_config.eos_token_id)
    grammar = xgr.GrammarCompiler(tokenizer_info).compile_json_schema(json.dumps(schema))

    completed = set()
    if a.output.exists():
        for line in a.output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                completed.add(str(json.loads(line).get("key")))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    errors = 0
    for row in rows:
        item = f"{row['video_id']}_q{row['qid']}"
        try:
            times = [float(x) for x in row["sampled_seconds"]]
            item_meta = meta[row_key(row)]
            evidence_spans = spans(item_meta)
            evid = evidence_indices(times, evidence_spans)
            control, control_method = matched_non_evidence_indices(len(times), evid)
            video = resolve_video(row, mapping, a.source_video_dir)
            original = extract_frames(video, times, a.frame_width, a.ffmpeg)
            if a.protocol == "forced":
                alphas = [float(x) for x in a.forced_alphas.split(",")]
                seeds = [a.forced_seed]
            else:
                alphas = [float(x) for x in a.alphas.split(",")]
                seeds = [int(x) for x in a.seeds.split(",")]
            plan = [("original", None, None, original)]
            for seed in seeds:
                for alpha in alphas:
                    plan.append(("evidence", alpha, seed, blend_indices(original, evid, alpha, item, seed)))
                    plan.append(("non_evidence", alpha, seed, blend_indices(original, control, alpha, item, seed)))
            for location, alpha, seed, frames in plan:
                key = response_key(item, a.protocol, location, alpha, seed)
                if key in completed:
                    continue
                prompt = selective_prompt(row) if a.protocol == "selective" else forced_choice_prompt(row)
                messages = make_messages(frames, times, prompt)
                inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                       return_tensors="pt", return_dict=True)
                inputs = {name: value.to(model.device) for name, value in inputs.items()}
                prompt_tokens = inputs["input_ids"].shape[-1]
                with torch.inference_mode():
                    output = model.generate(**inputs, max_new_tokens=a.max_new_tokens, do_sample=False,
                                            logits_processor=[LogitsProcessor(grammar)])
                text = processor.tokenizer.decode(output[0, prompt_tokens:], skip_special_tokens=True)
                record = {
                    "key": key, "status": "ok", "protocol": a.protocol,
                    "condition": location, "alpha": alpha, "noise_seed": seed,
                    "model_tag": a.model_tag, "video_id": str(row["video_id"]),
                    "qid": str(row["qid"]), "question_type": row.get("question_type"),
                    "question": row["question"], "choices": row["choices"], "answer": row["answer"],
                    "sampled_seconds": times, "evidence_intervals": [list(x) for x in evidence_spans],
                    "evidence_frame_indices": evid, "control_frame_indices": control,
                    "control_matching": control_method, "json_constraint": True,
                    "response": {"text": text, "input_tokens": prompt_tokens,
                                 "output_tokens": output.shape[-1] - prompt_tokens},
                }
                with a.output.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(key, flush=True)
        except Exception as exc:
            errors += 1
            record = {"key": f"{item}::ITEM_ERROR", "status": "error",
                      "model_tag": a.model_tag, "video_id": str(row["video_id"]),
                      "qid": str(row["qid"]), "error": f"{type(exc).__name__}: {exc}"}
            with a.output.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(json.dumps(record), flush=True)
    if errors:
        raise SystemExit(f"{errors} items failed")


if __name__ == "__main__":
    main()
