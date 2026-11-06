#!/usr/bin/env python3
"""Package selected non-Qwen3 leak discoveries for manual video review."""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
from pathlib import Path


def latest(pattern: str) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for filename in sorted(glob.glob(pattern), key=os.path.getmtime):
        with open(filename, encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    row = json.loads(line)
                    rows[str(row.get("key"))] = row
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--item", action="append", required=True)
    parser.add_argument("--new-classifications", required=True)
    parser.add_argument("--qwen3-classifications", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    return parser.parse_args()


def source_qa(row: dict) -> dict:
    path = Path(str(row["source_result_file"]))
    values = json.loads(path.read_text(encoding="utf-8"))
    for value in values:
        if str(value.get("video_id")) == str(row["video_id"]) and str(value.get("qid")) == str(row["qid"]):
            return value
    raise ValueError(f"QA not found in {path}: {row['item_key']}")


def markdown(payload: dict) -> str:
    lines = [
        f"# {payload['item_key']}", "",
        f"- 问题类型：`{payload['question_type']}`",
        f"- 问题：{payload['question']}",
        f"- 正确答案：**{payload['gold_letter']}. {payload['reference_answer']}**",
        f"- 人工审阅联合时段：`{payload['review_interval'][0]:.3f}s – {payload['review_interval'][1]:.3f}s`",
        f"- 新模型确认数：{len(payload['discoveries'])}", "", "## 选项", "",
    ]
    for index, choice in enumerate(payload["choices"]):
        lines.append(f"- {chr(65 + index)}. {choice}")
    lines.extend(["", "## 新模型泄露发现", ""])
    for discovery in payload["discoveries"]:
        lines.extend([
            f"### {discovery['model_tag']}", "",
            f"- 模型答案：`{discovery['model_answer']}`",
            f"- 模型声称证据时段：`{discovery['claimed_interval']}`",
            f"- 实际审计时段（含 padding）：`{discovery['audit_interval']}`",
            f"- 模型证据陈述：{discovery['model_evidence']}",
            f"- 无答案/选项 grounding 描述：{discovery['grounding_description']}",
            f"- Text judge 开放式答案：{discovery['judge_open_answer']}",
            f"- 判定理由：{discovery['judge_reason']}", "",
        ])
    lines.extend([
        "## 文件", "",
        "- `full_insufficient.mp4`：模型实际看到的完整 insufficient/frozen 视频。",
        "- `evidence_review_clip.mp4`：所有新模型审计时段的联合裁剪，便于快速检查。",
        "- `metadata.json`：完整机器可读信息。", "",
        "人工检查重点：裁剪时段是否真的包含回答完整问题所需的视觉事实，而不只是与正确答案相关的物体或场景线索。", "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    new = latest(args.new_classifications)
    qwen3 = latest(args.qwen3_classifications)
    old_leaks = {
        str(row.get("item_key")) for row in qwen3.values()
        if row.get("status") == "ok" and row.get("label") == "EVIDENCE_LEAK"
    }
    new_by_item: dict[str, list[dict]] = {}
    for row in new.values():
        if row.get("status") == "ok" and row.get("label") == "EVIDENCE_LEAK":
            new_by_item.setdefault(str(row.get("item_key")), []).append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for item in args.item:
        if item in old_leaks:
            raise SystemExit(f"selected item was already confirmed by Qwen3: {item}")
        rows = sorted(new_by_item.get(item, []), key=lambda row: str(row.get("source_eval_model_tag")))
        if not rows:
            raise SystemExit(f"selected item has no new-model EVIDENCE_LEAK record: {item}")
        qa = source_qa(rows[0])
        intervals = [tuple(map(float, row["candidate"]["interval"])) for row in rows]
        review_interval = [min(x[0] for x in intervals), max(x[1] for x in intervals)]
        discoveries = []
        for row in rows:
            claim = row.get("source_model_claim") or {}
            decision = row.get("decision") or {}
            discoveries.append({
                "model_tag": row.get("source_eval_model_tag"),
                "model_answer": claim.get("answer"),
                "claimed_interval": claim.get("evidence_seconds"),
                "audit_interval": row.get("candidate", {}).get("interval"),
                "model_evidence": claim.get("evidence"),
                "model_reason": claim.get("reason"),
                "grounding_description": row.get("grounding", {}).get("description"),
                "judge_open_answer": decision.get("answer"),
                "judge_reason": decision.get("reason"),
                "semantic_label": row.get("label"),
                "classification_key": row.get("key"),
            })
        payload = {
            "item_key": item,
            "selection_policy": "new-model EVIDENCE_LEAK and absent from Qwen3 confirmed leak item set",
            "video_id": str(rows[0]["video_id"]), "qid": str(rows[0]["qid"]),
            "question_type": rows[0].get("question_type"), "question": rows[0].get("question"),
            "choices": qa.get("choices"), "gold_letter": qa.get("answer"),
            "reference_answer": rows[0].get("reference_answer"),
            "review_interval": review_interval, "discoveries": discoveries,
        }
        folder = args.output_dir / item
        folder.mkdir(parents=True, exist_ok=True)
        video = Path(str(rows[0]["insufficient_video"]))
        if not video.is_file():
            raise FileNotFoundError(video)
        full = folder / "full_insufficient.mp4"
        shutil.copy2(video, full)
        clip = folder / "evidence_review_clip.mp4"
        subprocess.run([
            args.ffmpeg, "-nostdin", "-loglevel", "error", "-y",
            "-ss", str(review_interval[0]), "-to", str(review_interval[1]),
            "-i", str(video), "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "aac", "-movflags", "+faststart", str(clip),
        ], check=True)
        (folder / "metadata.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (folder / "README.md").write_text(markdown(payload), encoding="utf-8")
        manifest.append(payload)
        print(f"PACKAGED {item} models={len(rows)} interval={review_interval}")
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
