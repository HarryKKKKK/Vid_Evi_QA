#!/usr/bin/env python3
"""Rebuild the canonical model-run ledger from finalized result directories."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


RESULT_NAMES = {
    "sufficient.classify.jsonl": "sufficient",
    "insufficient.classify.jsonl": "insufficient",
    "partial.classify.jsonl": "partial",
}
EXCLUDED_DIR_MARKERS = ("pilot", "smoke", "retry", "no_evidence_leak")


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return rows


def normalized_prediction(row: dict) -> tuple[str, str | None]:
    parsed = row.get("model_parsed") or {}
    label = str(parsed.get("label") or "").strip().upper()
    answer = parsed.get("answer")

    if label in "ABCDE" and len(label) == 1:
        return "ANSWERABLE", label

    if isinstance(answer, str):
        match = re.match(r"\s*([A-E])(?:\b|[.:)])", answer.upper())
        if match:
            answer = match.group(1)

    raw = str(parsed.get("raw") or row.get("model_response") or "")
    if label not in ("ANSWERABLE", "UNANSWERABLE"):
        match = re.search(
            r"[\"']label[\"']\s*:\s*[\"'](ANSWERABLE|UNANSWERABLE)",
            raw,
            re.IGNORECASE,
        )
        if match:
            label = match.group(1).upper()

    if not (isinstance(answer, str) and answer in "ABCDE" and len(answer) == 1):
        match = re.search(
            r"[\"']answer[\"']\s*:\s*[\"']([A-E])",
            raw,
            re.IGNORECASE,
        )
        answer = match.group(1).upper() if match else None

    return label, answer


def display_model(model_tag: str) -> str:
    tag = model_tag.lower()
    if "qwen25" in tag or "qwen2.5" in tag:
        return "Qwen2.5-VL-32B-Instruct"
    if "llava" in tag:
        return "LLaVA-OneVision-Qwen2-7B-OV"
    if "internvl" in tag:
        return "InternVL3.5-8B"
    if "qwen3" in tag:
        return "Qwen3-VL"
    return model_tag or "UNKNOWN"


def pct(numerator: int, denominator: int) -> str:
    return "N/A" if denominator == 0 else f"{100 * numerator / denominator:.2f}%"


def summarize(path: Path, condition: str) -> dict:
    rows = load_jsonl(path)
    valid_rows = [
        row for row in rows if str(row.get("answer", "")).strip().upper() in "ABCDE"
        and len(str(row.get("answer", "")).strip()) == 1
    ]
    predictions = [normalized_prediction(row) for row in valid_rows]
    answerable = sum(label == "ANSWERABLE" for label, _ in predictions)
    unanswerable = sum(label == "UNANSWERABLE" for label, _ in predictions)
    invalid_output = len(predictions) - answerable - unanswerable
    correct = sum(
        label == "ANSWERABLE" and answer == str(row["answer"]).strip().upper()
        for row, (label, answer) in zip(valid_rows, predictions)
    )
    frame_counts = Counter(len(row.get("sampled_seconds") or []) for row in rows)
    media_modes = sorted({str(row.get("media_mode") or "unknown") for row in rows})
    model_tags = sorted({str(row.get("model_tag") or "UNKNOWN") for row in rows})
    per_alpha = defaultdict(list)
    if condition == "partial":
        for row in valid_rows:
            per_alpha[round(float(row["alpha"]), 1)].append(row)

    return {
        "path": path,
        "directory": path.parent.name,
        "condition": condition,
        "rows": len(rows),
        "valid": len(valid_rows),
        "malformed_gt": len(rows) - len(valid_rows),
        "answerable": answerable,
        "unanswerable": unanswerable,
        "invalid_output": invalid_output,
        "correct": correct,
        "frames": ", ".join(f"{n} ({count})" for n, count in sorted(frame_counts.items())),
        "media_mode": ", ".join(media_modes),
        "model_tag": ", ".join(model_tags),
        "model": display_model(model_tags[0] if model_tags else ""),
        "per_alpha": per_alpha,
    }


def discover(root: Path) -> list[dict]:
    summaries = []
    for path in sorted(root.glob("results/nextgqa_result_*/*.classify.jsonl")):
        condition = RESULT_NAMES.get(path.name)
        dirname = path.parent.name.lower()
        if not condition or any(marker in dirname for marker in EXCLUDED_DIR_MARKERS):
            continue
        # Canonical final runs use these suffixes; this excludes unrelated old outputs.
        if condition == "sufficient" and not dirname.endswith("_sufficient"):
            continue
        if condition == "insufficient" and not dirname.endswith("_insufficient"):
            continue
        if condition == "partial" and "_partial_" not in dirname:
            continue
        summaries.append(summarize(path, condition))
    return summaries


def alpha_metrics(rows: list[dict]) -> tuple[int, int, int, int]:
    predictions = [normalized_prediction(row) for row in rows]
    answerable = sum(label == "ANSWERABLE" for label, _ in predictions)
    unanswerable = sum(label == "UNANSWERABLE" for label, _ in predictions)
    correct = sum(
        label == "ANSWERABLE" and answer == str(row["answer"]).strip().upper()
        for row, (label, answer) in zip(rows, predictions)
    )
    invalid = len(rows) - answerable - unanswerable
    return answerable, unanswerable, correct, invalid


def render(root: Path, summaries: list[dict]) -> str:
    def expected_rows(entry: dict) -> int:
        if entry["condition"] != "partial":
            return 5280
        match = re.search(r"_n(\d+)(?:_|$)", entry["directory"])
        if match and entry["per_alpha"]:
            return int(match.group(1)) * len(entry["per_alpha"])
        return entry["rows"]

    lines = [
        "# Model Run Results",
        "",
        "> 这是自动生成的模型运行结果台账。请勿手工修改表格；运行 "
        "`python scripts/eval/update_model_results.py` 刷新。",
        "",
        f"- 最后更新时间：{datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"- 项目根目录：`{root}`",
        "- 统计口径：只纳入标准答案为 A–E 的样本；总体准确率把拒答计为不正确；条件准确率只在模型选择作答的样本上计算。",
        "- 输出归一化：兼容 `ANSWERABLE/UNANSWERABLE`、直接 A–E label，以及 `A. ...` 形式答案。",
        "",
        "## 总览",
        "",
        "| 模型 | 条件 | 状态 | 记录数 | 有效 GT | 可回答率 | 拒答率 | 总体准确率 | 作答条件准确率 | 无效输出 | 帧数（样本数） | 输入模式 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]

    condition_order = {"sufficient": 0, "insufficient": 1, "partial": 2}
    summaries.sort(key=lambda item: (item["model"], condition_order[item["condition"]]))
    for entry in summaries:
        target = expected_rows(entry)
        status = "完整" if entry["rows"] == target else f"缺 {target - entry['rows']}"
        lines.append(
            "| {model} | {condition} | {status} | {rows} | {valid} | {answer_rate} | "
            "{abstain_rate} | {accuracy} | {conditional_accuracy} | {invalid} | {frames} | {media} |".format(
                model=entry["model"],
                condition=entry["condition"],
                status=status,
                rows=entry["rows"],
                valid=entry["valid"],
                answer_rate=pct(entry["answerable"], entry["valid"]),
                abstain_rate=pct(entry["unanswerable"], entry["valid"]),
                accuracy=pct(entry["correct"], entry["valid"]),
                conditional_accuracy=pct(entry["correct"], entry["answerable"]),
                invalid=entry["invalid_output"],
                frames=entry["frames"],
                media=entry["media_mode"],
            )
        )

    partial = [entry for entry in summaries if entry["condition"] == "partial"]
    if partial:
        lines.extend(["", "## Partial α 曲线", ""])
        for entry in partial:
            lines.extend([
                f"### {entry['model']}",
                "",
                "| α | N | 可回答率 | 拒答率 | 总体准确率 | 作答条件准确率 | 无效输出 |",
                "|---:|---:|---:|---:|---:|---:|---:|",
            ])
            for alpha, rows in sorted(entry["per_alpha"].items()):
                answerable, unanswerable, correct, invalid = alpha_metrics(rows)
                lines.append(
                    f"| {alpha:.1f} | {len(rows)} | {pct(answerable, len(rows))} | "
                    f"{pct(unanswerable, len(rows))} | {pct(correct, len(rows))} | "
                    f"{pct(correct, answerable)} | {invalid} |"
                )
            lines.append("")

    lines.extend([
        "## 运行来源",
        "",
        "| 模型 | 条件 | model_tag | 结果目录 | 非 A–E GT |",
        "|---|---|---|---|---:|",
    ])
    for entry in summaries:
        lines.append(
            f"| {entry['model']} | {entry['condition']} | `{entry['model_tag']}` | "
            f"`{entry['directory']}` | {entry['malformed_gt']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", default="results/reports/results/MODEL_RESULTS.md")
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    root_hash = hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:16]
    lock_path = Path(tempfile.gettempdir()) / f"vid_evi_qa_results_{root_hash}.lock"

    with lock_path.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        content = render(root, discover(root))
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=output.parent, delete=False, prefix=".model-results-"
        ) as handle:
            handle.write(content)
            temp_name = handle.name
        os.replace(temp_name, output)
    print(f"MODEL_RESULTS_UPDATED path={output}")


if __name__ == "__main__":
    main()
