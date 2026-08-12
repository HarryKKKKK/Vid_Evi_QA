#!/usr/bin/env python3
"""Classify grounded NExT-GQA windows with a text-only LLM.

The first call sees only question type, question, and grounding description. It
must decide whether the complete question is answerable and provide an open
answer without seeing choices or gold.  Only after that answer is frozen does a
separate text-only call compare it with the reference answer.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
from pathlib import Path

from common import append_jsonl, latest_jsonl_records, parse_endpoints, selected_indices


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GROUNDINGS = str(ROOT / "nextgqa_pipeline/insufficient/window_groundings*.jsonl")
DEFAULT_OUTPUT = ROOT / "nextgqa_pipeline/insufficient/leak_classifications.jsonl"
CLASSIFIER_PROMPT_VERSION = "nextgqa-text-leak-classifier-v1.2"


TYPE_RULES = {
    "TC": "The description must support the relevant anchor event and answer event occurring at the same time.",
    "TN": "The description must support the answer event occurring after the anchor event in the question.",
    "TP": "The description must support the answer event occurring before the anchor event in the question.",
    "CW": (
        "The description must support both the event being explained and a visually justified cause. "
        "Mere co-occurrence or temporal order is not automatically causal."
    ),
    "CH": (
        "The description must support the target action and how it is performed, such as its visible "
        "manner, tool, mechanism, or intermediate process."
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groundings", default=DEFAULT_GROUNDINGS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--endpoint", action="append", required=True, metavar="TAG,BASE_URL,MODEL")
    parser.add_argument("--max-tokens", type=int, default=220)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--api-retries", type=int, default=3)
    parser.add_argument("--num-shards", type=int)
    parser.add_argument("--shard", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--retry-status", default="error,needs_review")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def extract_json(text: str) -> dict:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start, end = stripped.find("{"), stripped.rfind("}")
        if start < 0 or end <= start:
            raise ValueError(f"response contains no JSON object: {text[:300]!r}")
        value = json.loads(stripped[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("response JSON is not an object")
    return value


def analysis_prompt(question_type: str, question: str, description: str) -> str:
    rule = TYPE_RULES.get(
        question_type,
        "Every entity, event, condition, and relation required by the complete question must be supported.",
    )
    return f"""You are auditing whether a grounded video-window description contains sufficient visual evidence for a complete NExT-GQA question.

Use only facts stated in the grounding description. Do not assume an unseen event, fill missing context with common sense, or use knowledge of multiple-choice datasets. A description may contain an action that looks like a plausible answer while omitting the condition, anchor event, temporal relation, causal relation, or manner required by the question. In that case the question is UNANSWERABLE, not ANSWERABLE.

Question type: {question_type}
Type-specific requirement: {rule}
Question: {question}

Grounding description:
{description}

First decide whether the description supports every condition needed to answer the complete question. If yes, give a short open-ended answer. If not, do not guess an answer.

Return exactly one JSON object with these fields:
- verdict: "ANSWERABLE", "UNANSWERABLE", or "UNCERTAIN"
- answer: a short open answer when ANSWERABLE, otherwise null
- failure_mode: "NONE", "ANSWER_CORRELATED_ONLY", "MISSING_REQUIRED_CONTEXT", "IRRELEVANT", or "UNCERTAIN"
- reason: one concise sentence grounded in the description

Use ANSWER_CORRELATED_ONLY when the description contains a plausible answer-like action, object, or event but does not establish the question's required anchor, condition, or relation."""


def equivalence_prompt(candidate_answer: str, reference_answer: str) -> str:
    return f"""Decide whether two short answers are semantically equivalent for video question answering. Allow harmless paraphrases, synonyms, inflection changes, and differences in articles. Do not treat merely related actions or objects as equivalent.

Candidate answer: {candidate_answer}
Reference answer: {reference_answer}

Return exactly one JSON object:
{{"equivalent": true or false, "reason": "one concise sentence"}}"""


def reference_presence_prompt(description: str, reference_answer: str) -> str:
    return f"""Check only whether a grounded video description visibly contains the action, object, state, or event expressed by a reference answer. Do not decide whether the complete original question is answerable, and do not require its missing anchor or temporal/causal condition. Be conservative: related but different content is not a match.

Grounding description:
{description}

Reference answer:
{reference_answer}

Return exactly one JSON object:
{{"present": true or false, "reason": "one concise sentence"}}"""


class TextLeakClassifier:
    def __init__(self, endpoint, *, timeout: float, max_tokens: int, retries: int):
        from openai import OpenAI

        self.endpoint = endpoint
        self.max_tokens = max_tokens
        self.retries = max(1, retries)
        self.client = OpenAI(
            base_url=endpoint.base_url,
            api_key=endpoint.api_key,
            timeout=timeout,
        )

    def complete_json(self, prompt: str, *, max_tokens: int | None = None) -> tuple[dict, str]:
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.endpoint.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=max_tokens or self.max_tokens,
                )
                raw = (response.choices[0].message.content or "").strip()
                return extract_json(raw), raw
            except Exception as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(min(2 ** (attempt - 1), 8))
        assert last_error is not None
        raise last_error

    def classify(self, record: dict) -> dict:
        grounding = record.get("grounding", {})
        description = str(grounding.get("description", "")).strip()
        if not description:
            raise ValueError("grounding record has no description")
        question_type = str(record.get("question_type", ""))
        decision, decision_raw = self.complete_json(
            analysis_prompt(question_type, str(record.get("question", "")), description)
        )

        verdict = str(decision.get("verdict", "")).upper()
        failure_mode = str(decision.get("failure_mode", "")).upper()
        output_normalizations: list[str] = []
        # Qwen occasionally emits the intended failure mode as the verdict.
        # Preserve its semantic decision instead of failing an otherwise valid
        # record, while keeping the public verdict schema unchanged.
        if verdict == "ANSWER_CORRELATED_ONLY":
            verdict = "UNANSWERABLE"
            failure_mode = "ANSWER_CORRELATED_ONLY"
            output_normalizations.append(
                "verdict=ANSWER_CORRELATED_ONLY -> verdict=UNANSWERABLE, "
                "failure_mode=ANSWER_CORRELATED_ONLY"
            )
        if verdict not in {"ANSWERABLE", "UNANSWERABLE", "UNCERTAIN"}:
            raise ValueError(f"invalid verdict: {decision.get('verdict')!r}")
        allowed_failures = {
            "NONE", "ANSWER_CORRELATED_ONLY", "MISSING_REQUIRED_CONTEXT", "IRRELEVANT", "UNCERTAIN"
        }
        if failure_mode not in allowed_failures:
            raise ValueError(f"invalid failure_mode: {decision.get('failure_mode')!r}")

        answer = decision.get("answer")
        reference = record.get("reference_answer")
        equivalence = None
        equivalence_raw = None
        reference_presence = None
        reference_presence_raw = None
        if verdict == "ANSWERABLE":
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError("ANSWERABLE response has no open answer")
            if not isinstance(reference, str) or not reference.strip():
                raise ValueError("grounding record has no reference answer")
            equivalence, equivalence_raw = self.complete_json(
                equivalence_prompt(answer.strip(), reference.strip()), max_tokens=100
            )
            if not isinstance(equivalence.get("equivalent"), bool):
                raise ValueError("equivalence response has no boolean equivalent field")
        elif verdict == "UNANSWERABLE":
            if not isinstance(reference, str) or not reference.strip():
                raise ValueError("grounding record has no reference answer")
            reference_presence, reference_presence_raw = self.complete_json(
                reference_presence_prompt(description, reference.strip()), max_tokens=100
            )
            if not isinstance(reference_presence.get("present"), bool):
                raise ValueError("reference-presence response has no boolean present field")

        if verdict == "UNCERTAIN":
            label = "UNCERTAIN"
        elif verdict == "ANSWERABLE":
            label = "EVIDENCE_LEAK" if equivalence["equivalent"] else "ANSWERABLE_WRONG"
        elif reference_presence and reference_presence["present"]:
            label = "ANSWER_CORRELATED_ONLY"
        elif failure_mode == "IRRELEVANT":
            label = "IRRELEVANT"
        else:
            label = "INSUFFICIENT_CONTEXT"

        return {
            "label": label,
            "decision": {
                "verdict": verdict,
                "answer": answer.strip() if isinstance(answer, str) else None,
                "failure_mode": failure_mode,
                "reason": str(decision.get("reason", "")).strip(),
            },
            "answer_equivalence": equivalence,
            "reference_answer_presence": reference_presence,
            "output_normalizations": output_normalizations,
            "raw_responses": {
                "answerability": decision_raw,
                "answer_equivalence": equivalence_raw,
                "reference_answer_presence": reference_presence_raw,
            },
        }


def main() -> None:
    args = parse_args()
    endpoints = parse_endpoints(args.endpoint)
    if len(endpoints) != 1:
        raise SystemExit("classification requires exactly one --endpoint")
    classifier = TextLeakClassifier(
        endpoints[0], timeout=args.timeout, max_tokens=args.max_tokens, retries=args.api_retries
    )

    paths = [Path(path) for path in glob.glob(args.groundings)]
    if not paths:
        raise SystemExit(f"no grounding files match {args.groundings!r}")
    # A no-shard pilot file and later sharded full files can coexist.  Apply
    # files by modification time so the newest physical run wins per key.
    groundings: dict[str, dict] = {}
    for path in sorted(paths, key=lambda value: (value.stat().st_mtime, value.name)):
        groundings.update(latest_jsonl_records([path]))
    rows = [row for _, row in sorted(groundings.items()) if row.get("status") == "ok"]

    shard = args.shard
    if shard is None and os.environ.get("SLURM_ARRAY_TASK_ID") is not None:
        shard = int(os.environ["SLURM_ARRAY_TASK_ID"])
    indices = selected_indices(len(rows), shard, args.num_shards, args.limit)
    output = args.output
    if shard is not None:
        output = output.with_name(f"{output.stem}.{shard}{output.suffix}")
    previous = latest_jsonl_records([output]) if output.exists() else {}
    retry_statuses = {part.strip() for part in args.retry_status.split(",") if part.strip()}

    print(
        f"[START] groundings={len(rows)} selected={len(indices)} shard={shard}/{args.num_shards} "
        f"model={endpoints[0].tag} output={output}",
        flush=True,
    )
    written = errors = 0
    for position, index in enumerate(indices, start=1):
        grounding = rows[index]
        key = grounding["key"]
        old = previous.get(key)
        if old and not args.overwrite and old.get("status") not in retry_statuses:
            print(f"[{position}/{len(indices)}] {key} skip status={old.get('status')}", flush=True)
            continue
        started = time.time()
        base = {
            "key": key,
            "item_key": grounding.get("item_key"),
            "video_id": grounding.get("video_id"),
            "qid": grounding.get("qid"),
            "question": grounding.get("question"),
            "question_type": grounding.get("question_type"),
            "reference_answer": grounding.get("reference_answer"),
            "candidate_rank": grounding.get("candidate_rank"),
            "candidate": grounding.get("candidate"),
            "grounding": grounding.get("grounding"),
            "classifier_model_tag": endpoints[0].tag,
            "classifier_model": endpoints[0].model,
            "prompt_version": CLASSIFIER_PROMPT_VERSION,
            "classifier_input_policy": "question_type_question_grounding_no_choices_no_gold_for_first_call",
        }
        try:
            result = classifier.classify(grounding)
            record = {
                **base,
                "status": "ok",
                **result,
                "elapsed_seconds": round(time.time() - started, 3),
            }
        except Exception as exc:
            record = {
                **base,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": round(time.time() - started, 3),
            }
            errors += 1
        append_jsonl(output, record)
        previous[key] = record
        written += 1
        print(
            f"[{position}/{len(indices)}] {key} {record['status']} "
            f"label={record.get('label')} elapsed={record['elapsed_seconds']:.1f}s",
            flush=True,
        )
    print(f"[DONE] wrote={written} errors={errors} output={output}", flush=True)


if __name__ == "__main__":
    main()
