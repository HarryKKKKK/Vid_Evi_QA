#!/usr/bin/env python3
"""Classify and aggregate independently grounded NExT-GQA windows.

The first call sees only question type, question, and grounding description. It
must decide whether the complete question is answerable and provide an open
answer without seeing choices or gold.  Only after that answer is frozen does a
separate text-only call compares it with the reference answer. When multiple
grounding files contain the same candidate key, every distinct grounding model
is judged independently and a deterministic support threshold produces the
final label. A single dissenting or hallucinating grounder cannot confirm a leak.
"""

from __future__ import annotations

# Source location bootstrap; frozen copies remain standalone.
from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_file = _LayoutPath(__file__).resolve()
_layout_root = (_layout_file.parents[3]
                if _layout_file.as_posix().endswith('/pipelines/nextgqa/insufficient/classify_groundings.py') else None)
_LAYOUT_LEGACY_FILE = str(_layout_root / 'pipelines/nextgqa/insufficient/classify_groundings.py') if _layout_root else __file__
if _layout_root is not None:
    for _layout_dir in (_layout_root, _layout_root / 'scripts/src',
                        _layout_root / 'scripts/eval',
                        _layout_root / 'pipelines/nextgqa/insufficient', _layout_file.parent):
        if str(_layout_dir) not in _layout_sys.path:
            _layout_sys.path.insert(0, str(_layout_dir))


import argparse
import glob
import json
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

from common import append_jsonl, latest_jsonl_records, parse_endpoints, selected_indices


ROOT = Path(_LAYOUT_LEGACY_FILE).resolve().parents[3]
DEFAULT_GROUNDINGS = str(ROOT / "results/build/nextgqa/insufficient/window_groundings*.jsonl")
DEFAULT_OUTPUT = ROOT / "results/build/nextgqa/insufficient/leak_classifications.jsonl"
CLASSIFIER_PROMPT_VERSION = "nextgqa-text-leak-classifier-ensemble-v2.0"


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
    parser.add_argument(
        "--min-grounder-support",
        type=int,
        default=1,
        help=(
            "Distinct grounding models required to confirm EVIDENCE_LEAK. "
            "Legacy default is 1; the ensemble launcher explicitly uses 2."
        ),
    )
    parser.add_argument(
        "--expected-grounders",
        type=int,
        default=0,
        help="Expected distinct grounders per candidate; 0 disables completeness checking.",
    )
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Request a direct JSON answer from reasoning-capable chat templates.",
    )
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
        # Some OpenAI-compatible reasoning servers emit literal newlines or
        # tabs inside otherwise valid JSON strings. Python's non-strict mode
        # accepts those control characters without changing the JSON schema.
        try:
            value = json.loads(stripped, strict=False)
        except json.JSONDecodeError:
            value = None
    if value is None:
        start, end = stripped.find("{"), stripped.rfind("}")
        if start >= 0 and end > start:
            try:
                value = json.loads(stripped[start:end + 1])
            except json.JSONDecodeError:
                try:
                    value = json.loads(stripped[start:end + 1], strict=False)
                except json.JSONDecodeError:
                    value = None
        if value is None:
            # A server can stop after all decision fields but in the middle of
            # the final free-text reason. Recover only schema-known scalar
            # fields; the caller still validates every required field/type.
            value = {}
            for field in ("verdict", "answer", "failure_mode", "reason"):
                match = re.search(rf'"{field}"\s*:\s*"((?:\\.|[^"\\])*)', stripped)
                if match:
                    value[field] = bytes(match.group(1), "utf-8").decode("unicode_escape")
                elif re.search(rf'"{field}"\s*:\s*null\b', stripped):
                    value[field] = None
            for field in ("equivalent", "present"):
                match = re.search(rf'"{field}"\s*:\s*(true|false)\b', stripped, re.I)
                if match:
                    value[field] = match.group(1).lower() == "true"
            if not value:
                raise ValueError(f"response contains no recoverable JSON fields: {text[:300]!r}")
    if not isinstance(value, dict):
        raise ValueError("response JSON is not an object")
    return value


def grounding_evidence_text(grounding: dict) -> str:
    """Render structured grounding without discarding its legacy summary."""
    payload = {
        "summary": str(grounding.get("description", "")).strip(),
        "observations": grounding.get("observations", []),
        "missing_or_ambiguous": grounding.get("missing_or_ambiguous", []),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def analysis_prompt(question_type: str, question: str, evidence: str) -> str:
    rule = TYPE_RULES.get(
        question_type,
        "Every entity, event, condition, and relation required by the complete question must be supported.",
    )
    return f"""You are auditing whether a grounded video-window description contains sufficient visual evidence for a complete NExT-GQA question.

Use only facts stated in the grounding description. Do not assume an unseen event, fill missing context with common sense, or use knowledge of multiple-choice datasets. A description may contain an action that looks like a plausible answer while omitting the condition, anchor event, temporal relation, causal relation, or manner required by the question. In that case the question is UNANSWERABLE, not ANSWERABLE.

Question type: {question_type}
Type-specific requirement: {rule}
Question: {question}

Grounding evidence (summary plus atomic timestamped observations):
{evidence}

Treat DIRECT observations as primary evidence. AMBIGUOUS observations may
support a broader description but not an unsupported fine-grained claim.
INFERRED observations alone cannot establish a leak. A directly visible named
tool, contact target, conventional action phase, or temporal relation may entail
an ordinary-language answer even if the wording differs.

First decide whether the description supports every condition needed to answer the complete question. If yes, give a short open-ended answer. If not, do not guess an answer.

Return exactly one JSON object with these fields:
- verdict: "ANSWERABLE", "UNANSWERABLE", or "UNCERTAIN"
- answer: a short open answer when ANSWERABLE, otherwise null
- failure_mode: "NONE", "ANSWER_CORRELATED_ONLY", "MISSING_REQUIRED_CONTEXT", "IRRELEVANT", or "UNCERTAIN"
- reason: one concise sentence grounded in the description

Use ANSWER_CORRELATED_ONLY when the description contains a plausible answer-like action, object, or event but does not establish the question's required anchor, condition, or relation."""


def equivalence_prompt(candidate_answer: str, reference_answer: str) -> str:
    return f"""Decide whether a candidate answer semantically entails the reference answer for video question answering. This is meaning-level entailment, not exact-string similarity.

Accept harmless paraphrases, synonyms, inflection changes, more-specific answers that contain the reference fact, and ordinary visual granularity differences that do not change the answer. Examples: touching the back of a boy's head can support touching his hair when hair is visibly at the contact region; reeling a fish while pulling a fishing rod supports catching it with a rod; performing a golf-club swing supports playing golf. Reject answers that merely mention a related object/action, contradict the reference, identify the wrong participant, or omit a relation essential to the answer.

Candidate answer: {candidate_answer}
Reference answer: {reference_answer}

Return exactly one JSON object:
{{"equivalent": true or false, "reason": "one concise sentence"}}"""


def reference_presence_prompt(description: str, reference_answer: str) -> str:
    return f"""Check only whether grounded visual evidence semantically contains the action, object, state, or event expressed by a reference answer. Do not decide whether the complete original question is answerable, and do not require its missing anchor or temporal/causal condition. Accept a more-specific visible fact that entails the reference and harmless visual granularity variants. Reject merely associated objects/actions and wrong participants or relations.

Grounding description:
{description}

Reference answer:
{reference_answer}

Return exactly one JSON object:
{{"present": true or false, "reason": "one concise sentence"}}"""


class TextLeakClassifier:
    def __init__(
        self,
        endpoint,
        *,
        timeout: float,
        max_tokens: int,
        retries: int,
        disable_thinking: bool = False,
    ):
        from openai import OpenAI

        self.endpoint = endpoint
        self.max_tokens = max_tokens
        self.retries = max(1, retries)
        self.disable_thinking = disable_thinking
        self.client = OpenAI(
            base_url=endpoint.base_url,
            api_key=endpoint.api_key,
            timeout=timeout,
        )

    def complete_json(self, prompt: str, *, max_tokens: int | None = None) -> tuple[dict, str]:
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                request_kwargs = {}
                if self.disable_thinking:
                    request_kwargs["extra_body"] = {
                        "chat_template_kwargs": {"enable_thinking": False}
                    }
                    request_kwargs["response_format"] = {"type": "json_object"}
                response = self.client.chat.completions.create(
                    model=self.endpoint.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=max_tokens or self.max_tokens,
                    **request_kwargs,
                )
                raw = (response.choices[0].message.content or "").strip()
                return extract_json(raw), raw
            except Exception as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(min(2 ** (attempt - 1), 8))
        assert last_error is not None
        raise last_error

    def classify(self, record: dict, grounding: dict | None = None) -> dict:
        grounding = grounding if grounding is not None else record.get("grounding", {})
        description = str(grounding.get("description", "")).strip()
        if not description:
            raise ValueError("grounding record has no description")
        evidence = grounding_evidence_text(grounding)
        question_type = str(record.get("question_type", ""))
        decision, decision_raw = self.complete_json(
            analysis_prompt(question_type, str(record.get("question", "")), evidence)
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
                reference_presence_prompt(evidence, reference.strip()), max_tokens=100
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


def grounding_model_tag(record: dict) -> str:
    grounding = record.get("grounding") or {}
    return str(grounding.get("model_tag") or record.get("grounding_model_tag") or "legacy_unknown")


def load_grounding_groups(paths: list[Path]) -> dict[str, list[dict]]:
    """Keep the newest record per (candidate key, grounding model)."""
    latest: dict[tuple[str, str], dict] = {}
    for path in sorted(paths, key=lambda value: (value.stat().st_mtime, value.name)):
        for row in latest_jsonl_records([path]).values():
            if row.get("status") != "ok":
                continue
            latest[(str(row.get("key")), grounding_model_tag(row))] = row
    grouped: defaultdict[str, list[dict]] = defaultdict(list)
    for (key, _), row in latest.items():
        grouped[key].append(row)
    return {
        key: sorted(rows, key=grounding_model_tag)
        for key, rows in sorted(grouped.items())
    }


def aggregate_grounder_results(results: list[dict], min_support: int) -> tuple[str, dict]:
    """Aggregate independent semantic decisions without text-majority shortcuts."""
    if min_support < 1:
        raise ValueError("min_support must be positive")
    labels = Counter(str(result.get("label")) for result in results)
    leak_support = labels["EVIDENCE_LEAK"]
    if leak_support >= min_support:
        label = "EVIDENCE_LEAK"
    elif leak_support > 0 or labels["UNCERTAIN"] > 0:
        label = "UNCERTAIN"
    elif labels["ANSWER_CORRELATED_ONLY"] > 0:
        label = "ANSWER_CORRELATED_ONLY"
    elif labels["ANSWERABLE_WRONG"] > 0:
        label = "ANSWERABLE_WRONG"
    elif labels["IRRELEVANT"] == len(results):
        label = "IRRELEVANT"
    else:
        label = "INSUFFICIENT_CONTEXT"
    return label, {
        "policy": "independent_grounder_semantic_entailment_vote_v1",
        "distinct_grounders": len(results),
        "min_grounder_support": min_support,
        "leak_support": leak_support,
        "label_counts": dict(sorted(labels.items())),
        "disputed": 0 < leak_support < min_support,
    }


def main() -> None:
    args = parse_args()
    endpoints = parse_endpoints(args.endpoint)
    if len(endpoints) != 1:
        raise SystemExit("classification requires exactly one --endpoint")
    classifier = TextLeakClassifier(
        endpoints[0],
        timeout=args.timeout,
        max_tokens=args.max_tokens,
        retries=args.api_retries,
        disable_thinking=args.disable_thinking,
    )

    paths = [Path(path) for path in glob.glob(args.groundings)]
    if not paths:
        raise SystemExit(f"no grounding files match {args.groundings!r}")
    if args.min_grounder_support < 1:
        raise SystemExit("--min-grounder-support must be positive")
    if args.expected_grounders < 0:
        raise SystemExit("--expected-grounders must be non-negative")
    if args.expected_grounders and args.min_grounder_support > args.expected_grounders:
        raise SystemExit("--min-grounder-support cannot exceed --expected-grounders")
    grounding_groups = load_grounding_groups(paths)
    rows = list(grounding_groups.items())

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
        f"[START] candidates={len(rows)} source_records={sum(len(group) for _, group in rows)} "
        f"selected={len(indices)} shard={shard}/{args.num_shards} model={endpoints[0].tag} "
        f"min_grounder_support={args.min_grounder_support} output={output}",
        flush=True,
    )
    written = errors = 0
    for position, index in enumerate(indices, start=1):
        key, grounding_group = rows[index]
        grounding = grounding_group[0]
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
            # Keep the first field for legacy readers; new readers should use
            # groundings and per_grounder_results.
            "grounding": grounding.get("grounding"),
            "groundings": [row.get("grounding") for row in grounding_group],
            # Optional provenance used by audits that feed downstream-model
            # claimed evidence windows into this classifier. These fields are
            # absent for the original calibrated/dense discovery routes.
            "source_eval_model_tag": grounding.get("source_eval_model_tag"),
            "source_result_file": grounding.get("source_result_file"),
            "source_model_claim": grounding.get("source_model_claim"),
            "insufficient_video": grounding.get("insufficient_video"),
            "classifier_model_tag": endpoints[0].tag,
            "classifier_model": endpoints[0].model,
            "prompt_version": CLASSIFIER_PROMPT_VERSION,
            "classifier_input_policy": (
                "independent_per_grounder_question_type_question_structured_grounding_"
                "no_choices_no_gold_then_semantic_entailment_vote"
            ),
        }
        try:
            per_grounder_results = []
            grounder_errors = []
            for source in grounding_group:
                model_tag = grounding_model_tag(source)
                try:
                    result = classifier.classify(source)
                    per_grounder_results.append({
                        "grounding_model_tag": model_tag,
                        "grounding": source.get("grounding"),
                        **result,
                    })
                except Exception as exc:
                    grounder_errors.append({
                        "grounding_model_tag": model_tag,
                        "error": f"{type(exc).__name__}: {exc}",
                    })
            if not per_grounder_results:
                raise ValueError(f"all grounding judgments failed: {grounder_errors}")
            label, consensus = aggregate_grounder_results(
                per_grounder_results, args.min_grounder_support
            )
            consensus["expected_grounders"] = args.expected_grounders or None
            consensus["complete_grounder_set"] = (
                not args.expected_grounders
                or len(per_grounder_results) >= args.expected_grounders
            )
            if not consensus["complete_grounder_set"]:
                label = "UNCERTAIN"
                consensus["disputed"] = True
            representative = next(
                (row for row in per_grounder_results if row.get("label") == "EVIDENCE_LEAK"),
                per_grounder_results[0],
            )
            record = {
                **base,
                "status": "ok",
                "label": label,
                "consensus": consensus,
                "per_grounder_results": per_grounder_results,
                "grounder_errors": grounder_errors,
                "decision": representative.get("decision"),
                "answer_equivalence": representative.get("answer_equivalence"),
                "reference_answer_presence": representative.get("reference_answer_presence"),
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
