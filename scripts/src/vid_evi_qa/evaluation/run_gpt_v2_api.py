#!/usr/bin/env python3
"""Run the frozen v2 video-QA evaluator against an OpenAI-compatible API."""

from __future__ import annotations

# Source location bootstrap; frozen copies remain standalone.
from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_file = _LayoutPath(__file__).resolve()
_layout_root = (_layout_file.parents[4]
                if _layout_file.as_posix().endswith('/scripts/src/vid_evi_qa/evaluation/run_gpt_v2_api.py') else None)
_LAYOUT_LEGACY_FILE = str(_layout_root / 'scripts/eval/run_gpt_v2_api.py') if _layout_root else __file__
if _layout_root is not None:
    for _layout_dir in (_layout_root, _layout_root / 'scripts/src',
                        _layout_root / 'scripts/eval',
                        _layout_root / 'scripts/eval', _layout_file.parent):
        if str(_layout_dir) not in _layout_sys.path:
            _layout_sys.path.insert(0, str(_layout_dir))


import importlib.util
import json
import os
from pathlib import Path
import sys

from openai import OpenAI


ROOT = Path("/aifs4su/hansirui_2nd/harry/Vid_Evi_QA")
V2_PROMPT = ROOT / "results/open_v2_deployment_20260902_r4/eval/prompt_v2.py"
EVALUATOR = ROOT / "results/frontier_v22_full_20260907_r2/eval/evaluate.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    key_file = Path(os.environ.get(
        "XHUB_KEY_FILE", Path.home() / ".config/vid_evi_qa/xhub_api_key"
    ))
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key and key_file.is_file():
        api_key = key_file.read_text(encoding="utf-8").strip()
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is required")

    prompt_v2 = load_module("prompt_v2", V2_PROMPT)
    # The newer evaluator imports this symbol at module load time. The GPT
    # transport below deliberately requests JSON-object mode because the
    # original frozen v2 prompt predates the later strict JSON schema.
    prompt_v2.RESPONSE_JSON_SCHEMA = {}
    evaluate = load_module("evaluate_gpt_v2", EVALUATOR)

    base_url = os.environ.get("OPENAI_BASE_URL", "https://api3.xhub.chat/v1")
    model = os.environ.get("OPENAI_MODEL", "gpt-4o")
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=300.0, max_retries=0)
    evaluate._client = client
    evaluate.VLLM_BASE_URL = base_url
    evaluate.MODEL_NAME = model

    def call_llm(frames_b64, timestamps, question, choices, *, task="qa"):
        content = []
        for encoded, second in zip(frames_b64, timestamps):
            content.append({"type": "text", "text": f"Frame at {second:.1f}s:"})
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
            })
        content.append({
            "type": "text",
            "text": prompt_v2.build_prompt(question, choices),
        })
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            max_tokens=int(os.environ.get("GPT_MAX_TOKENS", "768")),
            temperature=0,
            response_format={"type": "json_object"},
        )
        choice = response.choices[0]
        usage = response.usage
        metadata = {
            "requested": False,
            "available": False,
            "tokens": [],
            "finish_reason": choice.finish_reason,
            "input_tokens": getattr(usage, "prompt_tokens", None),
            "output_tokens": getattr(usage, "completion_tokens", None),
        }
        return choice.message.content or "", metadata

    evaluate.call_llm = call_llm
    print(json.dumps({
        "backend": "openai_compatible_chat_completions",
        "base_url": base_url,
        "model": model,
        "prompt_version": prompt_v2.PROMPT_VERSION,
        "prompt_sha256": prompt_v2.PROMPT_SHA256,
        "media_mode": "frames",
    }), flush=True)
    evaluate.main()


if __name__ == "__main__":
    main()
