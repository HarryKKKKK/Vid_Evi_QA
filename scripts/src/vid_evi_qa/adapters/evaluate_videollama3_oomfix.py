"""Memory-bounded native VideoLLaMA3 adapter used by the OOM pilot.

The frozen prompt, 32 selected JPEG frames, and generation settings are kept
unchanged.  ``VIDEOLLAMA3_VISUAL_MAX_TOKENS`` only controls the checkpoint's
internal dynamic resize budget so that its memory/quality trade-off can be
measured explicitly.
"""

# Source location bootstrap; frozen copies remain standalone.
from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_file = _LayoutPath(__file__).resolve()
_layout_root = (_layout_file.parents[4]
                if _layout_file.as_posix().endswith('/scripts/src/vid_evi_qa/adapters/evaluate_videollama3_oomfix.py') else None)
_LAYOUT_LEGACY_FILE = str(_layout_root / 'scripts/eval/open_v2/evaluate_videollama3_oomfix.py') if _layout_root else __file__
if _layout_root is not None:
    for _layout_dir in (_layout_root, _layout_root / 'scripts/src',
                        _layout_root / 'scripts/eval',
                        _layout_root / 'scripts/eval/open_v2', _layout_file.parent):
        if str(_layout_dir) not in _layout_sys.path:
            _layout_sys.path.insert(0, str(_layout_dir))

import base64
import gc
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoProcessor, image_utils

if not hasattr(image_utils, "VideoInput"):
    image_utils.VideoInput = Any

adapter_path = Path(_LAYOUT_LEGACY_FILE).resolve()
eval_candidates = (adapter_path.parents[1], adapter_path.parents[1] / "eval")
eval_dir = next((path for path in eval_candidates if (path / "evaluate.py").is_file()), None)
if eval_dir is None:
    raise RuntimeError(f"Cannot locate evaluate.py from {adapter_path}")
sys.path.insert(0, str(eval_dir))
import evaluate


MODEL_PATH = os.environ["VIDEOLLAMA3_MODEL_PATH"]
MAX_OUTPUT_TOKENS = int(os.environ.get("V2_MAX_TOKENS", "8192"))
VISUAL_MAX_TOKENS = int(os.environ.get("VIDEOLLAMA3_VISUAL_MAX_TOKENS", "16384"))
ATTN_IMPLEMENTATION = os.environ.get("VIDEOLLAMA3_ATTN_IMPLEMENTATION", "sdpa")

processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
if not hasattr(processor, "image_processor"):
    raise RuntimeError("VideoLLaMA3 processor has no image_processor")
processor.image_processor.max_tokens = VISUAL_MAX_TOKENS

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    attn_implementation=ATTN_IMPLEMENTATION,
    low_cpu_mem_usage=True,
).cuda().eval()

print(
    f"VideoLLaMA3 OOM pilot: visual_max_tokens={VISUAL_MAX_TOKENS} "
    f"attention={ATTN_IMPLEMENTATION} max_new_tokens={MAX_OUTPUT_TOKENS}",
    flush=True,
)


def native_call(frames_b64, timestamps, question, choices, *, task="qa"):
    source_times = ", ".join(f"{timestamp:.1f}" for timestamp in timestamps)
    instruction = evaluate.build_instruction(question, choices, task=task)
    instruction = instruction.replace(
        "You are given frames sampled from the video. Each image is preceded by a line \"Frame at <t>s:\" giving its approximate timestamp in seconds from the start of the video.",
        "You are given a video formed from temporally ordered frames sampled from the source video. "
        f"Their source timestamps in order are: [{source_times}] seconds.",
    )
    inputs = output_ids = generated_ids = None
    try:
        with tempfile.TemporaryDirectory(prefix="videollama3_frames_") as temporary:
            directory = Path(temporary)
            for index, encoded in enumerate(frames_b64):
                (directory / f"{index:04d}.jpg").write_bytes(base64.b64decode(encoded))
            conversation = [{
                "role": "user",
                "content": [
                    {"type": "video", "video": {"video_path": str(directory), "max_frames": len(frames_b64)}},
                    {"type": "text", "text": instruction},
                ],
            }]
            inputs = processor(conversation=conversation, return_tensors="pt")
            inputs = {
                key: value.cuda(non_blocking=True) if isinstance(value, torch.Tensor) else value
                for key, value in inputs.items()
            }
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=MAX_OUTPUT_TOKENS,
                    do_sample=False,
                    use_cache=True,
                )
            prompt_length = inputs["input_ids"].shape[1]
            includes_prompt = (
                output_ids.shape[1] > prompt_length
                and torch.equal(output_ids[:, :prompt_length], inputs["input_ids"])
            )
            generated_ids = output_ids[:, prompt_length:] if includes_prompt else output_ids
            text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            return text, {}
    finally:
        # Drop the actual local references before emptying PyTorch's cache.
        # Passing them to a helper would keep these locals alive until return.
        inputs = output_ids = generated_ids = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


evaluate.call_llm_video = native_call
evaluate.main(backend_preflight=lambda: None)
