"""Pinned open-weight frontier VLMs proposed for v2.2 smoke evaluation."""

# Source location bootstrap; frozen copies remain standalone.
from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_file = _LayoutPath(__file__).resolve()
_layout_root = (_layout_file.parents[4]
                if _layout_file.as_posix().endswith('/scripts/src/vid_evi_qa/adapters/frontier_models.py') else None)
_LAYOUT_LEGACY_FILE = str(_layout_root / 'scripts/eval/open_v2/frontier_models.py') if _layout_root else __file__
if _layout_root is not None:
    for _layout_dir in (_layout_root, _layout_root / 'scripts/src',
                        _layout_root / 'scripts/eval',
                        _layout_root / 'scripts/eval/open_v2', _layout_file.parent):
        if str(_layout_dir) not in _layout_sys.path:
            _layout_sys.path.insert(0, str(_layout_dir))


MODELS = {
    "videollama3_7b": {
        "name": "VideoLLaMA3-7B",
        "repo": "DAMO-NLP-SG/VideoLLaMA3-7B",
        "revision": "d5b763e368861e7f5096e7ff1b49f92fbccf8ae6",
        "storage_gib": 14.99,
        "backend": "transformers_native",
        "gpus": 1,
        "min_free_gib": 20,
        "length": 32768,
        "media_mode": "video",
        "max_output_tokens": 8192,
    },
    "qwen36_35b_a3b_fp8": {
        "name": "Qwen3.6-35B-A3B-FP8",
        "repo": "Qwen/Qwen3.6-35B-A3B-FP8",
        "revision": "95a723d08a9490559dae23d0cff1d9466213d989",
        "storage_gib": 34.92,
        "backend": "vllm_0191",
        "gpus": 1,
        "min_free_gib": 45,
        "length": 32768,
        "media_mode": "video",
        "reasoning_parser": "qwen3",
        "request_extra": {"chat_template_kwargs": {"enable_thinking": False}},
        "max_output_tokens": 8192,
    },
    "glm45v_fp8": {
        "name": "GLM-4.5V-FP8",
        "repo": "zai-org/GLM-4.5V-FP8",
        "revision": "3ca028eac7af91c53109dcfa865e5c8da7b1faf3",
        "storage_gib": 102.47,
        "backend": "vllm_0110",
        "gpus": 4,
        "min_free_gib": 40,
        "length": 32768,
        "media_mode": "video",
        "reasoning_parser": "glm45",
        "max_output_tokens": 8192,
    },
    "glm45v_fp8_tp2": {
        "name": "GLM-4.5V-FP8",
        "repo": "zai-org/GLM-4.5V-FP8",
        "revision": "3ca028eac7af91c53109dcfa865e5c8da7b1faf3",
        "storage_gib": 102.47,
        "backend": "vllm_0110",
        "gpus": 2,
        "min_free_gib": 65,
        "length": 32768,
        "media_mode": "video",
        "reasoning_parser": "glm45",
        "max_output_tokens": 8192,
    },
    "glm45v_fp8_tp2_vllm0191": {
        "name": "GLM-4.5V-FP8",
        "repo": "zai-org/GLM-4.5V-FP8",
        "revision": "3ca028eac7af91c53109dcfa865e5c8da7b1faf3",
        "storage_gib": 102.47,
        "backend": "vllm_0191",
        "gpus": 2,
        "min_free_gib": 65,
        "length": 32768,
        "media_mode": "video",
        "reasoning_parser": "glm45",
        "max_output_tokens": 8192,
    },
    "qwen3vl_235b_thinking_fp8": {
        "name": "Qwen3-VL-235B-A22B-Thinking-FP8",
        "repo": "Qwen/Qwen3-VL-235B-A22B-Thinking-FP8",
        "revision": "4208801232c8ef5e6faea2ea1e4066193958776d",
        "storage_gib": 221.30,
        "backend": "vllm_0191",
        "gpus": 4,
        "min_free_gib": 65,
        "length": 32768,
        "media_mode": "video",
        "reasoning_parser": "qwen3",
        "max_output_tokens": 8192,
    },
    "kimi_k25_int4": {
        "name": "Kimi-K2.5",
        "repo": "moonshotai/Kimi-K2.5",
        "revision": "4d01dfe0332d63057c186e0b262165819efb6611",
        "storage_gib": 554.33,
        "backend": "vllm_0191",
        "gpus": 8,
        "min_free_gib": 72,
        "length": 16384,
        "media_mode": "video",
        "reasoning_parser": "kimi_k2",
        "max_output_tokens": 8192,
        "cpu_offload_gb": 16,
        "gpu_memory_utilization": 0.95,
        "mm_encoder_tp_mode": "data",
    },
}
