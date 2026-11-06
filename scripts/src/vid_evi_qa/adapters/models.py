"""Supported on-demand open vision model smoke configurations."""

# Source location bootstrap; frozen copies remain standalone.
from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_file = _LayoutPath(__file__).resolve()
_layout_root = (_layout_file.parents[4]
                if _layout_file.as_posix().endswith('/scripts/src/vid_evi_qa/adapters/models.py') else None)
_LAYOUT_LEGACY_FILE = str(_layout_root / 'scripts/eval/open_v2/models.py') if _layout_root else __file__
if _layout_root is not None:
    for _layout_dir in (_layout_root, _layout_root / 'scripts/src',
                        _layout_root / 'scripts/eval',
                        _layout_root / 'scripts/eval/open_v2', _layout_file.parent):
        if str(_layout_dir) not in _layout_sys.path:
            _layout_sys.path.insert(0, str(_layout_dir))

MODELS = {
    'molmo2_o_7b': dict(name='Molmo2-O-7B', length=65536, max_output_tokens=8192, processor={}, backend='transformers',
                       repo='allenai/Molmo2-O-7B', revision='784410650d12be9bc086118fdefa32d2c3bced86'),
    'glm41v_9b': dict(name='GLM-4.1V-9B-Thinking', length=32768, max_output_tokens=8192,
                     processor={'size': {'shortest_edge': 12544, 'longest_edge': 75264}},
                     repo='zai-org/GLM-4.1V-9B-Thinking', revision='3c1471e51dc811b589d4d12b1c1c7c1c941267c2'),
    'minicpm': dict(name='MiniCPM-V-4_5', length=16384, max_output_tokens=4096, processor={'max_slice_nums': 1}),
    'qwen3_8b': dict(name='Qwen3-VL-8B-Instruct', length=32768, max_output_tokens=8192, processor={'max_pixels': 75264},
                    repo='Qwen/Qwen3-VL-8B-Instruct', revision='0c351dd01ed87e9c1b53cbc748cba10e6187ff3b'),
    'qwen25_7b': dict(name='Qwen2.5-VL-7B-Instruct', length=32768, max_output_tokens=8192, processor={'max_pixels': 75264},
                     repo='Qwen/Qwen2.5-VL-7B-Instruct', revision='cc594898137f460bfe9f0759e9844b3ce807cfb5'),
    'internvl_4b': dict(name='InternVL3_5-4B', length=40960, max_output_tokens=8192, processor={'max_dynamic_patch': 3},
                       repo='OpenGVLab/InternVL3_5-4B', revision='481f6e32467eab4e922ccd7fd6cf420441a62331'),
    'llava_ov_7b': dict(name='LLaVA-OneVision-Qwen2-7B-OV', length=32768, max_output_tokens=8192, processor={},
                       media_mode='video', repo='llava-hf/llava-onevision-qwen2-7b-ov-hf',
                       revision='0d50680527681998e456c7b78950205bedd8a068'),
    'llava_next_video_7b': dict(name='LLaVA-NeXT-Video-7B-hf', length=10240, max_output_tokens=4096, processor={},
                               media_mode='video', repo='llava-hf/LLaVA-NeXT-Video-7B-hf',
                               revision='f417d9edeb69a9eaa88283780332d00ce3c66593'),
}
