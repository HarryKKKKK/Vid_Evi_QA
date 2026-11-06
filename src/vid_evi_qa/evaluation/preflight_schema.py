"""CPU-only compilation check for the frozen Molmo XGrammar schema."""

# Layout compatibility: data stays at its existing paths; frozen copies remain standalone.
from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_file = _LayoutPath(__file__).resolve()
_layout_root = (_layout_file.parents[3]
                if _layout_file.as_posix().endswith('/src/vid_evi_qa/evaluation/preflight_schema.py') else None)
_LAYOUT_LEGACY_FILE = str(_layout_root / 'scripts/eval/open_v2/preflight_schema.py') if _layout_root else __file__
if _layout_root is not None:
    for _layout_dir in (_layout_root, _layout_root / 'src',
                        _layout_root / 'scripts/eval',
                        _layout_root / 'scripts/eval/open_v2', _layout_file.parent):
        if str(_layout_dir) not in _layout_sys.path:
            _layout_sys.path.insert(0, str(_layout_dir))

import argparse
import json
from pathlib import Path
import sys

parent = Path(_LAYOUT_LEGACY_FILE).resolve().parents[1]
sys.path.insert(0, str(parent if (parent/'prompt_v2.py').is_file() else parent/'eval'))
from prompt_v2 import PROMPT_SHA256, PROMPT_VERSION, RESPONSE_JSON_SCHEMA


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=Path)
    args = parser.parse_args()
    import xgrammar as xgr
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(
        args.model_path, trust_remote_code=True, local_files_only=True)
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(processor.tokenizer)
    xgr.GrammarCompiler(tokenizer_info).compile_json_schema(
        json.dumps(RESPONSE_JSON_SCHEMA))
    print(json.dumps({'status': 'passed', 'prompt_version': PROMPT_VERSION,
                      'prompt_sha256': PROMPT_SHA256}))


if __name__ == '__main__':
    main()
