"""Use the frozen v2 evaluator with model-specific chat-template settings only."""

# Layout compatibility: data stays at its existing paths; frozen copies remain standalone.
from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_file = _LayoutPath(__file__).resolve()
_layout_root = (_layout_file.parents[3]
                if _layout_file.as_posix().endswith('/src/vid_evi_qa/adapters/evaluate_open.py') else None)
_LAYOUT_LEGACY_FILE = str(_layout_root / 'scripts/eval/open_v2/evaluate_open.py') if _layout_root else __file__
if _layout_root is not None:
    for _layout_dir in (_layout_root, _layout_root / 'src',
                        _layout_root / 'scripts/eval',
                        _layout_root / 'scripts/eval/open_v2', _layout_file.parent):
        if str(_layout_dir) not in _layout_sys.path:
            _layout_sys.path.insert(0, str(_layout_dir))

import json
import hashlib
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(_LAYOUT_LEGACY_FILE).resolve().parents[1]/'eval'))
import evaluate


def add_position_aware_image_ids(messages):
    """MiniCPM image-ID tokens depend on position; identical pixels can differ in
    embedding layout. Prevent vLLM 0.11 from aliasing positions in frozen clips.
    This changes cache metadata only, never the image bytes or prompt text.
    """
    index = 0
    for message in messages:
        content = message.get('content', [])
        if not isinstance(content, list):
            continue
        for item in content:
            if item.get('type') == 'image_url':
                url = item['image_url']['url']
                item['uuid'] = hashlib.sha256((str(index)+':'+url).encode()).hexdigest()
                index += 1


def main():
    extra = json.loads(os.environ.get('V2_REQUEST_EXTRA', '{}'))
    original = evaluate._client.chat.completions.create
    def create(*args, **kwargs):
        if os.environ.get('V2_POSITION_IMAGE_IDS') == '1':
            add_position_aware_image_ids(kwargs.get('messages', []))
        if extra:
            kwargs['extra_body'] = {**kwargs.get('extra_body', {}), **extra}
        if os.environ.get('V2_MAX_TOKENS'):
            kwargs['max_tokens'] = int(os.environ['V2_MAX_TOKENS'])
        return original(*args, **kwargs)
    evaluate._client.chat.completions.create = create
    print('MODEL_SPECIFIC_REQUEST_SETTINGS', json.dumps({
        'extra_body': extra,
        'max_tokens': int(os.environ['V2_MAX_TOKENS']) if os.environ.get('V2_MAX_TOKENS') else None,
    }), flush=True)
    evaluate.main()


if __name__ == '__main__':
    main()
