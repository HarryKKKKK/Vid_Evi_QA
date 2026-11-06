"""GLM thinking diagnostics. Preserve full raw output; parse final answer only."""

# Source location bootstrap; frozen copies remain standalone.
from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_file = _LayoutPath(__file__).resolve()
_layout_root = (_layout_file.parents[4]
                if _layout_file.as_posix().endswith('/scripts/src/vid_evi_qa/adapters/evaluate_glm.py') else None)
_LAYOUT_LEGACY_FILE = str(_layout_root / 'scripts/eval/open_v2/evaluate_glm.py') if _layout_root else __file__
if _layout_root is not None:
    for _layout_dir in (_layout_root, _layout_root / 'scripts/src',
                        _layout_root / 'scripts/eval',
                        _layout_root / 'scripts/eval/open_v2', _layout_file.parent):
        if str(_layout_dir) not in _layout_sys.path:
            _layout_sys.path.insert(0, str(_layout_dir))

import json
import os
from pathlib import Path
import sys
import threading

sys.path.insert(0, str(Path(_LAYOUT_LEGACY_FILE).resolve().parents[1]/'eval'))
import evaluate


def final_answer_text(raw):
    """Remove model transport delimiters only; never repair or infer an answer.

    An unfinished thinking block has no final answer, even if it quotes JSON.
    """
    tail = raw
    if '<think>' in tail:
        if '</think>' not in tail:
            return ''
        tail = tail.split('</think>', 1)[1]
    elif '</think>' in tail:
        tail = tail.split('</think>', 1)[1]
    if '<answer>' in tail:
        tail = tail.split('<answer>', 1)[1]
        # </answer> is a configured GLM EOS token. vLLM may consume it rather
        # than return it; the remaining JSON still has to pass strict parsing.
        tail = tail.split('</answer>', 1)[0]
    # GLM's natural output can include end-of-turn tokens.
    for token in ['<|endoftext|>', '<|user|>', '<|end_of_text|>']:
        tail = tail.split(token, 1)[0]
    return tail.strip()


def main():
    budget = int(os.environ.get('GLM_MAX_TOKENS', '8192'))
    constrained = os.environ.get('GLM_JSON_CONSTRAINT', '0') == '1'
    official_sampling = os.environ.get('GLM_OFFICIAL_SAMPLING', '0') == '1'
    state = threading.local()
    original_create = evaluate._client.chat.completions.create
    def create(*args, **kwargs):
        kwargs['max_tokens'] = budget
        kwargs['timeout'] = 600.0
        if not constrained:
            kwargs.pop('response_format', None)
        if official_sampling:
            kwargs.update(temperature=0.8, top_p=0.6, seed=0)
            kwargs['extra_body'] = {**kwargs.get('extra_body', {}), 'top_k':2}
        response = original_create(*args, **kwargs)
        choice = response.choices[0]
        raw = choice.message.content or ''
        final = final_answer_text(raw)
        state.capture = {'raw':raw, 'final':final, 'finish_reason':choice.finish_reason,
                         'usage':response.usage.model_dump() if response.usage else None}
        # Keep token logprobs untouched; they correspond to the full generated text.
        choice.message.content = final
        return response
    evaluate._client.chat.completions.create = create
    original_process = evaluate.process_entry
    def process_entry(*args, **kwargs):
        state.capture = None
        row = original_process(*args, **kwargs)
        if row.get('status') == 'ok' and state.capture is not None:
            capture = state.capture
            row['result'].update(model_response=capture['raw'], final_answer_text=capture['final'],
                response_format='json_object' if constrained else 'unconstrained_text',
                json_constraint=constrained, max_output_tokens=budget,
                finish_reason=capture['finish_reason'], generation_usage=capture['usage'],
                api_logprobs_scope='full_generated_text_including_reasoning',
                sampling_policy='official_generation_config_seed0' if official_sampling else 'greedy',
                evaluation_scope='engineering_smoke', inference_backend='vllm')
        return row
    evaluate.process_entry = process_entry
    print('GLM_DIAGNOSTIC_POLICY', json.dumps({'max_tokens':budget,'json_constraint':constrained,
          'official_sampling':official_sampling,'prompt_changed':False,'final_only_validation':True}), flush=True)
    evaluate.main()


if __name__ == '__main__':
    main()
