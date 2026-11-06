"""Native Molmo2 engineering smoke with optional frozen schema constraint."""

# Layout compatibility: data stays at its existing paths; frozen copies remain standalone.
from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_file = _LayoutPath(__file__).resolve()
_layout_root = (_layout_file.parents[3]
                if _layout_file.as_posix().endswith('/src/vid_evi_qa/adapters/evaluate_molmo.py') else None)
_LAYOUT_LEGACY_FILE = str(_layout_root / 'scripts/eval/open_v2/evaluate_molmo.py') if _layout_root else __file__
if _layout_root is not None:
    for _layout_dir in (_layout_root, _layout_root / 'src',
                        _layout_root / 'scripts/eval',
                        _layout_root / 'scripts/eval/open_v2', _layout_file.parent):
        if str(_layout_dir) not in _layout_sys.path:
            _layout_sys.path.insert(0, str(_layout_dir))

import base64
import io
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(_LAYOUT_LEGACY_FILE).resolve().parents[1]/'eval'))
import evaluate
from prompt_v2 import RESPONSE_JSON_SCHEMA

MAX_OUTPUT_TOKENS = int(os.environ.get('V2_MAX_TOKENS', '8192'))


def make_messages(frames_b64, timestamps, instruction):
    from PIL import Image
    if len(frames_b64) != len(timestamps):
        raise ValueError('Each frame must have its original timestamp')
    content = []
    for b64, ts in zip(frames_b64, timestamps):
        content.append({'type':'text', 'text':f'Frame at {ts:.1f}s:'})
        content.append({'type':'image', 'image':Image.open(io.BytesIO(base64.b64decode(b64))).convert('RGB')})
    content.append({'type':'text','text':instruction})
    return [{'role':'user','content':content}]


def main():
    if not os.environ.get('SLURM_JOB_ID'):
        raise SystemExit('GPU inference requires a Slurm allocation')
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    model_path = os.environ['MOLMO_MODEL_PATH']
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(model_path, trust_remote_code=True,
        local_files_only=True, dtype=torch.bfloat16, device_map='cuda:0', attn_implementation='sdpa')
    model.eval()
    constrained = os.environ.get('MOLMO_JSON_CONSTRAINT', '0') == '1'
    compiled_grammar = None
    if constrained:
        import xgrammar as xgr
        from xgrammar.contrib.hf import LogitsProcessor
        tokenizer_info = xgr.TokenizerInfo.from_huggingface(processor.tokenizer,
            vocab_size=model.lm_head.out_features, stop_token_ids=model.generation_config.eos_token_id)
        compiled_grammar = xgr.GrammarCompiler(tokenizer_info).compile_json_schema(
            json.dumps(RESPONSE_JSON_SCHEMA))

    def call_llm(frames_b64, timestamps, question, choices, *, task='qa'):
        messages = make_messages(frames_b64, timestamps,
            evaluate.build_instruction(question, choices, task=task))
        inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
            return_tensors='pt', return_dict=True)
        inputs = {k:v.to(model.device) for k,v in inputs.items()}
        prompt_tokens = inputs['input_ids'].shape[-1]
        if prompt_tokens + MAX_OUTPUT_TOKENS > 65536:
            raise ValueError(f'Input would exceed model context: {prompt_tokens}+{MAX_OUTPUT_TOKENS}')
        with torch.inference_mode():
            # A fresh matcher is required for each example; no grammar state leaks.
            grammar_processors = [LogitsProcessor(compiled_grammar)] if constrained else []
            outputs = model.generate(**inputs, max_new_tokens=MAX_OUTPUT_TOKENS, do_sample=False,
                                     logits_processor=grammar_processors)
        raw = processor.tokenizer.decode(outputs[0, prompt_tokens:], skip_special_tokens=True)
        print('NATIVE_GENERATION', {'input_tokens':prompt_tokens,
            'output_tokens':outputs.shape[-1]-prompt_tokens}, flush=True)
        return raw, {'requested':True, 'available':False, 'tokens':[],
                     'unavailable_reason':'Native adapter does not collect per-token logprobs'}

    evaluate.call_llm = call_llm
    original_process_entry = evaluate.process_entry
    def process_entry(*args, **kwargs):
        record = original_process_entry(*args, **kwargs)
        if record.get('status') == 'ok':
            record['result'].update(response_format='json_schema' if constrained else 'unconstrained_text',
                requested_response_format='json_schema', json_constraint=constrained,
                grammar_backend='xgrammar_json_schema' if constrained else None,
                inference_backend='transformers', evaluation_scope='engineering_smoke')
        return record
    evaluate.process_entry = process_entry
    def native_preflight():
        if model.device.type != 'cuda' or model.training:
            raise RuntimeError('Native model must be loaded on CUDA in eval mode')
        if not callable(getattr(processor, 'apply_chat_template', None)):
            raise RuntimeError('Native processor is not ready')
    evaluate.main(backend_preflight=native_preflight)


if __name__ == '__main__':
    main()
