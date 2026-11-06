"""CPU-only compilation check for the frozen Molmo XGrammar schema."""
import argparse
import json
from pathlib import Path
import sys

parent = Path(__file__).resolve().parents[1]
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
