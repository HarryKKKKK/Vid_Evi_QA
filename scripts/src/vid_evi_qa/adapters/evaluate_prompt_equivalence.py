"""Run the current evaluator while changing only the v2.1/v2.2 prompt text.

Both variants use the current JSON schema, parser, validator, sampling, and
transport.  This prevents decoding changes from being confounded with the
prompt-version comparison.
"""

# Source location bootstrap; frozen copies remain standalone.
from pathlib import Path as _LayoutPath
import sys as _layout_sys
_layout_file = _LayoutPath(__file__).resolve()
_layout_root = (_layout_file.parents[4]
                if _layout_file.as_posix().endswith('/scripts/src/vid_evi_qa/adapters/evaluate_prompt_equivalence.py') else None)
_LAYOUT_LEGACY_FILE = str(_layout_root / 'scripts/eval/open_v2/evaluate_prompt_equivalence.py') if _layout_root else __file__
if _layout_root is not None:
    for _layout_dir in (_layout_root, _layout_root / 'scripts/src',
                        _layout_root / 'scripts/eval',
                        _layout_root / 'scripts/eval/open_v2', _layout_file.parent):
        if str(_layout_dir) not in _layout_sys.path:
            _layout_sys.path.insert(0, str(_layout_dir))

import importlib.util
import os
from pathlib import Path
import runpy
import sys
import types


ROOT = Path("/aifs4su/hansirui_2nd/harry/Vid_Evi_QA")
V21_PATH = ROOT / "results/final_filtered_hierarchical_v2/release_20260902_v21/code/prompt_v2.py"
V22_PATH = ROOT / "scripts/eval/prompt_v2.py"
EVALUATOR = ROOT / "scripts/eval/evaluate.py"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


variant = os.environ.get("PROMPT_ABLATION_VARIANT")
if variant not in {"v2.1", "v2.2"}:
    raise SystemExit("PROMPT_ABLATION_VARIANT must be v2.1 or v2.2")

old = load("frozen_prompt_v21", V21_PATH)
current = load("frozen_prompt_v22", V22_PATH)
chosen = old if variant == "v2.1" else current

# Inject the interface expected by evaluate.py. Prompt construction comes from
# the chosen version; all decoding and validation machinery comes from v2.2.
shim = types.ModuleType("prompt_v2")
shim.PROMPT_VERSION = chosen.PROMPT_VERSION
shim.PROMPT_SHA256 = chosen.PROMPT_SHA256
shim.RESPONSE_JSON_SCHEMA = current.RESPONSE_JSON_SCHEMA
shim.build_prompt = chosen.build_prompt
shim.fingerprint = current.fingerprint
shim.response_errors = current.response_errors
shim.record_identity = current.record_identity


def reusable(result, expected_fingerprint):
    return (
        result.get("prompt_version") == shim.PROMPT_VERSION
        and result.get("prompt_sha256") == shim.PROMPT_SHA256
        and result.get("input_fingerprint") == expected_fingerprint
        and not shim.response_errors(result.get("model_parsed"))
    )


shim.reusable = reusable
sys.modules["prompt_v2"] = shim
runpy.run_path(str(EVALUATOR), run_name="__main__")
